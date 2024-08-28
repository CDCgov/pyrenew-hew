# numpydoc ignore=GL08
import json

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from pyrenew.arrayutils import repeat_until_n, tile_until_n
from pyrenew.latent import (
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.metaclass import Model
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.randomvariable import DistributionalVariable


class hosp_only_ww_model(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        state_pop,
        i0_over_n_rv,
        initialization_rate_rv,
        log_r_mu_intercept_rv,
        autoreg_rt_rv,  # ar process
        eta_sd_rv,  # sd of random walk for ar process
        generation_interval_pmf_rv,
        infection_feedback_strength_rv,
        infection_feedback_pmf_rv,
        p_hosp_mean_rv,
        p_hosp_w_sd_rv,
        autoreg_p_hosp_rv,
        hosp_wday_effect_rv,
        inf_to_hosp_rv,
        phi_rv,
        n_initialization_points,
        i0_t_offset,
    ):  # numpydoc ignore=GL08
        self.infection_initialization_process = InfectionInitializationProcess(
            "I0_initialization",
            i0_over_n_rv,
            InitializeInfectionsExponentialGrowth(
                n_initialization_points,
                initialization_rate_rv,
                t_pre_init=i0_t_offset,
            ),
            t_unit=1,
        )

        self.inf_with_feedback_proc = InfectionsWithFeedback(
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        )

        self.p_hosp_ar_proc = ARProcess("p_hosp")
        self.ar_diff = DifferencedProcess(
            fundamental_process=ARProcess(
                noise_rv_name="rtu_weekly_diff_first_diff_ar_process_noise"
            ),
            differencing_order=1,
        )

        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.log_r_mu_intercept_rv = log_r_mu_intercept_rv
        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.infection_feedback_pmf_rv = infection_feedback_pmf_rv
        self.p_hosp_mean_rv = p_hosp_mean_rv
        self.p_hosp_w_sd_rv = p_hosp_w_sd_rv
        self.autoreg_p_hosp_rv = autoreg_p_hosp_rv
        self.hosp_wday_effect_rv = hosp_wday_effect_rv
        self.inf_to_hosp_rv = inf_to_hosp_rv
        self.phi_rv = phi_rv
        self.state_pop = state_pop
        self.n_initialization_points = n_initialization_points
        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(
        self, n_datapoints=None, data_observed_hospital_admissions=None
    ):  # numpydoc ignore=GL08
        if n_datapoints is None and data_observed_hospital_admissions is None:
            raise ValueError(
                "Either n_datapoints or data_observed_hosp_admissions "
                "must be passed."
            )
        elif (
            n_datapoints is not None
            and data_observed_hospital_admissions is not None
        ):
            raise ValueError(
                "Cannot pass both n_datapoints and data_observed_hospital_admissions."
            )
        elif n_datapoints is None:
            n_datapoints = len(data_observed_hospital_admissions)
        else:
            n_datapoints = n_datapoints

        n_weeks_post_init = n_datapoints // 7 + 1
        i0 = self.infection_initialization_process()[0].value

        eta_sd = self.eta_sd_rv()[0].value
        autoreg_rt = self.autoreg_rt_rv()[0].value
        log_r_mu_intercept = self.log_r_mu_intercept_rv()[0].value
        init_rate_of_change_rv = DistributionalVariable(
            "init_rate_of_change",
            dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
        )
        init_rate_of_change = init_rate_of_change_rv()[0].value

        log_rtu_weekly = self.ar_diff(
            n=n_weeks_post_init,
            init_vals=jnp.array(log_r_mu_intercept),
            autoreg=jnp.array(autoreg_rt),
            noise_sd=jnp.array(eta_sd),
            fundamental_process_init_vals=jnp.array(init_rate_of_change),
        )[0].value

        rtu = repeat_until_n(
            data=jnp.exp(log_rtu_weekly),
            n_timepoints=n_datapoints,
            offset=0,
            period_size=7,
        )

        generation_interval_pmf = self.generation_interval_pmf_rv()[0].value

        inf_with_feedback_proc_sample = self.inf_with_feedback_proc(
            Rt=rtu,
            I0=i0,
            gen_int=generation_interval_pmf,
        )

        latent_infections = jnp.concat(
            [
                i0,
                inf_with_feedback_proc_sample.post_initialization_infections.value,
            ]
        )
        numpyro.deterministic("rtu", rtu)
        numpyro.deterministic("rt", inf_with_feedback_proc_sample.Rt.value)
        numpyro.deterministic("latent_infections", latent_infections)

        p_hosp_mean = self.p_hosp_mean_rv()[0].value
        p_hosp_w_sd = self.p_hosp_w_sd_rv()[0].value
        autoreg_p_hosp = self.autoreg_p_hosp_rv()[0].value

        p_hosp_ar_init_rv = DistributionalVariable(
            "p_hosp_ar_init",
            dist.Normal(
                0,
                p_hosp_w_sd / jnp.sqrt(1 - jnp.pow(autoreg_p_hosp, 2)),
            ),
        )
        p_hosp_ar_init = p_hosp_ar_init_rv()[0].value

        p_hosp_ar = self.p_hosp_ar_proc(
            n=n_weeks_post_init,
            autoreg=autoreg_p_hosp,
            init_vals=p_hosp_ar_init,
            noise_sd=p_hosp_w_sd,
        )[0].value

        ihr = jnp.repeat(
            transformation.SigmoidTransform()(p_hosp_ar + p_hosp_mean),
            repeats=7,
        )[
            :n_datapoints
        ]  # this is only applied after the hospital_admissions are generated, not to all the latent infections. This is why we cannot apply the ihr in compute_delay_ascertained_incidence

        numpyro.deterministic("ihr", ihr)

        hosp_wday_effect_raw = self.hosp_wday_effect_rv()[0].value
        hosp_wday_effect = tile_until_n(hosp_wday_effect_raw, n_datapoints)

        inf_to_hosp = self.inf_to_hosp_rv()[0].value

        potential_latent_hospital_admissions = jnp.convolve(
            latent_infections,
            inf_to_hosp,
            mode="valid",
        )[-n_datapoints:]

        latent_hospital_admissions = (
            potential_latent_hospital_admissions
            * ihr
            * hosp_wday_effect
            * self.state_pop
        )

        hospital_admission_obs_rv = NegativeBinomialObservation(
            "observed_hospital_admissions", concentration_rv=self.phi_rv
        )

        observed_hospital_admissions = hospital_admission_obs_rv(
            mu=latent_hospital_admissions,
            obs=data_observed_hospital_admissions,
        )

        # These are returned only for debugging purposes
        # We should record more deterministic variables for plotting and diagnostics
        return (
            i0,
            rtu,
            inf_with_feedback_proc_sample,
            p_hosp_ar,
            ihr,
            hosp_wday_effect,
            potential_latent_hospital_admissions,
            latent_hospital_admissions,
            observed_hospital_admissions,
        )


def fit_hosp_only_ww_model_from_stan_data(stan_data_file):
    with open(
        stan_data_file,
        "r",
    ) as file:
        stan_data = json.load(file)
    return stan_data
