# numpydoc ignore=GL08
import json

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from pyrenew.arrayutils import repeat_until_n, tile_until_n
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.metaclass import Model
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable


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
        rt_init_rate_of_change_rv = DistributionalVariable(
            "rt_init_rate_of_change",
            dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
        )
        rt_init_rate_of_change = rt_init_rate_of_change_rv()[0].value

        log_rtu_weekly = self.ar_diff(
            n=n_weeks_post_init,
            init_vals=jnp.array(log_r_mu_intercept),
            autoreg=jnp.array(autoreg_rt),
            noise_sd=jnp.array(eta_sd),
            fundamental_process_init_vals=jnp.array(rt_init_rate_of_change),
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
        numpyro.deterministic("rt", inf_with_feedback_proc_sample.rt.value)
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
        )[:n_datapoints]
        # this is only applied after the hospital_admissions are generated, not to all the latent infections. This is why we cannot apply the ihr in compute_delay_ascertained_incidence
        # see https://github.com/CDCgov/ww-inference-model/issues/43

        numpyro.deterministic("ihr", ihr)

        hosp_wday_effect_raw = self.hosp_wday_effect_rv()[0].value
        hosp_wday_effect = tile_until_n(hosp_wday_effect_raw, n_datapoints)

        inf_to_hosp = self.inf_to_hosp_rv()[0].value

        potential_latent_hospital_admissions = (
            compute_delay_ascertained_incidence(
                p_observed_given_incident=1,
                latent_incidence=latent_infections,
                delay_incidence_to_observation_pmf=inf_to_hosp,
            )[-n_datapoints:]
        )

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

        return observed_hospital_admissions


def create_hosp_only_ww_model_from_stan_data(stan_data_file):
    with open(
        stan_data_file,
        "r",
    ) as file:
        stan_data = json.load(file)

    def convert_to_logmean_log_sd(mean, sd):
        logmean = np.log(
            np.power(mean, 2) / np.sqrt(np.power(sd, 2) + np.power(mean, 2))
        )
        logsd = np.sqrt(np.log(1 + (np.power(sd, 2) / np.power(mean, 2))))
        return logmean, logsd

    i0_over_n_prior_a = stan_data["i0_over_n_prior_a"]
    i0_over_n_prior_b = stan_data["i0_over_n_prior_b"]
    i0_over_n_rv = DistributionalVariable(
        "i0_over_n_rv", dist.Beta(i0_over_n_prior_a, i0_over_n_prior_b)
    )

    initial_growth_prior_mean = stan_data["initial_growth_prior_mean"]
    initial_growth_prior_sd = stan_data["initial_growth_prior_sd"]
    initialization_rate_rv = DistributionalVariable(
        "rate",
        dist.TruncatedNormal(
            loc=initial_growth_prior_mean,
            scale=initial_growth_prior_sd,
            low=-1,
            high=1,
        ),
    )
    # could reasonably switch to non-Truncated

    r_prior_mean = stan_data["r_prior_mean"]
    r_prior_sd = stan_data["r_prior_sd"]
    r_logmean, r_logsd = convert_to_logmean_log_sd(r_prior_mean, r_prior_sd)
    log_r_mu_intercept_rv = DistributionalVariable(
        "log_r_mu_intercept_rv", dist.Normal(r_logmean, r_logsd)
    )

    eta_sd_sd = stan_data["eta_sd_sd"]
    eta_sd_rv = DistributionalVariable(
        "eta_sd", dist.TruncatedNormal(0, eta_sd_sd, low=0)
    )

    autoreg_rt_a = stan_data["autoreg_rt_a"]
    autoreg_rt_b = stan_data["autoreg_rt_b"]
    autoreg_rt_rv = DistributionalVariable(
        "autoreg_rt", dist.Beta(autoreg_rt_a, autoreg_rt_b)
    )

    generation_interval_pmf_rv = DeterministicVariable(
        "generation_interval_pmf", jnp.array(stan_data["generation_interval"])
    )

    infection_feedback_pmf_rv = DeterministicVariable(
        "infection_feedback_pmf",
        jnp.array(stan_data["infection_feedback_pmf"]),
    )

    inf_feedback_prior_logmean = stan_data["inf_feedback_prior_logmean"]
    inf_feedback_prior_logsd = stan_data["inf_feedback_prior_logsd"]
    inf_feedback_strength_rv = TransformedVariable(
        "inf_feedback",
        DistributionalVariable(
            "inf_feedback_raw",
            dist.LogNormal(
                inf_feedback_prior_logmean, inf_feedback_prior_logsd
            ),
        ),
        transforms=transformation.AffineTransform(loc=0, scale=-1),
    )
    # Could be reparameterized?

    p_hosp_prior_mean = stan_data["p_hosp_prior_mean"]
    p_hosp_sd_logit = stan_data["p_hosp_sd_logit"]

    p_hosp_mean_rv = DistributionalVariable(
        "p_hosp_mean",
        dist.Normal(
            transformation.SigmoidTransform().inv(p_hosp_prior_mean),
            p_hosp_sd_logit,
        ),
    )  # logit scale

    p_hosp_w_sd_sd = stan_data["p_hosp_w_sd_sd"]
    p_hosp_w_sd_rv = DistributionalVariable(
        "p_hosp_w_sd_sd", dist.TruncatedNormal(0, p_hosp_w_sd_sd, low=0)
    )

    autoreg_p_hosp_a = stan_data["autoreg_p_hosp_a"]
    autoreg_p_hosp_b = stan_data["autoreg_p_hosp_b"]
    autoreg_p_hosp_rv = DistributionalVariable(
        "autoreg_p_hosp", dist.Beta(autoreg_p_hosp_a, autoreg_p_hosp_b)
    )

    hosp_wday_effect_rv = TransformedVariable(
        "hosp_wday_effect",
        DistributionalVariable(
            "hosp_wday_effect_raw",
            dist.Dirichlet(
                jnp.array(stan_data["hosp_wday_effect_prior_alpha"])
            ),
        ),
        transformation.AffineTransform(loc=0, scale=7),
    )

    inf_to_hosp_rv = DeterministicVariable(
        "inf_to_hosp", jnp.array(stan_data["inf_to_hosp"])
    )

    inv_sqrt_phi_prior_mean = stan_data["inv_sqrt_phi_prior_mean"]
    inv_sqrt_phi_prior_sd = stan_data["inv_sqrt_phi_prior_sd"]

    phi_rv = TransformedVariable(
        "phi",
        DistributionalVariable(
            "inv_sqrt_phi",
            dist.TruncatedNormal(
                loc=inv_sqrt_phi_prior_mean,
                scale=inv_sqrt_phi_prior_sd,
                low=1 / jnp.sqrt(5000),
            ),
        ),
        transforms=transformation.PowerTransform(-2),
    )

    uot = stan_data["uot"]
    uot = len(jnp.array(stan_data["inf_to_hosp"]))
    state_pop = stan_data["state_pop"]

    data_observed_hospital_admissions = jnp.array(stan_data["hosp"])

    my_model = hosp_only_ww_model(
        state_pop=state_pop,
        i0_over_n_rv=i0_over_n_rv,
        initialization_rate_rv=initialization_rate_rv,
        log_r_mu_intercept_rv=log_r_mu_intercept_rv,
        autoreg_rt_rv=autoreg_rt_rv,
        eta_sd_rv=eta_sd_rv,  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        infection_feedback_strength_rv=inf_feedback_strength_rv,
        p_hosp_mean_rv=p_hosp_mean_rv,
        p_hosp_w_sd_rv=p_hosp_w_sd_rv,
        autoreg_p_hosp_rv=autoreg_p_hosp_rv,
        hosp_wday_effect_rv=hosp_wday_effect_rv,
        phi_rv=phi_rv,
        inf_to_hosp_rv=inf_to_hosp_rv,
        n_initialization_points=uot,
        i0_t_offset=0,
    )

    return my_model, data_observed_hospital_admissions
