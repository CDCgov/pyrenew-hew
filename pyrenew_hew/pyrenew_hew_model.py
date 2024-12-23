# numpydoc ignore=GL08
import json

import jax.numpy as jnp
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

from pyrenew_hew.utils import convert_to_logmean_log_sd


class pyrenew_hew_model(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        state_pop,
        i0_first_obs_n_rv,
        initialization_rate_rv,
        log_r_mu_intercept_rv,
        autoreg_rt_rv,  # ar process
        eta_sd_rv,  # sd of random walk for ar process
        generation_interval_pmf_rv,
        infection_feedback_strength_rv,
        infection_feedback_pmf_rv,
        p_ed_mean_rv,
        p_ed_w_sd_rv,
        autoreg_p_ed_rv,
        ed_wday_effect_rv,
        inf_to_ed_rv,
        phi_rv,
        right_truncation_pmf_rv,  # when unnamed deterministic variables are allowed, we could default this to 1.
        n_initialization_points,
    ):  # numpydoc ignore=GL08
        self.infection_initialization_process = InfectionInitializationProcess(
            "I0_initialization",
            i0_first_obs_n_rv,
            InitializeInfectionsExponentialGrowth(
                n_initialization_points, initialization_rate_rv, t_pre_init=0
            ),
        )

        self.inf_with_feedback_proc = InfectionsWithFeedback(
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        )

        self.p_ed_ar_proc = ARProcess()
        self.ar_diff = DifferencedProcess(
            fundamental_process=ARProcess(),
            differencing_order=1,
        )

        self.right_truncation_cdf_rv = TransformedVariable(
            "right_truncation_cdf", right_truncation_pmf_rv, jnp.cumsum
        )
        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.log_r_mu_intercept_rv = log_r_mu_intercept_rv
        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.infection_feedback_pmf_rv = infection_feedback_pmf_rv
        self.p_ed_mean_rv = p_ed_mean_rv
        self.p_ed_w_sd_rv = p_ed_w_sd_rv
        self.autoreg_p_ed_rv = autoreg_p_ed_rv
        self.ed_wday_effect_rv = ed_wday_effect_rv
        self.inf_to_ed_rv = inf_to_ed_rv
        self.phi_rv = phi_rv
        self.state_pop = state_pop
        self.n_initialization_points = n_initialization_points
        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        n_observed_disease_ed_visits_datapoints=None,
        data_observed_disease_ed_visits=None,
        right_truncation_offset=None,
    ):  # numpydoc ignore=GL08
        if (
            n_observed_disease_ed_visits_datapoints is None
            and data_observed_disease_ed_visits is None
        ):
            raise ValueError(
                "Either n_observed_disease_ed_visits_datapoints or data_observed_disease_ed_visits "
                "must be passed."
            )
        elif (
            n_observed_disease_ed_visits_datapoints is not None
            and data_observed_disease_ed_visits is not None
        ):
            raise ValueError(
                "Cannot pass both n_observed_disease_ed_visits_datapoints and data_observed_disease_ed_visits."
            )
        elif n_observed_disease_ed_visits_datapoints is None:
            n_observed_disease_ed_visits_datapoints = len(
                data_observed_disease_ed_visits
            )
        else:
            n_observed_disease_ed_visits_datapoints = (
                n_observed_disease_ed_visits_datapoints
            )

        n_weeks_post_init = n_observed_disease_ed_visits_datapoints // 7 + 1
        i0 = self.infection_initialization_process()

        eta_sd = self.eta_sd_rv()
        autoreg_rt = self.autoreg_rt_rv()
        log_r_mu_intercept = self.log_r_mu_intercept_rv()
        rt_init_rate_of_change = DistributionalVariable(
            "rt_init_rate_of_change",
            dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
        )()

        log_rtu_weekly = self.ar_diff(
            n=n_weeks_post_init,
            init_vals=jnp.array(log_r_mu_intercept),
            autoreg=jnp.array(autoreg_rt),
            noise_sd=jnp.array(eta_sd),
            fundamental_process_init_vals=jnp.array(rt_init_rate_of_change),
            noise_name="rtu_weekly_diff_first_diff_ar_process_noise",
        )

        rtu = repeat_until_n(
            data=jnp.exp(log_rtu_weekly),
            n_timepoints=n_observed_disease_ed_visits_datapoints,
            offset=0,
            period_size=7,
        )

        generation_interval_pmf = self.generation_interval_pmf_rv()

        inf_with_feedback_proc_sample = self.inf_with_feedback_proc(
            Rt=rtu,
            I0=i0,
            gen_int=generation_interval_pmf,
        )

        latent_infections = jnp.concat(
            [
                i0,
                inf_with_feedback_proc_sample.post_initialization_infections,
            ]
        )
        numpyro.deterministic("rtu", rtu)
        numpyro.deterministic("rt", inf_with_feedback_proc_sample.rt)
        numpyro.deterministic("latent_infections", latent_infections)

        p_ed_mean = self.p_ed_mean_rv()
        p_ed_w_sd = self.p_ed_w_sd_rv()
        autoreg_p_ed = self.autoreg_p_ed_rv()

        p_ed_ar_init_rv = DistributionalVariable(
            "p_ed_ar_init",
            dist.Normal(
                0,
                p_ed_w_sd / jnp.sqrt(1 - jnp.pow(autoreg_p_ed, 2)),
            ),
        )
        p_ed_ar_init = p_ed_ar_init_rv()

        p_ed_ar = self.p_ed_ar_proc(
            noise_name="p_ed",
            n=n_weeks_post_init,
            autoreg=autoreg_p_ed,
            init_vals=p_ed_ar_init,
            noise_sd=p_ed_w_sd,
        )

        iedr = jnp.repeat(
            transformation.SigmoidTransform()(p_ed_ar + p_ed_mean),
            repeats=7,
        )[:n_observed_disease_ed_visits_datapoints]
        # this is only applied after the ed visits are generated, not to all the latent infections. This is why we cannot apply the iedr in compute_delay_ascertained_incidence
        # see https://github.com/CDCgov/ww-inference-model/issues/43

        numpyro.deterministic("iedr", iedr)

        ed_wday_effect_raw = self.ed_wday_effect_rv()
        ed_wday_effect = tile_until_n(
            ed_wday_effect_raw, n_observed_disease_ed_visits_datapoints
        )

        inf_to_ed = self.inf_to_ed_rv()

        potential_latent_ed_visits = compute_delay_ascertained_incidence(
            p_observed_given_incident=1,
            latent_incidence=latent_infections,
            delay_incidence_to_observation_pmf=inf_to_ed,
        )[-n_observed_disease_ed_visits_datapoints:]

        latent_ed_visits_final = (
            potential_latent_ed_visits * iedr * ed_wday_effect * self.state_pop
        )

        if right_truncation_offset is not None:
            prop_already_reported_tail = jnp.flip(
                self.right_truncation_cdf_rv()[right_truncation_offset:]
            )
            n_points_to_prepend = (
                n_observed_disease_ed_visits_datapoints
                - prop_already_reported_tail.shape[0]
            )
            prop_already_reported = jnp.pad(
                prop_already_reported_tail,
                (n_points_to_prepend, 0),
                mode="constant",
                constant_values=(1, 0),
            )
            latent_ed_visits_now = (
                latent_ed_visits_final * prop_already_reported
            )
        else:
            latent_ed_visits_now = latent_ed_visits_final

        ed_visit_obs_rv = NegativeBinomialObservation(
            "observed_ed_visits", concentration_rv=self.phi_rv
        )

        observed_ed_visits = ed_visit_obs_rv(
            mu=latent_ed_visits_now,
            obs=data_observed_disease_ed_visits,
        )

        return observed_ed_visits


def create_pyrenew_hew_model_from_stan_data(stan_data_file):
    with open(
        stan_data_file,
        "r",
    ) as file:
        stan_data = json.load(file)

    i_first_obs_over_n_prior_a = stan_data["i_first_obs_over_n_prior_a"]
    i_first_obs_over_n_prior_b = stan_data["i_first_obs_over_n_prior_b"]
    i0_first_obs_n_rv = DistributionalVariable(
        "i0_first_obs_n_rv",
        dist.Beta(i_first_obs_over_n_prior_a, i_first_obs_over_n_prior_b),
    )

    mean_initial_exp_growth_rate_prior_mean = stan_data[
        "mean_initial_exp_growth_rate_prior_mean"
    ]
    mean_initial_exp_growth_rate_prior_sd = stan_data[
        "mean_initial_exp_growth_rate_prior_sd"
    ]
    initialization_rate_rv = DistributionalVariable(
        "rate",
        dist.TruncatedNormal(
            loc=mean_initial_exp_growth_rate_prior_mean,
            scale=mean_initial_exp_growth_rate_prior_sd,
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

    p_ed_mean_rv = DistributionalVariable(
        "p_ed_mean",
        dist.Normal(
            transformation.SigmoidTransform().inv(p_hosp_prior_mean),
            p_hosp_sd_logit,
        ),
    )  # logit scale

    p_ed_w_sd_sd = stan_data["p_hosp_w_sd_sd"]
    p_ed_w_sd_rv = DistributionalVariable(
        "p_ed_w_sd_sd", dist.TruncatedNormal(0, p_ed_w_sd_sd, low=0)
    )

    autoreg_p_ed_a = stan_data["autoreg_p_hosp_a"]
    autoreg_p_ed_b = stan_data["autoreg_p_hosp_b"]
    autoreg_p_ed_rv = DistributionalVariable(
        "autoreg_p_ed", dist.Beta(autoreg_p_ed_a, autoreg_p_ed_b)
    )

    ed_wday_effect_rv = TransformedVariable(
        "ed_wday_effect",
        DistributionalVariable(
            "ed_wday_effect_raw",
            dist.Dirichlet(
                jnp.array(stan_data["hosp_wday_effect_prior_alpha"])
            ),
        ),
        transformation.AffineTransform(loc=0, scale=7),
    )

    inf_to_ed_rv = DeterministicVariable(
        "inf_to_ed", jnp.array(stan_data["inf_to_hosp"])
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

    data_observed_disease_ed_visits = jnp.array(stan_data["hosp"])
    right_truncation_pmf_rv = DeterministicVariable(
        "right_truncation_pmf", jnp.array(1)
    )
    my_model = pyrenew_hew_model(
        state_pop=state_pop,
        i0_first_obs_n_rv=i0_first_obs_n_rv,
        initialization_rate_rv=initialization_rate_rv,
        log_r_mu_intercept_rv=log_r_mu_intercept_rv,
        autoreg_rt_rv=autoreg_rt_rv,
        eta_sd_rv=eta_sd_rv,  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        infection_feedback_strength_rv=inf_feedback_strength_rv,
        p_ed_mean_rv=p_ed_mean_rv,
        p_ed_w_sd_rv=p_ed_w_sd_rv,
        autoreg_p_ed_rv=autoreg_p_ed_rv,
        ed_wday_effect_rv=ed_wday_effect_rv,
        inf_to_ed_rv=inf_to_ed_rv,
        phi_rv=phi_rv,
        right_truncation_pmf_rv=right_truncation_pmf_rv,
        n_initialization_points=uot,
    )

    return my_model, data_observed_disease_ed_visits
