# numpydoc ignore=GL08
import json

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from jax.typing import ArrayLike
from numpyro.infer.reparam import LocScaleReparam
from pyrenew.arrayutils import repeat_until_n, tile_until_n
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.metaclass import Model, RandomVariable
from pyrenew.observation import NegativeBinomialObservation, PoissonObservation
from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

from pyrenew_hew.utils import convert_to_logmean_log_sd


class LatentInfectionProcess(RandomVariable):
    def __init__(
        self,
        i0_first_obs_n_rv: RandomVariable,
        initialization_rate_rv: RandomVariable,
        log_r_mu_intercept_rv: RandomVariable,
        autoreg_rt_rv: RandomVariable,  # ar coefficient of AR(1) process on R'(t)
        eta_sd_rv: RandomVariable,  # sd of random walk for ar process on R'(t)
        generation_interval_pmf_rv: RandomVariable,
        infection_feedback_strength_rv: RandomVariable,
        infection_feedback_pmf_rv: RandomVariable,
        n_initialization_points: int,
        pop_fraction: ArrayLike = jnp.array([1]),
        autoreg_rt_subpop_rv: RandomVariable = None,
        sigma_rt_rv: RandomVariable = None,
        sigma_i_first_obs_rv: RandomVariable = None,
        sigma_initial_exp_growth_rate_rv: RandomVariable = None,
        offset_ref_logit_i_first_obs_rv: RandomVariable = None,
        offset_ref_initial_exp_growth_rate_rv: RandomVariable = None,
        offset_ref_log_rt_rv: RandomVariable = None,
    ) -> None:
        self.inf_with_feedback_proc = InfectionsWithFeedback(
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        )

        self.ar_diff = DifferencedProcess(
            fundamental_process=ARProcess(),
            differencing_order=1,
        )

        self.log_r_mu_intercept_rv = log_r_mu_intercept_rv
        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.infection_feedback_pmf_rv = infection_feedback_pmf_rv
        self.i0_first_obs_n_rv = i0_first_obs_n_rv
        self.initialization_rate_rv = initialization_rate_rv
        self.offset_ref_logit_i_first_obs_rv = offset_ref_logit_i_first_obs_rv
        self.offset_ref_initial_exp_growth_rate_rv = (
            offset_ref_initial_exp_growth_rate_rv
        )
        self.offset_ref_log_rt_rv = offset_ref_log_rt_rv
        self.autoreg_rt_subpop_rv = autoreg_rt_subpop_rv
        self.sigma_rt_rv = sigma_rt_rv
        self.sigma_i_first_obs_rv = sigma_i_first_obs_rv
        self.sigma_initial_exp_growth_rate_rv = (
            sigma_initial_exp_growth_rate_rv
        )
        self.n_initialization_points = n_initialization_points
        self.pop_fraction = pop_fraction
        self.n_subpops = len(pop_fraction)

    def validate(self):
        pass

    def sample(self, n_days_post_init: int, n_weeks_post_init: int):
        """
        Sample latent infections.
        """
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

        i0_first_obs_n = self.i0_first_obs_n_rv()
        initial_exp_growth_rate = self.initialization_rate_rv()
        if self.n_subpops == 1:
            i_first_obs_over_n_subpop = i0_first_obs_n
            initial_exp_growth_rate_subpop = initial_exp_growth_rate
            log_rtu_weekly_subpop = log_rtu_weekly[:, jnp.newaxis]
        else:
            i_first_obs_over_n_ref_subpop = transformation.SigmoidTransform()(
                transformation.logit(i0_first_obs_n)
                + self.offset_ref_logit_i_first_obs_rv(),
            )
            initial_exp_growth_rate_ref_subpop = (
                initial_exp_growth_rate
                + self.offset_ref_initial_exp_growth_rate_rv()
            )

            log_rtu_weekly_ref_subpop = (
                log_rtu_weekly + self.offset_ref_log_rt_rv()
            )
            i_first_obs_over_n_non_ref_subpop_rv = TransformedVariable(
                "i_first_obs_over_n_non_ref_subpop",
                DistributionalVariable(
                    "i_first_obs_over_n_non_ref_subpop_raw",
                    dist.Normal(
                        transformation.logit(i0_first_obs_n),
                        self.sigma_i_first_obs_rv(),
                    ),
                    reparam=LocScaleReparam(0),
                ),
                transforms=transformation.SigmoidTransform(),
            )
            initial_exp_growth_rate_non_ref_subpop_rv = DistributionalVariable(
                "initial_exp_growth_rate_non_ref_subpop_raw",
                dist.Normal(
                    initial_exp_growth_rate,
                    self.sigma_initial_exp_growth_rate_rv(),
                ),
                reparam=LocScaleReparam(0),
            )

            autoreg_rt_subpop = self.autoreg_rt_subpop_rv()
            sigma_rt = self.sigma_rt_rv()
            rtu_subpop_ar_init_rv = DistributionalVariable(
                "rtu_subpop_ar_init",
                dist.Normal(
                    0,
                    sigma_rt / jnp.sqrt(1 - jnp.pow(autoreg_rt_subpop, 2)),
                ),
            )

            with numpyro.plate("n_subpops", self.n_subpops - 1):
                initial_exp_growth_rate_non_ref_subpop = (
                    initial_exp_growth_rate_non_ref_subpop_rv()
                )
                i_first_obs_over_n_non_ref_subpop = (
                    i_first_obs_over_n_non_ref_subpop_rv()
                )
                rtu_subpop_ar_init = rtu_subpop_ar_init_rv()

            i_first_obs_over_n_subpop = jnp.hstack(
                [
                    i_first_obs_over_n_ref_subpop,
                    i_first_obs_over_n_non_ref_subpop,
                ]
            )
            initial_exp_growth_rate_subpop = jnp.hstack(
                [
                    initial_exp_growth_rate_ref_subpop,
                    initial_exp_growth_rate_non_ref_subpop,
                ]
            )

            rtu_subpop_ar_proc = ARProcess()
            rtu_subpop_ar_weekly = rtu_subpop_ar_proc(
                noise_name="rtu_ar_proc",
                n=n_weeks_post_init,
                init_vals=rtu_subpop_ar_init[jnp.newaxis],
                autoreg=autoreg_rt_subpop[jnp.newaxis],
                noise_sd=sigma_rt,
            )

            log_rtu_weekly_non_ref_subpop = (
                rtu_subpop_ar_weekly + log_rtu_weekly[:, jnp.newaxis]
            )
            log_rtu_weekly_subpop = jnp.concat(
                [
                    log_rtu_weekly_ref_subpop[:, jnp.newaxis],
                    log_rtu_weekly_non_ref_subpop,
                ],
                axis=1,
            )

        rtu_subpop = jnp.squeeze(
            jnp.repeat(
                jnp.exp(log_rtu_weekly_subpop),
                repeats=7,
                axis=0,
            )[:n_days_post_init, :]
        )

        i0_subpop_rv = DeterministicVariable(
            "i0_subpop", i_first_obs_over_n_subpop
        )
        initial_exp_growth_rate_subpop_rv = DeterministicVariable(
            "initial_exp_growth_rate_subpop", initial_exp_growth_rate_subpop
        )
        infection_initialization_process = InfectionInitializationProcess(
            "I0_initialization",
            i0_subpop_rv,
            InitializeInfectionsExponentialGrowth(
                self.n_initialization_points,
                initial_exp_growth_rate_subpop_rv,
                t_pre_init=0,
            ),
        )

        generation_interval_pmf = self.generation_interval_pmf_rv()
        i0 = infection_initialization_process()

        inf_with_feedback_proc_sample = self.inf_with_feedback_proc(
            Rt=rtu_subpop,
            I0=i0,
            gen_int=generation_interval_pmf,
        )

        latent_infections_subpop = jnp.concat(
            [
                i0,
                inf_with_feedback_proc_sample.post_initialization_infections,
            ]
        )

        if self.n_subpops == 1:
            latent_infections = latent_infections_subpop
        else:
            latent_infections = jnp.sum(
                self.pop_fraction * latent_infections_subpop, axis=1
            )

        numpyro.deterministic("rtu_subpop", rtu_subpop)
        numpyro.deterministic("rt", inf_with_feedback_proc_sample.rt)
        numpyro.deterministic("latent_infections", latent_infections)

        return latent_infections


class EDVisitObservationProcess(RandomVariable):
    def __init__(
        self,
        p_ed_mean_rv: RandomVariable,
        p_ed_w_sd_rv: RandomVariable,
        autoreg_p_ed_rv: RandomVariable,
        ed_wday_effect_rv: RandomVariable,
        inf_to_ed_rv: RandomVariable,
        ed_neg_bin_concentration_rv: RandomVariable,
        ed_right_truncation_pmf_rv: RandomVariable,
    ) -> None:
        self.p_ed_ar_proc = ARProcess()
        self.p_ed_mean_rv = p_ed_mean_rv
        self.p_ed_w_sd_rv = p_ed_w_sd_rv
        self.autoreg_p_ed_rv = autoreg_p_ed_rv
        self.ed_wday_effect_rv = ed_wday_effect_rv
        self.inf_to_ed_rv = inf_to_ed_rv
        self.ed_right_truncation_cdf_rv = TransformedVariable(
            "ed_right_truncation_cdf", ed_right_truncation_pmf_rv, jnp.cumsum
        )
        self.ed_neg_bin_concentration_rv = ed_neg_bin_concentration_rv

    def validate(self):
        pass

    def sample(
        self,
        latent_infections: ArrayLike,
        population_size: int,
        data_observed_disease_ed_visits: ArrayLike,
        n_observed_disease_ed_visits_datapoints: int,
        n_weeks_post_init: int,
        right_truncation_offset: ArrayLike = None,
    ) -> ArrayLike:
        """
        Observe and/or predict ED visit values
        """
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
        # this is only applied after the ed visits are generated, not to all
        # the latent infections. This is why we cannot apply the iedr in
        # compute_delay_ascertained_incidence
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
            potential_latent_ed_visits
            * iedr
            * ed_wday_effect
            * population_size
        )

        if right_truncation_offset is not None:
            prop_already_reported_tail = jnp.flip(
                self.ed_right_truncation_cdf_rv()[right_truncation_offset:]
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
            "observed_ed_visits",
            concentration_rv=self.ed_neg_bin_concentration_rv,
        )

        observed_ed_visits = ed_visit_obs_rv(
            mu=latent_ed_visits_now,
            obs=data_observed_disease_ed_visits,
        )

        return observed_ed_visits


class HospAdmitObservationProcess(RandomVariable):
    def __init__(
        self,
        inf_to_hosp_admit_rv: RandomVariable,
        hosp_admit_neg_bin_concentration_rv: RandomVariable,
    ):
        self.inf_to_hosp_admit_rv = inf_to_hosp_admit_rv
        self.hosp_admit_neg_bin_concentration_rv = (
            hosp_admit_neg_bin_concentration_rv
        )

    def validate(self):
        pass

    def sample(
        self,
        latent_infections: ArrayLike,
        population_size: int,
        n_observed_hospital_admissions_datapoints: int,
        data_observed_disease_hospital_admissions: ArrayLike | None = None,
    ) -> ArrayLike:
        """
        Observe and/or predict incident hospital admissions.
        """
        inf_to_hosp_admit = self.inf_to_hosp_admit_rv()

        latent_hospital_admissions = compute_delay_ascertained_incidence(
            p_observed_given_incident=1,
            latent_incidence=latent_infections,
            delay_incidence_to_observation_pmf=inf_to_hosp_admit,
        )[-n_observed_hospital_admissions_datapoints:]

        hospital_admissions_obs_rv = NegativeBinomialObservation(
            "observed_hospital_admissions",
            concentration_rv=self.hosp_admit_neg_bin_concentration_rv,
        )
        predicted_admissions = latent_hospital_admissions
        sampled_admissions = hospital_admissions_obs_rv(
            mu=predicted_admissions,
            obs=data_observed_disease_hospital_admissions,
        )
        return sampled_admissions


class WastewaterObservationProcess(RandomVariable):
    """
    Placeholder for wasteater obs process
    """

    def __init__(self) -> None:
        pass

    def sample(self):
        pass

    def validate(self):
        pass


class PyrenewHEWModel(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        population_size: int,
        latent_infection_process_rv: LatentInfectionProcess,
        ed_visit_obs_process_rv: EDVisitObservationProcess,
        hosp_admit_obs_process_rv: HospAdmitObservationProcess,
        wastewater_obs_process_rv: WastewaterObservationProcess,
    ) -> None:  # numpydoc ignore=GL08
        self.population_size = population_size
        self.latent_infection_process_rv = latent_infection_process_rv
        self.ed_visit_obs_process_rv = ed_visit_obs_process_rv
        self.hosp_admit_obs_process_rv = hosp_admit_obs_process_rv
        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        n_observed_disease_ed_visits_datapoints=None,
        n_observed_hospital_admissions_datapoints=None,
        data_observed_disease_ed_visits=None,
        data_observed_disease_hospital_admissions=None,
        right_truncation_offset=None,
    ) -> ArrayLike:  # numpydoc ignore=GL08
        if (
            n_observed_disease_ed_visits_datapoints is None
            and data_observed_disease_ed_visits is None
        ):
            raise ValueError(
                "Either n_observed_disease_ed_visits_datapoints or "
                "data_observed_disease_ed_visits "
                "must be passed."
            )
        elif (
            n_observed_disease_ed_visits_datapoints is not None
            and data_observed_disease_ed_visits is not None
        ):
            raise ValueError(
                "Cannot pass both n_observed_disease_ed_visits_datapoints "
                "and data_observed_disease_ed_visits."
            )
        elif n_observed_disease_ed_visits_datapoints is None:
            n_observed_disease_ed_visits_datapoints = len(
                data_observed_disease_ed_visits
            )

        if (
            n_observed_hospital_admissions_datapoints is None
            and data_observed_disease_hospital_admissions is not None
        ):
            n_observed_hospital_admissions_datapoints = len(
                data_observed_disease_hospital_admissions
            )
        elif n_observed_hospital_admissions_datapoints is None:
            n_observed_hospital_admissions_datapoints = 0

        n_weeks_post_init = n_observed_disease_ed_visits_datapoints // 7 + 1
        n_days_post_init = n_observed_disease_ed_visits_datapoints

        latent_infections = self.latent_infection_process_rv(
            n_days_post_init=n_days_post_init,
            n_weeks_post_init=n_weeks_post_init,
        )

        sample_ed_visits = True
        sample_admissions = n_observed_hospital_admissions_datapoints > 0
        sample_wastewater = False

        sampled_ed_visits, sampled_admissions, sampled_wastewater = (
            None,
            None,
            None,
        )

        if sample_ed_visits:
            sampled_ed_visits = self.ed_visit_obs_process_rv(
                latent_infections=latent_infections,
                population_size=self.population_size,
                data_observed_disease_ed_visits=(
                    data_observed_disease_ed_visits
                ),
                n_observed_disease_ed_visits_datapoints=(
                    n_observed_disease_ed_visits_datapoints
                ),
                n_weeks_post_init=n_weeks_post_init,
                right_truncation_offset=right_truncation_offset,
            )

        if sample_admissions:
            sampled_admissions = self.hosp_admit_obs_process_rv(
                latent_infections=latent_infections,
                population_size=self.population_size,
                n_observed_hospital_admissions_datapoints=(
                    n_observed_hospital_admissions_datapoints
                ),
                data_observed_disease_hospital_admissions=(
                    data_observed_disease_hospital_admissions
                ),
            )
        if sample_wastewater:
            sampled_wastewater = self.wastewater_obs_process_rv()

        return {
            "ed_visits": sampled_ed_visits,
            "admissions": sampled_admissions,
            "wasewater": sampled_wastewater,
        }


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

    ed_neg_bin_concentration_rv = TransformedVariable(
        "ed_visit_neg_bin_concentration",
        DistributionalVariable(
            "inv_sqrt_ed_visit_neg_bin_conc",
            dist.TruncatedNormal(
                loc=inv_sqrt_phi_prior_mean,
                scale=inv_sqrt_phi_prior_sd,
                low=1 / jnp.sqrt(5000),
            ),
        ),
        transforms=transformation.PowerTransform(-2),
    )

    inf_to_hosp_admit_rv = DeterministicVariable(
        "inf_to_hosp_admit", jnp.array(stan_data["inf_to_hosp"])
    )

    hosp_admit_neg_bin_concentration_rv = TransformedVariable(
        "hosp_admit_neg_bin_concentration",
        DistributionalVariable(
            "inv_sqrt_hosp_admit_neg_bin_conc",
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
    population_size = stan_data["state_pop"]

    data_observed_disease_ed_visits = jnp.array(stan_data["hosp"])
    ed_right_truncation_pmf_rv = DeterministicVariable(
        "ed_visit_right_truncation_pmf", jnp.array(1)
    )

    my_latent_infection_model = LatentInfectionProcess(
        i0_first_obs_n_rv=i0_first_obs_n_rv,
        initialization_rate_rv=initialization_rate_rv,
        log_r_mu_intercept_rv=log_r_mu_intercept_rv,
        autoreg_rt_rv=autoreg_rt_rv,
        eta_sd_rv=eta_sd_rv,  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        infection_feedback_strength_rv=inf_feedback_strength_rv,
        n_initialization_points=uot,
    )

    my_ed_visit_obs_model = EDVisitObservationProcess(
        p_ed_mean_rv=p_ed_mean_rv,
        p_ed_w_sd_rv=p_ed_w_sd_rv,
        autoreg_p_ed_rv=autoreg_p_ed_rv,
        ed_wday_effect_rv=ed_wday_effect_rv,
        inf_to_ed_rv=inf_to_ed_rv,
        ed_neg_bin_concentration_rv=ed_neg_bin_concentration_rv,
        ed_right_truncation_pmf_rv=ed_right_truncation_pmf_rv,
    )

    my_hosp_admit_obs_model = HospAdmitObservationProcess(
        inf_to_hosp_admit_rv=inf_to_hosp_admit_rv,
        hosp_admit_neg_bin_concentration_rv=(
            hosp_admit_neg_bin_concentration_rv
        ),
    )

    my_wastewater_obs_model = WastewaterObservationProcess()

    my_model = PyrenewHEWModel(
        population_size=population_size,
        latent_infection_process_rv=my_latent_infection_model,
        ed_visit_obs_process_rv=my_ed_visit_obs_model,
        hosp_admit_obs_process_rv=my_hosp_admit_obs_model,
        wastewater_obs_process_rv=my_wastewater_obs_model,
    )

    return my_model, data_observed_disease_ed_visits
