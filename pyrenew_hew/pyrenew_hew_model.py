# numpydoc ignore=GL08
import datetime as dt
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from jax.typing import ArrayLike
from numpyro.handlers import scope
from numpyro.infer.reparam import LocScaleReparam
from pyrenew.arrayutils import tile_until_n
from pyrenew.convolve import compute_delay_ascertained_incidence
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.math import r_approx_from_R
from pyrenew.metaclass import Model, RandomVariable
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
from pyrenew.time import daily_to_mmwr_epiweekly

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData


class OffsetDiscretizedLognormalPMF(RandomVariable):
    """
    Discrete PMF modeled by offseting the location or
    scale of a lognormal distribution from central
    values, then discretizing and normalizing.

    Attributes
    ----------
    name
        Name for the `RandomVariable`.

    reference_loc
        Reference location(s) from which to offset the
        distribution's own location. If `offset_loc_rv`
        is `None`, this will be the (deterministic)
        location parameter(s) for the distribution.

    reference_scale
        Reference scale(s) from which to offset the
        distribution's own scale. If `offset_scale_rv`
        is `None`, this will be the (deterministic)
        scale parameter(s) for the distribution.

    n
       Number of points over which to discrete the distribution.
       The final PMF will have support on [0, n - 1],
       but with 0 mass at 0.

    offset_loc_rv
       `RandomVariable` representing the offset of the
        distribution's location parameter from the
       `reference_loc`. If `None`, use the `reference_loc`
        as a fixed location parameter (i.e. a use a fixed loc
        offset of 0).

    offset_scale_rv
       `RandomVariable` representing the offset of the
        distribution's scale parameter from the
       `reference_scale`. If `None`, use the `reference_scale`
        as a fixed location parameter (i.e. a use a fixed scale
        offset of 0).
    """

    def __init__(
        self,
        name: str,
        reference_loc: ArrayLike,
        reference_scale: ArrayLike,
        n: int,
        offset_loc_rv: RandomVariable = None,
        log_offset_scale_rv: RandomVariable = None,
    ):
        """
        Default constructor.
        """
        self.name = name
        self.reference_loc = reference_loc
        self.reference_scale = reference_scale
        self.n = n

        if offset_loc_rv is None:
            offset_loc_rv = DeterministicVariable("offset_loc", 0)

        if log_offset_scale_rv is None:
            log_offset_scale_rv = DeterministicVariable("log_offset_scale", 0)

        self.offset_loc_rv = offset_loc_rv
        self.log_offset_scale_rv = log_offset_scale_rv

    def sample(self):
        with scope(prefix=self.name, divider="_"):
            offset_loc = self.offset_loc_rv()
            log_offset_scale = self.log_offset_scale_rv()
        lognorm = dist.LogNormal(
            loc=self.reference_loc + offset_loc,
            scale=jnp.exp(jnp.log(self.reference_scale) + log_offset_scale),
        )
        unnormed = jnp.exp(lognorm.log_prob(jnp.arange(1, self.n)))
        pmf = jnp.pad(unnormed / jnp.sum(unnormed), [1, 0])
        numpyro.deterministic(self.name, pmf)
        return pmf

    def validate(self):
        pass

    def size(self):
        return self.n


class LatentInfectionProcess(RandomVariable):
    def __init__(
        self,
        i0_first_obs_n_rv: RandomVariable,
        log_r_mu_intercept_rv: RandomVariable,
        autoreg_rt_rv: RandomVariable,  # ar coeff for AR(1) on R'(t)
        eta_sd_rv: RandomVariable,  # sd of random walk for AR(1) on R'(t)
        generation_interval_pmf_rv: RandomVariable,
        infection_feedback_strength_rv: RandomVariable,
        infection_feedback_pmf_rv: RandomVariable,
        n_initialization_points: int,
        pop_fraction: ArrayLike = jnp.array([1]),
        autoreg_rt_subpop_rv: RandomVariable = None,
        sigma_rt_rv: RandomVariable = None,
        sigma_i_first_obs_rv: RandomVariable = None,
        offset_ref_logit_i_first_obs_rv: RandomVariable = None,
        offset_ref_log_rt_rv: RandomVariable = None,
        n_newton_steps: int = 4,
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
        self.offset_ref_logit_i_first_obs_rv = offset_ref_logit_i_first_obs_rv
        self.offset_ref_log_rt_rv = offset_ref_log_rt_rv
        self.autoreg_rt_subpop_rv = autoreg_rt_subpop_rv
        self.sigma_rt_rv = sigma_rt_rv
        self.sigma_i_first_obs_rv = sigma_i_first_obs_rv
        self.n_initialization_points = n_initialization_points
        self.pop_fraction = pop_fraction
        self.n_subpops = len(pop_fraction)
        self.n_newton_steps = n_newton_steps

    def validate(self):
        pass

    def sample(self, n_days_post_init: int):
        """
        Sample latent infections.

        Parameters
        ----------
        n_days_post_init
            Number of days of infections to sample, not including
            the initialization period.
        """
        eta_sd = self.eta_sd_rv()
        autoreg_rt = self.autoreg_rt_rv()
        log_r_mu_intercept = self.log_r_mu_intercept_rv()
        rt_init_rate_of_change = DistributionalVariable(
            "rt_init_rate_of_change",
            dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
        )()

        n_weeks_rt = n_days_post_init // 7 + 1

        log_rtu_weekly = self.ar_diff(
            n=n_weeks_rt,
            init_vals=jnp.array(log_r_mu_intercept),
            autoreg=jnp.array(autoreg_rt),
            noise_sd=jnp.array(eta_sd),
            fundamental_process_init_vals=jnp.array(rt_init_rate_of_change),
            noise_name="rtu_weekly_diff_first_diff_ar_process_noise",
        )
        generation_interval_pmf = self.generation_interval_pmf_rv()

        i0_first_obs_n = self.i0_first_obs_n_rv()
        if self.n_subpops == 1:
            i_first_obs_over_n_subpop = i0_first_obs_n
            log_rtu_weekly_subpop = log_rtu_weekly[:, jnp.newaxis]
        else:
            i_first_obs_over_n_ref_subpop = transformation.SigmoidTransform()(
                transformation.SigmoidTransform().inv(i0_first_obs_n)
                + self.offset_ref_logit_i_first_obs_rv(),
            )
            log_rtu_weekly_ref_subpop = log_rtu_weekly + self.offset_ref_log_rt_rv()
            i_first_obs_over_n_non_ref_subpop_rv = TransformedVariable(
                "i_first_obs_over_n_non_ref_subpop",
                DistributionalVariable(
                    "i_first_obs_over_n_non_ref_subpop_raw",
                    dist.Normal(
                        transformation.SigmoidTransform().inv(i0_first_obs_n),
                        self.sigma_i_first_obs_rv(),
                    ),
                    reparam=LocScaleReparam(0),
                ),
                transforms=transformation.SigmoidTransform(),
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

            rtu_subpop_ar_proc = ARProcess()
            rtu_subpop_ar_weekly = rtu_subpop_ar_proc(
                noise_name="rtu_ar_proc",
                n=n_weeks_rt,
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
            pass

        ## this can be replaced once r_approx_from_R is vectorized
        r_approx = jax.vmap(
            partial(
                r_approx_from_R,
                g=generation_interval_pmf,
                n_newton_steps=self.n_newton_steps,
            )
        )

        initial_exp_growth_rate_subpop = r_approx(jnp.exp(log_rtu_weekly_subpop[0]))

        rtu_subpop = jnp.squeeze(
            jnp.repeat(
                jnp.exp(log_rtu_weekly_subpop),
                repeats=7,
                axis=0,
            )[:n_days_post_init, :]
        )  # indexed rel to first post-init day.

        i0_subpop_rv = DeterministicVariable("i0_subpop", i_first_obs_over_n_subpop)
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
            latent_infections_subpop = jnp.expand_dims(latent_infections_subpop, axis=1)
        else:
            latent_infections = jnp.sum(
                self.pop_fraction * latent_infections_subpop, axis=1
            )
        assert latent_infections.size == self.n_initialization_points + n_days_post_init
        numpyro.deterministic("rtu_subpop", rtu_subpop)
        numpyro.deterministic("rt", inf_with_feedback_proc_sample.rt)
        numpyro.deterministic("latent_infections", latent_infections)

        return latent_infections, latent_infections_subpop


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
        data_observed: ArrayLike,
        model_t_observed: ArrayLike,
        model_t_first_latent_infection: int,
        right_truncation_offset: int = None,
    ) -> tuple[ArrayLike]:
        """
        Observe and/or predict ED visit values
        """
        inf_to_ed = self.inf_to_ed_rv()
        potential_latent_ed_visits, ed_visit_offset = (
            compute_delay_ascertained_incidence(
                p_observed_given_incident=1,
                latent_incidence=latent_infections,
                delay_incidence_to_observation_pmf=inf_to_ed,
            )
        )

        model_t_first_latent_ed_visit = ed_visit_offset + model_t_first_latent_infection

        if model_t_observed is None:  # True for forecasting/posterior prediction
            # slice the latent ed visits from model t0 to the end of the vector
            which_obs_ed_visits = np.s_[
                -model_t_first_latent_ed_visit : potential_latent_ed_visits.size
            ]
        else:
            which_obs_ed_visits = model_t_observed - model_t_first_latent_ed_visit

        p_ed_mean = self.p_ed_mean_rv()
        p_ed_w_sd = self.p_ed_w_sd_rv()
        autoreg_p_ed = self.autoreg_p_ed_rv()
        n_weeks_p_ed_ar = potential_latent_ed_visits.size // 7 + 1

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
            n=n_weeks_p_ed_ar,
            autoreg=autoreg_p_ed,
            init_vals=p_ed_ar_init,
            noise_sd=p_ed_w_sd,
        )

        iedr = jnp.repeat(
            transformation.SigmoidTransform()(p_ed_ar + p_ed_mean),
            repeats=7,
        )[: potential_latent_ed_visits.size]  # indexed rel to first ed report day
        # this is only applied after the ed visits are generated, not to all
        # the latent infections. This is why we cannot apply the iedr in
        # compute_delay_ascertained_incidence
        # see https://github.com/CDCgov/ww-inference-model/issues/43

        numpyro.deterministic("iedr", iedr)

        ed_wday_effect_raw = self.ed_wday_effect_rv()
        ed_wday_effect = tile_until_n(
            ed_wday_effect_raw, potential_latent_ed_visits.size
        )

        latent_ed_visits_final = (
            potential_latent_ed_visits * iedr * ed_wday_effect * population_size
        )

        if right_truncation_offset is not None:
            prop_already_reported_tail = jnp.flip(
                self.ed_right_truncation_cdf_rv()[right_truncation_offset:]
            )
            n_points_to_prepend = (
                potential_latent_ed_visits.size - prop_already_reported_tail.shape[0]
            )
            prop_already_reported = jnp.pad(
                prop_already_reported_tail,
                (n_points_to_prepend, 0),
                mode="constant",
                constant_values=(1, 0),
            )
            latent_ed_visits_now = latent_ed_visits_final * prop_already_reported
        else:
            latent_ed_visits_now = latent_ed_visits_final

        ed_visit_obs_rv = NegativeBinomialObservation(
            "observed_ed_visits",
            concentration_rv=self.ed_neg_bin_concentration_rv,
        )

        observed_ed_visits = ed_visit_obs_rv(
            mu=latent_ed_visits_now[which_obs_ed_visits],
            obs=data_observed,
        )

        return observed_ed_visits, iedr


class HospAdmitObservationProcess(RandomVariable):
    def __init__(
        self,
        inf_to_hosp_admit_rv: RandomVariable,
        hosp_admit_neg_bin_concentration_rv: RandomVariable,
        ihr_rv: RandomVariable = None,
        ihr_rel_iedr_rv: RandomVariable = None,
    ) -> None:
        self.inf_to_hosp_admit_rv = inf_to_hosp_admit_rv
        self.hosp_admit_neg_bin_concentration_rv = hosp_admit_neg_bin_concentration_rv
        self.ihr_rv = ihr_rv
        self.ihr_rel_iedr_rv = ihr_rel_iedr_rv

    def validate(self):
        pass

    @staticmethod
    def calculate_weekly_hosp_indices(
        first_latent_admission_dow: int,
        model_t_first_latent_admissions: int,
        model_t_observed: ArrayLike,
        n_datapoints: int,
    ):
        """
        Calculates indices of the predicted weekly
        hospital admissions vector corresponding to
        observed hospital admission date.

        Parameters
        ----------
        first_latent_admission_dow : int
            Day of the week (0=Monday, ..., 6=Sunday)
            of the first latent hospital admission.
        model_t_first_latent_admissions : int
            Time index in model time of the
            first latent hospital admission.
            Model time `t0` is the first overall data date.
        model_t_observed : ArrayLike
            Time indices in model time of observed hospital
            admissions (must be end of MMWR epiweek).
        n_datapoints : int
            Number of data points to sample

        Returns
        -------
        ArrayLike
            Vector of indices corresponding to the observed hospital admission.

        """
        # Days to truncate to get full epiweek of predicted admissions
        truncated_latent_admit_days = (6 - first_latent_admission_dow) % 7

        # First prediction is made for the week ending day (Saturday)
        # of the first full epiweek
        model_t_first_pred_admissions = (
            model_t_first_latent_admissions + truncated_latent_admit_days + 6
        )

        model_dow_first_pred_admissions = (
            first_latent_admission_dow + truncated_latent_admit_days + 6
        ) % 7

        # Check the first predicted admissions day is a Saturday (MMWR epiweek end)
        assert model_dow_first_pred_admissions == 5

        if model_t_observed is not None:
            if not all((model_t_observed - model_t_first_pred_admissions) >= 0):
                raise ValueError(
                    "Observed hospital admissions date is before predicted hospital admissions."
                )
            if not all((model_t_observed - model_t_first_pred_admissions) % 7 == 0):
                raise ValueError(
                    "Not all observed or predicted hospital admissions are on Saturdays."
                )
            which_obs_weekly_hosp_admissions = (
                model_t_observed - model_t_first_pred_admissions
            ) // 7
        else:
            which_obs_weekly_hosp_admissions = jnp.arange(n_datapoints)
            if model_t_first_pred_admissions < 0:
                which_obs_weekly_hosp_admissions = which_obs_weekly_hosp_admissions[
                    (-model_t_first_pred_admissions - 1) // 7 + 1 :
                ]
                # Truncate to include only the epiweek ending after
                # model t0 for posterior prediction

        return which_obs_weekly_hosp_admissions

    def sample(
        self,
        latent_infections: ArrayLike,
        first_latent_infection_dow: int,
        population_size: int,
        model_t_first_latent_infection: int,
        data_observed: ArrayLike = None,
        model_t_observed: ArrayLike = None,
        iedr: ArrayLike = None,
    ) -> ArrayLike:
        """
        Observe and/or predict incident hospital admissions.
        """
        inf_to_hosp_admit = self.inf_to_hosp_admit_rv()

        if self.ihr_rel_iedr_rv is not None and self.ihr_rv is not None:
            raise ValueError(
                "IHR must either be specified "
                "in absolute terms by a non-None "
                "`ihr_rv` or specified relative "
                "to the IEDR by a non-None "
                "`ihr_rel_iedr_rv`, but not both. "
                "Got non-None RVs for both "
                "quantities"
            )
        elif self.ihr_rel_iedr_rv is not None:
            if iedr is None:
                raise ValueError(
                    "Must pass in an IEDR to compute IHR relative to IEDR."
                )
            ihr = iedr[0] * self.ihr_rel_iedr_rv()
            numpyro.deterministic("ihr", ihr)
        elif self.ihr_rv is not None:
            ihr = self.ihr_rv()
        else:
            raise ValueError(
                "Must provide either an ihr_rv "
                "or an ihr_rel_iedr_rv. "
                "Got neither (both were None)."
            )
        latent_hospital_admissions, hospital_admissions_offset = (
            compute_delay_ascertained_incidence(
                p_observed_given_incident=1,
                latent_incidence=(population_size * ihr * latent_infections),
                delay_incidence_to_observation_pmf=(inf_to_hosp_admit),
            )
        )

        model_t_first_latent_admissions = (
            hospital_admissions_offset + model_t_first_latent_infection
        )

        first_latent_admission_dow = (
            first_latent_infection_dow + hospital_admissions_offset
        ) % 7

        predicted_weekly_admissions = daily_to_mmwr_epiweekly(
            latent_hospital_admissions,
            input_data_first_dow=first_latent_admission_dow,
        )

        which_obs_weekly_hosp_admissions = self.calculate_weekly_hosp_indices(
            first_latent_admission_dow,
            model_t_first_latent_admissions,
            model_t_observed,
            n_datapoints=predicted_weekly_admissions.size,
        )
        hospital_admissions_obs_rv = NegativeBinomialObservation(
            "observed_hospital_admissions",
            concentration_rv=self.hosp_admit_neg_bin_concentration_rv,
        )

        observed_hospital_admissions = hospital_admissions_obs_rv(
            mu=predicted_weekly_admissions[which_obs_weekly_hosp_admissions],
            obs=data_observed,
        )

        return observed_hospital_admissions


class WastewaterObservationProcess(RandomVariable):
    """
    Observe and/or predict wastewater concentration
    """

    def __init__(
        self,
        t_peak_rv: RandomVariable,
        duration_shed_after_peak_rv: RandomVariable,
        log10_genome_per_inf_ind_rv: RandomVariable,
        mode_sigma_ww_site_rv: RandomVariable,
        sd_log_sigma_ww_site_rv: RandomVariable,
        mode_sd_ww_site_rv: RandomVariable,
        max_shed_interval: float,
        ww_ml_produced_per_day: float,
        pop_fraction: ArrayLike,
    ) -> None:
        self.t_peak_rv = t_peak_rv
        self.duration_shed_after_peak_rv = duration_shed_after_peak_rv
        self.log10_genome_per_inf_ind_rv = log10_genome_per_inf_ind_rv
        self.mode_sigma_ww_site_rv = mode_sigma_ww_site_rv
        self.sd_log_sigma_ww_site_rv = sd_log_sigma_ww_site_rv
        self.mode_sd_ww_site_rv = mode_sd_ww_site_rv
        self.max_shed_interval = max_shed_interval
        self.ww_ml_produced_per_day = ww_ml_produced_per_day
        self.pop_fraction = pop_fraction

    def validate(self):
        pass

    @staticmethod
    def normed_shedding_cdf(
        time: ArrayLike, t_p: float, t_d: float, log_base: float
    ) -> ArrayLike:
        """
        calculates fraction of total fecal RNA shedding that has occurred
        by a given time post infection.


        Parameters
        ----------
        time: ArrayLike
            Time points to calculate the CDF of viral shedding.
        t_p : float
            Time (in days) from infection to peak shedding.
        t_d: float
            Time (in days) from peak shedding to the end of shedding.
        log_base: float
            Log base used for the shedding kinetics function.


        Returns
        -------
        ArrayLike
            Normalized CDF values of viral shedding at each time point.
        """
        norm_const = (t_p + t_d) * ((log_base - 1) / jnp.log(log_base) - 1)

        def ad_pre(x):
            return t_p / jnp.log(log_base) * jnp.exp(jnp.log(log_base) * x / t_p) - x

        def ad_post(x):
            return (
                -t_d
                / jnp.log(log_base)
                * jnp.exp(jnp.log(log_base) * (1 - ((x - t_p) / t_d)))
                - x
            )

        return (
            jnp.where(
                time < t_p + t_d,
                jnp.where(
                    time < t_p,
                    ad_pre(time) - ad_pre(0),
                    ad_pre(t_p) - ad_pre(0) + ad_post(time) - ad_post(t_p),
                ),
                norm_const,
            )
            / norm_const
        )

    def get_viral_trajectory(
        self,
        tpeak: float,
        duration_shed_after_peak: float,
    ) -> ArrayLike:
        """
        Computes the probability mass function (PMF) of
        daily viral shedding based on a normalized CDF.

        Parameters
        ----------
        tpeak: float
            Time (in days) from infection to peak viral load in shedding.
        duration_shed_after_peak: float
            Duration (in days) of detectable viral shedding after the peak.

        Returns
        -------
        ArrayLike
            Normalized daily viral shedding PMF
        """
        daily_shedding_pmf = self.normed_shedding_cdf(
            jnp.arange(1, self.max_shed_interval),
            tpeak,
            duration_shed_after_peak,
            10,
        ) - self.normed_shedding_cdf(
            jnp.arange(0, self.max_shed_interval - 1),
            tpeak,
            duration_shed_after_peak,
            10,
        )
        return daily_shedding_pmf

    def sample(
        self,
        latent_infections_subpop: ArrayLike,
        data_observed: ArrayLike,
        model_t_first_latent_infection: int,
        ww_uncensored: ArrayLike,
        ww_censored: ArrayLike,
        ww_observed_lab_sites: ArrayLike,
        ww_observed_subpops: ArrayLike,
        ww_model_t_observed: ArrayLike,
        ww_log_lod: ArrayLike,
        lab_site_to_subpop_map: ArrayLike,
        n_ww_lab_sites: int,
        shedding_offset: float,
    ):
        t_peak = self.t_peak_rv()
        dur_shed = self.duration_shed_after_peak_rv()
        viral_kinetics = self.get_viral_trajectory(t_peak, dur_shed)

        def batch_colvolve_fn(m):
            return jnp.convolve(m, viral_kinetics, mode="valid")

        model_net_inf_ind_shedding = jax.vmap(batch_colvolve_fn, in_axes=1, out_axes=1)(
            latent_infections_subpop
        )

        log10_genome_per_inf_ind = self.log10_genome_per_inf_ind_rv()
        expected_obs_viral_genomes = (
            jnp.log(10) * log10_genome_per_inf_ind
            + jnp.log(model_net_inf_ind_shedding + shedding_offset)
            - jnp.log(self.ww_ml_produced_per_day)
        )
        numpyro.deterministic("expected_obs_viral_genomes", expected_obs_viral_genomes)

        mode_sigma_ww_site = self.mode_sigma_ww_site_rv()
        sd_log_sigma_ww_site = self.sd_log_sigma_ww_site_rv()
        mode_sd_ww_site = self.mode_sd_ww_site_rv()

        mode_ww_site_rv = DistributionalVariable(
            "mode_ww_site",
            dist.Normal(0, mode_sd_ww_site),
            reparam=LocScaleReparam(0),
        )  # lab-site specific variation

        sigma_ww_site_rv = TransformedVariable(
            "sigma_ww_site",
            DistributionalVariable(
                "log_sigma_ww_site",
                dist.Normal(jnp.log(mode_sigma_ww_site), sd_log_sigma_ww_site),
                reparam=LocScaleReparam(0),
            ),
            transforms=transformation.ExpTransform(),
        )

        with numpyro.plate("n_ww_lab_sites", n_ww_lab_sites):
            mode_ww_site = mode_ww_site_rv()
            sigma_ww_site = sigma_ww_site_rv()

        viral_genome_offset = viral_kinetics.shape[0] - 1  # max_shed_interval-2
        model_t_first_latent_viral_genome = (
            viral_genome_offset + model_t_first_latent_infection
        )

        if data_observed is not None:
            which_obs_t_viral_genome = (
                ww_model_t_observed - model_t_first_latent_viral_genome
            )
            # multiply the expected observed genomes by the site-specific multiplier at that sampling time
            expected_obs_log_v_site = (
                expected_obs_viral_genomes[
                    which_obs_t_viral_genome, ww_observed_subpops
                ]
                + mode_ww_site[ww_observed_lab_sites]
            )
            site_level_log_ww_conc = DistributionalVariable(
                "site_level_log_ww_conc",
                dist.Normal(
                    loc=expected_obs_log_v_site[ww_uncensored],
                    scale=sigma_ww_site[ww_observed_lab_sites[ww_uncensored]],
                ),
            ).sample(
                obs=data_observed[ww_uncensored],
            )
            if ww_censored.shape[0] != 0:
                log_cdf_values = dist.Normal(
                    loc=expected_obs_log_v_site[ww_censored],
                    scale=sigma_ww_site[ww_observed_lab_sites[ww_censored]],
                ).log_cdf(ww_log_lod[ww_censored])
                numpyro.factor("log_prob_censored", log_cdf_values.sum())
        else:
            which_obs_t_viral_genome = np.s_[
                -model_t_first_latent_viral_genome:
            ]  # Slice time (first) dimension from model t0 to end of the vector
            site_level_log_ww_conc = DistributionalVariable(
                "site_level_log_ww_conc",
                dist.Normal(
                    loc=expected_obs_viral_genomes[
                        which_obs_t_viral_genome, lab_site_to_subpop_map
                    ]
                    + mode_ww_site,
                    scale=sigma_ww_site,
                ),
            )()

        pop_log_latent_viral_genome_conc = jax.scipy.special.logsumexp(
            expected_obs_viral_genomes, axis=1, b=self.pop_fraction
        )

        return site_level_log_ww_conc, pop_log_latent_viral_genome_conc


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
        self.wastewater_obs_process_rv = wastewater_obs_process_rv

    def validate(self) -> None:  # numpydoc ignore=GL08
        pass

    def sample(
        self,
        data: PyrenewHEWData = None,
        sample_ed_visits: bool = False,
        sample_hospital_admissions: bool = False,
        sample_wastewater: bool = False,
    ) -> dict[str, ArrayLike]:  # numpydoc ignore=GL08
        n_init_days = self.latent_infection_process_rv.n_initialization_points
        latent_infections, latent_infections_subpop = self.latent_infection_process_rv(
            n_days_post_init=data.n_days_post_init,
        )
        first_latent_infection_dow = (
            data.first_data_date_overall - dt.timedelta(days=n_init_days)
        ).weekday()

        observed_ed_visits = None
        observed_admissions = None
        site_level_observed_wastewater = None
        population_level_latent_wastewater = None

        iedr = None

        if sample_ed_visits:
            observed_ed_visits, iedr = self.ed_visit_obs_process_rv(
                latent_infections=latent_infections,
                population_size=self.population_size,
                data_observed=data.data_observed_disease_ed_visits,
                model_t_observed=data.model_t_obs_ed_visits,
                model_t_first_latent_infection=-n_init_days,
                right_truncation_offset=data.right_truncation_offset,
            )

        if sample_hospital_admissions:
            observed_admissions = self.hosp_admit_obs_process_rv(
                latent_infections=latent_infections,
                first_latent_infection_dow=first_latent_infection_dow,
                population_size=self.population_size,
                model_t_first_latent_infection=-n_init_days,
                data_observed=data.data_observed_disease_hospital_admissions,
                model_t_observed=data.model_t_obs_hospital_admissions,
                iedr=iedr,
            )
        if sample_wastewater:
            (
                site_level_observed_wastewater,
                population_level_latent_wastewater,
            ) = self.wastewater_obs_process_rv(
                latent_infections_subpop=latent_infections_subpop,
                data_observed=data.data_observed_disease_wastewater_conc,
                model_t_first_latent_infection=-n_init_days,
                ww_uncensored=data.ww_uncensored,
                ww_censored=data.ww_censored,
                ww_observed_lab_sites=data.ww_observed_lab_sites,
                ww_observed_subpops=data.ww_observed_subpops,
                ww_model_t_observed=data.model_t_obs_wastewater,
                ww_log_lod=data.ww_log_lod,
                lab_site_to_subpop_map=data.lab_site_to_subpop_map,
                n_ww_lab_sites=data.n_ww_lab_sites,
                shedding_offset=1e-8,
            )

        return {
            "ed_visits": observed_ed_visits,
            "hospital_admissions": observed_admissions,
            "site_level_wastewater_conc": site_level_observed_wastewater,
            "population_level_latent_wastewater_conc": population_level_latent_wastewater,
        }
