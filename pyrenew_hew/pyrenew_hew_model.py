# numpydoc ignore=GL08
import datetime

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from jax.typing import ArrayLike
from pyrenew.arrayutils import repeat_until_n, tile_until_n
from pyrenew.convolve import (
    compute_delay_ascertained_incidence,
    daily_to_mmwr_epiweekly,
)
from pyrenew.latent import (
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.metaclass import Model, RandomVariable
from pyrenew.observation import NegativeBinomialObservation
from pyrenew.process import ARProcess, DifferencedProcess
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData


class LatentInfectionProcess(RandomVariable):
    def __init__(
        self,
        i0_first_obs_n_rv: RandomVariable,
        initialization_rate_rv: RandomVariable,
        log_r_mu_intercept_rv: RandomVariable,
        autoreg_rt_rv: RandomVariable,  # ar coeff for AR(1) on R'(t)
        eta_sd_rv: RandomVariable,  # sd of random walk for AR(1) on R'(t)
        generation_interval_pmf_rv: RandomVariable,
        infection_feedback_strength_rv: RandomVariable,
        infection_feedback_pmf_rv: RandomVariable,
        n_initialization_points: int,
    ) -> None:
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

        self.ar_diff = DifferencedProcess(
            fundamental_process=ARProcess(),
            differencing_order=1,
        )

        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.log_r_mu_intercept_rv = log_r_mu_intercept_rv
        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.infection_feedback_pmf_rv = infection_feedback_pmf_rv

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
        i0 = self.infection_initialization_process()

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

        rtu = repeat_until_n(
            data=jnp.exp(log_rtu_weekly),
            n_timepoints=n_days_post_init,
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
        data_observed: ArrayLike,
        n_datapoints: int,
        right_truncation_offset: int = None,
    ) -> tuple[ArrayLike]:
        """
        Observe and/or predict ED visit values
        """
        p_ed_mean = self.p_ed_mean_rv()
        p_ed_w_sd = self.p_ed_w_sd_rv()
        autoreg_p_ed = self.autoreg_p_ed_rv()
        n_weeks_p_ed_ar = n_datapoints // 7 + 1

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
        )[:n_datapoints]
        # this is only applied after the ed visits are generated, not to all
        # the latent infections. This is why we cannot apply the iedr in
        # compute_delay_ascertained_incidence
        # see https://github.com/CDCgov/ww-inference-model/issues/43

        numpyro.deterministic("iedr", iedr)

        ed_wday_effect_raw = self.ed_wday_effect_rv()
        ed_wday_effect = tile_until_n(ed_wday_effect_raw, n_datapoints)

        inf_to_ed = self.inf_to_ed_rv()

        potential_latent_ed_visits = compute_delay_ascertained_incidence(
            p_observed_given_incident=1,
            latent_incidence=latent_infections,
            delay_incidence_to_observation_pmf=inf_to_ed,
        )[-n_datapoints:]

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
                n_datapoints - prop_already_reported_tail.shape[0]
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

        sampled_ed_visits = ed_visit_obs_rv(
            mu=latent_ed_visits_now,
            obs=data_observed,
        )

        return sampled_ed_visits, iedr


class HospAdmitObservationProcess(RandomVariable):
    def __init__(
        self,
        inf_to_hosp_admit_rv: RandomVariable,
        hosp_admit_neg_bin_concentration_rv: RandomVariable,
        ihr_rv: RandomVariable = None,
        ihr_rel_iedr_rv: RandomVariable = None,
    ) -> None:
        self.inf_to_hosp_admit_rv = inf_to_hosp_admit_rv
        self.hosp_admit_neg_bin_concentration_rv = (
            hosp_admit_neg_bin_concentration_rv
        )
        self.ihr_rv = ihr_rv
        self.ihr_rel_iedr_rv = ihr_rel_iedr_rv

    def validate(self):
        pass

    def sample(
        self,
        latent_infections: ArrayLike,
        first_latent_infection_dow: int,
        population_size: int,
        n_datapoints: int,
        data_observed: ArrayLike = None,
        iedr: ArrayLike = None,
    ) -> ArrayLike:
        """
        Observe and/or predict incident hospital admissions.
        """
        inf_to_hosp_admit = self.inf_to_hosp_admit_rv()

        if iedr is not None:
            ihr_rel_iedr = self.ihr_rel_iedr_rv()
            ihr = iedr[0] * ihr_rel_iedr
            numpyro.deterministic("ihr", ihr)
        else:
            ihr = self.ihr_rv()

        latent_admissions = population_size * ihr * latent_infections
        latent_hospital_admissions = compute_delay_ascertained_incidence(
            p_observed_given_incident=1,
            latent_incidence=latent_admissions,
            delay_incidence_to_observation_pmf=inf_to_hosp_admit,
        )

        longest_possible_delay = inf_to_hosp_admit.shape[0]

        # we should add functionality to automate this,
        # along with tests
        first_latent_admission_dow = (
            first_latent_infection_dow + longest_possible_delay
        ) % 7

        predicted_weekly_admissions = daily_to_mmwr_epiweekly(
            latent_hospital_admissions,
            input_data_first_dow=first_latent_admission_dow,
        )

        hospital_admissions_obs_rv = NegativeBinomialObservation(
            "observed_hospital_admissions",
            concentration_rv=self.hosp_admit_neg_bin_concentration_rv,
        )

        sampled_admissions = hospital_admissions_obs_rv(
            mu=predicted_weekly_admissions[-n_datapoints:], obs=data_observed
        )

        return sampled_admissions


class WastewaterObservationProcess(RandomVariable):
    """
    Placeholder for wastewater obs process
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
        self.wastewater_obs_process_rv = wastewater_obs_process_rv

    def validate(self) -> None:  # numpydoc ignore=GL08
        pass

    def sample(
        self,
        data: PyrenewHEWData = None,
        sample_ed_visits: bool = False,
        sample_hospital_admissions: bool = False,
        sample_wastewater: bool = False,
    ) -> ArrayLike:  # numpydoc ignore=GL08
        latent_infections = self.latent_infection_process_rv(
            n_days_post_init=data.n_days_post_init,
        )
        n_init_days = self.latent_infection_process_rv.infection_initialization_process.infection_init_method.n_timepoints
        first_latent_infection_dow = (
            data.first_data_date_overall - datetime.timedelta(days=n_init_days)
        ).weekday()

        sampled_ed_visits, sampled_admissions, sampled_wastewater = (
            None,
            None,
            None,
        )

        iedr = None

        if sample_ed_visits:
            sampled_ed_visits, iedr = self.ed_visit_obs_process_rv(
                latent_infections=latent_infections,
                population_size=self.population_size,
                data_observed=data.data_observed_disease_ed_visits,
                n_datapoints=data.n_ed_visits_datapoints,
                right_truncation_offset=data.right_truncation_offset,
            )

        if sample_hospital_admissions:
            sampled_admissions = self.hosp_admit_obs_process_rv(
                latent_infections=latent_infections,
                first_latent_infection_dow=first_latent_infection_dow,
                population_size=self.population_size,
                n_datapoints=data.n_hospital_admissions_datapoints,
                data_observed=(data.data_observed_disease_hospital_admissions),
                iedr=iedr,
            )
        if sample_wastewater:
            sampled_wastewater = self.wastewater_obs_process_rv()

        return {
            "ed_visits": sampled_ed_visits,
            "hospital_admissions": sampled_admissions,
            "wasewater": sampled_wastewater,
        }
