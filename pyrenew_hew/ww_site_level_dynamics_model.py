import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms
import pyrenew.transformation as transformation
from jax.typing import ArrayLike
from numpyro.infer.reparam import LocScaleReparam
from pyrenew.arrayutils import tile_until_n
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


class ww_site_level_dynamics_model(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        state_pop,
        unobs_time,
        n_initialization_points,
        i0_t_offset,
        log_r_t_first_obs_rv,
        autoreg_rt_rv,
        eta_sd_rv,
        i_first_obs_over_n_rv,
        mean_initial_exp_growth_rate_rv,
        offset_ref_logit_i_first_obs_rv,
        offset_ref_initial_exp_growth_rate_rv,
        offset_ref_log_r_t_rv,
        generation_interval_pmf_rv,
        infection_feedback_strength_rv,
        infection_feedback_pmf_rv,
        p_hosp_mean_rv,
        p_hosp_w_sd_rv,
        autoreg_p_hosp_rv,
        hosp_wday_effect_rv,
        inf_to_hosp_rv,
        phi_rv,
        hosp_times,
        pop_fraction,
        n_subpops,
        autoreg_rt_subpop_rv=None,
        sigma_rt_rv=None,
        sigma_i_first_obs_rv=None,
        sigma_initial_exp_growth_rate_rv=None,
        t_peak_rv=None,
        dur_shed_after_peak_rv=None,
        n_ww_lab_sites=None,
        max_shed_interval=None,
        log10_g_rv=None,
        mode_sigma_ww_site_rv=None,
        sd_log_sigma_ww_site_rv=None,
        ww_site_mod_sd_rv=None,
        ww_ml_produced_per_day=None,
        ww_uncensored=None,
        ww_censored=None,
        ww_sampled_lab_sites=None,
        ww_sampled_subpops=None,
        ww_sampled_times=None,
        ww_log_lod=None,
        lab_site_to_subpop_map=None,
        max_ww_sampled_days=None,
        include_ww=False,
    ):  # numpydoc ignore=GL08
        self.state_pop = state_pop
        self.n_subpops = n_subpops
        self.n_ww_lab_sites = n_ww_lab_sites
        self.unobs_time = unobs_time
        self.n_initialization_points = n_initialization_points
        self.max_shed_interval = max_shed_interval
        self.i0_t_offset = i0_t_offset
        self.log_r_t_first_obs_rv = log_r_t_first_obs_rv
        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.t_peak_rv = t_peak_rv
        self.dur_shed_after_peak_rv = dur_shed_after_peak_rv
        self.autoreg_rt_subpop_rv = autoreg_rt_subpop_rv
        self.sigma_rt_rv = sigma_rt_rv
        self.i_first_obs_over_n_rv = i_first_obs_over_n_rv
        self.sigma_i_first_obs_rv = sigma_i_first_obs_rv
        self.sigma_initial_exp_growth_rate_rv = (
            sigma_initial_exp_growth_rate_rv
        )
        self.mean_initial_exp_growth_rate_rv = mean_initial_exp_growth_rate_rv
        self.offset_ref_logit_i_first_obs_rv = offset_ref_logit_i_first_obs_rv
        self.offset_ref_initial_exp_growth_rate_rv = (
            offset_ref_initial_exp_growth_rate_rv
        )
        self.offset_ref_log_r_t_rv = offset_ref_log_r_t_rv
        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.p_hosp_mean_rv = p_hosp_mean_rv
        self.p_hosp_w_sd_rv = p_hosp_w_sd_rv
        self.autoreg_p_hosp_rv = autoreg_p_hosp_rv
        self.hosp_wday_effect_rv = hosp_wday_effect_rv
        self.inf_to_hosp_rv = inf_to_hosp_rv
        self.log10_g_rv = log10_g_rv
        self.mode_sigma_ww_site_rv = mode_sigma_ww_site_rv
        self.sd_log_sigma_ww_site_rv = sd_log_sigma_ww_site_rv
        self.ww_site_mod_sd_rv = ww_site_mod_sd_rv
        self.phi_rv = phi_rv
        self.ww_ml_produced_per_day = ww_ml_produced_per_day
        self.pop_fraction = pop_fraction
        self.ww_uncensored = ww_uncensored
        self.ww_censored = ww_censored
        self.ww_sampled_lab_sites = ww_sampled_lab_sites
        self.ww_sampled_subpops = ww_sampled_subpops
        self.ww_sampled_times = ww_sampled_times
        self.ww_log_lod = ww_log_lod
        self.lab_site_to_subpop_map = lab_site_to_subpop_map
        self.hosp_times = hosp_times
        self.max_ww_sampled_days = max_ww_sampled_days
        self.include_ww = include_ww

        self.inf_with_feedback_proc = InfectionsWithFeedback(
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        )

        self.ar_diff_rt = DifferencedProcess(
            fundamental_process=ARProcess(),
            differencing_order=1,
        )
        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        n_datapoints=None,
        data_observed_hospital_admissions=None,
        data_observed_log_conc=None,
        is_predictive=False,
    ):  # numpydoc ignore=GL08
        if (
            n_datapoints is None
        ):  # calculate model calibration period based on data
            if (
                data_observed_hospital_admissions is None
                and data_observed_log_conc is None
            ):  # no data for calibration
                raise ValueError(
                    "Either n_datapoints or data_observed_hosp_admissions "
                    "must be passed."
                )
            elif (
                data_observed_hospital_admissions is None
                and data_observed_log_conc is not None
            ):  # does not support fitting to just wastewater data
                raise ValueError(
                    "Either n_datapoints or data_observed_hosp_admissions "
                    "must be passed."
                )
            elif (
                data_observed_hospital_admissions is not None
                and data_observed_log_conc is None
            ):  # only fit hosp admissions data
                n_datapoints = len(data_observed_hospital_admissions)
            else:  # both hosp admisssions and ww data provided
                n_datapoints = max(
                    len(data_observed_hospital_admissions),
                    self.max_ww_sampled_days,
                )
        else:
            if (
                data_observed_hospital_admissions is not None
                or data_observed_log_conc is not None
            ):
                raise ValueError(
                    "Cannot pass both n_datapoints and "
                    "data_observed_hospital_admissions "
                    "or data_observed_log_conc"
                )
            else:
                n_datapoints = n_datapoints

        n_weeks_post_init = -((-n_datapoints) // 7)  # n_datapoints // 7 + 1

        eta_sd = self.eta_sd_rv()
        autoreg_rt = self.autoreg_rt_rv()
        log_r_t_first_obs = self.log_r_t_first_obs_rv()  # log_r_mu_intercept
        rt_init_rate_of_change = DistributionalVariable(
            "rt_init_rate_of_change",
            dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
        )()

        log_rtu_weekly = self.ar_diff_rt(
            noise_name="rtu_weekly_diff_first_diff_ar_process_noise",
            n=n_weeks_post_init,
            init_vals=jnp.array(log_r_t_first_obs),
            autoreg=jnp.array(autoreg_rt),
            noise_sd=jnp.array(eta_sd),
            fundamental_process_init_vals=jnp.array(rt_init_rate_of_change),
        )
        numpyro.deterministic("log_rtu_weekly", log_rtu_weekly)

        i_first_obs_over_n = self.i_first_obs_over_n_rv()
        mean_initial_exp_growth_rate = self.mean_initial_exp_growth_rate_rv()
        i_first_obs_over_n_ref_subpop = transforms.SigmoidTransform()(
            transforms.logit(i_first_obs_over_n)
            + jnp.where(
                self.n_subpops > 1, self.offset_ref_logit_i_first_obs_rv(), 0
            )
        )
        initial_exp_growth_rate_ref_subpop = (
            mean_initial_exp_growth_rate
            + jnp.where(
                self.n_subpops > 1,
                self.offset_ref_initial_exp_growth_rate_rv(),
                0,
            )
        )
        log_rtu_weekly_ref_subpop = log_rtu_weekly + jnp.where(
            self.n_subpops > 1, self.offset_ref_log_rt_rv(), 0
        )

        if self.n_subpops == 1:
            i_first_obs_over_n_subpop = i_first_obs_over_n_ref_subpop
            initial_exp_growth_rate_subpop = initial_exp_growth_rate_ref_subpop
            log_rtu_weekly_subpop = log_rtu_weekly_ref_subpop[:, jnp.newaxis]
        else:
            i_first_obs_over_n_non_ref_subpop_rv = TransformedVariable(
                "i_first_obs_over_n_non_ref_subpop",
                DistributionalVariable(
                    "i_first_obs_over_n_non_ref_subpop_raw",
                    dist.Normal(
                        transforms.logit(i_first_obs_over_n),
                        self.sigma_i_first_obs_rv(),
                    ),
                    reparam=LocScaleReparam(0),
                ),
                transforms=transforms.SigmoidTransform(),
            )
            initial_exp_growth_rate_non_ref_subpop_rv = DistributionalVariable(
                "initial_exp_growth_rate_non_ref_subpop_raw",
                dist.Normal(
                    mean_initial_exp_growth_rate,
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
            numpyro.deterministic("rtu_subpop_ar_weekly", rtu_subpop_ar_weekly)
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

        numpyro.deterministic(
            "i_first_obs_over_n_subpop", i_first_obs_over_n_subpop
        )
        numpyro.deterministic(
            "initial_exp_growth_rate_subpop", initial_exp_growth_rate_subpop
        )

        log_i0_subpop = (
            jnp.log(i_first_obs_over_n_subpop)
            - self.unobs_time * initial_exp_growth_rate_subpop
        )
        numpyro.deterministic("log_i0_subpop", log_i0_subpop)

        rtu_subpop = jnp.squeeze(
            jnp.repeat(
                jnp.exp(log_rtu_weekly_subpop),
                repeats=7,
                axis=0,
            )[:n_datapoints, :]
        )
        numpyro.deterministic("rtu_subpop", rtu_subpop)

        i0_subpop_rv = DeterministicVariable(
            "i0_subpop", jnp.exp(log_i0_subpop)
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
                t_pre_init=self.i0_t_offset,
            ),
        )

        generation_interval_pmf = self.generation_interval_pmf_rv()
        i0 = infection_initialization_process()
        numpyro.deterministic("i0", i0)

        inf_with_feedback_proc_sample = self.inf_with_feedback_proc.sample(
            Rt=rtu_subpop,
            I0=i0,
            gen_int=generation_interval_pmf,
        )

        latent_infections_subpop = jnp.atleast_2d(
            jnp.concat(
                [
                    i0,
                    inf_with_feedback_proc_sample.post_initialization_infections,
                ]
            )
        )

        if self.n_subpops == 1:
            latent_infections = jnp.squeeze(latent_infections_subpop)
        else:
            latent_infections = jnp.sum(
                self.pop_fraction * latent_infections_subpop, axis=1
            )

        numpyro.deterministic("latent_infections", latent_infections)
        numpyro.deterministic("rt", inf_with_feedback_proc_sample.rt)

        # Hospital admission component
        p_hosp_mean = self.p_hosp_mean_rv()
        p_hosp_w_sd = self.p_hosp_w_sd_rv()
        autoreg_p_hosp = self.autoreg_p_hosp_rv()

        p_hosp_ar_init_rv = DistributionalVariable(
            "p_hosp_ar_init",
            dist.Normal(
                0,
                p_hosp_w_sd / jnp.sqrt(1 - jnp.pow(autoreg_p_hosp, 2)),
            ),
        )

        p_hosp_ar_init = p_hosp_ar_init_rv()
        p_hosp_ar_proc = ARProcess()
        p_hosp_ar = p_hosp_ar_proc.sample(
            noise_name="p_hosp_noise",
            n=n_weeks_post_init,
            autoreg=autoreg_p_hosp,
            init_vals=p_hosp_ar_init,
            noise_sd=p_hosp_w_sd,
        )

        ihr = jnp.repeat(
            transformation.SigmoidTransform()(p_hosp_ar + p_hosp_mean),
            repeats=7,
        )[:n_datapoints]
        numpyro.deterministic("ihr", ihr)

        hosp_wday_effect_raw = self.hosp_wday_effect_rv()
        inf_to_hosp = self.inf_to_hosp_rv()

        hosp_wday_effect = tile_until_n(hosp_wday_effect_raw, n_datapoints)

        potential_latent_hospital_admissions, hospital_admissions_offset = (
            compute_delay_ascertained_incidence(
                p_observed_given_incident=1,
                latent_incidence=latent_infections,
                delay_incidence_to_observation_pmf=inf_to_hosp,
            )
        )

        latent_hospital_admissions = (
            potential_latent_hospital_admissions[-n_datapoints:]
            * ihr
            * hosp_wday_effect
            * self.state_pop
        )
        numpyro.deterministic(
            "latent_hospital_admissions", latent_hospital_admissions
        )

        hospital_admission_obs_rv = NegativeBinomialObservation(
            "observed_hospital_admissions", concentration_rv=self.phi_rv
        )

        if not is_predictive:
            mu_obs_hosp = latent_hospital_admissions[self.hosp_times]
        else:
            mu_obs_hosp = latent_hospital_admissions

        observed_hospital_admissions = hospital_admission_obs_rv(
            mu=mu_obs_hosp,
            obs=data_observed_hospital_admissions,
        )

        # wastewater component
        if self.include_ww:
            t_peak = self.t_peak_rv()
            dur_shed = self.dur_shed_after_peak_rv()
            s = get_viral_trajectory(t_peak, dur_shed, self.max_shed_interval)

            def batch_colvolve_fn(m):
                return jnp.convolve(m, s, mode="valid")

            model_net_i = jax.vmap(batch_colvolve_fn, in_axes=1, out_axes=1)(
                latent_infections_subpop
            )[-n_datapoints:, :]
            numpyro.deterministic("model_net_i", model_net_i)

            log10_g = self.log10_g_rv()
            model_log_v_ot = (
                jnp.log(10) * log10_g
                + jnp.log(model_net_i + 1e-8)
                - jnp.log(self.ww_ml_produced_per_day)
            )  # expected observed viral genomes/mL at all observed and forecasted times
            numpyro.deterministic("model_log_v_ot", model_log_v_ot)

            mode_sigma_ww_site = self.mode_sigma_ww_site_rv()
            sd_log_sigma_ww_site = self.sd_log_sigma_ww_site_rv()
            ww_site_mod_sd = self.ww_site_mod_sd_rv()

            ww_site_mod_rv = DistributionalVariable(
                "ww_site_mod",
                dist.Normal(0, ww_site_mod_sd),
                reparam=LocScaleReparam(0),
            )  # lab-site specific variation

            sigma_ww_site_rv = TransformedVariable(
                "sigma_ww_site",
                DistributionalVariable(
                    "log_sigma_ww_site",
                    dist.Normal(
                        jnp.log(mode_sigma_ww_site), sd_log_sigma_ww_site
                    ),
                    reparam=LocScaleReparam(0),
                ),
                transforms=transforms.ExpTransform(),
            )

            with numpyro.plate("n_ww_lab_sites", self.n_ww_lab_sites):
                ww_site_mod = ww_site_mod_rv()
                sigma_ww_site = sigma_ww_site_rv()

            # expected observations at each site in log scale
            exp_obs_log_v_true = model_log_v_ot[
                self.ww_sampled_times, self.ww_sampled_subpops
            ]

            # multiply the expected observed genomes by the site-specific multiplier at that sampling time
            exp_obs_log_v = (
                exp_obs_log_v_true + ww_site_mod[self.ww_sampled_lab_sites]
            )

            numpyro.sample(
                "log_conc_obs",
                dist.Normal(
                    loc=exp_obs_log_v[self.ww_uncensored],
                    scale=sigma_ww_site[
                        self.ww_sampled_lab_sites[self.ww_uncensored]
                    ],
                ),
                obs=(
                    data_observed_log_conc[self.ww_uncensored]
                    if data_observed_log_conc is not None
                    else None
                ),
            )
            if self.ww_censored.shape[0] != 0:
                log_cdf_values = dist.Normal(
                    loc=exp_obs_log_v[self.ww_censored],
                    scale=sigma_ww_site[
                        self.ww_sampled_lab_sites[self.ww_censored]
                    ],
                ).log_cdf(self.ww_log_lod[self.ww_censored])
                numpyro.factor("log_prob_censored", log_cdf_values.sum())

            site_ww_pred_log = numpyro.sample(
                "site_ww_pred_log",
                dist.Normal(
                    loc=model_log_v_ot[:, self.lab_site_to_subpop_map]
                    + ww_site_mod,
                    scale=sigma_ww_site,
                ),
            )

            state_model_net_i = jnp.convolve(
                latent_infections, s, mode="valid"
            )[-n_datapoints:]
            numpyro.deterministic("state_model_net_i", state_model_net_i)

            state_log_c = (
                jnp.log(10) * log10_g
                + jnp.log(state_model_net_i + 1e-8)
                - jnp.log(self.ww_ml_produced_per_day)
            )
            numpyro.deterministic("state_log_c", state_log_c)

            expected_state_ww_conc = jnp.exp(state_log_c)
            numpyro.deterministic(
                "expected_state_ww_conc", expected_state_ww_conc
            )

            state_rt = (
                latent_infections[-n_datapoints:]
                / jnp.convolve(
                    latent_infections,
                    jnp.hstack(
                        (jnp.array([0]), jnp.array(generation_interval_pmf))
                    ),
                    mode="valid",
                )[-n_datapoints:]
            )
            numpyro.deterministic("state_rt", state_rt)

        return (
            observed_hospital_admissions,
            site_ww_pred_log if self.include_ww else None,
        )


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
        return (
            t_p / jnp.log(log_base) * jnp.exp(jnp.log(log_base) * x / t_p) - x
        )

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
    tpeak: float, duration_shedding_after_peak: float, max_days: int
):
    """
    Computes the probability mass function (PMF) of
    daily viral shedding based on a normalized CDF.

    Parameters
    ----------
    tpeak: float
        Time (in days) from infection to peak viral load in shedding.
    duration_shedding_after_peak: float
        Duration (in days) of detectable viral shedding after the peak.
    max_days: int
        Maximum number of days to calculate the shedding trajectory.

    Returns
    -------
    ArrayLike
        Normalized daily viral shedding PMF
    """
    daily_shedding_pmf = normed_shedding_cdf(
        jnp.arange(1, max_days), tpeak, duration_shedding_after_peak, 10
    ) - normed_shedding_cdf(
        jnp.arange(0, max_days - 1), tpeak, duration_shedding_after_peak, 10
    )
    return daily_shedding_pmf
