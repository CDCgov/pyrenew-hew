import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms
import pyrenew.transformation as transformation
from pyrenew.arrayutils import tile_until_n
from pyrenew.deterministic import DeterministicVariable
from pyrenew.latent import (
    InfectionInitializationProcess,
    InfectionsWithFeedback,
    InitializeInfectionsExponentialGrowth,
)
from pyrenew.metaclass import Model
from pyrenew.process import ARProcess, RtWeeklyDiffARProcess
from pyrenew.randomvariable import DistributionalVariable
from pyrenew.observation import NegativeBinomialObservation
from pyrenew_covid_wastewater.utils import get_vl_trajectory


class ww_site_level_dynamics_model(Model):  # numpydoc ignore=GL08
    def __init__(
        self,
        state_pop,
        n_subpops,
        n_initialization_points,
        gt_max,
        i0_t_offset,
        log_r_mu_intercept_rv,
        autoreg_rt_rv,
        eta_sd_rv,
        t_peak_rv,
        viral_peak_rv,
        dur_shed_rv,
        autoreg_rt_site_rv,
        sigma_rt_rv,
        i0_over_n_rv,
        sigma_i0_rv,
        eta_i0_rv,
        initialization_rate_rv,
        eta_growth_rv,
        sigma_growth_rv,
        generation_interval_pmf_rv,
        infection_feedback_strength_rv,
        infection_feedback_pmf_rv,
        p_hosp_mean_rv,
        p_hosp_w_sd_rv,
        autoreg_p_hosp_rv,
        hosp_wday_effect_rv,
        inf_to_hosp_rv,
        log10_g_rv,
        mode_sigma_ww_site_rv,
        sd_log_sigma_ww_site_rv,
        eta_log_sigma_ww_site_rv,
        ww_site_mod_raw_rv,
        ww_site_mod_sd_rv,
        phi_rv,
        ww_ml_produced_per_day,
        pop_fraction_reshaped,
        ww_uncensored,
        ww_censored,
        ww_sampled_lab_sites,
        ww_sampled_sites,
        ww_sampled_times,
        ww_log_lod,
    ):  # numpydoc ignore=GL08
        self.state_pop = state_pop
        self.n_subpops = n_subpops
        self.n_initialization_points = n_initialization_points
        self.gt_max = gt_max
        self.i0_t_offset = i0_t_offset
        self.log_r_mu_intercept_rv = log_r_mu_intercept_rv
        self.autoreg_rt_rv = autoreg_rt_rv
        self.eta_sd_rv = eta_sd_rv
        self.t_peak_rv = t_peak_rv
        self.viral_peak_rv = viral_peak_rv
        self.dur_shed_rv = dur_shed_rv
        self.autoreg_rt_site_rv = autoreg_rt_site_rv
        self.sigma_rt_rv = sigma_rt_rv
        self.i0_over_n_rv = i0_over_n_rv
        self.sigma_i0_rv = sigma_i0_rv
        self.eta_i0_rv = eta_i0_rv
        self.initial_growth_rv = initialization_rate_rv
        self.eta_growth_rv = eta_growth_rv
        self.sigma_growth_rv = sigma_growth_rv
        self.generation_interval_pmf_rv = generation_interval_pmf_rv
        self.p_hosp_mean_rv = p_hosp_mean_rv
        self.p_hosp_w_sd_rv = p_hosp_w_sd_rv
        self.autoreg_p_hosp_rv = autoreg_p_hosp_rv
        self.hosp_wday_effect_rv = hosp_wday_effect_rv
        self.inf_to_hosp_rv = inf_to_hosp_rv
        self.log10_g_rv = log10_g_rv
        self.mode_sigma_ww_site_rv = mode_sigma_ww_site_rv
        self.sd_log_sigma_ww_site_rv = sd_log_sigma_ww_site_rv
        self.eta_log_sigma_ww_site_rv = eta_log_sigma_ww_site_rv
        self.ww_site_mod_raw_rv = ww_site_mod_raw_rv
        self.ww_site_mod_sd_rv = ww_site_mod_sd_rv
        self.phi_rv = phi_rv
        self.ww_ml_produced_per_day = ww_ml_produced_per_day
        self.pop_fraction_reshaped = pop_fraction_reshaped
        self.ww_uncensored = ww_uncensored
        self.ww_censored = ww_censored
        self.ww_sampled_lab_sites = ww_sampled_lab_sites
        self.ww_sampled_sites = ww_sampled_sites
        self.ww_sampled_times = ww_sampled_times
        self.ww_log_lod = ww_log_lod

        self.inf_with_feedback_proc = InfectionsWithFeedback(
            infection_feedback_strength=infection_feedback_strength_rv,
            infection_feedback_pmf=infection_feedback_pmf_rv,
        )
        return None

    def validate(self):  # numpydoc ignore=GL08
        return None

    def sample(
        self,
        n_datapoints=None,
        data_observed_hospital_admissions=None,
        data_observed_log_conc=None,
    ):  # numpydoc ignore=GL08
        if n_datapoints is None and data_observed_hospital_admissions is None:
            raise ValueError(
                "Either n_datapoints or data_observed_hosp_admissions "
                "must be passed."
            )
        elif n_datapoints is not None and data_observed_hospital_admissions is not None:
            raise ValueError(
                "Cannot pass both n_datapoints and data_observed_hospital_admissions."
            )
        elif n_datapoints is None:
            n_datapoints = len(data_observed_hospital_admissions)
        else:
            n_datapoints = n_datapoints

        n_weeks = n_datapoints // 7 + 1

        eta_sd = self.eta_sd_rv()[0].value
        autoreg_rt = self.autoreg_rt_rv()[0].value
        log_r_mu_intercept = self.log_r_mu_intercept_rv()[0].value

        autoreg_rt_det_rv = DeterministicVariable("autoreg_rt_det", autoreg_rt)
        init_rate_of_change_rv = DistributionalVariable(
            "init_rate_of_change",
            dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
        )

        init_rate_of_change = init_rate_of_change_rv()[0].value

        rt_proc = RtWeeklyDiffARProcess(
            name="rtu_state_weekly_diff",
            offset=0,
            log_rt_rv=DeterministicVariable(
                name="log_rt",
                value=jnp.array(
                    [
                        log_r_mu_intercept,
                        log_r_mu_intercept + init_rate_of_change,
                    ]
                ),
            ),
            autoreg_rv=autoreg_rt_det_rv,
            periodic_diff_sd_rv=DeterministicVariable(
                name="periodic_diff_sd", value=jnp.array(eta_sd)
            ),
        )

        rtu = rt_proc.sample(
            duration=n_datapoints
        )  # log_r_mu_t_in_weeks in stan - not log anymore and not weekly either

        t_peak = self.t_peak_rv()[0].value
        viral_peak = self.viral_peak_rv()[0].value
        dur_shed = self.dur_shed_rv()[0].value

        s = get_vl_trajectory(t_peak, viral_peak, dur_shed, self.gt_max)

        # Site-level Rt, to be repeated for each site
        r_site_t = jnp.zeros((self.n_subpops, n_datapoints))
        new_i_site_matrix = jnp.zeros(
            (self.n_subpops, n_datapoints + self.n_initialization_points)
        )
        model_log_v_ot = jnp.zeros((self.n_subpops, n_datapoints))

        for i in range(self.n_subpops):
            autoreg_rt_site = self.autoreg_rt_site_rv()[0].value
            sigma_rt = self.sigma_rt_rv()[0].value

            rtu_site_ar_init_rv = DistributionalVariable(
                "rtu_site_ar_init",
                dist.Normal(
                    0,
                    sigma_rt / jnp.sqrt(1 - jnp.pow(autoreg_rt_site, 2)),
                ),
            )

            rtu_site_ar_proc = ARProcess(noise_rv_name="rtu_ar_proc")

            rtu_site_ar_init = rtu_site_ar_init_rv()[0].value
            rtu_site_ar_weekly = rtu_site_ar_proc(
                n=n_weeks,
                init_vals=rtu_site_ar_init,
                autoreg=autoreg_rt_site,
                noise_sd=sigma_rt,
            )

            rtu_site_ar = jnp.repeat(
                transformation.ExpTransform()(rtu_site_ar_weekly[0].value),
                repeats=7,
            )[:n_datapoints]

            rtu_site = (
                rtu_site_ar + rtu.rt.value
            )  # this reults in more sensible values but it should be as below?
            # rtu_site = rtu_site_ar*rtu.rt.value

            # Site level disease dynamic estimates!
            i0_over_n = self.i0_over_n_rv()
            sigma_i0 = self.sigma_i0_rv()
            eta_i0 = self.eta_i0_rv()
            initial_growth = self.initial_growth_rv()
            eta_growth = self.eta_growth_rv()
            sigma_growth = self.sigma_growth_rv()

            # Calculate infection and adjusted Rt for each sight using site-level i0 `i0_site_over_n` and initialization rate `growth_site`
            # These are computed as a vector in stan code, but iid implementation is probably better for using numpyro.plate

            #  site level growth rate
            growth_site = (
                initial_growth[0].value + eta_growth[0].value * sigma_growth[0].value
            )

            growth_site_rv = DeterministicVariable(
                "growth_site_rv", jnp.array(growth_site)
            )

            # site-level initial per capita infection incidence
            i0_site_over_n = jax.nn.sigmoid(
                transforms.logit(i0_over_n[0].value)
                + eta_i0[0].value * sigma_i0[0].value
            )

            i0_site_over_n_rv = DeterministicVariable(
                "i0_site_over_n_rv", jnp.array(i0_site_over_n)
            )

            infection_initialization_process = InfectionInitializationProcess(
                "I0_initialization",
                i0_site_over_n_rv,
                InitializeInfectionsExponentialGrowth(
                    self.n_initialization_points,
                    growth_site_rv,
                    t_pre_init=self.i0_t_offset,
                ),
                t_unit=1,
            )

            generation_interval_pmf = self.generation_interval_pmf_rv()
            i0 = infection_initialization_process()

            inf_with_feedback_proc_sample = self.inf_with_feedback_proc.sample(
                Rt=rtu_site,
                I0=i0[0].value,
                gen_int=generation_interval_pmf[0].value,
            )

            new_i_site = jnp.concat(
                [
                    i0[0].value,
                    inf_with_feedback_proc_sample.post_initialization_infections.value,
                ]
            )
            r_site_t = r_site_t.at[i, :].set(inf_with_feedback_proc_sample.rt.value)
            new_i_site_matrix = new_i_site_matrix.at[i, :].set(new_i_site)

            # number of net infected individuals shedding on each day (sum of individuals in dift stages of infection)
            model_net_i = jnp.convolve(new_i_site, s, mode="valid")[-n_datapoints:]

            log10_g = self.log10_g_rv()

            # expected observed viral genomes/mL at all observed and forecasted times
            # [n_subpops, ot + ht] model_log_v_ot   aka do it for all subpop
            model_log_v_ot_site = (
                jnp.log(10) * log10_g[0].value
                + jnp.log(model_net_i[:(n_datapoints)] + 1e-8)
                - jnp.log(self.ww_ml_produced_per_day)
            )
            model_log_v_ot = model_log_v_ot.at[i, :].set(model_log_v_ot_site)

        state_inf_per_capita = jnp.sum(
            self.pop_fraction_reshaped * new_i_site_matrix, axis=0
        )
        # Hospital admission component

        # p_hosp_w is std_normal - weekly random walk for IHR

        p_hosp_mean = self.p_hosp_mean_rv()
        p_hosp_w_sd = self.p_hosp_w_sd_rv()
        autoreg_p_hosp = self.autoreg_p_hosp_rv()

        p_hosp_ar_proc = ARProcess("p_hosp")

        p_hosp_ar_init_rv = DistributionalVariable(
            "p_hosp_ar_init",
            dist.Normal(
                0,
                p_hosp_w_sd[0].value
                / jnp.sqrt(1 - jnp.pow(autoreg_p_hosp[0].value, 2)),
            ),
        )

        p_hosp_ar_init = p_hosp_ar_init_rv()
        p_hosp_ar = p_hosp_ar_proc.sample(
            n=n_weeks,
            autoreg=autoreg_p_hosp[0].value,
            init_vals=p_hosp_ar_init[0].value,
            noise_sd=p_hosp_w_sd[0].value,
        )

        ihr = jnp.repeat(
            transformation.SigmoidTransform()(
                p_hosp_ar[0].value + p_hosp_mean[0].value
            ),
            repeats=7,
        )[:n_datapoints]

        hosp_wday_effect_raw = self.hosp_wday_effect_rv()[0].value
        inf_to_hosp = self.inf_to_hosp_rv()[0].value

        hosp_wday_effect = tile_until_n(hosp_wday_effect_raw, n_datapoints)

        potential_latent_hospital_admissions = jnp.convolve(
            state_inf_per_capita,
            inf_to_hosp,
            mode="valid",
        )[-n_datapoints:]

        latent_hospital_admissions = (
            potential_latent_hospital_admissions
            * ihr
            * hosp_wday_effect
            * self.state_pop
        )

        mode_sigma_ww_site = self.mode_sigma_ww_site_rv()[0].value
        sd_log_sigma_ww_site = self.sd_log_sigma_ww_site_rv()[0].value
        eta_log_sigma_ww_site = self.eta_log_sigma_ww_site_rv()[0].value
        ww_site_mod_raw = self.ww_site_mod_raw_rv()[0].value
        ww_site_mod_sd = self.ww_site_mod_sd_rv()[0].value

        # These are the true expected genomes at the site level before observation error
        # (which is at the lab-site level)
        exp_obs_log_v_true = model_log_v_ot[
            self.ww_sampled_sites, self.ww_sampled_times
        ]

        # modify by lab-site specific variation (multiplier!)
        ww_site_mod = ww_site_mod_raw * ww_site_mod_sd

        # LHS log transformed obs genomes per person-day, RHS multiplies the expected observed
        # genomes by the site-specific multiplier at that sampling time
        exp_obs_log_v = exp_obs_log_v_true + ww_site_mod[self.ww_sampled_lab_sites]

        sigma_ww_site = jnp.exp(
            jnp.log(mode_sigma_ww_site) + sd_log_sigma_ww_site * eta_log_sigma_ww_site
        )

        # g = jnp.power(
        #     log10_g[0].value, 10
        # )  # Estimated genomes shed per infected individual

        log_conc_obs = numpyro.sample(
            "log_conc",
            dist.Normal(
                loc=exp_obs_log_v[self.ww_uncensored],
                scale=sigma_ww_site[self.ww_sampled_lab_sites[self.ww_uncensored]],
            ),
            obs=data_observed_log_conc[self.ww_uncensored],
        )

        if self.ww_censored.shape[0] != 0:
            log_cdf_values = dist.Normal(
                loc=exp_obs_log_v[self.ww_censored],
                scale=sigma_ww_site[self.ww_sampled_lab_sites[self.ww_censored]],
            ).log_cdf(self.ww_log_lod[self.ww_censored])

            numpyro.factor("log_prob_censored", log_cdf_values.sum())

        hospital_admission_obs_rv = NegativeBinomialObservation(
            "observed_hospital_admissions", concentration_rv=self.phi_rv
        )

        observed_hospital_admissions = hospital_admission_obs_rv(
            mu=latent_hospital_admissions,
            obs=data_observed_hospital_admissions,
        )

        return (observed_hospital_admissions, log_conc_obs)
