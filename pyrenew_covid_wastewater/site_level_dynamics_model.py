import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms
from pyrenew.deterministic import DeterministicVariable, DeterministicPMF
from pyrenew.process import ARProcess, RtWeeklyDiffARProcess
import pyrenew.transformation as transformation
from pyrenew.latent import (
    InfectionInitializationProcess,
    InitializeInfectionsExponentialGrowth,
    InfectionsWithFeedback,
)

from pyrenew.observation import NegativeBinomialObservation
from pyrenew.arrayutils import tile_until_n
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable


with numpyro.handlers.seed(rng_seed=223):
    eta_sd = eta_sd_rv()[0].value
    autoreg_rt = autoreg_rt_rv()[0].value
    log_r_mu_intercept = log_r_mu_intercept_rv()[0].value

autoreg_rt_det_rv = DeterministicVariable("autoreg_rt_det", autoreg_rt)

init_rate_of_change_rv = DistributionalVariable(
    "init_rate_of_change",
    dist.Normal(0, eta_sd / jnp.sqrt(1 - jnp.pow(autoreg_rt, 2))),
)

with numpyro.handlers.seed(rng_seed=223):
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

with numpyro.handlers.seed(rng_seed=223):
    rtu = rt_proc.sample(
        duration=n_datapoints
    )  # log_r_mu_t_in_weeks in stan - not log anymore and not weekly either


with numpyro.handlers.seed(rng_seed=223):
    t_peak = t_peak_rv()
    viral_peak = viral_peak_rv()
    dur_shed = dur_shed_rv()

s = get_vl_trajectory(t_peak[0].value, viral_peak[0].value, dur_shed[0].value, gt_max)


# Site-level Rt, to be repeated for each site
r_site_t = jnp.zeros((n_subpops, obs_time + horizon_time))
new_i_site_matrix = jnp.zeros((n_subpops, n_datapoints + n_initialization_points))
model_log_v_ot = jnp.zeros((n_subpops, obs_time + horizon_time))

for i in range(n_subpops):
    with numpyro.handlers.seed(rng_seed=223):
        autoreg_rt_site = autoreg_rt_site_rv()
        sigma_rt = sigma_rt_rv()

    rtu_site_ar_init_rv = DistributionalVariable(
        "rtu_site_ar_init",
        dist.Normal(
            0,
            sigma_rt[0].value / jnp.sqrt(1 - jnp.pow(autoreg_rt_site[0].value, 2)),
        ),
    )

    rtu_site_ar_proc = ARProcess(noise_rv_name="rtu_ar_proc")

    with numpyro.handlers.seed(rng_seed=223):
        rtu_site_ar_init = rtu_site_ar_init_rv()
        rtu_site_ar_weekly = rtu_site_ar_proc(
            n=n_weeks,
            init_vals=rtu_site_ar_init[0].value,
            autoreg=autoreg_rt_site[0].value,
            noise_sd=sigma_rt[0].value,
        )

    rtu_site_ar = jnp.repeat(
        transformation.ExpTransform()(rtu_site_ar_weekly[0].value), repeats=7
    )[:n_datapoints]

    rtu_site = (
        rtu_site_ar + rtu.rt.value
    )  # this reults in more sensible values but it should be as below?
    # rtu_site = rtu_site_ar*rtu.rt.value

    # Site level disease dynamic estimates!
    with numpyro.handlers.seed(rng_seed=223):
        i0_over_n = i0_over_n_rv()
        sigma_i0 = sigma_i0_rv()
        eta_i0 = eta_i0_rv()
        initial_growth = initialization_rate_rv()
        eta_growth = eta_growth_rv()
        sigma_growth = sigma_growth_rv()

    # Calculate infection and adjusted Rt for each sight using site-level i0 `i0_site_over_n` and initialization rate `growth_site`
    # These are computed as a vector in stan code, but iid implementation is probably better for using numpyro.plate

    #  site level growth rate
    growth_site = initial_growth[0].value + eta_growth[0].value * sigma_growth[0].value

    growth_site_rv = DeterministicVariable("growth_site_rv", jnp.array(growth_site))

    # site-level initial per capita infection incidence
    i0_site_over_n = jax.nn.sigmoid(
        transforms.logit(i0_over_n[0].value) + eta_i0[0].value * sigma_i0[0].value
    )

    i0_site_over_n_rv = DeterministicVariable(
        "i0_site_over_n_rv", jnp.array(i0_site_over_n)
    )

    infection_initialization_process = InfectionInitializationProcess(
        "I0_initialization",
        i0_site_over_n_rv,
        InitializeInfectionsExponentialGrowth(
            n_initialization_points,
            growth_site_rv,
            t_pre_init=i0_t_offset,
        ),
        t_unit=1,
    )

    with numpyro.handlers.seed(rng_seed=223):
        generation_interval_pmf = generation_interval_pmf_rv()
        i0 = infection_initialization_process()
        inf_with_feedback_proc_sample = inf_with_feedback_proc.sample(
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

    with numpyro.handlers.seed(rng_seed=223):
        log10_g = log10_g_rv()

    # expected observed viral genomes/mL at all observed and forecasted times
    # [n_subpops, ot + ht] model_log_v_ot   aka do it for all subpop
    model_log_v_ot_site = (
        jnp.log(10) * log10_g[0].value
        + jnp.log(model_net_i[: (obs_time + horizon_time)] + 1e-8)
        - jnp.log(ww_ml_produced_per_day)
    )
    model_log_v_ot = model_log_v_ot.at[i, :].set(model_log_v_ot_site)

state_inf_per_capita = jnp.sum(pop_fraction_reshaped * new_i_site_matrix, axis=0)
# Hospital admission component


# p_hosp_w is std_normal - weekly random walk for IHR

with numpyro.handlers.seed(rng_seed=223):
    p_hosp_mean = p_hosp_mean_rv()
    p_hosp_w_sd = p_hosp_w_sd_rv()
    autoreg_p_hosp = autoreg_p_hosp_rv()

p_hosp_ar_proc = ARProcess("p_hosp")

p_hosp_ar_init_rv = DistributionalVariable(
    "p_hosp_ar_init",
    dist.Normal(
        0,
        p_hosp_w_sd[0].value / jnp.sqrt(1 - jnp.pow(autoreg_p_hosp[0].value, 2)),
    ),
)

with numpyro.handlers.seed(rng_seed=223):
    p_hosp_ar_init = p_hosp_ar_init_rv()
    p_hosp_ar = p_hosp_ar_proc.sample(
        n=n_weeks,
        autoreg=autoreg_p_hosp[0].value,
        init_vals=p_hosp_ar_init[0].value,
        noise_sd=p_hosp_w_sd[0].value,
    )

ihr = jnp.repeat(
    transformation.SigmoidTransform()(p_hosp_ar[0].value + p_hosp_mean[0].value),
    repeats=7,
)[:n_datapoints]


with numpyro.handlers.seed(rng_seed=223):
    hosp_wday_effect_raw = hosp_wday_effect_rv()[0].value
    inf_to_hosp = inf_to_hosp_rv()[0].value

hosp_wday_effect = tile_until_n(hosp_wday_effect_raw, n_datapoints)

potential_latent_hospital_admissions = jnp.convolve(
    state_inf_per_capita,
    inf_to_hosp,
    mode="valid",
)[-n_datapoints:]

latent_hospital_admissions = (
    potential_latent_hospital_admissions * ihr * hosp_wday_effect * state_pop
)


with numpyro.handlers.seed(rng_seed=223):
    mode_sigma_ww_site = mode_sigma_ww_site_rv()[0].value
    sd_log_sigma_ww_site = sd_log_sigma_ww_site_rv()[0].value
    eta_log_sigma_ww_site = eta_log_sigma_ww_site_rv()[0].value
    ww_site_mod_raw = ww_site_mod_raw_rv()[0].value
    ww_site_mod_sd = ww_site_mod_sd_rv()[0].value


# These are the true expected genomes at the site level before observation error
# (which is at the lab-site level)
exp_obs_log_v_true = model_log_v_ot[ww_sampled_sites, ww_sampled_times]

# modify by lab-site specific variation (multiplier!)
ww_site_mod = ww_site_mod_raw * ww_site_mod_sd

# LHS log transformed obs genomes per person-day, RHS multiplies the expected observed
# genomes by the site-specific multiplier at that sampling time
exp_obs_log_v = exp_obs_log_v_true + ww_site_mod[ww_sampled_lab_sites]

sigma_ww_site = jnp.exp(
    jnp.log(mode_sigma_ww_site) + sd_log_sigma_ww_site * eta_log_sigma_ww_site
)

g = jnp.power(log10_g[0].value, 10)  # Estimated genomes shed per infected individual


log_conc_obs_rv = numpyro.sample(
    "log_conc",
    dist.Normal(
        loc=exp_obs_log_v[ww_uncensored],
        scale=sigma_ww_site[ww_sampled_lab_sites[ww_uncensored]],
    ),
    obs=data_observed_log_conc[ww_uncensored],
)

if ww_censored.shape[0] != 0:
    log_cdf_values = dist.Normal(
        loc=exp_obs_log_v[ww_censored],
        scale=sigma_ww_site[ww_sampled_lab_sites[ww_censored]],
    ).log_cdf(ww_log_lod[ww_censored])

    numpyro.factor("log_prob_censored", log_cdf_values.sum())


with numpyro.handlers.seed(rng_seed=223):
    observed_hospital_admissions = hospital_admission_obs_rv(
        mu=latent_hospital_admissions,
        obs=data_observed_hospital_admissions,
    )
