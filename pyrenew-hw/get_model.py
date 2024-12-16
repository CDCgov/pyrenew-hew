import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms
from pyrenew.deterministic import DeterministicVariable, DeterministicPMF
from pyrenew.randomvariable import DistributionalVariable, TransformedVariable
import jax.numpy as jnp
import polars as pl
from pyrenew_hew.utils import convert_to_logmean_log_sd
from pyrenew_hew.ww_site_level_dynamics_model import ww_site_level_dynamics_model


def get_model(
    params,
    config,
    hosp_data_joined,
    ww_data_joined,
    site_subpop_spine,
    lab_site_subpop_spine,
):

    state_pop = hosp_data_joined["total_pop"].unique().item()
    subpop_size = site_subpop_spine.sort(by="subpop_index", descending=False)[
        "subpop_pop"
    ].to_numpy()  # subpop_pops
    n_subpops = len(subpop_size)
    pop_fraction = jnp.array(subpop_size) / state_pop
    unobs_time = params["timescale"]["uot"]
    ww_ml_produced_per_day = params["wastewater_observation_process"][
        "ml_of_ww_per_person_day"
    ]

    ww_log_lod = ww_data_joined["log_lod"].to_numpy()
    ww_censored = ww_data_joined.filter(pl.col("below_lod") == 1)[
        "ind_rel_to_sampled_times"
    ].to_numpy()
    ww_uncensored = ww_data_joined.filter(pl.col("below_lod") == 0)[
        "ind_rel_to_sampled_times"
    ].to_numpy()
    assert len(ww_data_joined) == len(ww_censored) + len(ww_uncensored)
    ww_sampled_times = ww_data_joined["t"].to_numpy()
    ww_sampled_subpops = ww_data_joined["subpop_index"].to_numpy()
    ww_sampled_lab_sites = ww_data_joined["lab_site_index"].to_numpy()

    n_ww_lab_sites = len(ww_data_joined["lab_site_index"].unique())
    # n_ww_sites = len(ww_data_joined["site_index"].unique())
    # n_censored = sum(ww_data_joined["below_lod"] == 1)
    # n_uncensored = len(ww_data_joined) - n_censored

    lab_site_to_subpop_map = lab_site_subpop_spine["subpop_index"].to_numpy()
    max_ww_sampled_days = max(ww_sampled_times)

    hosp_times = hosp_data_joined["t"].to_numpy()

    i_first_obs_est = (
        hosp_data_joined["count"][0:6].mean()
        / params["hospital_admission_observation_process"]["p_hosp_mean"]
    ) / state_pop

    # rvs
    generation_interval = config["generation_interval"]
    generation_interval_pmf_rv = DeterministicPMF(
        "generation_interval_pmf", jnp.array(generation_interval)
    )

    infection_feedback_pmf = config["infection_feedback_pmf"]
    infection_feedback_pmf_rv = DeterministicPMF(
        "infection_feedback_pmf", jnp.array(infection_feedback_pmf)
    )

    inf_to_hosp = config["inf_to_hosp"]
    inf_to_hosp_rv = DeterministicVariable("inf_to_hosp", jnp.array(inf_to_hosp))

    hosp_delay_max = len(inf_to_hosp)
    gt_max = min(
        len(generation_interval), params["continuous_distribution_parameters"]["gt_max"]
    )
    n_initialization_points = max(gt_max, hosp_delay_max)

    t_peak_mean = params["wastewater_observation_process"]["t_peak_mean"]
    t_peak_sd = params["wastewater_observation_process"]["t_peak_sd"]
    t_peak_rv = DistributionalVariable(
        "t_peak", dist.TruncatedNormal(t_peak_mean, t_peak_sd, low=0)
    )

    dur_shed_mean = params["wastewater_observation_process"]["duration_shedding_mean"]
    dur_shed_sd = params["wastewater_observation_process"]["duration_shedding_sd"]
    dur_shed_after_peak_rv = DistributionalVariable(
        "dur_shed_after_peak",
        dist.TruncatedNormal(
            dur_shed_mean - t_peak_mean, jnp.sqrt(dur_shed_sd**2 + t_peak_sd**2), low=0
        ),
    )
    max_shed_interval = dur_shed_mean + 3 * dur_shed_sd
    # set priors
    autoreg_rt_a = params["infection_process"]["autoreg_rt_a"]
    autoreg_rt_b = params["infection_process"]["autoreg_rt_b"]
    autoreg_rt_rv = DistributionalVariable(
        "autoreg_rt", dist.Beta(autoreg_rt_a, autoreg_rt_b)
    )

    r_prior_mean = params["infection_process"]["r_prior_mean"]
    r_prior_sd = params["infection_process"]["r_prior_sd"]
    r_logmean, r_logsd = convert_to_logmean_log_sd(r_prior_mean, r_prior_sd)
    log_r_t_first_obs_rv = DistributionalVariable(
        "log_r_t_first_obs", dist.Normal(r_logmean, r_logsd)
    )

    offset_ref_log_r_t_prior_mean = params["infection_process"][
        "offset_ref_log_r_t_prior_mean"
    ]
    offset_ref_log_r_t_prior_sd = params["infection_process"][
        "offset_ref_log_r_t_prior_sd"
    ]
    offset_ref_log_r_t_rv = DistributionalVariable(
        "offset_ref_log_r_t",
        dist.Normal(offset_ref_log_r_t_prior_mean, offset_ref_log_r_t_prior_sd),
    )

    offset_ref_logit_i_first_obs_prior_mean = params["infection_process"][
        "offset_ref_logit_i_first_obs_prior_mean"
    ]
    offset_ref_logit_i_first_obs_prior_sd = params["infection_process"][
        "offset_ref_logit_i_first_obs_prior_sd"
    ]
    offset_ref_logit_i_first_obs_rv = DistributionalVariable(
        "offset_ref_logit_i_first_obs",
        dist.Normal(
            offset_ref_logit_i_first_obs_prior_mean,
            offset_ref_logit_i_first_obs_prior_sd,
        ),
    )

    offset_ref_initial_exp_growth_rate_prior_mean = params["infection_process"][
        "offset_ref_initial_exp_growth_rate_prior_mean"
    ]
    offset_ref_initial_exp_growth_rate_prior_sd = params["infection_process"][
        "offset_ref_initial_exp_growth_rate_prior_sd"
    ]
    offset_ref_initial_exp_growth_rate_rv = DistributionalVariable(
        "offset_ref_initial_exp_growth_rate",
        dist.TruncatedNormal(
            offset_ref_initial_exp_growth_rate_prior_mean,
            offset_ref_initial_exp_growth_rate_prior_sd,
            low=-0.01,
            high=0.01,
        ),
    )

    autoreg_rt_subpop_a = params["infection_process"]["autoreg_rt_subpop_a"]
    autoreg_rt_subpop_b = params["infection_process"]["autoreg_rt_subpop_b"]
    autoreg_rt_subpop_rv = DistributionalVariable(
        "autoreg_rt_subpop", dist.Beta(autoreg_rt_subpop_a, autoreg_rt_subpop_b)
    )

    autoreg_p_hosp_a = params["infection_process"]["autoreg_p_hosp_a"]
    autoreg_p_hosp_b = params["infection_process"]["autoreg_p_hosp_b"]
    autoreg_p_hosp_rv = DistributionalVariable(
        "autoreg_p_hosp", dist.Beta(autoreg_p_hosp_a, autoreg_p_hosp_b)
    )

    inv_sqrt_phi_prior_mean = params["hospital_admission_observation_process"][
        "inv_sqrt_phi_prior_mean"
    ]
    inv_sqrt_phi_prior_sd = params["hospital_admission_observation_process"][
        "inv_sqrt_phi_prior_sd"
    ]
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
        transforms=transforms.PowerTransform(-2),
    )

    log10_g_prior_mean = params["wastewater_observation_process"]["log10_g_prior_mean"]
    log10_g_prior_sd = params["wastewater_observation_process"]["log10_g_prior_sd"]
    log10_g_rv = DistributionalVariable(
        "log10_g", dist.Normal(log10_g_prior_mean, log10_g_prior_sd)
    )

    hosp_wday_effect_prior_alpha = params["hospital_admission_observation_process"][
        "hosp_wday_effect_prior_alpha"
    ]
    hosp_wday_effect_rv = TransformedVariable(
        "hosp_wday_effect",
        DistributionalVariable(
            "hosp_wday_effect_raw",
            dist.Dirichlet(jnp.array(hosp_wday_effect_prior_alpha)),
        ),
        transforms.AffineTransform(loc=0, scale=7),
    )

    mean_initial_exp_growth_rate_prior_mean = params["infection_process"][
        "mean_initial_exp_growth_rate_prior_mean"
    ]
    mean_initial_exp_growth_rate_prior_sd = params["infection_process"][
        "mean_initial_exp_growth_rate_prior_sd"
    ]
    mean_initial_exp_growth_rate_rv = DistributionalVariable(
        "mean_initial_exp_growth_rate",
        dist.TruncatedNormal(
            loc=mean_initial_exp_growth_rate_prior_mean,
            scale=mean_initial_exp_growth_rate_prior_sd,
            low=-0.01,
            high=0.01,
        ),
    )

    sigma_initial_exp_growth_rate_prior_mode = params["infection_process"][
        "sigma_initial_exp_growth_rate_prior_mode"
    ]
    sigma_initial_exp_growth_rate_prior_sd = params["infection_process"][
        "sigma_initial_exp_growth_rate_prior_sd"
    ]
    sigma_initial_exp_growth_rate_rv = DistributionalVariable(
        "sigma_initial_exp_growth_rate",
        dist.TruncatedNormal(
            sigma_initial_exp_growth_rate_prior_mode,
            sigma_initial_exp_growth_rate_prior_sd,
            low=0,
        ),
    )

    mode_sigma_ww_site_prior_mode = params["wastewater_observation_process"][
        "mode_sigma_ww_site_prior_mode"
    ]
    mode_sigma_ww_site_prior_sd = params["wastewater_observation_process"][
        "mode_sigma_ww_site_prior_sd"
    ]
    mode_sigma_ww_site_rv = DistributionalVariable(
        "mode_sigma_ww_site",
        dist.TruncatedNormal(
            mode_sigma_ww_site_prior_mode, mode_sigma_ww_site_prior_sd, low=0
        ),
    )

    sd_log_sigma_ww_site_prior_mode = params["wastewater_observation_process"][
        "sd_log_sigma_ww_site_prior_mode"
    ]
    sd_log_sigma_ww_site_prior_sd = params["wastewater_observation_process"][
        "sd_log_sigma_ww_site_prior_sd"
    ]
    sd_log_sigma_ww_site_rv = DistributionalVariable(
        "sd_log_sigma_ww_site",
        dist.TruncatedNormal(
            sd_log_sigma_ww_site_prior_mode, sd_log_sigma_ww_site_prior_sd, low=0
        ),
    )

    eta_sd_sd = params["infection_process"]["eta_sd_sd"]
    eta_sd_rv = DistributionalVariable(
        "eta_sd", dist.TruncatedNormal(0, eta_sd_sd, low=0)
    )
    # eta_sd_mean = params['infection_process']['eta_sd_mean']

    sigma_i_first_obs_prior_mode = params["infection_process"][
        "sigma_i_first_obs_prior_mode"
    ]
    sigma_i_first_obs_prior_sd = params["infection_process"][
        "sigma_i_first_obs_prior_sd"
    ]
    sigma_i_first_obs_rv = DistributionalVariable(
        "sigma_i_first_obs",
        dist.TruncatedNormal(
            sigma_i_first_obs_prior_mode, sigma_i_first_obs_prior_sd, low=0
        ),
    )

    p_hosp_prior_mean = params["hospital_admission_observation_process"]["p_hosp_mean"]
    p_hosp_sd_logit = params["hospital_admission_observation_process"][
        "p_hosp_sd_logit"
    ]
    p_hosp_mean_rv = DistributionalVariable(
        "p_hosp_mean",
        dist.Normal(transforms.logit(p_hosp_prior_mean), p_hosp_sd_logit),
    )  # logit scale

    p_hosp_w_sd_sd = params["hospital_admission_observation_process"]["p_hosp_w_sd_sd"]
    p_hosp_w_sd_rv = DistributionalVariable(
        "p_hosp_w_sd", dist.TruncatedNormal(0, p_hosp_w_sd_sd, low=0)
    )

    ww_site_mod_sd_sd = params["wastewater_observation_process"]["ww_site_mod_sd_sd"]
    ww_site_mod_sd_rv = DistributionalVariable(
        "ww_site_mod_sd", dist.TruncatedNormal(0, ww_site_mod_sd_sd, low=0)
    )

    infection_feedback_prior_logmean = params["infection_process"][
        "infection_feedback_prior_logmean"
    ]
    infection_feedback_prior_logsd = params["infection_process"][
        "infection_feedback_prior_logsd"
    ]
    infection_feedback_strength_rv = TransformedVariable(
        "inf_feedback",
        DistributionalVariable(
            "inf_feedback_raw",
            dist.LogNormal(
                infection_feedback_prior_logmean, infection_feedback_prior_logsd
            ),
        ),
        transforms=transforms.AffineTransform(loc=0, scale=-1),
    )

    sigma_rt_prior = params["infection_process"]["sigma_rt_prior"]
    sigma_rt_rv = DistributionalVariable(
        "sigma_rt", dist.TruncatedNormal(0, sigma_rt_prior, low=0)
    )

    # log_phi_g_prior_mean = params['wastewater_observation_process']['log_phi_g_prior_mean']
    # log_phi_g_prior_sd = params['wastewater_observation_process']['log_phi_g_prior_sd']

    # Calculate i_first_obs_over_n_prior_a and i_first_obs_over_n_prior_b
    i_first_obs_over_n_prior_a = 1 + params["infection_process"][
        "i_first_obs_certainty"
    ] * (i_first_obs_est)
    i_first_obs_over_n_prior_b = 1 + params["infection_process"][
        "i_first_obs_certainty"
    ] * (1 - i_first_obs_est)
    i_first_obs_over_n_rv = DistributionalVariable(
        "i_first_obs_over_n",
        dist.Beta(i_first_obs_over_n_prior_a, i_first_obs_over_n_prior_b),
    )
    i0_t_offset = 0

    my_model = ww_site_level_dynamics_model(
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
        autoreg_rt_subpop_rv,
        sigma_rt_rv,
        sigma_i_first_obs_rv,
        sigma_initial_exp_growth_rate_rv,
        t_peak_rv,
        dur_shed_after_peak_rv,
        n_ww_lab_sites,
        max_shed_interval,
        log10_g_rv,
        mode_sigma_ww_site_rv,
        sd_log_sigma_ww_site_rv,
        ww_site_mod_sd_rv,
        ww_ml_produced_per_day,
        ww_uncensored,
        ww_censored,
        ww_sampled_lab_sites,
        ww_sampled_subpops,
        ww_sampled_times,
        ww_log_lod,
        lab_site_to_subpop_map,
        max_ww_sampled_days,
        include_ww=True,
    )

    return my_model
