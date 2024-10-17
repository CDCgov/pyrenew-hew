import json

import jax.numpy as jnp

# load priors
# have to run this from the right directory
from priors import (  # noqa: E402
    autoreg_p_hosp_rv,
    autoreg_rt_rv,
    eta_sd_rv,
    hosp_wday_effect_rv,
    i0_first_obs_n_rv,
    inf_feedback_strength_rv,
    initialization_rate_rv,
    log_r_mu_intercept_rv,
    p_hosp_mean_rv,
    p_hosp_w_sd_rv,
    phi_rv,
)
from pyrenew.deterministic import DeterministicVariable

from pyrenew_hew.hosp_only_ww_model import hosp_only_ww_model


def build_model_from_dir(model_dir):
    data_path = model_dir / "data_for_model_fit.json"

    with open(
        data_path,
        "r",
    ) as file:
        model_data = json.load(file)

    inf_to_hosp_rv = DeterministicVariable(
        "inf_to_hosp", jnp.array(model_data["inf_to_hosp_pmf"])
    )  # check if off by 1 or reversed

    generation_interval_pmf_rv = DeterministicVariable(
        "generation_interval_pmf",
        jnp.array(model_data["generation_interval_pmf"]),
    )  # check if off by 1 or reversed

    infection_feedback_pmf_rv = DeterministicVariable(
        "infection_feedback_pmf",
        jnp.array(model_data["generation_interval_pmf"]),
    )  # check if off by 1 or reversed

    data_observed_hospital_admissions = jnp.array(
        model_data["data_observed_hospital_admissions"]
    )
    state_pop = jnp.array(model_data["state_pop"])

    right_truncation_pmf_rv = DeterministicVariable(
        "right_truncation_pmf", jnp.array(model_data["right_truncation_pmf"])
    )

    uot = (
        max(
            len(model_data["generation_interval_pmf"]),
            len(model_data["inf_to_hosp_pmf"]),
        )
        - 1
    )

    right_truncation_offset = model_data["right_truncation_offset"]

    my_model = hosp_only_ww_model(
        state_pop=state_pop,
        i0_first_obs_n_rv=i0_first_obs_n_rv,
        initialization_rate_rv=initialization_rate_rv,
        log_r_mu_intercept_rv=log_r_mu_intercept_rv,
        autoreg_rt_rv=autoreg_rt_rv,
        eta_sd_rv=eta_sd_rv,  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_strength_rv=inf_feedback_strength_rv,
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        p_hosp_mean_rv=p_hosp_mean_rv,
        p_hosp_w_sd_rv=p_hosp_w_sd_rv,
        autoreg_p_hosp_rv=autoreg_p_hosp_rv,
        hosp_wday_effect_rv=hosp_wday_effect_rv,
        inf_to_hosp_rv=inf_to_hosp_rv,
        phi_rv=phi_rv,
        right_truncation_pmf_rv=right_truncation_pmf_rv,
        n_initialization_points=uot,
    )

    return my_model, data_observed_hospital_admissions, right_truncation_offset
