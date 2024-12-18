import json
import runpy

import jax.numpy as jnp
from pyrenew.deterministic import DeterministicVariable

from pyrenew_hew.pyrenew_hew_model import pyrenew_hew_model


def build_model_from_dir(model_dir):
    data_path = model_dir / "data" / "data_for_model_fit.json"
    prior_path = model_dir / "priors.py"

    with open(
        data_path,
        "r",
    ) as file:
        model_data = json.load(file)

    inf_to_ed_rv = DeterministicVariable(
        "inf_to_ed", jnp.array(model_data["inf_to_ed_pmf"])
    )  # check if off by 1 or reversed

    generation_interval_pmf_rv = DeterministicVariable(
        "generation_interval_pmf",
        jnp.array(model_data["generation_interval_pmf"]),
    )  # check if off by 1 or reversed

    infection_feedback_pmf_rv = DeterministicVariable(
        "infection_feedback_pmf",
        jnp.array(model_data["generation_interval_pmf"]),
    )  # check if off by 1 or reversed

    data_observed_disease_ed_visits = jnp.array(
        model_data["data_observed_disease_ed_visits"]
    )
    state_pop = jnp.array(model_data["state_pop"])

    right_truncation_pmf_rv = DeterministicVariable(
        "right_truncation_pmf", jnp.array(model_data["right_truncation_pmf"])
    )

    uot = (
        max(
            len(model_data["generation_interval_pmf"]),
            len(model_data["inf_to_ed_pmf"]),
        )
        - 1
    )

    priors = runpy.run_path(str(prior_path))

    right_truncation_offset = model_data["right_truncation_offset"]

    my_model = pyrenew_hew_model(
        state_pop=state_pop,
        i0_first_obs_n_rv=priors["i0_first_obs_n_rv"],
        initialization_rate_rv=priors["initialization_rate_rv"],
        log_r_mu_intercept_rv=priors["log_r_mu_intercept_rv"],
        autoreg_rt_rv=priors["autoreg_rt_rv"],
        eta_sd_rv=priors["eta_sd_rv"],  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_strength_rv=priors["inf_feedback_strength_rv"],
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        p_ed_mean_rv=priors["p_ed_visit_mean_rv"],
        p_ed_w_sd_rv=priors["p_ed_visit_w_sd_rv"],
        autoreg_p_ed_rv=priors["autoreg_p_ed_visit_rv"],
        ed_wday_effect_rv=priors["ed_visit_wday_effect_rv"],
        inf_to_ed_rv=inf_to_ed_rv,
        phi_rv=priors["phi_rv"],
        right_truncation_pmf_rv=right_truncation_pmf_rv,
        n_initialization_points=uot,
    )

    return (
        my_model,
        data_observed_disease_ed_visits,
        right_truncation_offset,
    )
