import json

import jax.numpy as jnp
import numpyro.distributions as dist
import pyrenew.transformation as transformation
from numpyro.infer.reparam import LocScaleReparam

# load priors
# have to run this from the right directory
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import (
    DistributionalVariable,
    TransformedVariable,
)

from pyrenew_hew.hosp_only_ww_model import hosp_only_ww_model


def parametrize_priors(model_data: dict) -> dict:
    """
    Parameterize prior distribution RVs
    from a model data dictionary.

    Parameters
    ----------
    model_data
       Dictionary of model data from which to parametrize
       prior distribution random variables. Ignored currently.

    Returns
    -------
    dict
        A dictionary containing the parameterized
        :class:`pyrenew.metaclass.RandomVariable`s.
        as its values.
    """

    prior_dict = dict(
        i0_first_obs_n_rv=DistributionalVariable(
            "i0_first_obs_n_rv",
            dist.Beta(1, 10),
        ),
        initialization_rate_rv=DistributionalVariable(
            "rate", dist.Normal(0, 0.01), reparam=LocScaleReparam(0)
        ),
        log_r_mu_intercept_rv=DistributionalVariable(
            "log_r_mu_intercept_rv",
            dist.Normal(jnp.log(1), jnp.log(jnp.sqrt(2))),
        ),
        eta_sd_rv=DistributionalVariable(
            "eta_sd", dist.TruncatedNormal(0.04, 0.02, low=0)
        ),
        autoreg_rt_rv=DistributionalVariable("autoreg_rt", dist.Beta(2, 40)),
        inf_feedback_strength_rv=TransformedVariable(
            "inf_feedback",
            DistributionalVariable(
                "inf_feedback_raw",
                dist.LogNormal(jnp.log(50), jnp.log(2)),
            ),
            transforms=transformation.AffineTransform(loc=0, scale=-1),
        ),
        p_ed_visit_mean_rv=DistributionalVariable(
            "p_ed_visit_mean",
            dist.Normal(
                transformation.SigmoidTransform().inv(0.005),
                0.3,
            ),  # logit-Normal prior
        ),
        p_ed_visit_w_sd_rv=DistributionalVariable(
            "p_ed_visit_w_sd_sd", dist.TruncatedNormal(0, 0.01, low=0)
        ),
        autoreg_p_ed_visit_rv=DistributionalVariable(
            "autoreg_p_ed_visit", dist.Beta(1, 100)
        ),
        ed_visit_wday_effect_rv=TransformedVariable(
            "hosp_wday_effect",
            DistributionalVariable(
                "hosp_wday_effect_raw",
                dist.Dirichlet(jnp.array([5, 5, 5, 5, 5, 5, 5])),
            ),
            transformation.AffineTransform(loc=0, scale=7),
        ),
        # Based on looking at some historical posteriors.
        phi_rv=DistributionalVariable("phi", dist.LogNormal(6, 1)),
    )

    return prior_dict


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

    data_observed_disease_hospital_admissions = jnp.array(
        model_data["data_observed_disease_hospital_admissions"]
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

    priors = parameterize_priors(model_data)

    right_truncation_offset = model_data["right_truncation_offset"]

    my_model = hosp_only_ww_model(
        state_pop=state_pop,
        i0_first_obs_n_rv=priors["i0_first_obs_n_rv"],
        initialization_rate_rv=priors["initialization_rate_rv"],
        log_r_mu_intercept_rv=priors["log_r_mu_intercept_rv"],
        autoreg_rt_rv=priors["autoreg_rt_rv"],
        eta_sd_rv=priors["eta_sd_rv"],  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_strength_rv=priors["inf_feedback_strength_rv"],
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        p_hosp_mean_rv=priors["p_ed_visit_mean_rv"],
        p_hosp_w_sd_rv=priors["p_ed_visit_w_sd_rv"],
        autoreg_p_hosp_rv=priors["autoreg_p_ed_visit_rv"],
        hosp_wday_effect_rv=priors["ed_visit_wday_effect_rv"],
        inf_to_hosp_rv=inf_to_hosp_rv,
        phi_rv=priors["phi_rv"],
        right_truncation_pmf_rv=right_truncation_pmf_rv,
        n_initialization_points=uot,
    )

    return (
        my_model,
        data_observed_disease_hospital_admissions,
        right_truncation_offset,
    )
