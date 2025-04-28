import itertools

import jax.numpy as jnp
import pytest

from pyrenew_hew.pyrenew_hew_model import HospAdmitObservationProcess


@pytest.mark.parametrize(
    [
        "first_latent_admission_dow",
        "model_t_first_latent_admissions",
        "model_t_observed",
    ],
    [
        [0, -2, jnp.array([10, 17, 24])],
        [1, -4, jnp.array([7, 14, 21])],
        [4, -2, jnp.array([6, 13, 20])],
        [3, 1, jnp.array([17, 24, 31])],
    ],
)
def test_calculate_weekly_hosp_indices(
    first_latent_admission_dow,
    model_t_first_latent_admissions,
    model_t_observed,
):
    result = HospAdmitObservationProcess.calculate_weekly_hosp_indices(
        first_latent_admission_dow,
        model_t_first_latent_admissions,
        model_t_observed,
    )
    expected_result = (
        model_t_observed
        - model_t_first_latent_admissions
        - (6 - first_latent_admission_dow) % 7
        - 6
    ) // 7
    assert jnp.array_equal(result, expected_result)
