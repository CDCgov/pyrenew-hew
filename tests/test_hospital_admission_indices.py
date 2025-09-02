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
        [2, 10, jnp.array([27, 34])],
    ],
)
def test_calculate_weekly_hosp_valid_indices(
    first_latent_admission_dow,
    model_t_first_latent_admissions,
    model_t_observed,
):
    result = HospAdmitObservationProcess.calculate_weekly_hosp_indices(
        first_latent_admission_dow,
        model_t_first_latent_admissions,
        model_t_observed,
        n_datapoints=None,
    )
    expected_result = (
        model_t_observed
        - model_t_first_latent_admissions
        - (6 - first_latent_admission_dow) % 7
        - 6
    ) // 7
    assert jnp.array_equal(result, expected_result)


@pytest.mark.parametrize(
    [
        "first_latent_admission_dow",
        "model_t_first_latent_admissions",
        "model_t_observed",
        "expected_error",
    ],
    [
        [
            2,
            10,
            jnp.array([26, 34]),
            "Not all observed or predicted hospital admissions are on Saturdays.",
        ],
        [
            2,
            10,
            jnp.array([19, 34]),
            "Observed hospital admissions date is before predicted hospital admissions.",
        ],
    ],
)
def test_calculate_weekly_hosp_invalid_indices(
    first_latent_admission_dow,
    model_t_first_latent_admissions,
    model_t_observed,
    expected_error,
):
    with pytest.raises(
        ValueError,
        match=expected_error,
    ):
        HospAdmitObservationProcess.calculate_weekly_hosp_indices(
            first_latent_admission_dow,
            model_t_first_latent_admissions,
            model_t_observed,
            n_datapoints=None,
        )
