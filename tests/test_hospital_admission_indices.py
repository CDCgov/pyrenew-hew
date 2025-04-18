import jax.numpy as jnp
import pytest

from pyrenew_hew.pyrenew_hew_model import HospAdmitObservationProcess


@pytest.mark.parametrize(
    [
        "first_latent_admission_dow",
        "model_t_first_latent_admissions",
        "n_datapoints",
        "model_t_observed",
        "expected_result",
    ],
    [
        [2, 0, 7, None, jnp.arange(7)],
        [2, 0, None, jnp.array([10, 17, 24]), jnp.array([0, 1, 2])],
        [2, 0, 5, jnp.array([10, 17, 24]), jnp.array([0, 1, 2])],
    ],
)
def test_calculate_weekly_hosp_indices(
    first_latent_admission_dow,
    model_t_first_latent_admissions,
    n_datapoints,
    model_t_observed,
    expected_result,
):
    result = HospAdmitObservationProcess.calculate_weekly_hosp_indices(
        first_latent_admission_dow,
        model_t_first_latent_admissions,
        n_datapoints,
        model_t_observed,
    )
    assert jnp.array_equal(result, expected_result)
