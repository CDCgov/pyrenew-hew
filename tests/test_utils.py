"""
Tests for the pyrenew-hew utils.
"""

import jax.numpy as jnp
import pytest

from pyrenew_hew.utils import convert_to_logmean_log_sd


@pytest.mark.parametrize(
    ["mean", "sd"],
    [
        [
            jnp.array([10]),
            jnp.array([0]),
        ]
    ],
)
def test_convert_to_logmean_log_sd_edge_case_zero_sd(mean, sd):
    logmean, logsd = convert_to_logmean_log_sd(mean, sd)

    expected_logmean = jnp.log(10.0)
    expected_logsd = 0.0

    assert jnp.isclose(logmean, expected_logmean)
    assert jnp.isclose(logsd, expected_logsd)
