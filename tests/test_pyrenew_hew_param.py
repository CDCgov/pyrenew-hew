import json

import pytest
from jax.typing import ArrayLike

from pyrenew_hew.pyrenew_hew_param import PyrenewHEWParam


@pytest.fixture
def mock_param_path(tmpdir):
    params = {
        "population_size": 1000000,
        "pop_fraction": [0.2, 0.3, 0.5],
        "generation_interval_pmf": [0.1, 0.3, 0.4, 0.2],
        "right_truncation_pmf": [0.5, 0.3, 0.2],
        "inf_to_hosp_admit_pmf": [0.05, 0.2, 0.35, 0.3, 0.1],
        "inf_to_hosp_admit_lognormal_loc": 1.1,
        "inf_to_hosp_admit_lognormal_scale": 0.4,
    }
    param_path = tmpdir.join("param.json")
    with open(param_path, "w") as f:
        json.dump(params, f)
    return param_path


def test_build_from_json(mock_param_path):
    """Test loading parameters from JSON."""
    params = PyrenewHEWParam.from_json(mock_param_path)

    assert isinstance(params.population_size, int)
    assert isinstance(params.pop_fraction, ArrayLike)
    assert isinstance(params.generation_interval_pmf, ArrayLike)
    assert isinstance(params.right_truncation_pmf, ArrayLike)
    assert isinstance(params.inf_to_hosp_admit_pmf, ArrayLike)
    assert isinstance(params.inf_to_hosp_admit_lognormal_loc, float)
    assert isinstance(params.inf_to_hosp_admit_lognormal_scale, float)
