import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from pipelines.build_pyrenew_model import build_model_from_dir


@pytest.fixture
def mock_data():
    return json.dumps(
        {
            "data_observed_disease_ed_visits": [1, 2, 3],
            "data_observed_disease_hospital_admissions": [4, 5, 6],
            "state_pop": [7, 8, 9],
            "generation_interval_pmf": [0.1, 0.2, 0.7],
            "inf_to_ed_pmf": [0.4, 0.5, 0.1],
            "right_truncation_pmf": [0.7, 0.1, 0.2],
            "nssp_training_dates": ["2025-01-01"],
            "nhsn_training_dates": ["2025-01-02"],
            "right_truncation_offset": 10,
        }
    )


@pytest.fixture
def mock_priors():
    return """
from pyrenew.deterministic import NullVariable

i0_first_obs_n_rv = None
initialization_rate_rv = None
log_r_mu_intercept_rv = None
autoreg_rt_rv = None
eta_sd_rv = None
inf_feedback_strength_rv = NullVariable()
p_ed_visit_mean_rv = None
p_ed_visit_w_sd_rv = None
autoreg_p_ed_visit_rv = None
ed_visit_wday_effect_rv = None
ed_neg_bin_concentration_rv = None
hosp_admit_neg_bin_concentration_rv = None
ihr_rv = None
t_peak_rv = None
duration_shed_after_peak_rv = None
log10_genome_per_inf_ind_rv = None
mode_sigma_ww_site_rv = None
sd_log_sigma_ww_site_rv = None
mode_sd_ww_site_rv = None
max_shed_interval = None
ww_ml_produced_per_day = None
"""


def test_build_model_from_dir(tmp_path, mock_data, mock_priors):
    model_dir = tmp_path / "model_dir"
    data_dir = model_dir / "data"
    data_dir.mkdir(parents=True)

    (model_dir / "priors.py").write_text(mock_priors)

    data_path = data_dir / "data_for_model_fit.json"
    data_path.write_text(mock_data)

    model_data = json.loads(mock_data)

    # Test when all sample arguments are False
    _, data = build_model_from_dir(model_dir)
    assert data.data_observed_disease_ed_visits is None
    assert data.data_observed_disease_hospital_admissions is None
    assert data.data_observed_disease_wastewater is None

    # Test when all sample arguments are True
    _, data = build_model_from_dir(
        model_dir,
        sample_ed_visits=True,
        sample_hospital_admissions=True,
        sample_wastewater=True,
    )
    assert jnp.array_equal(
        data.data_observed_disease_ed_visits,
        jnp.array(model_data["data_observed_disease_ed_visits"]),
    )
    assert jnp.array_equal(
        data.data_observed_disease_hospital_admissions,
        jnp.array(model_data["data_observed_disease_hospital_admissions"]),
    )
    assert data.data_observed_disease_wastewater is None
    ## Update this if wastewater data is added later
