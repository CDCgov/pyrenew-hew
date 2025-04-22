import json

import jax.numpy as jnp
import polars as pl
import pytest

from pipelines.build_pyrenew_model import build_model_from_dir


@pytest.fixture
def mock_data():
    return json.dumps(
        {
            "data_observed_disease_ed_visits": [1, 2, 3],
            "data_observed_disease_hospital_admissions": [4, 5, 6],
            "state_pop": [10000],
            "generation_interval_pmf": [0.1, 0.2, 0.7],
            "inf_to_ed_pmf": [0.4, 0.5, 0.1],
            "inf_to_hosp_admit_pmf": [0.0, 0.7, 0.1, 0.1, 0.1],
            "inf_to_hosp_admit_lognormal_loc": 0.015,
            "inf_to_hosp_admit_lognormal_scale": 0.851,
            "right_truncation_pmf": [0.7, 0.1, 0.2],
            "nssp_training_dates": ["2025-01-01"],
            "nhsn_training_dates": ["2025-01-04"],
            "right_truncation_offset": 10,
            "data_observed_disease_wastewater": {
                "date": [
                    "2025-01-01",
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-02",
                ],
                "site": ["1.0", "1.0", "2.0", "2.0"],
                "lab": ["1.0", "1.0", "1.0", "1.0"],
                "site_pop": [4000, 4000, 2000, 2000],
                "site_index": [1, 1, 0, 0],
                "lab_site_index": [1, 1, 0, 0],
                "log_genome_copies_per_ml": [0.1, 0.1, 0.5, 0.4],
                "log_lod": [1.1, 2.0, 1.5, 2.1],
                "below_lod": [False, False, False, False],
            },
            "pop_fraction": [0.4, 0.4, 0.2],
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
ihr_rel_iedr_rv = None
t_peak_rv = None
duration_shed_after_peak_rv = None
delay_offset_loc_rv = None
delay_log_offset_scale_rv = None
log10_genome_per_inf_ind_rv = None
mode_sigma_ww_site_rv = None
sd_log_sigma_ww_site_rv = None
mode_sd_ww_site_rv = None
max_shed_interval = 10
ww_ml_produced_per_day = None
pop_fraction=None
autoreg_rt_subpop_rv=None
sigma_rt_rv=None
sigma_i_first_obs_rv=None
offset_ref_logit_i_first_obs_rv=None
offset_ref_log_rt_rv=None
"""


def test_build_model_from_dir(tmp_path, mock_data, mock_priors):
    model_dir = tmp_path / "model_dir"
    data_dir = model_dir / "data"
    data_dir.mkdir(parents=True)

    (model_dir / "priors.py").write_text(mock_priors)

    data_path = data_dir / "data_for_model_fit.json"
    data_path.write_text(mock_data)

    model_data = json.loads(mock_data)

    # Test when all `fit_` arguments are False
    _, data = build_model_from_dir(model_dir)
    assert data.data_observed_disease_ed_visits is None
    assert data.data_observed_disease_hospital_admissions is None
    assert data.data_observed_disease_wastewater_conc is None

    # Test when all `fit_` arguments are True
    _, data = build_model_from_dir(
        model_dir,
        fit_ed_visits=True,
        fit_hospital_admissions=True,
        fit_wastewater=True,
    )
    assert jnp.array_equal(
        data.data_observed_disease_ed_visits,
        jnp.array(model_data["data_observed_disease_ed_visits"]),
    )
    assert jnp.array_equal(
        data.data_observed_disease_hospital_admissions,
        jnp.array(model_data["data_observed_disease_hospital_admissions"]),
    )
    assert data.data_observed_disease_wastewater_conc is not None
