import pytest
from pyrenew.deterministic import NullVariable

from pyrenew_hew.utils import build_pyrenew_hew_model


@pytest.fixture
def mock_data():
    return {
        "nssp_training_data": {
            "date": [
                "2025-01-01",
                "2025-01-02",
            ],
            "geo_value": ["CA"] * 2,
            "other_ed_visits": [200, 400],
            "observed_ed_visits": [10, 3],
            "data_type": ["train"] * 2,
        },
        "nhsn_training_data": {
            "weekendingdate": ["2025-01-01", "2025-01-02"],
            "jurisdiction": ["CA"] * 2,
            "hospital_admissions": [5, 1],
            "data_type": ["train"] * 2,
        },
        "loc_pop": [10000],
        "nssp_training_dates": ["2025-01-01"],
        "nhsn_training_dates": ["2025-01-04"],
        "right_truncation_offset": 10,
        "nwss_training_data": {
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
        "nhsn_step_size": 7,
        "nssp_step_size": 1,
        "nwss_step_size": 1,
    }


@pytest.fixture
def mock_pmfs():
    return {
        "generation_interval_pmf": [0.1, 0.2, 0.7],
        "delay_pmf": [0.1, 0.2, 0.7],
        "inf_to_hosp_admit_lognormal_loc": 0.015,
        "inf_to_hosp_admit_lognormal_scale": 0.851,
        "inf_to_hosp_admit_pmf": [0.0, 0.7, 0.1, 0.1, 0.1],
        "right_truncation_pmf": [0.7, 0.1, 0.2],
    }


@pytest.fixture
def mock_priors():
    return {
        "i0_first_obs_n_rv": None,
        "initialization_rate_rv": None,
        "log_r_mu_intercept_rv": None,
        "autoreg_rt_rv": None,
        "eta_sd_rv": None,
        "inf_feedback_strength_rv": NullVariable(),
        "p_ed_visit_mean_rv": None,
        "p_ed_visit_w_sd_rv": None,
        "autoreg_p_ed_visit_rv": None,
        "ed_visit_wday_effect_rv": None,
        "ed_neg_bin_concentration_rv": None,
        "hosp_admit_neg_bin_concentration_rv": None,
        "ihr_rv": None,
        "ihr_rel_iedr_rv": None,
        "t_peak_rv": None,
        "duration_shed_after_peak_rv": None,
        "delay_offset_loc_rv": None,
        "delay_log_offset_scale_rv": None,
        "log10_genome_per_inf_ind_rv": None,
        "mode_sigma_ww_site_rv": None,
        "sd_log_sigma_ww_site_rv": None,
        "mode_sd_ww_site_rv": None,
        "max_shed_interval": 10,
        "ww_ml_produced_per_day": None,
        "pop_fraction": None,
        "autoreg_rt_subpop_rv": None,
        "sigma_rt_rv": None,
        "sigma_i_first_obs_rv": None,
        "offset_ref_logit_i_first_obs_rv": None,
        "offset_ref_log_rt_rv": None,
    }


def test_build_pyrenew_hew_model(mock_data, mock_priors, mock_pmfs):
    # Test when all `fit_` arguments are False

    _, data = build_pyrenew_hew_model(mock_data, mock_priors, **mock_pmfs)
    assert data.data_observed_disease_ed_visits is None
    assert data.data_observed_disease_hospital_admissions is None
    assert data.data_observed_disease_wastewater_conc is None

    # Test when all `fit_` arguments are True
    _, data = build_pyrenew_hew_model(
        mock_data,
        mock_priors,
        **mock_pmfs,
        fit_ed_visits=True,
        fit_hospital_admissions=True,
        fit_wastewater=True,
    )
    assert data.data_observed_disease_ed_visits is not None
    assert data.data_observed_disease_hospital_admissions is not None
    assert data.data_observed_disease_wastewater_conc is not None
