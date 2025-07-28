import json

import pytest

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData


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
        "population_size": 1e5,
        "nhsn_step_size": 7,
        "nssp_step_size": 1,
        "nwss_step_size": 1,
    }


@pytest.fixture
def mock_data_dir(mock_data, tmpdir):
    data_path = tmpdir.join("data.json")
    with open(data_path, "w") as f:
        json.dump(mock_data, f)
    return data_path


def test_build_pyrenew_hew_model(mock_data_dir):
    # Test when all `fit_` arguments are False

    data = PyrenewHEWData.from_json(mock_data_dir)
    assert isinstance(data, PyrenewHEWData)
    assert data.data_observed_disease_ed_visits is None
    assert data.data_observed_disease_hospital_admissions is None
    assert data.data_observed_disease_wastewater_conc is None

    # Test when all `fit_` arguments are True

    data = PyrenewHEWData.from_json(
        mock_data_dir,
        fit_ed_visits=True,
        fit_hospital_admissions=True,
        fit_wastewater=True,
    )
    assert isinstance(data, PyrenewHEWData)
    assert data.data_observed_disease_ed_visits is not None
    assert data.data_observed_disease_hospital_admissions is not None
    assert data.data_observed_disease_wastewater_conc is not None
