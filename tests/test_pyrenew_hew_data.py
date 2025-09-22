import datetime as dt
import itertools
import json

import numpy as np
import polars as pl
import pytest

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData


@pytest.mark.parametrize(
    "erroring_date",
    [np.datetime64("2025-03-08") + np.timedelta64(x, "D") for x in range(1, 7)],
)
def test_validation(erroring_date):
    """
    Confirm that constructor errors with non-Saturday
    first_hospital_admissions_date(s)
    """
    with pytest.raises(ValueError, match="Saturdays"):
        PyrenewHEWData(
            n_ed_visits_data_days=5,
            n_hospital_admissions_data_days=10,
            first_ed_visits_date=np.datetime64("2025-03-05"),
            first_hospital_admissions_date=erroring_date,
            right_truncation_offset=0,
        )


@pytest.mark.parametrize(
    [
        "n_ed_visits_data_days",
        "n_hospital_admissions_data_days",
        "n_wastewater_data_days",
        "right_truncation_offset",
        "first_ed_visits_date",
        "first_hospital_admissions_date",
        "first_wastewater_date",
        "n_forecast_points",
    ],
    [
        [
            50,
            0,
            0,
            5,
            np.datetime64("2023-01-01"),
            np.datetime64("2022-02-05"),
            np.datetime64("2025-12-05"),
            10,
        ],
        [
            0,
            325,
            2,
            5,
            np.datetime64("2025-01-01"),
            np.datetime64("2023-05-27"),
            np.datetime64("2022-04-05"),
            10,
        ],
        [
            0,
            0,
            2,
            3,
            np.datetime64("2025-01-01"),
            np.datetime64("2025-02-08"),
            np.datetime64("2024-12-05"),
            30,
        ],
        [
            0,
            0,
            23,
            3,
            np.datetime64("2025-01-01"),
            np.datetime64("2025-02-08"),
            np.datetime64("2024-12-05"),
            30,
        ],
    ],
)
def test_to_forecast_data(
    n_ed_visits_data_days: int,
    n_hospital_admissions_data_days: int,
    n_wastewater_data_days: int,
    right_truncation_offset: int,
    first_ed_visits_date: np.datetime64,
    first_hospital_admissions_date: np.datetime64,
    first_wastewater_date: np.datetime64,
    n_forecast_points: int,
) -> None:
    """
    Test the to_forecast_data method
    """
    data = PyrenewHEWData(
        n_ed_visits_data_days=n_ed_visits_data_days,
        n_hospital_admissions_data_days=n_hospital_admissions_data_days,
        first_ed_visits_date=first_ed_visits_date,
        first_hospital_admissions_date=first_hospital_admissions_date,
        right_truncation_offset=right_truncation_offset,
    )

    assert data.right_truncation_offset == right_truncation_offset
    assert data.right_truncation_offset is not None

    forecast_data = data.to_forecast_data(n_forecast_points)
    n_days_expected = data.n_days_post_init + n_forecast_points
    n_weeks_expected = n_days_expected // 7
    assert forecast_data.n_ed_visits_data_days == n_days_expected
    assert forecast_data.n_wastewater_data_days == n_days_expected
    assert forecast_data.n_hospital_admissions_data_days == n_weeks_expected
    assert forecast_data.right_truncation_offset is None
    assert forecast_data.first_ed_visits_date == data.first_data_date_overall

    ## hosp admit date should be the first Saturday
    assert forecast_data.first_hospital_admissions_date >= data.first_data_date_overall
    assert (
        forecast_data.first_hospital_admissions_date.astype(dt.datetime).weekday() == 5
    )

    assert (
        (forecast_data.first_hospital_admissions_date - data.first_data_date_overall)
        / np.timedelta64(1, "D")
    ).item() <= 6

    assert forecast_data.first_wastewater_date == data.first_data_date_overall
    assert forecast_data.data_observed_disease_wastewater_conc is None


def test_pyrenew_wastewater_data():
    first_training_date = np.datetime64("2023-01-01")
    last_training_date = np.datetime64("2023-07-23")
    dates = pl.date_range(
        first_training_date,
        last_training_date,
        interval="1w",
        closed="both",
        eager=True,
    )

    ww_raw = pl.DataFrame(
        {
            "date": dates.extend(dates),
            "site": [200] * 30 + [100] * 30,
            "lab": [21] * 60,
            "lab_site_index": [1] * 30 + [2] * 30,
            "site_index": [1] * 30 + [2] * 30,
            "log_genome_copies_per_ml": np.log(
                np.abs(np.random.normal(loc=500, scale=50, size=60))
            ),
            "log_lod": np.log([20] * 30 + [15] * 30),
            "site_pop": [200_000] * 30 + [400_000] * 30,
        }
    )

    ww_data = ww_raw.with_columns(
        (pl.col("log_genome_copies_per_ml") <= pl.col("log_lod"))
        .cast(pl.Int8)
        .alias("below_lod")
    )

    first_ed_visits_date = np.datetime64("2023-01-01")
    first_hospital_admissions_date = np.datetime64("2023-01-07")  # Saturday
    first_wastewater_date = np.datetime64("2023-01-01")
    n_forecast_points = 10

    data = PyrenewHEWData(
        first_ed_visits_date=first_ed_visits_date,
        first_hospital_admissions_date=first_hospital_admissions_date,
        first_wastewater_date=first_wastewater_date,
        nwss_training_data=ww_data,
        population_size=1e6,
    )

    forecast_data = data.to_forecast_data(n_forecast_points)
    assert forecast_data.data_observed_disease_wastewater_conc is None
    assert data.data_observed_disease_wastewater_conc is not None

    assert forecast_data.ww_censored is None
    assert data.ww_censored is not None

    assert forecast_data.ww_uncensored is None
    assert data.ww_uncensored is not None

    assert forecast_data.ww_log_lod is None
    assert data.ww_log_lod is not None

    assert forecast_data.ww_observed_lab_sites is None
    assert data.ww_observed_lab_sites is not None

    assert forecast_data.ww_observed_subpops is None
    assert data.ww_observed_subpops is not None

    assert data.model_t_obs_wastewater is not None
    assert forecast_data.model_t_obs_wastewater is None

    assert np.array_equal(data.n_ww_lab_sites, forecast_data.n_ww_lab_sites)

    assert np.array_equal(
        data.data_observed_disease_wastewater_conc,
        ww_data["log_genome_copies_per_ml"],
    )
    assert len(data.ww_censored) == len(ww_data.filter(pl.col("below_lod") == 1))
    assert len(data.ww_uncensored) == len(ww_data.filter(pl.col("below_lod") == 0))
    assert np.array_equal(data.ww_log_lod, ww_data["log_lod"])
    assert data.n_ww_lab_sites == ww_data["lab_site_index"].n_unique()


def mock_data(fit_ed_visits=False, fit_hospital_admissions=False, fit_wastewater=False):
    data_dict = {
        "population_size": 1e5,
        "loc_pop": [10000],
        "right_truncation_offset": 10,
    }

    if fit_ed_visits:
        ed_visit_dict = {
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
            "nssp_training_dates": ["2025-01-01"],
            "nssp_step_size": 1,
        }
        data_dict.update(ed_visit_dict)

    if fit_hospital_admissions:
        ed_visit_dict = {
            "nhsn_training_data": {
                "weekendingdate": ["2025-01-01", "2025-01-02"],
                "jurisdiction": ["CA"] * 2,
                "hospital_admissions": [5, 1],
                "data_type": ["train"] * 2,
            },
            "nhsn_training_dates": ["2025-01-04"],
            "nhsn_step_size": 7,
        }
        data_dict.update(ed_visit_dict)

    if fit_wastewater:
        wastewater_dict = {
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
            "nwss_step_size": 1,
        }
        data_dict.update(wastewater_dict)

    return data_dict


@pytest.fixture
def mock_data_dir(mock_data, tmpdir):
    data_path = tmpdir.join("data.json")
    with open(data_path, "w") as f:
        json.dump(mock_data, f)
    return data_path


@pytest.mark.parametrize(
    "fit_ed_visits,fit_hospital_admissions,fit_wastewater",
    list(itertools.product([False, True], repeat=3)),
)
def test_json_roundtrip(
    tmp_path, fit_ed_visits, fit_hospital_admissions, fit_wastewater
):
    # Use the mock_data fixture to generate the data dictionary
    data_dict = mock_data(
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )

    # Write to json
    json_path = tmp_path / "data.json"
    with open(json_path, "w") as f:
        json.dump(data_dict, f)

    # Read from json
    data = PyrenewHEWData.from_json(
        json_path,
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )
    assert isinstance(data, PyrenewHEWData)
