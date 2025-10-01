import datetime as dt
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
    ).item() <= 7

    assert forecast_data.first_wastewater_date == data.first_data_date_overall
    assert forecast_data.data_observed_disease_wastewater_conc is None


def test_to_forecast_data_saturday_edge_case():
    """
    Test that to_forecast_data correctly handles the edge case where
    first_data_date_overall is already a Saturday. In this case,
    first_hospital_admissions_date should be the NEXT Saturday (7 days later),
    not the same day.
    """
    # 2025-03-08 is a Saturday
    first_date_saturday = np.datetime64("2025-03-08")

    data = PyrenewHEWData(
        n_ed_visits_data_days=10,
        n_hospital_admissions_data_days=2,
        first_ed_visits_date=first_date_saturday,
        first_hospital_admissions_date=first_date_saturday,
        right_truncation_offset=0,
    )

    forecast_data = data.to_forecast_data(n_forecast_points=14)

    # Verify that first_hospital_admissions_date is 7 days after the start Saturday
    expected_hosp_date = first_date_saturday + np.timedelta64(7, "D")
    assert forecast_data.first_hospital_admissions_date == expected_hosp_date

    # Verify it's still a Saturday
    assert forecast_data.first_hospital_admissions_date.astype(dt.datetime).weekday() == 5


def test_pyrenew_wastewater_data():
    first_training_date = dt.date(2023, 1, 1)
    last_training_date = dt.date(2023, 7, 23)
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


def test_build_pyrenew_hew_data_from_json(mock_data_dir):
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


# ============================================================================
# NEW CRITICAL TESTS
# ============================================================================


def test_model_t_conversions():
    """
    Test model_t_obs_* properties calculate correct indices.

    Verifies that observation dates are correctly converted to model time
    indices relative to first_data_date_overall.
    """
    ed_dates = ["2023-01-05", "2023-01-08"]
    hosp_dates = ["2023-01-07", "2023-01-14"]

    nssp_data = pl.DataFrame(
        {
            "date": ed_dates,
            "geo_value": ["CA", "CA"],
            "observed_ed_visits": [10, 20],
            "other_ed_visits": [100, 200],
            "data_type": ["train", "train"],
        },
        schema={
            "date": pl.Date,
            "geo_value": pl.String,
            "observed_ed_visits": pl.Int64,
            "other_ed_visits": pl.Int64,
            "data_type": pl.String,
        },
    )

    nhsn_data = pl.DataFrame(
        {
            "weekendingdate": hosp_dates,
            "jurisdiction": ["CA", "CA"],
            "hospital_admissions": [5, 10],
            "data_type": ["train", "train"],
        },
        schema={
            "weekendingdate": pl.Date,
            "jurisdiction": pl.String,
            "hospital_admissions": pl.Int64,
            "data_type": pl.String,
        },
    )

    data = PyrenewHEWData(
        nssp_training_data=nssp_data,
        nhsn_training_data=nhsn_data,
        first_ed_visits_date=np.datetime64(ed_dates[0]),
        first_hospital_admissions_date=np.datetime64(hosp_dates[0]),
    )

    # first_data_date_overall should be min of all dates
    assert data.first_data_date_overall == np.datetime64(ed_dates[0])

    # model_t should be 0 for first date, 3 for second ED visit
    assert data.model_t_obs_ed_visits[0] == 0
    assert data.model_t_obs_ed_visits[1] == 3

    # Hospital admissions at days 2 and 9
    assert data.model_t_obs_hospital_admissions[0] == 2
    assert data.model_t_obs_hospital_admissions[1] == 9


def test_properties_with_no_data():
    """
    Test behavior when no training data provided.

    Ensures properties return None appropriately when no DataFrames
    are provided to the constructor.
    """
    data = PyrenewHEWData(
        n_ed_visits_data_days=10, first_ed_visits_date=np.datetime64("2023-01-01")
    )

    assert data.data_observed_disease_ed_visits is None
    assert data.data_observed_disease_hospital_admissions is None
    assert data.data_observed_disease_wastewater_conc is None
    assert data.model_t_obs_ed_visits is None
    assert data.model_t_obs_hospital_admissions is None
    assert data.model_t_obs_wastewater is None


def test_mixed_data_sources():
    """
    Test with only some data sources present.

    Verifies correct behavior when only ED data is provided
    but hospital and wastewater data are absent.
    """
    ed_data = pl.DataFrame(
        {
            "date": ["2023-01-01"],
            "geo_value": ["CA"],
            "observed_ed_visits": [10],
            "other_ed_visits": [100],
            "data_type": ["train"],
        },
        schema={
            "date": pl.Date,
            "geo_value": pl.String,
            "observed_ed_visits": pl.Int64,
            "other_ed_visits": pl.Int64,
            "data_type": pl.String,
        },
    )

    data = PyrenewHEWData(
        nssp_training_data=ed_data, first_ed_visits_date=np.datetime64("2023-01-01")
    )

    # ED data should work
    assert data.data_observed_disease_ed_visits is not None
    assert len(data.data_observed_disease_ed_visits) == 1

    # Others should be None
    assert data.first_hospital_admissions_date is None
    assert data.first_wastewater_date is None
    assert data.data_observed_disease_hospital_admissions is None
    assert data.data_observed_disease_wastewater_conc is None


def test_site_subpop_spine_with_auxiliary():
    """
    Test subpopulation creation when WW sites don't cover full population.

    When wastewater sampling sites don't cover the entire population,
    an auxiliary subpopulation should be created for the remainder.
    """
    ww_data = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01"],
            "site": ["site1", "site2"],
            "site_index": [0, 1],
            "site_pop": [200_000, 300_000],
            "lab_site_index": [0, 1],
            "log_genome_copies_per_ml": [1.0, 2.0],
            "log_lod": [0.5, 0.5],
            "below_lod": [0, 0],
        },
        schema={
            "date": pl.Date,
            "site": pl.String,
            "site_index": pl.Int64,
            "site_pop": pl.Int64,
            "lab_site_index": pl.Int64,
            "log_genome_copies_per_ml": pl.Float64,
            "log_lod": pl.Float64,
            "below_lod": pl.Int64,
        },
    )

    data = PyrenewHEWData(
        nwss_training_data=ww_data,
        population_size=1_000_000,  # 500k not covered by WW
    )

    spine = data.site_subpop_spine
    # Should have 3 subpops: 2 sites + 1 auxiliary
    assert len(spine) == 3
    assert spine.filter(pl.col("site").is_null()).height == 1
    # Auxiliary subpop should have remaining population
    assert spine.filter(pl.col("site").is_null())["subpop_pop"][0] == 500_000


def test_site_subpop_spine_no_auxiliary():
    """
    Test when WW sites cover entire population.

    When sampling sites cover the full population, no auxiliary
    subpopulation should be created.
    """
    ww_data = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-01"],
            "site": ["site1", "site2"],
            "site_index": [0, 1],
            "site_pop": [400_000, 600_000],
            "lab_site_index": [0, 1],
            "log_genome_copies_per_ml": [1.0, 2.0],
            "log_lod": [0.5, 0.5],
            "below_lod": [0, 0],
        },
        schema={
            "date": pl.Date,
            "site": pl.String,
            "site_index": pl.Int64,
            "site_pop": pl.Int64,
            "lab_site_index": pl.Int64,
            "log_genome_copies_per_ml": pl.Float64,
            "log_lod": pl.Float64,
            "below_lod": pl.Int64,
        },
    )

    data = PyrenewHEWData(nwss_training_data=ww_data, population_size=1_000_000)

    spine = data.site_subpop_spine
    # Should have only 2 subpops
    assert len(spine) == 2
    assert spine.filter(pl.col("site").is_null()).height == 0


def test_censored_uncensored_split():
    """
    Test correct identification of censored vs uncensored observations.

    Wastewater observations below the limit of detection are censored.
    This test verifies correct indexing of censored/uncensored data.
    """
    ww_data = pl.DataFrame(
        {
            "date": ["2023-01-01"] * 4,
            "site": ["site1"] * 4,
            "site_index": [0] * 4,
            "site_pop": [500_000] * 4,
            "lab_site_index": [0] * 4,
            "log_genome_copies_per_ml": [0.5, 1.5, 0.3, 2.0],
            "log_lod": [1.0, 1.0, 1.0, 1.0],
            "below_lod": [1, 0, 1, 0],  # 2 censored, 2 uncensored
        },
        schema={
            "date": pl.Date,
            "site": pl.String,
            "site_index": pl.Int64,
            "site_pop": pl.Int64,
            "lab_site_index": pl.Int64,
            "log_genome_copies_per_ml": pl.Float64,
            "log_lod": pl.Float64,
            "below_lod": pl.Int64,
        },
    )

    data = PyrenewHEWData(nwss_training_data=ww_data, population_size=1_000_000)

    assert len(data.ww_censored) == 2
    assert len(data.ww_uncensored) == 2
    # Censored indices should be 0 and 2
    assert np.array_equal(data.ww_censored, [0, 2])
    assert np.array_equal(data.ww_uncensored, [1, 3])


def test_n_days_post_init_single_source():
    """
    Test n_days_post_init with only one data source.

    With a single data source, n_days_post_init should equal
    the number of days in that source.
    """
    data = PyrenewHEWData(
        n_ed_visits_data_days=30, first_ed_visits_date=np.datetime64("2023-01-01")
    )

    # Should be 30 days (Jan 1 to Jan 30 inclusive)
    assert data.n_days_post_init == 30


def test_n_days_post_init_multiple_sources():
    """
    Test with multiple overlapping/non-overlapping data sources.

    With multiple data sources, n_days_post_init should span from
    the earliest first date to the latest last date.
    """
    data = PyrenewHEWData(
        n_ed_visits_data_days=20,
        n_hospital_admissions_data_days=3,  # 3 weeks = 21 days
        first_ed_visits_date=np.datetime64("2023-01-01"),
        first_hospital_admissions_date=np.datetime64("2023-01-07"),
    )

    # ED: Jan 1-20 (20 days)
    # Hosp: Jan 7, 14, 21 (3 weeks ending Jan 21, so 21 days from start)
    # Overall: Jan 1 - Jan 21 = 21 days
    assert data.n_days_post_init == 21


def test_lab_site_to_subpop_map():
    """
    Test correct mapping from lab sites to subpopulations.

    Multiple labs can sample from the same site. The mapping
    should correctly associate each lab with its subpopulation.
    """
    ww_data = pl.DataFrame(
        {
            "date": ["2023-01-01"] * 4,
            "site": ["site1", "site1", "site2", "site2"],
            "site_index": [0, 0, 1, 1],
            "site_pop": [400_000] * 2 + [200_000] * 2,
            "lab_site_index": [0, 1, 2, 3],  # 4 labs, 2 sites
            "log_genome_copies_per_ml": [1.0, 1.5, 2.0, 2.5],
            "log_lod": [0.5] * 4,
            "below_lod": [0] * 4,
        },
        schema={
            "date": pl.Date,
            "site": pl.String,
            "site_index": pl.Int64,
            "site_pop": pl.Int64,
            "lab_site_index": pl.Int64,
            "log_genome_copies_per_ml": pl.Float64,
            "log_lod": pl.Float64,
            "below_lod": pl.Int64,
        },
    )

    data = PyrenewHEWData(
        nwss_training_data=ww_data,
        population_size=1_000_000,  # Creates auxiliary subpop 0
    )

    # First 2 labs map to subpop 1 (site_index 0 + 1 for auxiliary)
    # Next 2 labs map to subpop 2 (site_index 1 + 1 for auxiliary)
    mapping = data.lab_site_to_subpop_map
    assert len(mapping) == 4
    assert mapping[0] == 1  # lab 0 -> subpop 1
    assert mapping[1] == 1  # lab 1 -> subpop 1
    assert mapping[2] == 2  # lab 2 -> subpop 2
    assert mapping[3] == 2  # lab 3 -> subpop 2


def test_date_time_spine():
    """
    Test creation of date-time spine for temporal indexing.

    The date_time_spine should map each date to its model time index.
    """
    data = PyrenewHEWData(
        n_ed_visits_data_days=10, first_ed_visits_date=np.datetime64("2023-01-01")
    )

    spine = data.date_time_spine
    assert len(spine) == 10
    assert spine["t"][0] == 0
    assert spine["t"][9] == 9
    assert spine["date"][0] == dt.date(2023, 1, 1)
    assert spine["date"][9] == dt.date(2023, 1, 10)
