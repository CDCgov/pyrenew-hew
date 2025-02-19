import datetime

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData


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
            datetime.date(2023, 1, 1),
            datetime.date(2022, 2, 5),
            datetime.date(2025, 12, 5),
            10,
        ],
        [
            0,
            325,
            2,
            5,
            datetime.date(2025, 1, 1),
            datetime.date(2023, 5, 25),
            datetime.date(2022, 4, 5),
            10,
        ],
        [
            0,
            0,
            2,
            3,
            datetime.date(2025, 1, 1),
            datetime.date(2025, 2, 5),
            datetime.date(2024, 12, 5),
            30,
        ],
        [
            0,
            0,
            23,
            3,
            datetime.date(2025, 1, 1),
            datetime.date(2025, 2, 5),
            datetime.date(2024, 12, 5),
            30,
        ],
    ],
)
def test_to_forecast_data(
    n_ed_visits_data_days: int,
    n_hospital_admissions_data_days: int,
    n_wastewater_data_days: int,
    right_truncation_offset: int,
    first_ed_visits_date: datetime.date,
    first_hospital_admissions_date: datetime.date,
    first_wastewater_date: datetime.date,
    n_forecast_points: int,
) -> None:
    """
    Test the to_forecast_data method
    """
    data = PyrenewHEWData(
        n_ed_visits_data_days=n_ed_visits_data_days,
        n_hospital_admissions_data_days=n_hospital_admissions_data_days,
        n_wastewater_data_days=n_wastewater_data_days,
        first_ed_visits_date=first_ed_visits_date,
        first_hospital_admissions_date=first_hospital_admissions_date,
        first_wastewater_date=first_wastewater_date,
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
    assert (
        forecast_data.first_hospital_admissions_date
        == data.first_data_date_overall
    )
    assert forecast_data.first_wastewater_date == data.first_data_date_overall
    assert forecast_data.data_observed_disease_wastewater is None


def test_wastewater_data_properties():
    first_training_date = datetime.date(2023, 1, 1)
    last_training_date = datetime.date(2023, 7, 23)
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

    first_ed_visits_date = datetime.date(2023, 1, 1)
    first_hospital_admissions_date = datetime.date(2023, 1, 1)
    first_wastewater_date = datetime.date(2023, 1, 1)
    n_forecast_points = 10

    data = PyrenewHEWData(
        first_ed_visits_date=first_ed_visits_date,
        first_hospital_admissions_date=first_hospital_admissions_date,
        first_wastewater_date=first_wastewater_date,
        data_observed_disease_wastewater=ww_data,
        population_size=1e6,
    )

    forecast_data = data.to_forecast_data(n_forecast_points)
    assert forecast_data.data_observed_disease_wastewater_conc is None
    assert data.data_observed_disease_wastewater_conc is not None

    assert jnp.array_equal(
        data.data_observed_disease_wastewater_conc,
        ww_data["log_genome_copies_per_ml"],
    )
    assert len(data.ww_censored) == len(
        ww_data.filter(pl.col("below_lod") == 1)
    )
    assert len(data.ww_uncensored) == len(
        ww_data.filter(pl.col("below_lod") == 0)
    )
    assert jnp.array_equal(data.ww_log_lod, ww_data["log_lod"])
    assert data.n_ww_lab_sites == ww_data["lab_site_index"].n_unique()
