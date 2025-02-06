import datetime

import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from pipelines.prep_ww_data import get_date_time_spine
from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData


@pytest.mark.parametrize(
    [
        "n_ed_visits_datapoints",
        "n_hospital_admissions_datapoints",
        "n_wastewater_datapoints",
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
    n_ed_visits_datapoints: int,
    n_hospital_admissions_datapoints: int,
    n_wastewater_datapoints: int,
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
        n_ed_visits_datapoints=n_ed_visits_datapoints,
        n_hospital_admissions_datapoints=n_hospital_admissions_datapoints,
        n_wastewater_datapoints=n_wastewater_datapoints,
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
    assert forecast_data.n_ed_visits_datapoints == n_days_expected
    assert forecast_data.n_wastewater_datapoints == n_days_expected
    assert forecast_data.n_hospital_admissions_datapoints == n_weeks_expected
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
            "lab_site_index": [1] * 30 + [2] * 30,
            "subpop_index": [1] * 30 + [2] * 30,
            "log_genome_copies_per_ml": np.log(
                np.abs(np.random.normal(loc=500, scale=50, size=60))
            ),
            "log_lod": np.log([20] * 30 + [15] * 30),
            "subpop_pop": [200_000] * 30 + [400_000] * 30,
        }
    )

    date_time_spine = get_date_time_spine(
        start_date=first_training_date, end_date=last_training_date
    )

    ww_data = ww_raw.join(
        date_time_spine, on="date", how="left", coalesce=True
    ).with_columns(
        pl.arange(0, pl.len()).alias("ind_rel_to_observed_times"),
        (pl.col("log_genome_copies_per_ml") <= pl.col("log_lod"))
        .cast(pl.Int8)
        .alias("below_lod"),
    )

    first_ed_visits_date = datetime.date(2023, 1, 1)
    first_hospital_admissions_date = datetime.date(2023, 1, 1)
    first_wastewater_date = datetime.date(2023, 1, 1)
    n_forecast_points = 10

    data = PyrenewHEWData(
        first_ed_visits_date=first_ed_visits_date,
        first_hospital_admissions_date=first_hospital_admissions_date,
        first_wastewater_date=first_wastewater_date,
        wastewater_data=ww_data,
        population_size=1e6,
    )

    forecast_data = data.to_forecast_data(10)

    assert forecast_data.data_observed_disease_wastewater is not None
    assert jnp.array_equal(forecast_data.ww_uncensored, data.ww_uncensored)
