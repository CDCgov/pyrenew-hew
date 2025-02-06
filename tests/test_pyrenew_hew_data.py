import datetime

import pytest

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
    assert forecast_data.ww_censored is None
