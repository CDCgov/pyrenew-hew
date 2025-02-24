import datetime

import pytest

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData
from pyrenew_hew.pyrenew_wastewater_data import PyrenewWastewaterData


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
    assert (
        forecast_data.first_hospital_admissions_date
        == data.first_data_date_overall
    )
    assert forecast_data.first_wastewater_date == data.first_data_date_overall
    assert forecast_data.data_observed_disease_wastewater_conc is None
