"""Missing likelihood terms must retain their original calendar coordinates."""

import datetime as dt

import numpy as np
import polars as pl
import pytest

from cfa.stf.routine.pyrenew_hew.observation_data import PyRenewObservationData


def test_independent_signal_masks_preserve_time_grid_and_tail():
    dates = [dt.date(2026, 1, 4) + dt.timedelta(days=i) for i in range(21)]
    values = [float(i + 1) if i not in {0, 9, 19, 20} else None for i in range(21)]
    data = PyRenewObservationData(
        nssp_training_data=pl.DataFrame(
            {
                "date": dates,
                "observed_ed_visits": values,
                "other_ed_visits": [100.0] * 21,
            }
        ),
        nhsn_training_data=pl.DataFrame(
            {"weekendingdate": dates[6::7], "hospital_admissions": [None, 20.0, None]}
        ),
        nssp_step_size=1,
        nhsn_step_size=7,
        right_truncation_offset=0,
    )
    assert data.n_days_post_init == 21
    assert data.first_ed_visits_date == np.datetime64(dates[0])
    assert data.last_data_date_overall == np.datetime64(dates[-1])
    np.testing.assert_array_equal(
        data.model_t_obs_ed_visits,
        [i for i, value in enumerate(values) if value is not None],
    )
    np.testing.assert_array_equal(
        data.data_observed_disease_ed_visits,
        [value for value in values if value is not None],
    )
    np.testing.assert_array_equal(data.model_t_obs_hospital_admissions, [13])
    np.testing.assert_array_equal(
        data.data_observed_disease_hospital_admissions, [20.0]
    )
    assert data.right_truncation_offset == 0
    forecast = data.to_forecast_data(28)
    assert forecast.n_days_post_init == 49
    assert forecast.data_observed_disease_ed_visits is None


def test_all_missing_signal_is_rejected():
    data = PyRenewObservationData(
        nssp_training_data=pl.DataFrame(
            {
                "date": [dt.date(2026, 1, 1)],
                "observed_ed_visits": [float("nan")],
            }
        ),
        nssp_step_size=1,
    )
    with pytest.raises(ValueError, match="finite observation"):
        _ = data.data_observed_disease_ed_visits
