import datetime

import numpy as np
import polars as pl

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData
from pyrenew_hew.pyrenew_wastewater_data import PyrenewWastewaterData


def test_pyrenew_wastewater_data():
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

    wastewater_data = PyrenewWastewaterData(
        data_observed_disease_wastewater=ww_data,
        population_size=1e6,
    )

    data = PyrenewHEWData(
        first_ed_visits_date=first_ed_visits_date,
        first_hospital_admissions_date=first_hospital_admissions_date,
        first_wastewater_date=first_wastewater_date,
        **wastewater_data.to_pyrenew_hew_data_args(),
    )

    forecast_data = data.to_forecast_data(n_forecast_points)
    assert forecast_data.data_observed_disease_wastewater_conc is None
    assert data.data_observed_disease_wastewater_conc is not None

    assert np.array_equal(
        data.data_observed_disease_wastewater_conc,
        ww_data["log_genome_copies_per_ml"],
    )
    assert len(data.ww_censored) == len(
        ww_data.filter(pl.col("below_lod") == 1)
    )
    assert len(data.ww_uncensored) == len(
        ww_data.filter(pl.col("below_lod") == 0)
    )
    assert np.array_equal(data.ww_log_lod, ww_data["log_lod"])
    assert data.n_ww_lab_sites == ww_data["lab_site_index"].n_unique()
