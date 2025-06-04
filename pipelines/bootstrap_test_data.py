from pathlib import Path

import numpy as np
import polars as pl

from pipelines.prep_data import process_and_save_loc

states_to_simulate = ["MT", "CA"]
diseases_to_simulate = ["COVID-19", "Influenza", "RSV"]
max_train_date = "2024-12-21"
# Verify this is a Saturday
assert dt.datetime.strptime(max_train_date, "%Y-%m-%d").weekday() == 5
n_training_weeks = 16
n_training_days = n_training_weeks * 7
n_forecast_weeks = 4
n_forecast_days = 7 * n_forecast_weeks

np.arange(-n_training_days, n_forecast_days + 1)

n_nssp_sites = 5
ww_flag_prob = 0.1

bootstrap_private_data_dir = Path(
    "pipelines/tests/end_to_end_test_output/bootstrap_private_data"
)
bootstrap_private_data_dir.mkdir(parents=True, exist_ok=True)

# replace with tempfile utilities
# https://docs.python.org/3/library/tempfile.html

nhsn_data_path = Path(bootstrap_private_data_dir, "nhsn_data.parquet")
nhsn_data = (
    pl.DataFrame(
        {
            "jurisdiction": states_to_simulate[0],
            "time": np.arange(-n_training_weeks, 0 + 1),
            "hospital_admissions": 0,
        }
    )
    .with_columns(
        (
            pl.lit(max_train_date).str.to_date()
            + pl.duration(weeks=pl.col("time"))
        ).alias("weekendingdate")
    )
    .select(["jurisdiction", "weekendingdate", "hospital_admissions"])
)

nhsn_data.write_parquet(nhsn_data_path)


process_and_save_loc(
    loc_abb=loc,
    disease=disease,
    facility_level_nssp_data=facility_level_nssp_data,
    loc_level_nssp_data=loc_level_nssp_data,
    loc_level_nwss_data=loc_level_nwss_data,
    report_date=report_date,
    first_training_date=first_training_date,
    last_training_date=last_training_date,
    param_estimates=param_estimates,
    model_run_dir=model_run_dir,
    logger=logger,
    credentials_dict=credentials_dict,
    nhsn_data_path=nhsn_data_path,
)
