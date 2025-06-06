import datetime as dt
import itertools
from pathlib import Path

import forecasttools
import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.stats import expon, norm

from pipelines.prep_data import process_and_save_loc
from pipelines.prep_ww_data import clean_nwss_data, preprocess_ww_data

states_to_simulate = ["MT", "CA"]
diseases_to_simulate = ["COVID-19", "Influenza", "RSV"]
loc = states_to_simulate[1]
disease = diseases_to_simulate[1]
max_train_date_str = "2024-12-21"
max_train_date = dt.datetime.strptime(max_train_date_str, "%Y-%m-%d").date()
# Verify this is a Saturday
assert max_train_date.weekday() == 5
n_training_weeks = 16
n_training_days = n_training_weeks * 7
n_forecast_weeks = 4
n_forecast_days = 7 * n_forecast_weeks

n_nssp_sites = 5
n_ww_sites = 5
ww_flag_prob = 0.1


def dirichlet_integer_split(n, k, alpha=1.0):
    proportions = np.random.dirichlet(np.full(k, alpha))
    scaled = proportions * n
    counts = np.floor(scaled).astype(int)

    remainder = n - counts.sum()
    if remainder > 0:
        frac_parts = scaled - counts
        indices = np.argpartition(-frac_parts, remainder)[:remainder]
        counts[indices] += 1

    return counts


# %% facility_level_nssp_data
facility_level_nssp_data_cols = [
    "reference_date",
    "report_date",
    "geo_type",
    "geo_value",
    "asof",
    "metric",
    "run_id",
    "facility",
    "disease",
    "value",
]

facility_level_nssp_data = (
    pl.DataFrame(
        itertools.product(
            np.arange(-n_training_days, 0 + 1),
            np.arange(1, n_nssp_sites + 1),
            [disease] + ["Total"],
        ),
        schema=["time", "facility", "disease"],
    )
    .with_columns(
        (pl.lit(max_train_date) + pl.duration(days=pl.col("time"))).alias(
            "reference_date"
        ),
        pl.lit(max_train_date).alias("report_date"),
        pl.lit("state").alias("geo_type"),
        pl.lit(loc).alias("geo_value"),
        pl.lit(max_train_date).alias("asof"),
        pl.lit("count_ed_visits").alias("metric"),
        pl.lit(0).alias("run_id"),
        pl.lit(0).alias("value"),
    )
    .select(cs.by_name(facility_level_nssp_data_cols))
)

# %% loc_level_nssp_data
loc_level_nssp_data_cols = [
    "reference_date",
    "report_date",
    "geo_type",
    "geo_value",
    "metric",
    "disease",
    "value",
    "any_update_this_day",
]


loc_level_nssp_data = (
    facility_level_nssp_data.with_columns(
        pl.lit(True).alias("any_update_this_day")
    )
    .select(cs.by_name(loc_level_nssp_data_cols))
    .unique()
)

first_training_date = loc_level_nssp_data.get_column("reference_date").min()

# %% loc_level_nwss_data

loc_level_nwss_data_columns = [
    "sample_collect_date",
    "lab_id",
    "wwtp_id",
    "pcr_target_avg_conc",
    "sample_location",
    "sample_matrix",
    "pcr_target_units",
    "pcr_target",
    "wwtp_jurisdiction",
    "population_served",
    "quality_flag",
    "lod_sewage",
]


nwss_etl_base = (
    pl.DataFrame(
        itertools.product(
            np.arange(-n_training_days, 0 + 1), np.arange(1, n_nssp_sites + 1)
        ),
        schema=["time", "site"],
    )
    .with_columns(
        (
            pl.lit(max_train_date)
            + pl.duration(
                days=(pl.col("time") - pl.col("time").max() + n_forecast_days)
            )
        ).alias("sample_collect_date"),
        pl.lit(loc).alias("state"),
        pl.lit("wwtp").alias("sample_location"),
        pl.lit("raw wastewater").alias("sample_matrix"),
        pl.lit("copies/l wastewater").alias("pcr_target_units"),
        pl.lit("sars-cov-2").alias("pcr_target"),
        pl.lit(0).alias("site_level_log_ww_conc"),
        pl.lit("n").alias("quality_flag"),
    )
    .with_columns(
        pl.col("site_level_log_ww_conc").exp().alias("pcr_target_avg_conc"),
        pl.col("site").alias("lab_id"),
        pl.col("site").alias("wwtp_id"),
    )
    .with_columns(
        pl.quantile("pcr_target_avg_conc", 0.05)
        .over("state", "site")
        .alias("lod_sewage")
    )
    .rename({"state": "wwtp_jurisdiction"})
    .select(cs.by_name(loc_level_nwss_data_columns, require_all=False))
)


nwss_site_pop = (
    nwss_etl_base.select(["wwtp_jurisdiction", "wwtp_id"])
    .unique(["wwtp_jurisdiction", "wwtp_id"])
    .group_by("wwtp_jurisdiction")
    .agg("wwtp_id")
    .join(
        forecasttools.location_table.rename(
            {"short_name": "wwtp_jurisdiction"}
        ).select("wwtp_jurisdiction", "population"),
        on="wwtp_jurisdiction",
    )
    .with_columns(
        pl.struct(["population", "wwtp_id"])
        .map_elements(
            lambda x: dirichlet_integer_split(
                x["population"], len(x["wwtp_id"]) + 1
            )[1:],
            pl.List(pl.Int64),
        )
        .alias("population_served")
    )
    .explode("wwtp_id", "population_served")
)

loc_level_nwss_data = nwss_etl_base.join(nwss_site_pop, on="wwtp_id").select(
    cs.by_name(loc_level_nwss_data_columns)
)

# %% param_estimates
param_estimates_cols = [
    "id",
    "start_date",
    "end_date",
    "reference_date",
    "disease",
    "format",
    "parameter",
    "geo_value",
    "value",
]

# GI PMF: Exponential on discrete times from 0.5 to 6.5
gi_support = np.arange(0.5, 7.0)  # Equivalent to seq(0.5, 6.5)
gi_pmf = expon.pdf(gi_support)
gi_pmf = gi_pmf / gi_pmf.sum()

# Delay PMF: Normal on log-transformed support, normalized and prepended with 0
delay_support = np.log(np.arange(1, 12))
delay_pmf = norm.pdf(delay_support, loc=np.log(3), scale=0.5)
delay_pmf = delay_pmf / delay_pmf.sum()
delay_pmf = np.insert(delay_pmf, 0, 0)

# RT Truncation PMF
rt_truncation_pmf = np.array([1.0, 0, 0, 0])

param_estimates = (
    (
        pl.DataFrame(
            {
                "parameter": [
                    "generation_interval",
                    "right_truncation",
                    "delay",
                ],
                "value": [rt_truncation_pmf, gi_pmf, delay_pmf],
            }
        )
        .join(
            pl.DataFrame(
                {
                    "geo_value": states_to_simulate + ["US"],
                    "parameter": "right_truncation",
                }
            ),
            on="parameter",
            how="left",
        )
        .join(pl.DataFrame({"disease": diseases_to_simulate}), how="cross")
    )
    .with_columns(
        pl.lit("PMF").alias("format"),
        pl.lit(max_train_date_str).alias("reference_date"),
        pl.lit(None).cast(pl.Date).alias("end_date"),
        pl.lit(max_train_date).alias("start_date") - pl.duration(days=180),
    )
    .with_row_index("id")
    .select(cs.by_name(param_estimates_cols))
)
# %% nhsn_data_path
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
        (pl.lit(max_train_date) + pl.duration(weeks=pl.col("time"))).alias(
            "weekendingdate"
        )
    )
    .select(["jurisdiction", "weekendingdate", "hospital_admissions"])
)

nhsn_data.write_parquet(nhsn_data_path)

# %% Run it

nwss_data_raw = loc_level_nwss_data.lazy()
nwss_data_cleaned = clean_nwss_data(nwss_data_raw).filter(
    (pl.col("location") == loc) & (pl.col("date") >= first_training_date)
)
loc_level_nwss_data = preprocess_ww_data(nwss_data_cleaned.collect())

process_and_save_loc(
    loc_abb=loc,
    disease=disease,
    facility_level_nssp_data=facility_level_nssp_data.lazy(),
    loc_level_nssp_data=loc_level_nssp_data.lazy(),
    loc_level_nwss_data=loc_level_nwss_data,
    report_date=max_train_date,
    first_training_date=first_training_date,
    last_training_date=max_train_date,
    param_estimates=param_estimates.lazy(),
    model_run_dir=bootstrap_private_data_dir,
    nhsn_data_path=nhsn_data_path,
)
