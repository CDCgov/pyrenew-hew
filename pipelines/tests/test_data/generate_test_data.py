# %% Setup
import argparse
import datetime as dt
import itertools
import pickle
from pathlib import Path

import arviz as az
import forecasttools
import jax.random as jr
import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.stats import expon, norm

from pipelines.build_pyrenew_model import (
    build_model_from_dir,
)
from pyrenew_hew.util import flags_from_pyrenew_model_name

max_train_date = "2024-12-21"
# Verify this is a Saturday
assert dt.datetime.strptime(max_train_date, "%Y-%m-%d").weekday() == 5
n_forecast_weeks = 4
n_forecast_days = 7 * n_forecast_weeks

n_nssp_sites = 5
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


# %% Use an existing model to simulate data
model_run_dir = "pipelines/tests/end_to_end_test_output/2024-12-21_forecasts/covid-19_r_2024-12-21_f_2024-10-22_t_2024-12-20/model_runs/CA"
model_name = "pyrenew_hew"

states_to_simulate = ["MT", "CA"]
diseases_to_simulate = ["COVID-19", "Influenza", "RSV"]

state_disease_key = pl.DataFrame(
    itertools.product(states_to_simulate, diseases_to_simulate),
    schema=["state", "disease"],
).with_row_index("draw")

max_draw = state_disease_key.height

model_run_dir = Path(model_run_dir)

model_dir = Path(model_run_dir, model_name)
if not model_dir.exists():
    raise FileNotFoundError(f"The directory {model_dir} does not exist.")

(my_model, my_data) = build_model_from_dir(
    model_run_dir, **flags_from_pyrenew_model_name(model_name)
)


with open(
    model_dir / "posterior_samples.pickle",
    "rb",
) as file:
    my_model.mcmc = pickle.load(file)

prior_predictive_samples = my_model.prior_predictive(
    rng_key=jr.key(20),
    numpyro_predictive_args={"num_samples": max_draw},
    data=my_data.to_forecast_data(n_forecast_points=0),
    sample_ed_visits=True,
    sample_hospital_admissions=True,
    sample_wastewater=True,
)

posterior_predictive_samples = my_model.posterior_predictive(
    data=my_data.to_forecast_data(n_forecast_points=0),
    sample_ed_visits=True,
    sample_hospital_admissions=True,
    sample_wastewater=True,
)

predictive_var_names = [
    "observed_ed_visits",
    "observed_hospital_admissions",
    "site_level_log_ww_conc",
]

idata = az.from_numpyro(
    prior=prior_predictive_samples,
    posterior_predictive=posterior_predictive_samples,
).sel(draw=slice(0, max_draw - 1))


# %% Get dfs per var
def create_var_df(idata: az.InferenceData, var: str):
    df = (
        pl.from_pandas(
            idata.posterior_predictive[var].to_dataframe(),
            include_index=True,
        )
        .join(state_disease_key, on="draw")
        .select(cs.exclude("draw", "chain"))
    )

    dim_0_col = f"{var}_dim_0"
    dim_1_col = f"{var}_dim_1"

    rename_dict = {}

    if dim_0_col in df.columns:
        rename_dict[dim_0_col] = "time"
    if dim_1_col in df.columns:
        rename_dict[dim_1_col] = "site"

    renamed_df = df.select(
        "state",
        "disease",
        var,
        cs.by_name([dim_0_col, dim_1_col], require_all=False),
    ).rename(rename_dict)
    return renamed_df


# Create individual dataframes for each variable
dfs = {var: create_var_df(idata, var) for var in predictive_var_names}


# %% Save data
private_data_dir = Path(
    "pipelines/tests/end_to_end_test_output/private_data_sim"
)
private_data_dir.mkdir(parents=True, exist_ok=True)

nssp_disease_key = {"COVID-19": "COVID-19/Omicron"}

# %% nssp_etl_gold/2024-12-21.parquet
nssp_etl_gold_no_total = (
    dfs["observed_ed_visits"]
    .with_columns(
        (
            pl.lit(max_train_date).str.to_date()
            + pl.duration(days=(pl.col("time") - n_forecast_days + 1))
        ).alias("reference_date"),
        pl.lit(max_train_date).str.to_date().alias("report_date"),
        pl.lit("state").alias("geo_type"),
        pl.lit("count_ed_visits").alias("metric"),
        pl.col("disease").replace(nssp_disease_key),
        pl.lit(True).alias("any_update_this_day"),
        pl.lit(np.arange(1, n_nssp_sites + 1).tolist()).alias("facility"),
        pl.lit(max_train_date).alias("asof"),
        pl.lit(0).alias("run_id"),
        pl.col("observed_ed_visits").map_elements(
            lambda x: dirichlet_integer_split(x, k=n_nssp_sites).tolist(),
            # return_dtype=pl.Array(pl.Int64, n_nssp_sites),
            # Seems like this should work if you omit the .tolist, but it doesn't
            pl.List(pl.Int64),
        ),
    )
    .rename({"state": "geo_value", "observed_ed_visits": "value"})
    .explode(["value", "facility"])
    .select(
        [
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
    )
)

nssp_etl_gold_total = (
    nssp_etl_gold_no_total.group_by(cs.exclude("disease", "value"))
    .agg(pl.col("value").sum())
    .with_columns(pl.lit("Total").alias("disease"))
    .select(nssp_etl_gold_no_total.columns)
    .sort(["reference_date", "geo_value", "facility", "disease"])
)


nssp_etl_gold = pl.concat([nssp_etl_gold_no_total, nssp_etl_gold_total]).sort(
    ["reference_date", "geo_value", "facility", "disease"]
)

nssp_etl_gold_dir = Path(private_data_dir, "nssp_etl_gold")
nssp_etl_gold_dir.mkdir(parents=True, exist_ok=True)
nssp_etl_gold.filter(
    pl.col("reference_date") <= pl.lit(max_train_date).str.to_date()
).write_parquet(Path(nssp_etl_gold_dir, f"{max_train_date}.parquet"))


# %% nssp_state_level_gold/2024-12-21.parquet
# I think this should have somewhat different dates available compared to
# nssp_etl_gold, but I'm not sure

nssp_state_level_gold = (
    nssp_etl_gold.group_by(cs.exclude("facility", "value"))
    .agg(pl.col("value").sum())
    .with_columns(pl.lit(True).alias("any_update_this_day"))
    .sort(["reference_date", "geo_value", "disease"])
    .select(
        [
            "reference_date",
            "report_date",
            "geo_type",
            "geo_value",
            "metric",
            "disease",
            "value",
            "any_update_this_day",
        ]
    )
)

nssp_state_level_gold_dir = Path(private_data_dir, "nssp_state_level_gold")
nssp_state_level_gold_dir.mkdir(parents=True, exist_ok=True)
nssp_state_level_gold.filter(
    pl.col("reference_date") <= pl.lit(max_train_date).str.to_date()
).write_parquet(Path(nssp_state_level_gold_dir, f"{max_train_date}.parquet"))


# %% nssp-etl/latest_comprehensive.parquet
nssp_etl_dir = Path(private_data_dir, "nssp-etl")
nssp_etl_dir.mkdir(parents=True, exist_ok=True)
nssp_state_level_gold.select(cs.exclude("any_update_this_day")).write_parquet(
    Path(nssp_etl_dir, "latest_comprehensive.parquet")
)


# %% nwss_vintages/NWSS-ETL-covid-2024-12-21/bronze.parquet
nwss_cols = [
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
    dfs["site_level_log_ww_conc"]
    .filter(pl.col("disease") == "COVID-19")
    .with_row_index()
    .with_columns(
        (
            pl.lit(max_train_date).str.to_date()
            - pl.duration(
                days=pl.col("time").max() - pl.col("time") - n_forecast_days
            )
        ).alias("sample_collect_date"),
        pl.first("index").over("state", "site").rank("dense").alias("site"),
        pl.lit("wwtp").alias("sample_location"),
        pl.lit("raw wastewater").alias("sample_matrix"),
        pl.lit("copies/l wastewater").alias("pcr_target_units"),
        pl.lit("sars-cov-2").alias("pcr_target"),
        pl.col("site_level_log_ww_conc").exp().alias("pcr_target_avg_conc"),
    )
    .with_columns(
        pl.col("site").alias("lab_id"),
        pl.col("site").alias("wwtp_id"),
    )
    .with_columns(
        pl.quantile("pcr_target_avg_conc", 0.05)
        .over("state", "site")
        .alias("lod_sewage")
    )
    .rename({"state": "wwtp_jurisdiction"})
    .pipe(
        lambda df: df.with_columns(
            quality_flag=np.random.choice(
                ["n", "y"], size=df.height, p=[1 - ww_flag_prob, ww_flag_prob]
            )
        )
    )
    .select(cs.by_name(nwss_cols, require_all=False))
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
nwss_etl = nwss_etl_base.join(nwss_site_pop, on="wwtp_id").select(
    cs.by_name(nwss_cols)
)
nwss_etl_dir = Path(
    private_data_dir, "nwss_vintages", f"NWSS-ETL-covid-{max_train_date}"
)
nwss_etl_dir.mkdir(parents=True, exist_ok=True)
nwss_etl.filter(
    pl.col("sample_collect_date") <= pl.lit(max_train_date).str.to_date()
).write_parquet(Path(nwss_etl_dir, "bronze.parquet"))


# %% nhsn_test_data/nhsn_test_data.parquet
nhsn_cols = ["jurisdiction", "weekendingdate", "hospital_admissions"]


nhsn_data_sates = (
    dfs["observed_hospital_admissions"]
    .with_columns(
        (
            pl.lit(max_train_date).str.to_date()
            + pl.duration(weeks=(pl.col("time") - n_forecast_weeks + 1))
        ).alias("weekendingdate")
    )
    .rename(
        {
            "state": "jurisdiction",
            "observed_hospital_admissions": "hospital_admissions",
        }
    )
    .select("disease", cs.by_name(nhsn_cols))
)

# Create us data by summing across jurisdictions
nhsn_data_us = (
    nhsn_data_sates.group_by(["disease", "weekendingdate"])
    .agg(pl.col("hospital_admissions").sum())
    .with_columns(pl.lit("US").alias("jurisdiction"))
    .select("disease", cs.by_name(nhsn_cols))
)

# Combine with state data
nhsn_data_combined = pl.concat([nhsn_data_sates, nhsn_data_us]).sort(
    "disease", "jurisdiction", "weekendingdate"
)

# Create directory for NHSN data
nhsn_dir = Path(private_data_dir, "nhsn_test_data")
nhsn_dir.mkdir(parents=True, exist_ok=True)

for name, data in nhsn_data_combined.group_by("disease", "jurisdiction"):
    print(f"{name[0]}_{name[1]}")
    print(data.select(cs.by_name(nhsn_cols)))
    data.select(cs.by_name(nhsn_cols)).write_parquet(
        Path(nhsn_dir, f"{name[0]}_{name[1]}.parquet")
    )

# %% prod_param_estimates/prod.parquet
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

prod_param_estimates = (
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
        pl.lit(max_train_date).alias("reference_date"),
        pl.lit(None).alias("end_date"),
        pl.lit(max_train_date).str.to_date().alias("start_date")
        - pl.duration(days=180),
    )
    .with_row_index("id")
    .select(cs.by_name(param_estimates_cols))
)

prod_param_estimates_dir = Path(private_data_dir, "prod_param_estimates")
prod_param_estimates_dir.mkdir(parents=True, exist_ok=True)
prod_param_estimates.write_parquet(
    Path(prod_param_estimates_dir, "prod.parquet")
)
