import argparse
import datetime as dt
import itertools
import shutil
from pathlib import Path

import arviz as az
import forecasttools
import jax.random as jr
import numpy as np
import polars as pl
import polars.selectors as cs
from scipy.stats import expon, norm

from pipelines.prep_data import (
    process_and_save_loc_data,
    process_and_save_loc_param,
)
from pipelines.prep_ww_data import clean_nwss_data, preprocess_ww_data
from pipelines.utils import build_pyrenew_hew_model_from_dir
from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData

parser = argparse.ArgumentParser(
    description="Create fit data for disease modeling."
)

parser.add_argument(
    "base_dir",
    type=Path,
    help="Base directory for output data.",
)

args = parser.parse_args()
base_dir = args.base_dir

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

nwss_data_disease_map = {
    "COVID-19": "covid",
    "Influenza": "flu",
}


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


def create_var_df(
    idata: az.InferenceData, var: str, state_disease_key: pl.DataFrame
):
    df = (
        pl.from_pandas(
            idata.prior[var].to_dataframe(),
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


def create_param_estimates(
    gi_pmf,
    rt_truncation_pmf,
    delay_pmf,
    states_to_simulate,
    diseases_to_simulate,
    max_train_date_str,
    max_train_date,
    param_estimates_cols,
):
    return (
        (
            pl.DataFrame(
                {
                    "parameter": [
                        "generation_interval",
                        "right_truncation",
                        "delay",
                    ],
                    "value": [gi_pmf, rt_truncation_pmf, delay_pmf],
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


def simulate_data_from_bootstrap(
    n_training_days,
    max_train_date,
    n_nssp_sites,
    facility_level_nssp_data_cols,
    loc_level_nssp_data_cols,
    loc_level_nwss_data_columns,
    n_training_weeks,
    nhsn_cols,
    bootstrap_private_data_dir,
    param_estimates_dir,
    n_forecast_days,
    predictive_var_names,
    n_ww_sites,
    states_to_simulate,
    diseases_to_simulate,
):
    bootstrap_loc = states_to_simulate[0]
    bootstrap_disease = diseases_to_simulate[0]
    # facility_level_nssp_data
    bootstrap_facility_level_nssp_data = (
        pl.DataFrame(
            itertools.product(
                np.arange(-n_training_days, 0 + 1),
                np.arange(1, n_nssp_sites + 1),
                [bootstrap_disease] + ["Total"],
            ),
            schema=["time", "facility", "disease"],
        )
        .with_columns(
            (pl.lit(max_train_date) + pl.duration(days=pl.col("time"))).alias(
                "reference_date"
            ),
            pl.lit(max_train_date).alias("report_date"),
            pl.lit("state").alias("geo_type"),
            pl.lit(bootstrap_loc).alias("geo_value"),
            pl.lit(max_train_date).alias("asof"),
            pl.lit("count_ed_visits").alias("metric"),
            pl.lit(0).alias("run_id"),
            pl.lit(0).alias("value"),
        )
        .select(cs.by_name(facility_level_nssp_data_cols))
    )
    # loc_level_nssp_data
    bootstrap_loc_level_nssp_data = (
        bootstrap_facility_level_nssp_data.with_columns(
            pl.lit(True).alias("any_update_this_day")
        )
        .select(cs.by_name(loc_level_nssp_data_cols))
        .unique()
    )

    first_training_date = bootstrap_loc_level_nssp_data.get_column(
        "reference_date"
    ).min()

    # loc_level_nwss_data
    bootstrap_nwss_etl_base = (
        pl.DataFrame(
            itertools.product(
                np.arange(-n_training_days, 0 + 1), np.arange(n_ww_sites)
            ),
            schema=["time", "site"],
        )
        .with_columns(
            (
                pl.lit(max_train_date)
                + pl.duration(days=(pl.col("time") - pl.col("time").max()))
            ).alias("sample_collect_date"),
            pl.lit(bootstrap_loc).alias("state"),
            pl.lit("wwtp").alias("sample_location"),
            pl.lit("raw wastewater").alias("sample_matrix"),
            pl.lit("copies/l wastewater").alias("pcr_target_units"),
            pl.lit("sars-cov-2").alias("pcr_target"),
            pl.lit(0).alias("site_level_log_ww_conc"),
            pl.lit("n").alias("quality_flag"),
        )
        .with_columns(
            pl.col("site_level_log_ww_conc")
            .exp()
            .alias("pcr_target_avg_conc"),
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

    bootstrap_nwss_site_pop = (
        bootstrap_nwss_etl_base.select(["wwtp_jurisdiction", "wwtp_id"])
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

    bootstrap_loc_level_nwss_data = bootstrap_nwss_etl_base.join(
        bootstrap_nwss_site_pop, on="wwtp_id"
    ).select(cs.by_name(loc_level_nwss_data_columns))

    bootstrap_loc_level_nwss_data = preprocess_ww_data(
        clean_nwss_data(bootstrap_loc_level_nwss_data).filter(
            (pl.col("location") == bootstrap_loc)
            & (pl.col("date") >= first_training_date)
        )
    )
    # nhsn_data_path
    bootstrap_nhsn_data = (
        pl.DataFrame(
            {
                "jurisdiction": bootstrap_loc,
                "time": np.arange(-n_training_weeks, 0 + 1),
                "hospital_admissions": 0,
            }
        )
        .with_columns(
            (pl.lit(max_train_date) + pl.duration(weeks=pl.col("time"))).alias(
                "weekendingdate"
            )
        )
        .select(cs.by_name(nhsn_cols))
    )

    # replace with tempfile utilities
    # https://docs.python.org/3/library/tempfile.html
    bootstrap_nhsn_data_path = Path(
        bootstrap_private_data_dir, "nhsn_data.parquet"
    )
    bootstrap_nhsn_data.write_parquet(bootstrap_nhsn_data_path)

    model_run_dir = Path(bootstrap_private_data_dir, bootstrap_loc)
    model_run_dir.mkdir(parents=True, exist_ok=True)

    process_and_save_loc_data(
        loc_abb=bootstrap_loc,
        disease=bootstrap_disease,
        facility_level_nssp_data=bootstrap_facility_level_nssp_data.lazy(),
        loc_level_nssp_data=bootstrap_loc_level_nssp_data.lazy(),
        loc_level_nwss_data=bootstrap_loc_level_nwss_data,
        report_date=max_train_date,
        first_training_date=first_training_date,
        last_training_date=max_train_date,
        model_run_dir=model_run_dir,
        nhsn_data_path=bootstrap_nhsn_data_path,
    )

    shutil.copy(
        Path("pipelines/priors/prod_priors.py"),
        Path(model_run_dir, "priors.py"),
    )
    process_and_save_loc_param(
        loc_abb=bootstrap_loc,
        disease=bootstrap_disease,
        loc_level_nwss_data=bootstrap_loc_level_nwss_data,
        param_data_dir=param_estimates_dir,
        model_run_dir=model_run_dir,
    )
    my_data = PyrenewHEWData.from_json(
        json_file_path=Path(model_run_dir)
        / "data"
        / "data_for_model_fit.json",
        fit_ed_visits=True,
        fit_hospital_admissions=True,
        fit_wastewater=True,
    )

    my_model = build_pyrenew_hew_model_from_dir(
        model_run_dir,
        fit_ed_visits=True,
        fit_hospital_admissions=True,
        fit_wastewater=True,
    )

    state_disease_key = pl.DataFrame(
        itertools.product(states_to_simulate, diseases_to_simulate),
        schema=["state", "disease"],
    ).with_row_index("draw")

    max_draw = state_disease_key.height

    prior_predictive_samples = my_model.prior_predictive(
        rng_key=jr.key(20),
        numpyro_predictive_args={"num_samples": max_draw},
        data=my_data.to_forecast_data(n_forecast_points=n_forecast_days),
        sample_ed_visits=True,
        sample_hospital_admissions=True,
        sample_wastewater=True,
    )

    idata = az.from_numpyro(
        prior=prior_predictive_samples,
    ).sel(draw=slice(0, max_draw - 1))

    return {
        var: create_var_df(idata, var, state_disease_key)
        for var in predictive_var_names
    }


# %% param_estimates
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

max_train_date_str = "2024-12-21"
max_train_date = dt.datetime.strptime(max_train_date_str, "%Y-%m-%d").date()
# Verify this is a Saturday
assert max_train_date.weekday() == 5

param_estimates = create_param_estimates(
    gi_pmf,
    rt_truncation_pmf,
    delay_pmf,
    states_to_simulate=["MT", "CA", "DC"],
    diseases_to_simulate=["Influenza", "COVID-19", "RSV"],
    max_train_date_str=max_train_date_str,
    max_train_date=max_train_date,
    param_estimates_cols=param_estimates_cols,
)

# %% Generate Bootstrap Data
bootstrap_dir_name = "bootstrap_private_data"
private_data_dir_name = "private_data"
bootstrap_private_data_dir = Path(base_dir, bootstrap_dir_name)
bootstrap_private_data_dir.mkdir(parents=True, exist_ok=True)

private_data_dir = Path(base_dir, private_data_dir_name)
private_data_dir.mkdir(parents=True, exist_ok=True)

nhsn_cols = ["jurisdiction", "weekendingdate", "hospital_admissions"]

n_training_weeks = 16
n_training_days = n_training_weeks * 7
n_forecast_weeks = 4
n_forecast_days = 7 * n_forecast_weeks
n_nssp_sites = 5
n_ww_sites = 5
ww_flag_prob = 0.1

# %% Simulate Data
param_estimates_dir = Path(private_data_dir, "prod_param_estimates")
param_estimates_dir.mkdir(parents=True, exist_ok=True)
param_estimates.write_parquet(Path(param_estimates_dir, "prod.parquet"))

predictive_var_names = [
    "observed_ed_visits",
    "observed_hospital_admissions",
    "site_level_log_ww_conc",
]

dfs_ref_subpop = simulate_data_from_bootstrap(
    n_training_days,
    max_train_date,
    n_nssp_sites,
    facility_level_nssp_data_cols,
    loc_level_nssp_data_cols,
    loc_level_nwss_data_columns,
    n_training_weeks,
    nhsn_cols,
    bootstrap_private_data_dir,
    param_estimates_dir,
    n_forecast_days,
    predictive_var_names,
    n_ww_sites,
    states_to_simulate=["MT", "CA"],
    diseases_to_simulate=["Influenza", "COVID-19", "RSV"],
)

dfs_no_ref_subpop = simulate_data_from_bootstrap(
    n_training_days,
    max_train_date,
    n_nssp_sites,
    facility_level_nssp_data_cols,
    loc_level_nssp_data_cols,
    loc_level_nwss_data_columns,
    n_training_weeks,
    nhsn_cols,
    bootstrap_private_data_dir,
    param_estimates_dir,
    n_forecast_days,
    predictive_var_names,
    n_ww_sites=1,
    states_to_simulate=["DC"],
    diseases_to_simulate=["Influenza", "COVID-19", "RSV"],
)

# Concatenate dataframes by variable names
dfs = {}
for var in predictive_var_names:
    dfs[var] = pl.concat([dfs_ref_subpop[var], dfs_no_ref_subpop[var]])

# %% nssp_etl_gold/2024-12-21.parquet
nssp_etl_gold_no_total = (
    dfs["observed_ed_visits"]
    .with_columns(
        (
            pl.lit(max_train_date)
            + pl.duration(
                days=(pl.col("time") - pl.col("time").max() + n_forecast_days)
            )
        ).alias("reference_date"),
        pl.lit(max_train_date).alias("report_date"),
        pl.lit("state").alias("geo_type"),
        pl.lit("count_ed_visits").alias("metric"),
        pl.col("disease").replace({"COVID-19": "COVID-19/Omicron"}),
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
    .select(cs.by_name(facility_level_nssp_data_cols))
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
nssp_etl_gold.filter(pl.col("reference_date") <= max_train_date).write_parquet(
    Path(nssp_etl_gold_dir, f"{max_train_date}.parquet")
)

# %% nssp_state_level_gold/2024-12-21.parquet
nssp_state_level_gold = (
    nssp_etl_gold.group_by(cs.exclude("facility", "value"))
    .agg(pl.col("value").sum())
    .with_columns(pl.lit(True).alias("any_update_this_day"))
    .sort(["reference_date", "geo_value", "disease"])
    .select(cs.by_name(loc_level_nssp_data_cols))
)

nssp_state_level_gold_dir = Path(private_data_dir, "nssp_state_level_gold")
nssp_state_level_gold_dir.mkdir(parents=True, exist_ok=True)
nssp_state_level_gold.filter(
    pl.col("reference_date") <= max_train_date
).write_parquet(Path(nssp_state_level_gold_dir, f"{max_train_date}.parquet"))


# %% nssp-etl/latest_comprehensive.parquet
nssp_etl_dir = Path(private_data_dir, "nssp-etl")
nssp_etl_dir.mkdir(parents=True, exist_ok=True)
nssp_state_level_gold.select(cs.exclude("any_update_this_day")).write_parquet(
    Path(nssp_etl_dir, "latest_comprehensive.parquet")
)


# %% nwss_vinatges/NWSS-ETL-{disease}-2024-12-21/bronze.parquet
def save_nwss_etl_by_disease(dfs, disease):
    nwss_etl_base = (
        dfs["site_level_log_ww_conc"]
        .filter(pl.col("disease") == disease)
        .with_row_index()
        .with_columns(
            (
                pl.lit(max_train_date)
                + pl.duration(
                    days=(
                        pl.col("time") - pl.col("time").max() + n_forecast_days
                    )
                )
            ).alias("sample_collect_date"),
            pl.first("index")
            .over("state", "site")
            .rank("dense")
            .alias("site"),
            pl.lit("wwtp").alias("sample_location"),
            pl.lit("raw wastewater").alias("sample_matrix"),
            pl.lit("copies/l wastewater").alias("pcr_target_units"),
            pl.lit("sars-cov-2").alias("pcr_target"),
            pl.col("site_level_log_ww_conc")
            .exp()
            .alias("pcr_target_avg_conc"),
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
                    ["n", "y"],
                    size=df.height,
                    p=[1 - ww_flag_prob, ww_flag_prob],
                )
            )
        )
        .select(cs.by_name(loc_level_nwss_data_columns, require_all=False))
        .pipe(
            lambda df: pl.concat(
                [
                    df,
                    df.sample(n=5).with_columns(
                        (pl.col("pcr_target_avg_conc") + np.random.rand(5))
                        .cast(pl.Float32)
                        .alias("pcr_target_avg_conc"),
                    ),
                ]
            )
        )
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
            pl.when(pl.col("wwtp_jurisdiction") == "DC")
            .then(pl.concat_list([pl.col("population") * 2]))
            # Simulates nwss data in DC where pop served
            # by ww surveillance > state population
            .otherwise(
                pl.struct(["population", "wwtp_id"]).map_elements(
                    lambda x: dirichlet_integer_split(
                        x["population"], len(x["wwtp_id"]) + 1
                    )[1:],
                    pl.List(pl.Int64),
                )
            )
            .alias("population_served")
        )
        .explode("wwtp_id", "population_served")
    )
    nwss_etl = nwss_etl_base.join(nwss_site_pop, on="wwtp_id").select(
        cs.by_name(loc_level_nwss_data_columns)
    )
    nwss_etl_dir = Path(
        private_data_dir,
        "nwss_vintages",
        f"NWSS-ETL-{nwss_data_disease_map[disease]}-{max_train_date}",
    )
    nwss_etl_dir.mkdir(parents=True, exist_ok=True)
    nwss_etl.filter(
        pl.col("sample_collect_date") <= max_train_date
    ).write_parquet(Path(nwss_etl_dir, "bronze.parquet"))


[
    save_nwss_etl_by_disease(dfs, disease)
    for disease in ["COVID-19", "Influenza"]
]

# %% nhsn_test_data/nhsn_test_data.parquet
nhsn_data_sates = (
    dfs["observed_hospital_admissions"]
    .with_columns(
        (
            pl.lit(max_train_date)
            + pl.duration(
                weeks=(
                    pl.col("time") - pl.col("time").max() + n_forecast_weeks
                )
            )
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
    data.select(cs.by_name(nhsn_cols)).write_parquet(
        Path(nhsn_dir, f"{name[0]}_{name[1]}.parquet")
    )
