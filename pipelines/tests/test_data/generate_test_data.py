# %% Setup
import argparse
import datetime as dt
import itertools
import pickle
from pathlib import Path

import arviz as az
import jax.random as jr
import numpy as np
import polars as pl
import polars.selectors as cs

from pipelines.build_pyrenew_model import (
    build_model_from_dir,
)
from pyrenew_hew.util import flags_from_pyrenew_model_name

max_date = "2024-12-21"


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

# %% nssp_state_level_gold/2024-12-21.parquet

observed_ed_visits_total = (
    (
        dfs["observed_ed_visits"]
        .group_by(["state", "time"])
        .agg(pl.col("observed_ed_visits").sum())
        .with_columns(pl.lit("Total").alias("disease"))
    )
    .select(dfs["observed_ed_visits"].columns)
    .sort(["state", "disease", "time"])
)

nssp_state_level_gold = (
    pl.concat([dfs["observed_ed_visits"], observed_ed_visits_total])
    .filter(pl.col("time") != 0)
    .with_columns(
        (
            pl.lit(max_date).str.to_date()
            - pl.duration(days=pl.col("time").max() - pl.col("time"))
        ).alias("reference_date"),
        pl.lit(max_date).str.to_date().alias("report_date"),
        pl.lit("state").alias("geo_type"),
        pl.lit("count_ed_visits").alias("metric"),
        pl.col("disease").replace(nssp_disease_key),
        pl.lit(True).alias("any_update_this_day"),
    )
    .rename({"state": "geo_value", "observed_ed_visits": "value"})
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
nssp_state_level_gold.write_parquet(
    Path(nssp_state_level_gold_dir, f"{max_date}.parquet")
)

# nssp_etl_gold/2024-12-21.parquet

# nwss_vintages/NWSS-ETL-covid-2024-12-21/bronze.parquet
# nssp-etl/latest_comprehensive.parquet
# prod_param_estimates/prod.parquet
# don't forget to also make NHSN like data
