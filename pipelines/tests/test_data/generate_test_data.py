# %% Use an existing model
model_run_dir = "tests/end_to_end_test_output/2024-12-21_forecasts/covid-19_r_2024-12-21_f_2024-10-22_t_2024-12-20/model_runs/CA"
model_name = "pyrenew_hew"

states_to_simulate = ["MT", "CA"]
n_states = len(states_to_simulate)

import argparse
import pickle
from pathlib import Path

import arviz as az
import jax.random as jr
import numpy as np
import polars as pl
import polars.selectors as cs
from build_pyrenew_model import (
    build_model_from_dir,
)

from pyrenew_hew.util import flags_from_pyrenew_model_name

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
    numpyro_predictive_args={"num_samples": n_states},
    data=my_data.to_forecast_data(n_forecast_points=0),
    sample_ed_visits=True,
    sample_hospital_admissions=True,
    sample_wastewater=True,
)

posterior_predictive_samples = posterior_predictive = (
    my_model.posterior_predictive(
        data=my_data.to_forecast_data(n_forecast_points=0),
        sample_ed_visits=True,
        sample_hospital_admissions=True,
        sample_wastewater=True,
    )
)

predictive_var_names = [
    "observed_ed_visits",
    "observed_hospital_admissions",
    "site_level_log_ww_conc",
]

idata = az.from_numpyro(
    prior=prior_predictive_samples,
    posterior_predictive=posterior_predictive_samples,
)

original_df = pl.from_pandas(
    idata.posterior_predictive[predictive_var_names].to_dataframe(),
    include_index=True,
)

original_df.unpivot(index=["chain", "draw", cs.contains("_dim_")])

original_df = pl.from_pandas(
    idata.sel(draw=slice(0, n_states - 1))
    .posterior_predictive[predictive_var_names]
    .to_dataframe(),
    include_index=True,
).insert_column(
    2,
    pl.col("draw")
    .replace_strict(np.arange(n_states), states_to_simulate)
    .alias("state"),
)


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


len(prior_predictive_samples["site_level_log_ww_conc"][0])
len(prior_predictive_samples["observed_ed_visits"][0])
