import argparse
import json
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from pyrenew.deterministic import DeterministicVariable

import pyrenew_covid_wastewater.plotting as plotting
from pyrenew_covid_wastewater.hosp_only_ww_model import hosp_only_ww_model

n_chains = 4
numpyro.set_host_device_count(n_chains)

# load priors
# have to run this from the right directory
from priors import (  # noqa: E402
    autoreg_p_hosp_rv,
    autoreg_rt_rv,
    eta_sd_rv,
    hosp_wday_effect_rv,
    i0_first_obs_n_rv,
    inf_feedback_strength_rv,
    initialization_rate_rv,
    log_r_mu_intercept_rv,
    p_hosp_mean_rv,
    p_hosp_w_sd_rv,
    phi_rv,
    uot,
)

parser = argparse.ArgumentParser(
    description="Fit the hospital-only wastewater model."
)
parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Path to the model directory containing the data.",
)
args = parser.parse_args()

model_dir = Path(args.model_dir)
data_path = model_dir / "data_for_model_fit.json"

with open(
    data_path,
    "r",
) as file:
    model_data = json.load(file)


inf_to_hosp_rv = DeterministicVariable(
    "inf_to_hosp", jnp.array(model_data["inf_to_hosp_pmf"])
)  # check if off by 1 or reversed

generation_interval_pmf_rv = DeterministicVariable(
    "generation_interval_pmf", jnp.array(model_data["generation_interval_pmf"])
)  # check if off by 1 or reversed

infection_feedback_pmf_rv = DeterministicVariable(
    "infection_feedback_pmf", jnp.array(model_data["generation_interval_pmf"])
)  # check if off by 1 or reversed

data_observed_hospital_admissions = jnp.array(
    model_data["data_observed_hospital_admissions"]
)
state_pop = jnp.array(model_data["state_pop"])
n_forecast_points = len(model_data["test_ed_admissions"])

my_model = hosp_only_ww_model(
    state_pop=state_pop,
    i0_first_obs_n_rv=i0_first_obs_n_rv,
    initialization_rate_rv=initialization_rate_rv,
    log_r_mu_intercept_rv=log_r_mu_intercept_rv,
    autoreg_rt_rv=autoreg_rt_rv,
    eta_sd_rv=eta_sd_rv,  # sd of random walk for ar process,
    generation_interval_pmf_rv=generation_interval_pmf_rv,
    infection_feedback_pmf_rv=infection_feedback_pmf_rv,
    infection_feedback_strength_rv=inf_feedback_strength_rv,
    p_hosp_mean_rv=p_hosp_mean_rv,
    p_hosp_w_sd_rv=p_hosp_w_sd_rv,
    autoreg_p_hosp_rv=autoreg_p_hosp_rv,
    hosp_wday_effect_rv=hosp_wday_effect_rv,
    phi_rv=phi_rv,
    inf_to_hosp_rv=inf_to_hosp_rv,
    n_initialization_points=uot,
    i0_t_offset=0,
)


my_model.run(
    num_warmup=500,
    num_samples=500,
    rng_key=jax.random.key(200),
    data_observed_hospital_admissions=data_observed_hospital_admissions,
    mcmc_args=dict(num_chains=n_chains, progress_bar=True),
    nuts_args=dict(find_heuristic_step_size=True),
)


posterior_predictive = my_model.posterior_predictive(
    n_datapoints=len(data_observed_hospital_admissions) + n_forecast_points
)

idata = az.from_numpyro(
    my_model.mcmc,
    posterior_predictive=posterior_predictive,
)

chain_ll = (
    idata["log_likelihood"]
    .mean(dim=["observed_hospital_admissions_dim_0", "draw"])[
        "observed_hospital_admissions"
    ]
    .values
)

chains_to_keep = np.arange(n_chains)[
    ((chain_ll - chain_ll.max()) / chain_ll.max()) < 2
]
# would like to not have to run this

idata = idata.sel(chain=chains_to_keep)


plotting.plot_predictive(idata)

idata.to_dataframe().to_csv(
    model_dir / "pyrenew_inference_data.csv", index=False
)
