import argparse
import pickle
from pathlib import Path

import arviz as az
import numpyro

n_chains = 4
numpyro.set_host_device_count(n_chains)
from build_model import build_model_from_dir  # noqa: E402

parser = argparse.ArgumentParser(
    description="Fit the hospital-only wastewater model."
)
parser.add_argument(
    "--model_dir",
    type=str,
    required=True,
    help="Path to the model directory containing the data.",
)
parser.add_argument(
    "--n_forecast_points",
    type=int,
    default=0,
    help="Number of time points to forecast",
)
args = parser.parse_args()
model_dir = Path(args.model_dir)
n_forecast_points = args.n_forecast_points
my_model, data_observed_hospital_admissions, right_truncation_offset = (
    build_model_from_dir(model_dir)
)

my_model._init_model(1, 1)
fresh_sampler = my_model.mcmc.sampler

with open(
    model_dir / "posterior_samples.pickle",
    "rb",
) as file:
    my_model.mcmc = pickle.load(file)

my_model.mcmc.sampler = fresh_sampler

# prior_predictive = my_model.prior_predictive(
#     numpyro_predictive_args={
#         "num_samples": my_model.mcmc.num_samples * my_model.mcmc.num_chains,
#         "batch_ndims":1
#     },
#     n_datapoints=len(data_observed_hospital_admissions) + n_forecast_points,
# )
# need to figure out a way to generate these as distinct chains, so that the result of the to_datarame method is more compact

posterior_predictive = my_model.posterior_predictive(
    n_datapoints=len(data_observed_hospital_admissions) + n_forecast_points
)

idata = az.from_numpyro(
    my_model.mcmc,
    # prior=prior_predictive,
    posterior_predictive=posterior_predictive,
)

idata.to_dataframe().to_csv(model_dir / "inference_data.csv", index=False)

# Save one netcdf for reloading
idata.to_netcdf(model_dir / "inference_data.nc")

# R cannot read netcdf files with groups, so we split them into separate files.
for group in idata._groups_all:
    idata[group].to_netcdf(model_dir / f"inference_data_{group}.nc")
