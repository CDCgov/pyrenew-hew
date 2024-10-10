import argparse
import pickle
from pathlib import Path

import jax
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
args = parser.parse_args()
model_dir = Path(args.model_dir)

my_model, data_observed_hospital_admissions, right_truncation_offset = (
    build_model_from_dir(model_dir)
)
my_model.run(
    num_warmup=500,
    num_samples=500,
    rng_key=jax.random.key(200),
    data_observed_hospital_admissions=data_observed_hospital_admissions,
    right_truncation_offset=right_truncation_offset,
    mcmc_args=dict(num_chains=n_chains, progress_bar=True),
    nuts_args=dict(find_heuristic_step_size=True),
)

my_model.mcmc.sampler = None

with open(
    model_dir / "posterior_samples.pickle",
    "wb",
) as file:
    pickle.dump(my_model.mcmc, file)
