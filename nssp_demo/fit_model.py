import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
import numpyro
from build_model import build_model_from_dir


def fit_and_save_model(
    model_dir,
    num_warmup=1000,
    num_samples=1000,
    n_chains=4,
    rng_key=jax.random.key(np.random.randint(0, 10000)),
) -> None:
    (
        my_model,
        data_observed_disease_hospital_admissions,
        right_truncation_offset,
    ) = build_model_from_dir(model_dir)
    my_model.run(
        num_warmup=num_warmup,
        num_samples=num_samples,
        rng_key=rng_key,
        data_observed_disease_hospital_admissions=(
            data_observed_disease_hospital_admissions
        ),
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


if __name__ == "__main__":
    n_chains = 4
    numpyro.set_host_device_count(n_chains)

    parser = argparse.ArgumentParser(
        description="Fit the hospital-only wastewater model."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Path to the model directory containing the data.",
    )
    args = parser.parse_args()

    fit_and_save_model(**vars(args), n_chains=n_chains)
