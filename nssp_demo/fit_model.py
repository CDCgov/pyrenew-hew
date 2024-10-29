import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
from build_model import build_model_from_dir
from jax.typing import ArrayLike


def fit_and_save_model(
    model_run_dir: str,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    n_chains: int = 4,
    rng_key: int | ArrayLike = None,
) -> None:
    if rng_key is None:
        rng_key = np.random.randint(0, 10000)
    if not isinstance(rng_key, ArrayLike):
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        else:
            raise ValueError(
                "rng_key must be key array "
                "created by :func:`jax.random.key``"
                "object or an integer to use when "
                "seeding :func:`jax.random.key`."
            )
    (
        my_model,
        data_observed_disease_hospital_admissions,
        right_truncation_offset,
    ) = build_model_from_dir(model_run_dir)
    my_model.run(
        num_warmup=n_warmup,
        num_samples=n_samples,
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
        model_run_dir / "posterior_samples.pickle",
        "wb",
    ) as file:
        pickle.dump(my_model.mcmc, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit the hospital-only wastewater model."
    )
    parser.add_argument(
        "model-run-dir",
        type=Path,
        help=(
            "Path to a directory containing model fitting data. "
            "The completed fit will be saved here."
        ),
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=1000,
        help=(
            "Number of warmup iterations for the No-U-Turn sampler "
            "(Default: 1000)."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help=(
            "Number of sampling iterations after warmup "
            "for the No-U-Turn sampler "
            "(Default: 1000)."
        ),
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=1000,
        help=("Number of duplicate MCMC chains to run " "(Default 4)."),
    )
    parser.add_argument(
        "--rng-key",
        type=int,
        default=None,
        help=(
            "Integer with which to seed the pseudorandom"
            "number generator. If none is specified, a "
            "pseudorandom seed will be drawn via "
            "np.random.randint"
        ),
    )

    args = parser.parse_args()

    fit_and_save_model(**vars(args))
