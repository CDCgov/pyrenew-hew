import argparse
import pickle
from pathlib import Path

import jax
import numpy as np
from build_pyrenew_model import (
    build_model_from_dir,
)


def fit_and_save_model(
    model_run_dir: str,
    model_name: str,
    fit_ed_visits: bool = False,
    fit_hospital_admissions: bool = False,
    fit_wastewater: bool = False,
    n_warmup: int = 1000,
    n_samples: int = 1000,
    n_chains: int = 4,
    rng_key: int = None,
) -> None:
    if rng_key is None:
        rng_key = np.random.randint(0, 10000)
    if isinstance(rng_key, int):
        rng_key = jax.random.key(rng_key)
    else:
        raise ValueError(
            "rng_key must be an integer with which "
            "to seed :func:`jax.random.key`"
        )
    (my_model, my_data) = build_model_from_dir(
        model_run_dir,
        sample_ed_visits=fit_ed_visits,
        sample_hospital_admissions=fit_hospital_admissions,
        sample_wastewater=fit_wastewater,
    )
    my_model.run(
        data=my_data,
        sample_ed_visits=fit_ed_visits,
        sample_hospital_admissions=fit_hospital_admissions,
        sample_wastewater=fit_wastewater,
        num_warmup=n_warmup,
        num_samples=n_samples,
        rng_key=rng_key,
        mcmc_args=dict(num_chains=n_chains, progress_bar=True),
        nuts_args=dict(find_heuristic_step_size=True),
    )

    my_model.mcmc.sampler = None
    model_dir = Path(model_run_dir, model_name)
    model_dir.mkdir(exist_ok=True)
    with open(
        model_dir / "posterior_samples.pickle",
        "wb",
    ) as file:
        pickle.dump(my_model.mcmc, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit the hospital-only wastewater model."
    )
    parser.add_argument(
        "model_run_dir",
        type=Path,
        help=(
            "Path to a directory containing model fitting data. "
            "The completed fit will be saved here."
        ),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use for generating predictions.",
    )

    parser.add_argument(
        "--fit-ed-visits",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="If provided, fit to ED visit data.",
    )
    parser.add_argument(
        "--fit-hospital-admissions",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help=("If provided, fit to hospital admissions data."),
    )
    parser.add_argument(
        "--fit-wastewater",
        type=bool,
        action=argparse.BooleanOptionalAction,
        help="If provided, fit to wastewater data.",
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
        default=4,
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
