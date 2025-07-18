import argparse
import pickle
from pathlib import Path

import jax
import numpy as np

from pipelines.utils import get_model_data_and_priors_from_dir
from pyrenew_hew.utils import build_pyrenew_hew_model


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

    (model_data, priors) = get_model_data_and_priors_from_dir(model_run_dir)
    (my_model, my_data) = build_pyrenew_hew_model(
        model_data,
        priors,
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
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
