import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import jax.numpy as jnp
from jax.typing import ArrayLike


@dataclass
class PyrenewHEWParam:
    """
    Class for holding PyrenewHEW model parameters
    """

    population_size: int
    pop_fraction: ArrayLike
    generation_interval_pmf: ArrayLike
    inf_to_hosp_admit_lognormal_loc: float
    inf_to_hosp_admit_lognormal_scale: float
    inf_to_hosp_admit_pmf: ArrayLike
    right_truncation_pmf: ArrayLike

    @classmethod
    def from_json(cls, file_path: Path | str) -> Self:
        """
        Load parameters from a JSON file

        Parameters
        ----------
        file_path : Path | str
            Path to the JSON file containing parameters

        Returns
        -------
        PyrenewHEWParam
            An instance of PyrenewHEWParam
        """
        with open(file_path) as f:
            param_dict = json.load(f)

        return cls(
            population_size=param_dict["population_size"],
            pop_fraction=jnp.array(param_dict["pop_fraction"]),
            generation_interval_pmf=jnp.array(param_dict["generation_interval_pmf"]),
            right_truncation_pmf=jnp.array(param_dict["right_truncation_pmf"]),
            inf_to_hosp_admit_pmf=jnp.array(param_dict["inf_to_hosp_admit_pmf"]),
            inf_to_hosp_admit_lognormal_loc=param_dict[
                "inf_to_hosp_admit_lognormal_loc"
            ],
            inf_to_hosp_admit_lognormal_scale=param_dict[
                "inf_to_hosp_admit_lognormal_scale"
            ],
        )
