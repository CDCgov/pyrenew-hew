"""
Python utilities for the NSSP ED visit forecasting
pipeline.
"""

import datetime as dt
import os
import re
import runpy
from pathlib import Path

from forecasttools import ensure_listlike, location_table

from pyrenew_hew.pyrenew_hew_param import PyrenewHEWParam
from pyrenew_hew.utils import build_pyrenew_hew_model

disease_map_lower_ = {
    "influenza": "Influenza",
    "covid-19": "COVID-19",
    "rsv": "RSV",
}
loc_abbrs_ = location_table["short_name"].to_list()


def parse_model_batch_dir_name(model_batch_dir_name: str) -> dict:
    """
    Parse the name of a model batch directory,
    returning a dictionary of parsed values.

    Parameters
    ----------
    model_batch_dir_name
       Model batch directory name to parse.

    Returns
    -------
    dict
       A dictionary with keys 'disease', 'report_date',
       'first_training_date', and 'last_training_date'.
    """
    regex_match = re.match(r"(.+)_r_(.+)_f_(.+)_t_(.+)", model_batch_dir_name)
    if regex_match:
        disease, report_date, first_training_date, last_training_date = (
            regex_match.groups()
        )
    else:
        raise ValueError(
            f"Invalid model batch directory name format: {model_batch_dir_name}"
        )
    return dict(
        disease=disease_map_lower_[disease],
        report_date=dt.datetime.strptime(report_date, "%Y-%m-%d").date(),
        first_training_date=dt.datetime.strptime(
            first_training_date, "%Y-%m-%d"
        ).date(),
        last_training_date=dt.datetime.strptime(last_training_date, "%Y-%m-%d").date(),
    )


def get_all_forecast_dirs(
    parent_dir: Path | str,
    diseases: str | list[str],
    report_date: str | dt.date = None,
) -> list[str]:
    """
    Get all the subdirectories within a parent directory
    that match the pattern for a forecast run for a
    given disease and optionally a given report date.

    Parameters
    ----------
    parent_dir
       Directory in which to look for forecast subdirectories.

    diseases
       Name of the diseases to match, as a list of strings,
       or a single disease as a string.

    Returns
    -------
    list[str]
        Names of matching directories, if any, otherwise an empty
        list.

    Raises
    ------
    ValueError
        Given an invalid ``report_date``.
    """
    diseases = ensure_listlike(diseases)

    if report_date is None:
        report_date_str = ""
    elif isinstance(report_date, str):
        report_date_str = report_date
    elif isinstance(report_date, dt.date):
        report_date_str = f"{report_date:%Y-%m-%d}"
    else:
        raise ValueError(
            "report_date must be one of None, "
            "a string in the format YYYY-MM-DD "
            "or a datetime.date instance. "
            f"Got {type(report_date)}."
        )
    valid_starts = tuple(
        [f"{disease.lower()}_r_{report_date_str}" for disease in diseases]
    )
    # by convention, disease names are
    # lowercase in directory patterns

    return [
        f.name
        for f in os.scandir(parent_dir)
        if f.is_dir() and f.name.startswith(valid_starts)
    ]


def get_all_model_run_dirs(parent_dir: Path) -> list[str]:
    """
    Get all the subdirectories within a parent directory
    that are valid model run directories (by convention,
    named with the two-letter code of a forecast location).

    Parameters
    ----------
    parent_dir
       Directory in which to look for model run subdirectories.

    Returns
    -------
    list[str]
        Names of matching directories, if any, otherwise an empty
        list.
    """

    return [
        f.name for f in os.scandir(parent_dir) if f.is_dir() and f.name in loc_abbrs_
    ]


def build_pyrenew_hew_model_from_dir(
    model_run_dir: Path | str = None,
    prior_path: Path | str = None,
    model_params_path: Path | str = None,
    fit_ed_visits: bool = False,
    fit_hospital_admissions: bool = False,
    fit_wastewater: bool = False,
):
    if model_run_dir is not None:
        prior_path = Path(model_run_dir) / "priors.py"
        model_params_path = Path(model_run_dir) / "model_params.json"
    else:
        if prior_path is None or model_params_path is None:
            raise ValueError(
                "Either model_run_dir must be provided, "
                "or both prior_path and model_params_path "
                "must be provided."
            )
        prior_path = Path(prior_path)
        model_params_path = Path(model_params_path)

    priors = runpy.run_path(str(prior_path))
    model_params = PyrenewHEWParam.from_json(model_params_path)
    my_model = build_pyrenew_hew_model(
        priors,
        model_params,
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )
    return my_model
