"""
Python utilities for the NSSP ED visit forecasting
pipeline.
"""

import datetime
import os
from collections.abc import MutableSequence
from pathlib import Path


def ensure_listlike(x):
    """
    Ensure that an object either behaves like a
    :class:`MutableSequence` and if not return a
    one-item :class:`list` containing the object.

    Useful for handling list-of-strings inputs
    alongside single strings.

    Based on this _`StackOverflow approach
    <https://stackoverflow.com/a/66485952>`.

    Parameters
    ----------
    x
        The item to ensure is :class:`list`-like.

    Returns
    -------
    MutableSequence
        ``x`` if ``x`` is a :class:`MutableSequence`
        otherwise ``[x]`` (i.e. a one-item list containing
        ``x``.
    """
    return x if isinstance(x, MutableSequence) else [x]


def get_all_forecast_dirs(
    parent_dir: Path | str,
    diseases: str | list[str],
    report_date: str | datetime.date = None,
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
        Matching directories, if any, otherwise an empty
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
    elif isinstance(report_date, datetime.date):
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
