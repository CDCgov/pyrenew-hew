"""
Pyrenew-HEW utilities
"""

from itertools import chain, combinations
from typing import Iterable


def powerset(iterable: Iterable) -> Iterable:
    """
    Subsequences of the iterable from shortest to longest,
    considering only unique elements.

    Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    s = set(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def hew_models(with_null: bool = True) -> Iterable:
    """
    Return an iterable of the Pyrenew-HEW models
    as their lowercase letters.

    Parameters
    ----------
    with_null
       Include the null model ("pyrenew_null"), represented as
       the empty tuple `()`? Default ``True``.

    Returns
    -------
    Iterable
       An iterable yielding tuples of model letters.
    """
    result = powerset(("h", "e", "w"))
    if not with_null:
        result = filter(None, result)
    return result


def hew_letters_from_flags(
    fit_ed_visits: bool = False,
    fit_hospital_admissions: bool = False,
    fit_wastewater: bool = False,
) -> str:
    """
    Get the {h, e, w} letters defining
    a model from a set of flags indicating which
    of the datastreams, if any, were used in fitting.
    If none of them were, return the string "null"

    Parameters
    ----------
    fit_ed_visits
        ED visit data used in fitting?

    fit_hospital_admissions
        Hospital admissions data used in fitting?

    fit_wastewater
        Wastewater data used in fitting?

    Returns
    -------
    str
        The relevant HEW letters, or 'null',
    """
    result = (
        f"{'h' if fit_hospital_admissions else ''}"
        f"{'e' if fit_ed_visits else ''}"
        f"{'w' if fit_wastewater else ''}"
    )
    if not result:
        result = "null"
    return result


def flags_from_hew_letters(
    hew_letters: str, flag_prefix: str = "fit"
) -> dict[str, bool]:
    """
    Get a set of boolean flags indicating which
    datastreams, if any, correspond to the set of
    {h, e, w} letters (or 'null') in a model name.

    Inverse of :func:`hew_letters_from_flags` when used
    with the default ``flag_prefix``, ``"fit"``.

    Parameters
    ----------
    hew_letters
        The relevant HEW letters, or 'null'.

    flag_prefix
        Prefix for the names of the boolean flags in the
        output dictionary, which will be
        ``{flag_prefix}_hospital_admissions``,
        ``{flag_prefix}_ed_visits``, and
        ``{flag_prefix}_wastewater``. Default ``"fit"``,
        since {H,E,W} models are named according to the
        datastreams used in fitting.

    Returns
    -------
    dict[str, bool]
        Dictionary of boolean values with keys named
        ``{flag_prefix}_hospital_admissions``,
        ``{flag_prefix)_ed_visits``, and
        ``{flag_prefix}_wastewater`` (with the given
        ``flag_prefix``).

    Raises
    ------
    ValueError
        if input does is neither ``'null'`` nor a
        combination of the letters
        ``'h'``, ``'e'``, and ``'w'``.
    """
    valid_letters = {"h", "e", "w"}
    letterset = set(hew_letters)
    if not (
        hew_letters.lower() == "null" or letterset.issubset(valid_letters)
    ):
        raise ValueError(
            "Input failed validation. Must either be "
            "a string consisting only of the letters "
            f"in {valid_letters} or the string 'null'"
        )
    return {
        f"{flag_prefix}_hospital_admissions": "h" in hew_letters,
        f"{flag_prefix}_ed_visits": "e" in hew_letters,
        f"{flag_prefix}_wastewater": "w" in hew_letters,
    }


def pyrenew_model_name_from_flags(
    fit_ed_visits: bool = False,
    fit_hospital_admissions: bool = False,
    fit_wastewater: bool = False,
) -> str:
    """
    Get a "pyrenew_{h,e,w}" model name
    string from a set of flags indicating which
    of the datastreams, if any, were used in fitting.
    If none of them were, call the model "pyrenew_null".

    Parameters
    ----------
    fit_ed_visits
        ED visit data used in fitting?

    fit_hospital_admissions
        Hospital admissions data used in fitting?

    fit_wastewater
        Wastewater data used in fitting?

    Returns
    -------
    str
        The model name.
    """
    hew_letters = hew_letters_from_flags(
        fit_ed_visits=fit_ed_visits,
        fit_hospital_admissions=fit_hospital_admissions,
        fit_wastewater=fit_wastewater,
    )
    return f"pyrenew_{hew_letters}"


def flags_from_pyrenew_model_name(model_name: str) -> dict[str, bool]:
    """
    Get a set of boolean flags indicating which
    datastreams, if any, were used in fitting
    from a "pyrenew_{h,e,w}" model name.

    Parameters
    ----------
    model_name
        The model name.

    Returns
    -------
    dict[str, bool]
       Dictionary with boolean entries named
       ``fit_hospital_admissions``, ``fit_ed_visits``,
       and ``fit_wastewater``.

    Raises
    ------
    ValueError
        If the input is not a valid pyrenew_{h, e, w} model name.

    """
    if not model_name.startswith("pyrenew_"):
        raise ValueError(
            "Expected a model_name beginning with "
            f"'pyrenew_'. Got {model_name}."
        )
    hew_letters = model_name.removeprefix("pyrenew_")
    return flags_from_hew_letters(hew_letters, flag_prefix="fit")
