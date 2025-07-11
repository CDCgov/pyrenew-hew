"""
Pyrenew-HEW utilities
"""

from itertools import chain, combinations
from typing import Iterable

import jax.numpy as jnp
import polars as pl
from pyrenew.deterministic import DeterministicPMF

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData
from pyrenew_hew.pyrenew_hew_model import (
    EDVisitObservationProcess,
    HospAdmitObservationProcess,
    LatentInfectionProcess,
    OffsetDiscretizedLognormalPMF,
    PyrenewHEWModel,
    WastewaterObservationProcess,
)


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


def validate_hew_letters(letters: str) -> None:
    """
    Confirm that a string of letters defines a
    valid Pyrenew-HEW family model, raising
    an error if not.

    Valid letters are combinations of
    'h', 'e', and 'w' (repeats permitted and ignored)
    or the string 'null'.

    Parameters
    ----------
    letters
        The string of letters to validate.

    Returns
    -------
    None
        If validation passes.

    Raises
    ------
    ValueError
        if input is neither ``'null'`` nor a
        combination of the letters
        ``'h'``, ``'e'``, and ``'w'``.
    """
    valid_letters = {"h", "e", "w"}
    letterset = set(letters)
    if not (letters.lower() == "null" or letterset.issubset(valid_letters)):
        raise ValueError(
            f"The letters in '{letters}' do not "
            "define a valid Pyrenew-HEW-family model. "
            "Expected either a string consisting "
            "only of combinations of the letters "
            f"{valid_letters} or the string 'null'"
        )
    return None


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
    validate_hew_letters(hew_letters)
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


def build_pyrenew_model(
    model_data: dict,
    priors: dict,
    fit_ed_visits: bool = False,
    fit_hospital_admissions: bool = False,
    fit_wastewater: bool = False,
) -> tuple[PyrenewHEWModel, PyrenewHEWData]:
    """
    Build a pyrenew-family model from dictionaries specifying model_data and priors

    Parameters
    ----------
    model_data : dict
        Dictionary containing the data to fit the model.

    priors : dict
        Dictionary containing the priors for the model.

    fit_ed_visits
        Fit ED visit data in the built model? Default ``False``.

    fit_ed_visits
        Fit hospital admissions data in the built model?
        Default ``False``.

    fit_wastewater
        Fit wastewater pathogen genome concentration data
        in the built model? Default ``False``.

    Returns
    -------
    tuple[PyrenewHEWModel, PyrenewHEWData]
        Instantiated model and data objects representing
        the model and its fitting data, respectively.
    """
    he = fit_hospital_admissions and fit_ed_visits

    inf_to_ed_rv = DeterministicPMF(
        "inf_to_ed", jnp.array(model_data["inf_to_hosp_admit_pmf"])
    )
    # For now follow NNH in just substituting
    # (eventually will use a different inferred fixed).
    if he:
        # offset from approx inf to ed distribution
        # when fitting admissions
        inf_to_hosp_admit_rv = OffsetDiscretizedLognormalPMF(
            "inf_to_hosp_admit",
            reference_loc=model_data["inf_to_hosp_admit_lognormal_loc"],
            reference_scale=model_data["inf_to_hosp_admit_lognormal_scale"],
            n=jnp.array(model_data["inf_to_hosp_admit_pmf"]).size * 2,
            # Flexibility to infer delays with a longer tail, up to a point.
            offset_loc_rv=priors["delay_offset_loc_rv"],
            log_offset_scale_rv=priors["delay_log_offset_scale_rv"],
        )
    else:
        inf_to_hosp_admit_rv = DeterministicPMF(
            "inf_to_hosp_admit", jnp.array(model_data["inf_to_hosp_admit_pmf"])
        )  # else use same, following NNH

    generation_interval_pmf_rv = DeterministicPMF(
        "generation_interval_pmf",
        jnp.array(model_data["generation_interval_pmf"]),
    )  # check if off by 1 or reversed

    infection_feedback_pmf_rv = DeterministicPMF(
        "infection_feedback_pmf",
        jnp.array(model_data["generation_interval_pmf"]),
    )  # check if off by 1 or reversed

    nssp_training_data = (
        pl.DataFrame(
            model_data["nssp_training_data"],
            schema={
                "date": pl.Date,
                "geo_value": pl.String,
                "observed_ed_visits": pl.Float64,
                "other_ed_visits": pl.Float64,
                "data_type": pl.String,
            },
        )
        if fit_ed_visits
        else None
    )
    nhsn_training_data = (
        pl.DataFrame(
            model_data["nhsn_training_data"],
            schema={
                "weekendingdate": pl.Date,
                "jurisdiction": pl.String,
                "hospital_admissions": pl.Float64,
                "data_type": pl.String,
            },
        )
        if fit_hospital_admissions
        else None
    )

    nwss_training_data = (
        pl.DataFrame(
            model_data["nwss_training_data"],
            schema_overrides={
                "date": pl.Date,
                "lab_index": pl.Int64,
                "site_index": pl.Int64,
            },
        )
        if fit_wastewater
        else None
    )

    population_size = jnp.array(model_data["loc_pop"]).item()

    pop_fraction = jnp.array(model_data["pop_fraction"])

    ed_right_truncation_pmf_rv = DeterministicPMF(
        "right_truncation_pmf", jnp.array(model_data["right_truncation_pmf"])
    )

    n_initialization_points = (
        max(
            generation_interval_pmf_rv.size(),
            infection_feedback_pmf_rv.size(),
            inf_to_ed_rv.size() if fit_ed_visits else 1,
            inf_to_hosp_admit_rv.size() + 6 if fit_hospital_admissions else 1,
            priors["max_shed_interval"] if fit_wastewater else 1,
        )
        - 1
    )

    right_truncation_offset = model_data["right_truncation_offset"]

    latent_infections_rv = LatentInfectionProcess(
        i0_first_obs_n_rv=priors["i0_first_obs_n_rv"],
        log_r_mu_intercept_rv=priors["log_r_mu_intercept_rv"],
        autoreg_rt_rv=priors["autoreg_rt_rv"],
        eta_sd_rv=priors["eta_sd_rv"],  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_strength_rv=priors["inf_feedback_strength_rv"],
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        n_initialization_points=n_initialization_points,
        pop_fraction=pop_fraction,
        autoreg_rt_subpop_rv=priors["autoreg_rt_subpop_rv"],
        sigma_rt_rv=priors["sigma_rt_rv"],
        sigma_i_first_obs_rv=priors["sigma_i_first_obs_rv"],
        offset_ref_logit_i_first_obs_rv=priors[
            "offset_ref_logit_i_first_obs_rv"
        ],
        offset_ref_log_rt_rv=priors["offset_ref_log_rt_rv"],
    )

    ed_visit_obs_rv = EDVisitObservationProcess(
        p_ed_mean_rv=priors["p_ed_visit_mean_rv"],
        p_ed_w_sd_rv=priors["p_ed_visit_w_sd_rv"],
        autoreg_p_ed_rv=priors["autoreg_p_ed_visit_rv"],
        ed_wday_effect_rv=priors["ed_visit_wday_effect_rv"],
        inf_to_ed_rv=inf_to_ed_rv,
        ed_neg_bin_concentration_rv=(priors["ed_neg_bin_concentration_rv"]),
        ed_right_truncation_pmf_rv=ed_right_truncation_pmf_rv,
    )

    hosp_admit_obs_rv = HospAdmitObservationProcess(
        inf_to_hosp_admit_rv=inf_to_hosp_admit_rv,
        hosp_admit_neg_bin_concentration_rv=(
            priors["hosp_admit_neg_bin_concentration_rv"]
        ),
        ihr_rel_iedr_rv=priors["ihr_rel_iedr_rv"] if he else None,
        ihr_rv=None if he else priors["ihr_rv"],
    )

    wastewater_obs_rv = WastewaterObservationProcess(
        t_peak_rv=priors["t_peak_rv"],
        duration_shed_after_peak_rv=priors["duration_shed_after_peak_rv"],
        log10_genome_per_inf_ind_rv=priors["log10_genome_per_inf_ind_rv"],
        mode_sigma_ww_site_rv=priors["mode_sigma_ww_site_rv"],
        sd_log_sigma_ww_site_rv=priors["sd_log_sigma_ww_site_rv"],
        mode_sd_ww_site_rv=priors["mode_sd_ww_site_rv"],
        max_shed_interval=priors["max_shed_interval"],
        ww_ml_produced_per_day=priors["ww_ml_produced_per_day"],
    )

    mod = PyrenewHEWModel(
        population_size=population_size,
        latent_infection_process_rv=latent_infections_rv,
        ed_visit_obs_process_rv=ed_visit_obs_rv,
        hosp_admit_obs_process_rv=hosp_admit_obs_rv,
        wastewater_obs_process_rv=wastewater_obs_rv,
    )

    dat = PyrenewHEWData(
        nssp_training_data=nssp_training_data,
        nhsn_training_data=nhsn_training_data,
        nwss_training_data=nwss_training_data,
        right_truncation_offset=right_truncation_offset,
        pop_fraction=pop_fraction,
        population_size=population_size,
    )

    return (mod, dat)
