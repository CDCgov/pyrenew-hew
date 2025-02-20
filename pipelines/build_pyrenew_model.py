import datetime
import json
import runpy
from pathlib import Path

import jax.numpy as jnp
import polars as pl
from pyrenew.deterministic import DeterministicVariable

from pyrenew_hew.pyrenew_hew_data import PyrenewHEWData
from pyrenew_hew.pyrenew_hew_model import (
    EDVisitObservationProcess,
    HospAdmitObservationProcess,
    LatentInfectionProcess,
    PyrenewHEWModel,
    WastewaterObservationProcess,
)


def build_model_from_dir(
    model_dir: Path,
    fit_ed_visits: bool = False,
    fit_hospital_admissions: bool = False,
    fit_wastewater: bool = False,
) -> tuple[PyrenewHEWModel, PyrenewHEWData]:
    """
    Build a pyrenew-family model from a model run directory
    containing data (as a .json file) and priors (as a .py file)

    Parameters
    ----------
    model_dir
        The model directory, containing a priors file and a
        data subdirectory.

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
    data_path = Path(model_dir) / "data" / "data_for_model_fit.json"
    prior_path = Path(model_dir) / "priors.py"

    with open(
        data_path,
        "r",
    ) as file:
        model_data = json.load(file)

    inf_to_ed_rv = DeterministicVariable(
        "inf_to_ed", jnp.array(model_data["inf_to_ed_pmf"])
    )  # check if off by 1 or reversed

    # use same as inf to ed, per NNH guidelines
    inf_to_hosp_admit_rv = DeterministicVariable(
        "inf_to_hosp_admit", jnp.array(model_data["inf_to_ed_pmf"])
    )  # check if off by 1 or reversed

    generation_interval_pmf_rv = DeterministicVariable(
        "generation_interval_pmf",
        jnp.array(model_data["generation_interval_pmf"]),
    )  # check if off by 1 or reversed

    infection_feedback_pmf_rv = DeterministicVariable(
        "infection_feedback_pmf",
        jnp.array(model_data["generation_interval_pmf"]),
    )  # check if off by 1 or reversed

    data_observed_disease_ed_visits = (
        jnp.array(model_data["data_observed_disease_ed_visits"])
        if fit_ed_visits
        else None
    )
    data_observed_disease_hospital_admissions = (
        jnp.array(model_data["data_observed_disease_hospital_admissions"])
        if fit_hospital_admissions
        else None
    )

    data_observed_disease_wastewater = (
        pl.DataFrame(
            model_data["data_observed_disease_wastewater"],
            schema_overrides={"date": pl.Date},
        )
        if fit_wastewater
        else None
    )

    population_size = jnp.array(model_data["state_pop"])

    ed_right_truncation_pmf_rv = DeterministicVariable(
        "right_truncation_pmf", jnp.array(model_data["right_truncation_pmf"])
    )

    n_initialization_points = (
        max(
            len(model_data["generation_interval_pmf"]),
            len(model_data["inf_to_ed_pmf"]),
        )
        - 1
    )

    first_ed_visits_date = datetime.datetime.strptime(
        model_data["nssp_training_dates"][0], "%Y-%m-%d"
    ).date()
    first_hospital_admissions_date = datetime.datetime.strptime(
        model_data["nhsn_training_dates"][0], "%Y-%m-%d"
    ).date()

    priors = runpy.run_path(str(prior_path))

    right_truncation_offset = model_data["right_truncation_offset"]

    dat = PyrenewHEWData(
        data_observed_disease_ed_visits=data_observed_disease_ed_visits,
        data_observed_disease_hospital_admissions=(
            data_observed_disease_hospital_admissions
        ),
        data_observed_disease_wastewater=data_observed_disease_wastewater,
        right_truncation_offset=right_truncation_offset,
        first_ed_visits_date=first_ed_visits_date,
        first_hospital_admissions_date=first_hospital_admissions_date,
        population_size=population_size,
    )

    latent_infections_rv = LatentInfectionProcess(
        i0_first_obs_n_rv=priors["i0_first_obs_n_rv"],
        initialization_rate_rv=priors["initialization_rate_rv"],
        log_r_mu_intercept_rv=priors["log_r_mu_intercept_rv"],
        autoreg_rt_rv=priors["autoreg_rt_rv"],
        eta_sd_rv=priors["eta_sd_rv"],  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_strength_rv=priors["inf_feedback_strength_rv"],
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        n_initialization_points=n_initialization_points,
        pop_fraction=dat.pop_fraction if fit_wastewater else jnp.array([1]),
        autoreg_rt_subpop_rv=priors["autoreg_rt_subpop_rv"],
        sigma_rt_rv=priors["sigma_rt_rv"],
        sigma_i_first_obs_rv=priors["sigma_i_first_obs_rv"],
        sigma_initial_exp_growth_rate_rv=priors[
            "sigma_initial_exp_growth_rate_rv"
        ],
        offset_ref_logit_i_first_obs_rv=priors[
            "offset_ref_logit_i_first_obs_rv"
        ],
        offset_ref_initial_exp_growth_rate_rv=priors[
            "offset_ref_initial_exp_growth_rate_rv"
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

    eh = fit_hospital_admissions and fit_ed_visits

    hosp_admit_obs_rv = HospAdmitObservationProcess(
        inf_to_hosp_admit_rv=inf_to_hosp_admit_rv,
        hosp_admit_neg_bin_concentration_rv=(
            priors["hosp_admit_neg_bin_concentration_rv"]
        ),
        ihr_rel_iedr_rv=priors["ihr_rel_iedr_rv"] if eh else None,
        ihr_rv=None if eh else priors["ihr_rv"],
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

    return (mod, dat)
