import datetime
import json
import runpy
from pathlib import Path

import jax.numpy as jnp
import numpyro.distributions as dist
from pyrenew.deterministic import DeterministicVariable
from pyrenew.randomvariable import DistributionalVariable

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
) -> tuple[PyrenewHEWModel, PyrenewHEWData]:
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

    data_observed_disease_ed_visits = jnp.array(
        model_data["data_observed_disease_ed_visits"]
    )
    data_observed_disease_hospital_admissions = jnp.array(
        model_data["data_observed_disease_hospital_admissions"]
    )
    population_size = jnp.array(model_data["state_pop"])

    ed_right_truncation_pmf_rv = DeterministicVariable(
        "right_truncation_pmf", jnp.array(model_data["right_truncation_pmf"])
    )

    uot = (
        max(
            len(model_data["generation_interval_pmf"]),
            len(model_data["inf_to_ed_pmf"]),
        )
        - 1
    )

    first_ed_visits_date = datetime.datetime.strptime(
        model_data["nssp_training_dates"][0], "%Y-%m-%d"
    )
    first_hospital_admissions_date = datetime.datetime.strptime(
        model_data["nhsn_training_dates"][0], "%Y-%m-%d"
    )

    priors = runpy.run_path(str(prior_path))

    right_truncation_offset = model_data["right_truncation_offset"]

    my_latent_infection_model = LatentInfectionProcess(
        i0_first_obs_n_rv=priors["i0_first_obs_n_rv"],
        initialization_rate_rv=priors["initialization_rate_rv"],
        log_r_mu_intercept_rv=priors["log_r_mu_intercept_rv"],
        autoreg_rt_rv=priors["autoreg_rt_rv"],
        eta_sd_rv=priors["eta_sd_rv"],  # sd of random walk for ar process,
        generation_interval_pmf_rv=generation_interval_pmf_rv,
        infection_feedback_strength_rv=priors["inf_feedback_strength_rv"],
        infection_feedback_pmf_rv=infection_feedback_pmf_rv,
        n_initialization_points=uot,
    )

    my_ed_visit_obs_model = EDVisitObservationProcess(
        p_ed_mean_rv=priors["p_ed_visit_mean_rv"],
        p_ed_w_sd_rv=priors["p_ed_visit_w_sd_rv"],
        autoreg_p_ed_rv=priors["autoreg_p_ed_visit_rv"],
        ed_wday_effect_rv=priors["ed_visit_wday_effect_rv"],
        inf_to_ed_rv=inf_to_ed_rv,
        ed_neg_bin_concentration_rv=(priors["ed_neg_bin_concentration_rv"]),
        ed_right_truncation_pmf_rv=ed_right_truncation_pmf_rv,
    )

    my_hosp_admit_obs_model = HospAdmitObservationProcess(
        inf_to_hosp_admit_rv=inf_to_hosp_admit_rv,
        hosp_admit_neg_bin_concentration_rv=(
            priors["hosp_admit_neg_bin_concentration_rv"]
        ),
        ihr_rel_iedr_rv=None,  # since for now we only use H or E, not HE
        ihr_rv=priors["ihr_rv"],
    )

    # placeholder
    my_wastewater_obs_model = WastewaterObservationProcess(
        t_peak_rv=DistributionalVariable(
            "t_peak", dist.TruncatedNormal(5, 1, low=0)
        ),
        dur_shed_after_peak_rv=DistributionalVariable(
            "dur_shed_after_peak", dist.TruncatedNormal(12, 3, low=0)
        ),
        log10_genome_per_inf_ind_rv=DeterministicVariable(
            "log10_genome_per_inf_ind", dist.Normal(12, 2)
        ),
        mode_sigma_ww_site_rv=DistributionalVariable(
            "mode_sigma_ww_site",
            dist.TruncatedNormal(1, 1, low=0),
        ),
        sd_log_sigma_ww_site_rv=DistributionalVariable(
            "sd_log_sigma_ww_site", dist.TruncatedNormal(0, 0.693, low=0)
        ),
        mode_sd_ww_site_rv=DistributionalVariable(
            "mode_sd_ww_site", dist.TruncatedNormal(0, 0.25, low=0)
        ),
        ww_ml_produced_per_day=None,
        ww_uncensored=None,
        ww_censored=None,
        ww_sampled_lab_sites=None,
        ww_sampled_subpops=None,
        ww_sampled_times=None,
        ww_log_lod=None,
        lab_site_to_subpop_map=None,
        max_ww_sampled_days=None,
        n_ww_lab_sites=None,
        max_shed_interval=None,
    )

    my_model = PyrenewHEWModel(
        population_size=population_size,
        latent_infection_process_rv=my_latent_infection_model,
        ed_visit_obs_process_rv=my_ed_visit_obs_model,
        hosp_admit_obs_process_rv=my_hosp_admit_obs_model,
        wastewater_obs_process_rv=my_wastewater_obs_model,
    )

    my_data = PyrenewHEWData(
        data_observed_disease_ed_visits=data_observed_disease_ed_visits,
        data_observed_disease_hospital_admissions=(
            data_observed_disease_hospital_admissions
        ),
        data_observed_disease_wastewater=None,  # placeholder
        right_truncation_offset=right_truncation_offset,
        first_ed_visits_date=first_ed_visits_date,
        first_hospital_admissions_date=first_hospital_admissions_date,
    )

    return (my_model, my_data)
