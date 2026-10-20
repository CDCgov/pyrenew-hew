"""Validate the experiment's model inputs and calendar-based truncation oracle."""

import copy
import runpy
from pathlib import Path

import numpy as np
import pytest
from numpyro.handlers import seed, trace

from cfa.stf.routine.pyrenew_hew.observation_data import PyRenewObservationData
from cfa.stf.routine.pyrenew_hew.utils import build_pyrenew_hew_model_from_dir


@pytest.fixture(scope="module")
def experiment():
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "experiments/canonical_exclusions/run.py"
        )
    )


@pytest.mark.parametrize(
    "model",
    [
        "fable_daily",
        "fable_weekly",
        "pyrenew_h",
        "pyrenew_e",
        "pyrenew_he",
        "gp_daily",
        "gp_other",
        "gp_weekly",
        "gp_hubverse",
    ],
)
@pytest.mark.parametrize(
    "scenario",
    ["beginning", "middle", "tail_short", "tail_long", "isolated", "combined"],
)
def test_excluded_contamination_cannot_change_effective_inputs(
    experiment, tmp_path, model, scenario
):
    prepare, fingerprint = experiment["prepare"], experiment["fingerprint"]
    clean = prepare(tmp_path / "missing", model, scenario, "missing")
    expected = fingerprint(clean, model, "smoke")
    for condition in ["excluded20", "excluded100"]:
        contaminated = prepare(tmp_path / condition, model, scenario, condition)
        assert fingerprint(contaminated, model, "smoke") == expected
        assert contaminated.report_date == clean.report_date
        assert contaminated.forecast_through == clean.forecast_through


@pytest.mark.parametrize("scenario", ["tail_short", "tail_long", "combined"])
def test_pyrenew_reporting_cdf_uses_calendar_age(experiment, tmp_path, scenario):
    run = experiment["prepare"](tmp_path, "pyrenew_he", scenario, "missing")
    data = PyRenewObservationData.from_json(
        run.data_dir / "data_for_model_fit.json",
        fit_ed_visits=True,
        fit_hospital_admissions=True,
    )
    model = build_pyrenew_hew_model_from_dir(
        run.model_dir, fit_ed_visits=True, fit_hospital_admissions=True
    )
    kwargs = dict(sample_ed_visits=True, sample_hospital_admissions=True)
    observed = trace(seed(model.sample, rng_seed=1343)).get_trace(data=data, **kwargs)
    untruncated = copy.copy(data)
    untruncated.right_truncation_offset = None
    final = trace(seed(model.sample, rng_seed=1343)).get_trace(
        data=untruncated, **kwargs
    )
    ratios = np.asarray(
        observed["observed_ed_visits"]["fn"].mean
        / final["observed_ed_visits"]["fn"].mean
    )
    ages = (data.n_days_post_init - 1 - data.model_t_obs_ed_visits).astype(int)
    cdf = experiment["CDF"]
    np.testing.assert_allclose(ratios, cdf[np.minimum(ages, len(cdf) - 1)], rtol=1e-6)
    for variable in ["observed_ed_visits", "observed_hospital_admissions"]:
        site = observed[variable]
        assert np.isfinite(np.asarray(site["fn"].log_prob(site["value"]))).all()


def test_percentage_control_changes_the_ratio(experiment, tmp_path):
    import json

    prepare = experiment["prepare"]
    clean = prepare(tmp_path / "clean", "gp_weekly", "clean", "clean")
    contaminated = prepare(tmp_path / "outliers", "gp_weekly", "middle", "contaminated")
    clean_input = json.loads(
        (clean.model_dir / f"{clean.model_name}_input.json").read_text()
    )
    contaminated_input = json.loads(
        (contaminated.model_dir / f"{contaminated.model_name}_input.json").read_text()
    )
    assert max(contaminated_input["reports"]) > 5 * max(clean_input["reports"])
