"""Canonical review tails through loading, aggregation, and serialization."""

import datetime as dt
import json
from unittest.mock import Mock

import polars as pl
import pytest
from tests.test_forecast_pipeline import TestPipeline

from cfa.stf.routine.data import data_access

REPORT_DATE = dt.date(2026, 1, 21)


def _raw_nssp(dates, excluded=()):
    return pl.DataFrame(
        {
            "date": [date for date in dates for _ in range(2)],
            "state_abb": ["CA"] * (2 * len(dates)),
            "disease": ["covid", "total"] * len(dates),
            "value": [10, 100] * len(dates),
            "exclude": [date in excluded for date in dates for _ in range(2)],
        },
        schema_overrides={"date": pl.Date, "exclude": pl.Boolean},
    )


@pytest.mark.parametrize(
    ("days", "excluded", "expected"),
    [
        ([17, 18, 19, 20], [], 0),
        ([17, 18, 19, 20], [20], 1),
        ([17, 18, 19, 20], [18, 19, 20], 3),
        ([17, 18, 19, 20], [17, 19], 0),
        ([20, 17, 19, 18], [19, 20], 2),
        ([17, 19, 20], [19, 20], 3),
        ([17, 18], [18], 3),
        ([17, 18], [], 0),
        ([17, 18, 21, 22], [18], 3),
        ([17, 18, 21, 22], [21, 22], 0),
    ],
)
def test_tail_uses_unique_pre_report_dates_and_calendar_days(days, excluded, expected):
    dates = [dt.date(2026, 1, day) for day in days]
    exclusions = [dt.date(2026, 1, day) for day in excluded]
    assert (
        data_access.infer_nssp_exclusion_tail(_raw_nssp(dates, exclusions), REPORT_DATE)
        == expected
    )


@pytest.mark.parametrize("days", [[], [21, 22], [18, 19, 20, 21]])
def test_tail_rejects_no_retained_training_observations(days):
    dates = [dt.date(2026, 1, day) for day in days]
    with pytest.raises(ValueError, match="no retained training observations"):
        data_access.infer_nssp_exclusion_tail(_raw_nssp(dates, dates), REPORT_DATE)


def _patch_sources(monkeypatch, *, excluded, stale=False):
    dates = pl.date_range(dt.date(2026, 1, 4), dt.date(2026, 1, 24), eager=True)
    get_nssp = Mock(return_value=_raw_nssp(dates, excluded))
    monkeypatch.setattr(data_access, "get_nssp", get_nssp)
    monkeypatch.setattr(
        data_access,
        "resolve_nssp_report_date",
        lambda: REPORT_DATE - dt.timedelta(days=int(stale)),
    )
    monkeypatch.setattr(
        data_access, "select_latest_nhsn_release", lambda: (True, REPORT_DATE)
    )
    monkeypatch.setattr(
        data_access,
        "get_nhsn_hrd",
        lambda **kwargs: pl.DataFrame(
            {
                "date": [dt.date(2026, 1, day) for day in (10, 17, 24)],
                "state_abb": ["CA"] * 3,
                "value": [5, 6, 7],
            }
        ),
    )
    monkeypatch.setattr(
        data_access,
        "get_us_loc_pop_tbl",
        lambda: pl.DataFrame({"abbr": ["CA"], "population": [1000]}),
    )
    return get_nssp


class MixedPipeline(TestPipeline):
    @property
    def sources(self):
        return {"nssp", "nhsn"}


@pytest.mark.parametrize(("tail", "minimum"), [(0, 0), (1, 0), (2, 4), (6, 3)])
@pytest.mark.parametrize("resolution", ["daily", "epiweekly"])
def test_canonical_tail_preserves_source_windows_and_serialized_rows(
    monkeypatch,
    tmp_path,
    tail,
    minimum,
    resolution,
):
    excluded = [REPORT_DATE - dt.timedelta(days=i) for i in range(1, tail + 1)]
    get_nssp = _patch_sources(monkeypatch, excluded=excluded)
    pipeline = MixedPipeline(
        disease="covid",
        loc="CA",
        output_dir=tmp_path,
        run_date=REPORT_DATE,
        n_lookback_days=17,
        minimum_exclude_last_n_days=minimum,
        ed_visit_input_resolution=resolution,
    )
    run = pipeline.build_forecast_run()
    pipeline.prepare_input_artifacts(run)

    get_nssp.assert_called_once_with(
        disease=["covid", "total"],
        state_abb="CA",
        dataset="comprehensive",
        start_date=dt.date(2026, 1, 4),
        get_exclusions=True,
        lazy=False,
    )
    effective = max(tail, minimum)
    cutoff = REPORT_DATE - dt.timedelta(days=effective + 1)
    nssp_end = (
        cutoff
        if resolution == "daily"
        else cutoff - dt.timedelta(days=(cutoff.weekday() - 5) % 7)
    )
    nhsn_end = dt.date(2026, 1, 17 if minimum < 4 else 10)
    assert run.nssp.last_training_date == nssp_end
    assert run.nhsn.last_training_date == nhsn_end
    assert run.canonical_window.exclude_last_n_days == tail
    assert run.forecast_window.exclude_last_n_days == effective
    assert run.nhsn_window.exclude_last_n_days == minimum
    assert run.model_batch_dir.name == f"covid_lookback-17_omit-{tail}"
    assert run.forecast_through == dt.date(2026, 2, 14)
    last_training_date = max(nssp_end, nhsn_end)
    assert run.n_forecast_days == (run.forecast_through - last_training_date).days
    assert run.right_truncation_offset == (REPORT_DATE - last_training_date).days - 1

    serialized = json.loads((run.data_dir / "data_for_model_fit.json").read_text())
    assert max(serialized["nssp_training_data"]["date"]) == nssp_end.isoformat()
    assert (
        max(serialized["nhsn_training_data"]["weekendingdate"]) == nhsn_end.isoformat()
    )
    assert serialized["right_truncation_offset"] == run.right_truncation_offset
    combined = pl.read_csv(
        run.data_dir / "combined_data.tsv", separator="\t", try_parse_dates=True
    )
    for variable, end in [
        ("observed_ed_visits", nssp_end),
        ("observed_hospital_admissions", nhsn_end),
    ]:
        data = combined.filter(pl.col(".variable") == variable)
        assert data.filter(pl.col("data_type") == "train")["date"].max() == end
        expected_eval_end = (
            None
            if variable == "observed_ed_visits"
            and resolution == "epiweekly"
            and cutoff > dt.date(2026, 1, 17)
            else dt.date(2026, 1, 24)
        )
        assert (
            data.filter(pl.col("data_type") == "eval")["date"].max()
            == expected_eval_end
        )


@pytest.mark.parametrize("fail_on_stale_data", [False, True])
def test_stale_policy_is_preserved(monkeypatch, tmp_path, caplog, fail_on_stale_data):
    _patch_sources(monkeypatch, excluded=[dt.date(2026, 1, 20)], stale=True)
    pipeline = TestPipeline(
        disease="covid",
        loc="CA",
        output_dir=tmp_path,
        run_date=REPORT_DATE,
        fail_on_stale_data=fail_on_stale_data,
    )
    if fail_on_stale_data:
        with pytest.raises(RuntimeError, match="Stale input data"):
            pipeline.build_forecast_run()
    else:
        assert pipeline.build_forecast_run().canonical_window.exclude_last_n_days == 1
        assert "Stale input data" in caplog.text


@pytest.mark.parametrize(
    ("lookback", "minimum", "resolution"),
    [(3, 0, "daily"), (17, 16, "epiweekly"), (1, 0, "epiweekly")],
)
def test_no_training_window_is_rejected(
    monkeypatch, tmp_path, lookback, minimum, resolution
):
    _patch_sources(
        monkeypatch, excluded=[dt.date(2026, 1, day) for day in (18, 19, 20)]
    )
    pipeline = TestPipeline(
        disease="covid",
        loc="CA",
        output_dir=tmp_path,
        run_date=REPORT_DATE,
        n_lookback_days=lookback,
        minimum_exclude_last_n_days=minimum,
        ed_visit_input_resolution=resolution,
    )
    with pytest.raises(ValueError, match="no (retained )?training observations"):
        pipeline.build_forecast_run()
