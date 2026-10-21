from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import dagster as dg
import pytest

from cfa.stf.routine.dagster.defs import asset_config, asset_helpers, assets


def _model_base_config() -> asset_config.ModelBaseConfig:
    config = asset_config.ModelBaseConfig(
        output_basedir="default-output",
        fable_pyrenew_n_lookback_days=100,
        epiautogp_n_lookback_days=200,
        fail_on_stale_data=True,
        config_overrides=[
            asset_config.ConfigOverride(
                location="CA",
                output_basedir="ca-output",
                fail_on_stale_data=False,
            ).as_dict()
        ],
    )
    config.diseases._current_value = "covid"
    config.locations._current_value = "CA"
    return config


def test_launchpad_has_model_defaults_and_shared_location_override():
    base_fields = asset_config.ModelBaseConfig.model_fields
    override_fields = asset_config.ConfigOverride.model_fields
    config = asset_config.ModelBaseConfig()

    assert "exclude_last_n_days" not in base_fields
    assert "exclude_last_n_days" not in override_fields
    assert "n_lookback_days" not in base_fields
    assert "n_lookback_days" in override_fields
    assert "fable_pyrenew_n_lookback_days" not in override_fields
    assert "epiautogp_n_lookback_days" not in override_fields
    assert config.fable_pyrenew_n_lookback_days == 150
    assert (
        "n_lookback_days" not in asset_config.EpiAutoGPEPctEpiweeklyConfig.model_fields
    )


def test_postprocess_always_copies_to_daily_output(monkeypatch):
    postprocess = Mock()
    monkeypatch.setattr(assets, "_throw_if_backfill", Mock())
    monkeypatch.setattr(assets, "postprocess", postprocess)

    with dg.build_asset_context(partition_key="2026-09-09") as context:
        assets.postprocess_forecasts(
            context,
            asset_config.PostProcessConfig(
                output_basedir="custom-output",
                postprocess_diseases=["flu"],
                skip_existing=True,
            ),
        )

    daily_output = Path("custom-output/2026-09-09_forecasts")
    postprocess.assert_called_once_with(
        base_forecast_dir=daily_output,
        diseases=["flu"],
        skip_existing=True,
        local_copy_dir=daily_output,
    )


@pytest.mark.parametrize(
    ("override_fields", "expected_lookbacks"),
    [
        pytest.param(
            {"fail_on_stale_data": False},
            (100, 200),
            id="omitted-retains-model-defaults",
        ),
        pytest.param(
            {"n_lookback_days": 101},
            (101, 101),
            id="value-applies-to-all-models",
        ),
        pytest.param(
            {"n_lookback_days": None},
            (None, None),
            id="explicit-null-applies-to-all-models",
        ),
    ],
)
def test_location_lookback_override(
    override_fields: dict[str, object],
    expected_lookbacks: tuple[int | None, int | None],
):
    config = asset_config.ModelBaseConfig(
        fable_pyrenew_n_lookback_days=100,
        epiautogp_n_lookback_days=200,
        config_overrides=[
            asset_config.ConfigOverride(
                location="CA",
                **override_fields,
            ).as_dict()
        ],
    )

    loc_config = config.get_by_location("CA")

    assert (
        loc_config.fable_pyrenew_n_lookback_days,
        loc_config.epiautogp_n_lookback_days,
    ) == expected_lookbacks


def test_model_runners_use_their_named_lookbacks(monkeypatch):
    context = SimpleNamespace(
        partition_key="2026-09-09",
        log=Mock(),
    )
    model_base_config = _model_base_config()
    forecast_fable = Mock()
    forecast_pyrenew = Mock()
    forecast_epiautogp = Mock()

    monkeypatch.setattr(asset_helpers, "_throw_if_backfill", Mock())
    monkeypatch.setattr(asset_helpers, "forecast_fable", forecast_fable)
    monkeypatch.setattr(asset_helpers, "forecast_pyrenew", forecast_pyrenew)
    monkeypatch.setattr(asset_helpers, "forecast_epiautogp", forecast_epiautogp)

    asset_helpers._run_fable_e_other(
        context,
        asset_config.FableEOtherConfig(),
        model_base_config,
        ed_visit_input_resolution="daily",
    )
    asset_helpers._run_pyrenew_model(
        context,
        asset_config.PyrenewConfig(),
        model_base_config,
        model_letters="e",
    )
    asset_helpers._run_epiautogp_e_pct_epiweekly(
        context,
        asset_config.EpiAutoGPEPctEpiweeklyConfig(),
        model_base_config,
    )

    shared_arguments = {
        "output_dir": Path("ca-output/2026-09-09_forecasts"),
        "fail_on_stale_data": False,
    }
    for forecast in (forecast_fable, forecast_pyrenew):
        for name, expected in shared_arguments.items():
            assert forecast.call_args.kwargs[name] == expected
        assert forecast.call_args.kwargs["n_lookback_days"] == 100

    for name, expected in shared_arguments.items():
        assert forecast_epiautogp.call_args.kwargs[name] == expected
    assert forecast_epiautogp.call_args.kwargs["n_lookback_days"] == 200


def _write_fusion_inputs(
    tmp_path,
    batch="covid_lookback-100_omit-2",
    *,
    location="CA",
    report="2026-09-09",
    models=("pyrenew_e", "daily_fable_e_other"),
):
    model_loc_dir = tmp_path / f"{report}_forecasts" / batch / "model_runs" / location
    for model in models:
        (model_loc_dir / model / "data").mkdir(parents=True, exist_ok=True)
        (model_loc_dir / model / "samples.parquet").touch()
        (model_loc_dir / model / "data/combined_data.tsv").touch()
    return model_loc_dir


@pytest.mark.parametrize("lookback", [100, None])
def test_fusion_directory_discovers_unique_batch(tmp_path, lookback):
    context = SimpleNamespace(partition_key="2026-09-09", log=Mock())
    config = _model_base_config().model_copy(
        update={
            "output_basedir": str(tmp_path),
            "config_overrides": [],
            "fable_pyrenew_n_lookback_days": lookback,
        }
    )
    expected = _write_fusion_inputs(
        tmp_path, f"covid_lookback-{lookback or 'all'}_omit-6"
    )
    _write_fusion_inputs(tmp_path, "flu_lookback-100_omit-6")
    _write_fusion_inputs(tmp_path, "covid_lookback-200_omit-6")
    _write_fusion_inputs(tmp_path, location="NY")
    _write_fusion_inputs(tmp_path, report="2026-09-08")
    _write_fusion_inputs(tmp_path, models=("pyrenew_e",))
    _write_fusion_inputs(tmp_path, "covid_lookback-100_omit-invalid")
    assert (
        asset_helpers._get_model_loc_dir(
            context, config, "pyrenew_e", "daily_fable_e_other"
        )
        == expected
    )


@pytest.mark.parametrize("n_matches", [0, 2])
def test_fusion_directory_rejects_missing_or_ambiguous_outputs(tmp_path, n_matches):
    context = SimpleNamespace(partition_key="2026-09-09", log=Mock())
    config = _model_base_config().model_copy(
        update={
            "output_basedir": str(tmp_path),
            "config_overrides": [],
        }
    )
    for tail in range(n_matches):
        _write_fusion_inputs(tmp_path, f"covid_lookback-100_omit-{tail}")
    # Models in different batches cannot be fused together.
    _write_fusion_inputs(tmp_path, "covid_lookback-100_omit-10", models=("pyrenew_e",))
    _write_fusion_inputs(
        tmp_path, "covid_lookback-100_omit-11", models=("daily_fable_e_other",)
    )
    with pytest.raises(RuntimeError, match=f"found {n_matches}"):
        asset_helpers._get_model_loc_dir(
            context, config, "pyrenew_e", "daily_fable_e_other"
        )


def test_fusion_requires_data_and_samples(tmp_path):
    context = SimpleNamespace(partition_key="2026-09-09", log=Mock())
    config = _model_base_config().model_copy(
        update={
            "output_basedir": str(tmp_path),
            "config_overrides": [],
        }
    )
    model_dir = _write_fusion_inputs(tmp_path, models=("pyrenew_e",))
    (model_dir / "daily_fable_e_other").mkdir()
    (model_dir / "daily_fable_e_other/samples.parquet").touch()
    with pytest.raises(RuntimeError, match="found 0"):
        asset_helpers._get_model_loc_dir(
            context, config, "pyrenew_e", "daily_fable_e_other"
        )
