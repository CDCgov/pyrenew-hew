"""Unit tests for forecast pipeline utility functions.

These tests use mocking via `@patch` decorators to isolate the units under test.
We mock these dependencies to:

1. Test the logic and control flow without side effects
2. Avoid requiring actual data files or external services
3. Make tests fast and deterministic
4. Focus on verifying correct function calls and parameter passing

Each test mocks the minimum dependencies needed for that specific test case.

The end-to-end functionality of the forecast pipeline is verified in a separate
shell-based integration test.
"""

import datetime as dt
import logging
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from pipelines.epiautogp.forecast_utils import (
    ForecastPipelineContext,
    ModelPaths,
    postprocess_forecast,
    prepare_model_data,
    setup_forecast_pipeline,
)


class TestForecastPipelineContext:
    """Tests for the ForecastPipelineContext dataclass."""

    def test_context_initialization(self):
        """Test that ForecastPipelineContext can be initialized with all fields."""
        context = ForecastPipelineContext(
            disease="COVID-19",
            loc="CA",
            report_date=dt.date(2024, 12, 20),
            first_training_date=dt.date(2024, 9, 22),
            last_training_date=dt.date(2024, 12, 20),
            n_forecast_days=28,
            exclude_last_n_days=0,
            model_batch_dir=Path("/output/batch"),
            model_run_dir=Path("/output/batch/model_runs/CA"),
            credentials_dict={"key": "value"},
            facility_level_nssp_data=pl.LazyFrame(),
            loc_level_nssp_data=pl.LazyFrame(),
            logger=logging.getLogger(),
        )

        assert context.disease == "COVID-19"
        assert context.loc == "CA"
        assert context.n_forecast_days == 28
        assert context.exclude_last_n_days == 0


class TestModelPaths:
    """Tests for the ModelPaths dataclass."""

    def test_paths_initialization(self):
        """Test that ModelPaths can be initialized with all fields."""
        paths = ModelPaths(
            model_output_dir=Path("/output/model"),
            data_dir=Path("/output/model/data"),
            daily_training_data=Path("/output/model/data/combined_training_data.tsv"),
            epiweekly_training_data=Path(
                "/output/model/data/epiweekly_combined_training_data.tsv"
            ),
        )

        assert paths.model_output_dir == Path("/output/model")
        assert paths.data_dir == Path("/output/model/data")
        assert paths.daily_training_data.name == "combined_training_data.tsv"
        assert (
            paths.epiweekly_training_data.name == "epiweekly_combined_training_data.tsv"
        )


class TestSetupForecastPipeline:
    """Tests for the setup_forecast_pipeline function."""

    @patch("pipelines.forecast_utils.load_credentials")
    @patch("pipelines.forecast_utils.get_available_reports")
    @patch("pipelines.forecast_utils.parse_and_validate_report_date")
    @patch("pipelines.forecast_utils.calculate_training_dates")
    @patch("pipelines.forecast_utils.load_nssp_data")
    def test_setup_pipeline_returns_context(
        self,
        mock_load_nssp,
        mock_calc_dates,
        mock_parse_date,
        mock_get_reports,
        mock_load_creds,
        tmp_path,
    ):
        """Test that setup_forecast_pipeline returns a properly configured context."""
        # Setup mocks
        mock_get_reports.return_value = [dt.date(2024, 12, 20)]
        mock_parse_date.return_value = (dt.date(2024, 12, 20), dt.date(2024, 12, 20))
        mock_calc_dates.return_value = (dt.date(2024, 9, 22), dt.date(2024, 12, 20))
        mock_load_nssp.return_value = (pl.LazyFrame(), pl.LazyFrame())

        context = setup_forecast_pipeline(
            disease="COVID-19",
            report_date="latest",
            loc="CA",
            facility_level_nssp_data_dir=tmp_path,
            state_level_nssp_data_dir=tmp_path,
            output_dir=tmp_path,
            n_training_days=90,
            n_forecast_days=28,
            exclude_last_n_days=0,
            credentials_path=None,
            logger=None,
        )

        assert isinstance(context, ForecastPipelineContext)
        assert context.disease == "COVID-19"
        assert context.loc == "CA"
        assert context.n_forecast_days == 28
        assert context.report_date == dt.date(2024, 12, 20)
        assert context.first_training_date == dt.date(2024, 9, 22)
        assert context.last_training_date == dt.date(2024, 12, 20)

    @patch("pipelines.forecast_utils.load_credentials")
    @patch("pipelines.forecast_utils.get_available_reports")
    @patch("pipelines.forecast_utils.parse_and_validate_report_date")
    @patch("pipelines.forecast_utils.calculate_training_dates")
    @patch("pipelines.forecast_utils.load_nssp_data")
    def test_setup_pipeline_creates_directory_structure(
        self,
        mock_load_nssp,
        mock_calc_dates,
        mock_parse_date,
        mock_get_reports,
        mock_load_creds,
        tmp_path,
    ):
        """Test that setup creates the expected directory structure."""
        mock_load_creds.return_value = {}
        mock_get_reports.return_value = [dt.date(2024, 12, 20)]
        mock_parse_date.return_value = (dt.date(2024, 12, 20), dt.date(2024, 12, 20))
        mock_calc_dates.return_value = (dt.date(2024, 9, 22), dt.date(2024, 12, 20))
        mock_load_nssp.return_value = (pl.LazyFrame(), pl.LazyFrame())

        context = setup_forecast_pipeline(
            disease="COVID-19",
            report_date="latest",
            loc="CA",
            facility_level_nssp_data_dir=tmp_path,
            state_level_nssp_data_dir=tmp_path,
            output_dir=tmp_path,
            n_training_days=90,
            n_forecast_days=28,
        )

        expected_batch_dir = (
            tmp_path / "covid-19_r_2024-12-20_f_2024-09-22_t_2024-12-20"
        )
        expected_run_dir = expected_batch_dir / "model_runs" / "CA"

        assert context.model_batch_dir == expected_batch_dir
        assert context.model_run_dir == expected_run_dir


class TestPrepareModelData:
    """Tests for the prepare_model_data function."""

    @patch("pipelines.forecast_utils.generate_epiweekly_data")
    @patch("pipelines.forecast_utils.save_eval_data")
    @patch("pipelines.forecast_utils.process_and_save_loc_data")
    def test_prepare_model_data_returns_paths(
        self,
        mock_process_loc,
        mock_save_eval,
        mock_gen_epiweekly,
        tmp_path,
    ):
        """Test that prepare_model_data returns ModelPaths."""
        context = ForecastPipelineContext(
            disease="COVID-19",
            loc="CA",
            report_date=dt.date(2024, 12, 20),
            first_training_date=dt.date(2024, 9, 22),
            last_training_date=dt.date(2024, 12, 20),
            n_forecast_days=28,
            exclude_last_n_days=0,
            model_batch_dir=tmp_path / "batch",
            model_run_dir=tmp_path / "batch" / "model_runs" / "CA",
            credentials_dict={},
            facility_level_nssp_data=pl.LazyFrame(),
            loc_level_nssp_data=pl.LazyFrame(),
            logger=logging.getLogger(),
        )

        paths = prepare_model_data(
            context=context,
            model_name="test_model",
            eval_data_path=tmp_path / "eval.parquet",
        )

        assert isinstance(paths, ModelPaths)
        assert paths.model_output_dir.name == "test_model"
        assert paths.data_dir.name == "data"
        assert paths.daily_training_data.name == "combined_training_data.tsv"
        assert (
            paths.epiweekly_training_data.name == "epiweekly_combined_training_data.tsv"
        )

    @patch("pipelines.forecast_utils.generate_epiweekly_data")
    @patch("pipelines.forecast_utils.save_eval_data")
    @patch("pipelines.forecast_utils.process_and_save_loc_data")
    def test_prepare_model_data_creates_directories(
        self,
        mock_process_loc,
        mock_save_eval,
        mock_gen_epiweekly,
        tmp_path,
    ):
        """Test that prepare_model_data creates the required directories."""
        model_run_dir = tmp_path / "batch" / "model_runs" / "CA"
        context = ForecastPipelineContext(
            disease="COVID-19",
            loc="CA",
            report_date=dt.date(2024, 12, 20),
            first_training_date=dt.date(2024, 9, 22),
            last_training_date=dt.date(2024, 12, 20),
            n_forecast_days=28,
            exclude_last_n_days=0,
            model_batch_dir=tmp_path / "batch",
            model_run_dir=model_run_dir,
            credentials_dict={},
            facility_level_nssp_data=pl.LazyFrame(),
            loc_level_nssp_data=pl.LazyFrame(),
            logger=logging.getLogger(),
        )

        paths = prepare_model_data(
            context=context,
            model_name="test_model",
            eval_data_path=tmp_path / "eval.parquet",
        )

        assert paths.model_output_dir.exists()
        assert paths.data_dir.exists()

    @patch("pipelines.forecast_utils.process_and_save_loc_data")
    def test_prepare_model_data_raises_without_eval_path(self, mock_process, tmp_path):
        """Test that prepare_model_data raises ValueError without eval_data_path."""
        context = ForecastPipelineContext(
            disease="COVID-19",
            loc="CA",
            report_date=dt.date(2024, 12, 20),
            first_training_date=dt.date(2024, 9, 22),
            last_training_date=dt.date(2024, 12, 20),
            n_forecast_days=28,
            exclude_last_n_days=0,
            model_batch_dir=tmp_path / "batch",
            model_run_dir=tmp_path / "batch" / "model_runs" / "CA",
            credentials_dict={},
            facility_level_nssp_data=pl.LazyFrame(),
            loc_level_nssp_data=pl.LazyFrame(),
            logger=logging.getLogger(),
        )

        with pytest.raises(ValueError, match="No path to an evaluation dataset"):
            prepare_model_data(
                context=context,
                model_name="test_model",
                eval_data_path=None,
            )


class TestPostprocessForecast:
    """Tests for the postprocess_forecast function."""

    @patch("pipelines.forecast_utils.create_hubverse_table")
    @patch("pipelines.forecast_utils.plot_and_save_loc_forecast")
    def test_postprocess_calls_required_functions(
        self,
        mock_plot,
        mock_hubverse,
        tmp_path,
    ):
        """Test that postprocess_forecast calls plotting and hubverse creation."""
        context = ForecastPipelineContext(
            disease="COVID-19",
            loc="CA",
            report_date=dt.date(2024, 12, 20),
            first_training_date=dt.date(2024, 9, 22),
            last_training_date=dt.date(2024, 12, 20),
            n_forecast_days=28,
            exclude_last_n_days=5,
            model_batch_dir=tmp_path / "batch",
            model_run_dir=tmp_path / "batch" / "model_runs" / "CA",
            credentials_dict={},
            facility_level_nssp_data=pl.LazyFrame(),
            loc_level_nssp_data=pl.LazyFrame(),
            logger=logging.getLogger(),
        )

        postprocess_forecast(context=context, model_name="test_model")

        # Verify functions were called
        mock_plot.assert_called_once()
        mock_hubverse.assert_called_once()

        # Verify correct arguments
        assert mock_plot.call_args[0][0] == context.model_run_dir
        assert mock_plot.call_args[0][1] == 33  # n_forecast_days + exclude_last_n_days
        assert mock_plot.call_args[1]["timeseries_model_name"] == "test_model"

    @patch("pipelines.forecast_utils.create_hubverse_table")
    @patch("pipelines.forecast_utils.plot_and_save_loc_forecast")
    def test_postprocess_calculates_correct_forecast_period(
        self,
        mock_plot,
        mock_hubverse,
        tmp_path,
    ):
        """Test that n_days_past_last_training is calculated correctly."""
        context = ForecastPipelineContext(
            disease="COVID-19",
            loc="CA",
            report_date=dt.date(2024, 12, 20),
            first_training_date=dt.date(2024, 9, 22),
            last_training_date=dt.date(2024, 12, 20),
            n_forecast_days=28,
            exclude_last_n_days=0,
            model_batch_dir=tmp_path / "batch",
            model_run_dir=tmp_path / "batch" / "model_runs" / "CA",
            credentials_dict={},
            facility_level_nssp_data=pl.LazyFrame(),
            loc_level_nssp_data=pl.LazyFrame(),
            logger=logging.getLogger(),
        )

        postprocess_forecast(context=context, model_name="test_model")

        # Should be exactly n_forecast_days when exclude_last_n_days is 0
        assert mock_plot.call_args[0][1] == 28
