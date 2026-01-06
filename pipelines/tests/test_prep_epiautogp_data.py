"""Unit tests for EpiAutoGP data preparation functions.

These tests verify the data conversion and filtering functionality,
particularly the exclude_date_ranges feature for removing problematic
reporting periods from the training data.
"""

import datetime as dt
import json
import logging

import polars as pl
import pytest

from pipelines.epiautogp.epiautogp_forecast_utils import (
    ForecastPipelineContext,
    ModelPaths,
)
from pipelines.epiautogp.prep_epiautogp_data import (
    _read_tsv_data,
    convert_to_epiautogp_json,
)


@pytest.fixture
def sample_tsv_data():
    """Create a sample TSV file with test data."""
    # Create sample data with dates spanning multiple months
    dates = [dt.date(2024, 1, i) for i in range(1, 32)]  # January 2024
    dates.extend([dt.date(2024, 2, i) for i in range(1, 29)])  # February 2024
    dates.extend([dt.date(2024, 3, i) for i in range(1, 11)])  # March 1-10

    data = []
    for date in dates:
        # Add observed_ed_visits data
        data.append(
            {
                "date": date,
                "geo_value": "CA",
                "disease": "COVID-19",
                "data_type": "observed",
                ".variable": "observed_ed_visits",
                ".value": 100.0,
            }
        )
        # Add other_ed_visits data
        data.append(
            {
                "date": date,
                "geo_value": "CA",
                "disease": "COVID-19",
                "data_type": "observed",
                ".variable": "other_ed_visits",
                ".value": 900.0,
            }
        )

    return pl.DataFrame(data)


@pytest.fixture
def sample_context(tmp_path):
    """Fixture providing a ForecastPipelineContext for testing."""
    return ForecastPipelineContext(
        disease="COVID-19",
        loc="CA",
        target="nssp",
        frequency="daily",
        use_percentage=False,
        ed_visit_type="observed",
        model_name="test_epiautogp",
        param_data_dir=None,
        eval_data_path=tmp_path / "eval.parquet",
        nhsn_data_path=None,
        report_date=dt.date(2024, 3, 10),
        first_training_date=dt.date(2024, 1, 1),
        last_training_date=dt.date(2024, 3, 10),
        n_forecast_days=28,
        exclude_last_n_days=0,
        model_batch_dir=tmp_path / "batch",
        model_run_dir=tmp_path / "batch" / "model_runs" / "CA",
        credentials_dict={},
        facility_level_nssp_data=pl.LazyFrame(),
        loc_level_nssp_data=pl.LazyFrame(),
        logger=logging.getLogger(__name__),
    )


class TestReadTsvDataWithExcludeDateRanges:
    """Tests for _read_tsv_data with exclude_date_ranges parameter."""

    def test_read_without_exclusions(self, tmp_path, sample_tsv_data):
        """Test reading data without any date exclusions."""
        # Write sample data to TSV
        tsv_path = tmp_path / "test_data.tsv"
        sample_tsv_data.write_csv(tsv_path, separator="\t")

        # Read data without exclusions
        dates, reports = _read_tsv_data(
            tsv_path=tsv_path,
            disease="COVID-19",
            location="CA",
            target="nssp",
            frequency="daily",
            use_percentage=False,
            ed_visit_type="observed",
            logger=logging.getLogger(__name__),
            exclude_date_ranges=None,
        )

        # Should have all dates (31 + 28 + 10 = 69 days)
        assert len(dates) == 69
        assert len(reports) == 69
        assert dates[0] == dt.date(2024, 1, 1)
        assert dates[-1] == dt.date(2024, 3, 10)

    def test_read_with_single_exclusion(self, tmp_path, sample_tsv_data):
        """Test reading data with a single date range exclusion."""
        # Write sample data to TSV
        tsv_path = tmp_path / "test_data.tsv"
        sample_tsv_data.write_csv(tsv_path, separator="\t")

        # Exclude January 15-20 (6 days)
        exclude_ranges = [(dt.date(2024, 1, 15), dt.date(2024, 1, 20))]

        dates, reports = _read_tsv_data(
            tsv_path=tsv_path,
            disease="COVID-19",
            location="CA",
            target="nssp",
            frequency="daily",
            use_percentage=False,
            ed_visit_type="observed",
            logger=logging.getLogger(__name__),
            exclude_date_ranges=exclude_ranges,
        )

        # Should have 69 - 6 = 63 days
        assert len(dates) == 63
        assert len(reports) == 63

        # Verify excluded dates are not present
        for date in dates:
            assert not (dt.date(2024, 1, 15) <= date <= dt.date(2024, 1, 20))

        # Verify dates before and after exclusion are present
        assert dt.date(2024, 1, 14) in dates
        assert dt.date(2024, 1, 21) in dates

    def test_read_with_multiple_exclusions(self, tmp_path, sample_tsv_data):
        """Test reading data with multiple date range exclusions."""
        # Write sample data to TSV
        tsv_path = tmp_path / "test_data.tsv"
        sample_tsv_data.write_csv(tsv_path, separator="\t")

        # Exclude two periods: Jan 15-20 (6 days) and Feb 10-15 (6 days)
        exclude_ranges = [
            (dt.date(2024, 1, 15), dt.date(2024, 1, 20)),
            (dt.date(2024, 2, 10), dt.date(2024, 2, 15)),
        ]

        dates, reports = _read_tsv_data(
            tsv_path=tsv_path,
            disease="COVID-19",
            location="CA",
            target="nssp",
            frequency="daily",
            use_percentage=False,
            ed_visit_type="observed",
            logger=logging.getLogger(__name__),
            exclude_date_ranges=exclude_ranges,
        )

        # Should have 69 - 12 = 57 days
        assert len(dates) == 57
        assert len(reports) == 57

        # Verify both excluded ranges are not present
        for date in dates:
            assert not (dt.date(2024, 1, 15) <= date <= dt.date(2024, 1, 20))
            assert not (dt.date(2024, 2, 10) <= date <= dt.date(2024, 2, 15))

    def test_read_with_empty_exclusion_list(self, tmp_path, sample_tsv_data):
        """Test that empty exclusion list behaves same as None."""
        # Write sample data to TSV
        tsv_path = tmp_path / "test_data.tsv"
        sample_tsv_data.write_csv(tsv_path, separator="\t")

        dates, reports = _read_tsv_data(
            tsv_path=tsv_path,
            disease="COVID-19",
            location="CA",
            target="nssp",
            frequency="daily",
            use_percentage=False,
            ed_visit_type="observed",
            logger=logging.getLogger(__name__),
            exclude_date_ranges=[],
        )

        # Should have all dates
        assert len(dates) == 69
        assert len(reports) == 69

    def test_read_with_percentage_and_exclusions(self, tmp_path, sample_tsv_data):
        """Test that exclusions work with percentage calculations."""
        # Write sample data to TSV
        tsv_path = tmp_path / "test_data.tsv"
        sample_tsv_data.write_csv(tsv_path, separator="\t")

        # Exclude one date range
        exclude_ranges = [(dt.date(2024, 1, 15), dt.date(2024, 1, 20))]

        dates, reports = _read_tsv_data(
            tsv_path=tsv_path,
            disease="COVID-19",
            location="CA",
            target="nssp",
            frequency="daily",
            use_percentage=True,  # Use percentage
            ed_visit_type="observed",
            logger=logging.getLogger(__name__),
            exclude_date_ranges=exclude_ranges,
        )

        # Should have 63 days
        assert len(dates) == 63
        assert len(reports) == 63

        # Verify percentage calculation (100 / (100 + 900) * 100 = 10.0)
        assert all(abs(r - 10.0) < 0.01 for r in reports)


class TestConvertToEpiautogpJsonWithExclusions:
    """Tests for convert_to_epiautogp_json with exclude_date_ranges."""

    def test_convert_with_exclusions(self, tmp_path, sample_context, sample_tsv_data):
        """Test that convert_to_epiautogp_json properly handles date exclusions."""
        # Setup paths
        model_output_dir = tmp_path / "model_output"
        data_dir = model_output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write sample data
        daily_data_path = data_dir / "combined_training_data.tsv"
        sample_tsv_data.write_csv(daily_data_path, separator="\t")

        paths = ModelPaths(
            model_output_dir=model_output_dir,
            data_dir=data_dir,
            daily_training_data=daily_data_path,
            epiweekly_training_data=data_dir / "epiweekly_combined_training_data.tsv",
        )

        # Exclude a date range
        exclude_ranges = [(dt.date(2024, 1, 15), dt.date(2024, 1, 20))]

        # Convert to JSON
        json_path = convert_to_epiautogp_json(
            context=sample_context,
            paths=paths,
            exclude_date_ranges=exclude_ranges,
        )

        # Verify JSON file was created
        assert json_path.exists()

        # Read and verify JSON content
        with open(json_path) as f:
            json_data = json.load(f)

        # Check that dates were excluded
        assert len(json_data["dates"]) == 63  # 69 - 6 excluded days
        assert len(json_data["reports"]) == 63

        # Verify excluded dates are not in the JSON
        json_dates = [dt.date.fromisoformat(d) for d in json_data["dates"]]
        for date in json_dates:
            assert not (dt.date(2024, 1, 15) <= date <= dt.date(2024, 1, 20))

    def test_convert_without_exclusions(
        self, tmp_path, sample_context, sample_tsv_data
    ):
        """Test that convert_to_epiautogp_json works without exclusions."""
        # Setup paths
        model_output_dir = tmp_path / "model_output"
        data_dir = model_output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write sample data
        daily_data_path = data_dir / "combined_training_data.tsv"
        sample_tsv_data.write_csv(daily_data_path, separator="\t")

        paths = ModelPaths(
            model_output_dir=model_output_dir,
            data_dir=data_dir,
            daily_training_data=daily_data_path,
            epiweekly_training_data=data_dir / "epiweekly_combined_training_data.tsv",
        )

        # Convert to JSON without exclusions
        json_path = convert_to_epiautogp_json(
            context=sample_context,
            paths=paths,
            exclude_date_ranges=None,
        )

        # Verify JSON file was created
        assert json_path.exists()

        # Read and verify JSON content
        with open(json_path) as f:
            json_data = json.load(f)

        # Check that all dates are present
        assert len(json_data["dates"]) == 69
        assert len(json_data["reports"]) == 69
