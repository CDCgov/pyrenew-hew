"""Unit tests for common utility functions in pipelines/common_utils.py"""

import datetime as dt
import logging

import polars as pl
import pytest

from pipelines.common_utils import (
    calculate_training_dates,
    get_available_reports,
    load_credentials,
    load_nssp_data,
    parse_and_validate_report_date,
)


class TestValidationUtils:
    """Tests for validation and configuration utilities."""

    def test_load_credentials_with_none_returns_none(self):
        """Test that None credentials path returns None."""
        logger = logging.getLogger(__name__)

        result = load_credentials(None, logger)

        assert result is None

    def test_load_credentials_with_invalid_extension_raises_error(self, tmp_path):
        """Test that non-TOML file extension raises ValueError."""
        invalid_file = tmp_path / "credentials.txt"
        invalid_file.write_text("not a toml file")
        logger = logging.getLogger(__name__)

        with pytest.raises(ValueError, match="must have the extension '.toml'"):
            load_credentials(invalid_file, logger)

    @pytest.mark.parametrize(
        "input_date,available_facility,available_loc,expected_report,expected_loc",
        [
            (
                "latest",
                [dt.date(2024, 12, 18), dt.date(2024, 12, 19), dt.date(2024, 12, 20)],
                [dt.date(2024, 12, 18), dt.date(2024, 12, 19)],
                dt.date(2024, 12, 20),
                dt.date(2024, 12, 19),
            ),
            (
                "2024-12-20",
                [dt.date(2024, 12, 15), dt.date(2024, 12, 20)],
                [dt.date(2024, 12, 15), dt.date(2024, 12, 20)],
                dt.date(2024, 12, 20),
                dt.date(2024, 12, 20),
            ),
        ],
    )
    def test_parse_and_validate_report_date(
        self,
        input_date,
        available_facility,
        available_loc,
        expected_report,
        expected_loc,
    ):
        """Test parsing report dates with various inputs."""
        logger = logging.getLogger(__name__)

        report_date, loc_report_date = parse_and_validate_report_date(
            input_date, available_facility, available_loc, logger
        )

        assert report_date == expected_report
        assert loc_report_date == expected_loc

    @pytest.mark.parametrize(
        "n_training_days,exclude_last_n_days,expected_first,expected_last",
        [
            (90, 0, dt.date(2024, 9, 22), dt.date(2024, 12, 20)),
            (90, 5, dt.date(2024, 9, 17), dt.date(2024, 12, 15)),
        ],
    )
    def test_calculate_training_dates(
        self, n_training_days, exclude_last_n_days, expected_first, expected_last
    ):
        """Test training date calculation with various parameters."""
        report_date = dt.date(2024, 12, 21)
        logger = logging.getLogger(__name__)

        first_date, last_date = calculate_training_dates(
            report_date, n_training_days, exclude_last_n_days, logger
        )

        assert first_date == expected_first
        assert last_date == expected_last
        assert (last_date - first_date).days == n_training_days - 1


class TestDataWranglingUtils:
    """Tests for data loading and processing utilities."""

    def test_get_available_reports_with_parquet_files(self, tmp_path):
        """Test discovering available report dates from parquet files."""
        (tmp_path / "2024-12-01.parquet").touch()
        (tmp_path / "2024-12-15.parquet").touch()
        (tmp_path / "2024-12-20.parquet").touch()

        result = get_available_reports(tmp_path)

        assert len(result) == 3
        assert dt.date(2024, 12, 1) in result
        assert dt.date(2024, 12, 15) in result
        assert dt.date(2024, 12, 20) in result

    def test_load_nssp_data_both_available(self, tmp_path):
        """Test loading NSSP data when both facility and location data available."""
        facility_dir = tmp_path / "facility"
        loc_dir = tmp_path / "location"
        facility_dir.mkdir()
        loc_dir.mkdir()

        # Create minimal valid parquet files
        facility_file = facility_dir / "2024-12-20.parquet"
        loc_file = loc_dir / "2024-12-20.parquet"

        df = pl.DataFrame({"col1": [1, 2, 3]})
        df.write_parquet(facility_file)
        df.write_parquet(loc_file)

        logger = logging.getLogger(__name__)
        report_date = dt.date(2024, 12, 20)
        available = [report_date]

        facility_data, loc_data = load_nssp_data(
            report_date,
            report_date,
            available,
            available,
            facility_dir,
            loc_dir,
            logger,
        )

        assert facility_data is not None
        assert loc_data is not None
        assert isinstance(facility_data, pl.LazyFrame)
        assert isinstance(loc_data, pl.LazyFrame)
