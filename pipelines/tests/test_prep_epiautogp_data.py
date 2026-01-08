"""Unit tests for EpiAutoGP data preparation date exclusion functionality.

Note: The filter_dates_by_exclusions function below is a standalone copy of the
filtering logic used in prep_epiautogp_data._read_tsv_data. This duplication is
intentional to allow testing the filtering logic in isolation without importing
the full epiautogp module, which has many external dependencies (pygit2, etc.).
The test logic mirrors the production implementation.
"""

import datetime as dt

import pytest


def filter_dates_by_exclusions(
    dates: list[dt.date],
    reports: list[float],
    exclude_date_ranges: list[tuple[dt.date, dt.date]] | None,
) -> tuple[list[dt.date], list[float]]:
    """
    Filter dates and reports based on exclusion ranges.
    
    This is a standalone version of the filtering logic used in _read_tsv_data
    for testing purposes.
    """
    if exclude_date_ranges is None or len(exclude_date_ranges) == 0:
        return dates, reports
    
    filtered_indices = []
    excluded_count = 0
    
    for i, date in enumerate(dates):
        should_exclude = False
        for start_date, end_date in exclude_date_ranges:
            if start_date <= date <= end_date:
                should_exclude = True
                excluded_count += 1
                break
        if not should_exclude:
            filtered_indices.append(i)
    
    # Filter both dates and reports using the same indices
    filtered_dates = [dates[i] for i in filtered_indices]
    filtered_reports = [reports[i] for i in filtered_indices]
    
    return filtered_dates, filtered_reports


class TestDateExclusionFiltering:
    """Tests for date exclusion filtering logic."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dates and reports."""
        dates = [dt.date(2024, 1, i) for i in range(1, 11)]
        reports = [float(i * 10) for i in range(1, 11)]
        return dates, reports

    def test_no_exclusions(self, sample_data):
        """Test filtering with no exclusions."""
        dates, reports = sample_data
        
        filtered_dates, filtered_reports = filter_dates_by_exclusions(
            dates, reports, None
        )
        
        assert len(filtered_dates) == 10
        assert len(filtered_reports) == 10
        assert filtered_dates == dates
        assert filtered_reports == reports

    def test_exclude_single_date(self, sample_data):
        """Test excluding a single date."""
        dates, reports = sample_data
        
        # Exclude Jan 5
        exclude_ranges = [(dt.date(2024, 1, 5), dt.date(2024, 1, 5))]
        
        filtered_dates, filtered_reports = filter_dates_by_exclusions(
            dates, reports, exclude_ranges
        )
        
        assert len(filtered_dates) == 9
        assert dt.date(2024, 1, 5) not in filtered_dates
        assert dt.date(2024, 1, 4) in filtered_dates
        assert dt.date(2024, 1, 6) in filtered_dates
        assert 50.0 not in filtered_reports  # Jan 5's value
        assert 40.0 in filtered_reports  # Jan 4's value
        assert 60.0 in filtered_reports  # Jan 6's value

    def test_exclude_date_range(self, sample_data):
        """Test excluding a range of dates."""
        dates, reports = sample_data
        
        # Exclude Jan 3-5 (inclusive)
        exclude_ranges = [(dt.date(2024, 1, 3), dt.date(2024, 1, 5))]
        
        filtered_dates, filtered_reports = filter_dates_by_exclusions(
            dates, reports, exclude_ranges
        )
        
        assert len(filtered_dates) == 7
        assert dt.date(2024, 1, 3) not in filtered_dates
        assert dt.date(2024, 1, 4) not in filtered_dates
        assert dt.date(2024, 1, 5) not in filtered_dates
        assert dt.date(2024, 1, 2) in filtered_dates
        assert dt.date(2024, 1, 6) in filtered_dates
        
        # Check reports
        assert 30.0 not in filtered_reports  # Jan 3's value
        assert 40.0 not in filtered_reports  # Jan 4's value
        assert 50.0 not in filtered_reports  # Jan 5's value

    def test_exclude_multiple_ranges(self, sample_data):
        """Test excluding multiple date ranges."""
        dates, reports = sample_data
        
        # Exclude Jan 2-3 and Jan 7-8
        exclude_ranges = [
            (dt.date(2024, 1, 2), dt.date(2024, 1, 3)),
            (dt.date(2024, 1, 7), dt.date(2024, 1, 8)),
        ]
        
        filtered_dates, filtered_reports = filter_dates_by_exclusions(
            dates, reports, exclude_ranges
        )
        
        assert len(filtered_dates) == 6
        assert dt.date(2024, 1, 2) not in filtered_dates
        assert dt.date(2024, 1, 3) not in filtered_dates
        assert dt.date(2024, 1, 7) not in filtered_dates
        assert dt.date(2024, 1, 8) not in filtered_dates
        assert dt.date(2024, 1, 1) in filtered_dates
        assert dt.date(2024, 1, 4) in filtered_dates
        assert dt.date(2024, 1, 9) in filtered_dates

    def test_empty_exclusions_list(self, sample_data):
        """Test that an empty exclusions list is treated the same as None."""
        dates, reports = sample_data
        
        filtered_dates, filtered_reports = filter_dates_by_exclusions(
            dates, reports, []
        )
        
        assert len(filtered_dates) == 10
        assert filtered_dates == dates
        assert filtered_reports == reports

    def test_reports_filtered_correctly(self, sample_data):
        """Test that reports are filtered along with dates."""
        dates, reports = sample_data
        
        # Exclude Jan 5 which should have value 50.0
        exclude_ranges = [(dt.date(2024, 1, 5), dt.date(2024, 1, 5))]
        
        filtered_dates, filtered_reports = filter_dates_by_exclusions(
            dates, reports, exclude_ranges
        )
        
        # Check that reports correspond to dates
        assert len(filtered_dates) == len(filtered_reports)
        assert 50.0 not in filtered_reports  # Jan 5's value
        
        # Check specific date-report pairs
        for i, date in enumerate(filtered_dates):
            if date == dt.date(2024, 1, 1):
                assert filtered_reports[i] == 10.0
            elif date == dt.date(2024, 1, 6):
                assert filtered_reports[i] == 60.0

    def test_exclude_dates_outside_range(self, sample_data):
        """Test excluding dates that don't exist in the data."""
        dates, reports = sample_data
        
        # Exclude dates before and after the data range
        exclude_ranges = [(dt.date(2023, 12, 1), dt.date(2023, 12, 31))]
        
        filtered_dates, filtered_reports = filter_dates_by_exclusions(
            dates, reports, exclude_ranges
        )
        
        # All dates should still be present
        assert len(filtered_dates) == 10
        assert filtered_dates == dates

    def test_exclude_all_dates_returns_empty(self, sample_data):
        """Test that excluding all dates returns empty lists (handled by calling code)."""
        dates, reports = sample_data
        
        # Exclude entire range
        exclude_ranges = [(dt.date(2024, 1, 1), dt.date(2024, 1, 10))]
        
        filtered_dates, filtered_reports = filter_dates_by_exclusions(
            dates, reports, exclude_ranges
        )
        
        # Should return empty lists - the calling code will raise an error
        assert len(filtered_dates) == 0
        assert len(filtered_reports) == 0

