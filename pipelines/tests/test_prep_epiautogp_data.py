"""
Tests for EpiAutoGP data conversion functions.
"""

import datetime as dt
import json

import polars as pl
import pytest

from pipelines.epiautogp.prep_epiautogp_data import convert_to_epiautogp_json


@pytest.fixture
def sample_combined_training_data(tmp_path):
    """Create a sample combined training data TSV file."""
    data = pl.DataFrame(
        {
            "date": [
                dt.date(2024, 9, 1),
                dt.date(2024, 9, 1),
                dt.date(2024, 9, 8),
                dt.date(2024, 9, 8),
                dt.date(2024, 9, 15),
                dt.date(2024, 9, 15),
            ],
            "geo_value": ["CA", "CA", "CA", "CA", "CA", "CA"],
            "disease": [
                "COVID-19",
                "COVID-19",
                "COVID-19",
                "COVID-19",
                "COVID-19",
                "COVID-19",
            ],
            "data_type": ["train", "train", "train", "train", "train", "train"],
            ".variable": [
                "observed_ed_visits",
                "other_ed_visits",
                "observed_ed_visits",
                "other_ed_visits",
                "observed_ed_visits",
                "other_ed_visits",
            ],
            ".value": [50.0, 450.0, 60.0, 540.0, 70.0, 630.0],
            "lab_site_index": [None, None, None, None, None, None],
        }
    )
    file_path = tmp_path / "combined_training_data.tsv"
    data.write_csv(file_path, separator="\t")
    return file_path


@pytest.fixture
def sample_nhsn_data(tmp_path):
    """Create a sample NHSN parquet file."""
    data = pl.DataFrame(
        {
            "jurisdiction": ["CA", "CA", "CA"],
            "weekendingdate": [
                dt.date(2024, 9, 7),
                dt.date(2024, 9, 14),
                dt.date(2024, 9, 21),
            ],
            "hospital_admissions": [45, 52, 38],
        }
    )
    file_path = tmp_path / "nhsn_test_data.parquet"
    data.write_parquet(file_path)
    return file_path


def test_convert_nssp_to_epiautogp_json(sample_combined_training_data, tmp_path):
    """Test conversion of NSSP data to EpiAutoGP JSON format."""
    output_path = tmp_path / "epiautogp_input.json"

    convert_to_epiautogp_json(
        combined_training_data_path=sample_combined_training_data,
        nhsn_data_path=None,
        output_json_path=output_path,
        disease="COVID-19",
        location="CA",
        forecast_date=dt.date(2024, 9, 15),
        target="nssp",
    )

    # Verify file was created
    assert output_path.exists()

    # Read and verify JSON content
    with open(output_path, "r") as f:
        result = json.load(f)

    # Check structure
    assert "dates" in result
    assert "reports" in result
    assert "pathogen" in result
    assert "location" in result
    assert "target" in result
    assert "forecast_date" in result
    assert "nowcast_dates" in result
    assert "nowcast_reports" in result

    # Check values
    assert result["pathogen"] == "COVID-19"
    assert result["location"] == "CA"
    assert result["target"] == "nssp"
    assert result["forecast_date"] == "2024-09-15"
    assert len(result["dates"]) == 3
    assert result["dates"] == ["2024-09-01", "2024-09-08", "2024-09-15"]

    # Check ED visit percentages are calculated correctly
    # Week 1: 50 / (50 + 450) * 100 = 10.0%
    # Week 2: 60 / (60 + 540) * 100 = 10.0%
    # Week 3: 70 / (70 + 630) * 100 = 10.0%
    assert len(result["reports"]) == 3
    assert result["reports"][0] == pytest.approx(10.0)
    assert result["reports"][1] == pytest.approx(10.0)
    assert result["reports"][2] == pytest.approx(10.0)


def test_convert_nhsn_to_epiautogp_json(sample_nhsn_data, tmp_path):
    """Test conversion of NHSN data to EpiAutoGP JSON format."""
    output_path = tmp_path / "epiautogp_input.json"

    # Create a dummy combined_training_data path (not used for NHSN)
    dummy_combined_path = tmp_path / "dummy.tsv"

    convert_to_epiautogp_json(
        combined_training_data_path=dummy_combined_path,
        nhsn_data_path=sample_nhsn_data,
        output_json_path=output_path,
        disease="COVID-19",
        location="CA",
        forecast_date=dt.date(2024, 9, 21),
        target="nhsn",
    )

    # Verify file was created
    assert output_path.exists()

    # Read and verify JSON content
    with open(output_path, "r") as f:
        result = json.load(f)

    # Check structure
    assert result["pathogen"] == "COVID-19"
    assert result["location"] == "CA"
    assert result["target"] == "nhsn"
    assert result["forecast_date"] == "2024-09-21"
    assert len(result["dates"]) == 3
    assert result["dates"] == ["2024-09-07", "2024-09-14", "2024-09-21"]

    # Check hospital admission counts
    assert len(result["reports"]) == 3
    assert result["reports"] == [45.0, 52.0, 38.0]


def test_convert_with_nowcast_data(sample_combined_training_data, tmp_path):
    """Test conversion with nowcast dates and reports."""
    output_path = tmp_path / "epiautogp_input.json"

    nowcast_dates = [dt.date(2024, 9, 8), dt.date(2024, 9, 15)]
    nowcast_reports = [[58.0, 60.0, 62.0], [68.0, 70.0, 72.0]]

    convert_to_epiautogp_json(
        combined_training_data_path=sample_combined_training_data,
        nhsn_data_path=None,
        output_json_path=output_path,
        disease="COVID-19",
        location="CA",
        forecast_date=dt.date(2024, 9, 15),
        target="nssp",
        nowcast_dates=nowcast_dates,
        nowcast_reports=nowcast_reports,
    )

    with open(output_path, "r") as f:
        result = json.load(f)

    assert result["nowcast_dates"] == ["2024-09-08", "2024-09-15"]
    assert result["nowcast_reports"] == [[58.0, 60.0, 62.0], [68.0, 70.0, 72.0]]


def test_invalid_target_raises_error(sample_combined_training_data, tmp_path):
    """Test that invalid target raises ValueError."""
    output_path = tmp_path / "epiautogp_input.json"

    with pytest.raises(ValueError, match="target must be 'nssp' or 'nhsn'"):
        convert_to_epiautogp_json(
            combined_training_data_path=sample_combined_training_data,
            nhsn_data_path=None,
            output_json_path=output_path,
            disease="COVID-19",
            location="CA",
            forecast_date=dt.date(2024, 9, 15),
            target="invalid_target",
        )


def test_nhsn_without_path_raises_error(sample_combined_training_data, tmp_path):
    """Test that NHSN target without data path raises ValueError."""
    output_path = tmp_path / "epiautogp_input.json"

    with pytest.raises(ValueError, match="nhsn_data_path is required"):
        convert_to_epiautogp_json(
            combined_training_data_path=sample_combined_training_data,
            nhsn_data_path=None,
            output_json_path=output_path,
            disease="COVID-19",
            location="CA",
            forecast_date=dt.date(2024, 9, 15),
            target="nhsn",
        )


def test_missing_data_raises_error(sample_combined_training_data, tmp_path):
    """Test that missing data raises ValueError."""
    output_path = tmp_path / "epiautogp_input.json"

    with pytest.raises(ValueError, match="No NSSP data found"):
        convert_to_epiautogp_json(
            combined_training_data_path=sample_combined_training_data,
            nhsn_data_path=None,
            output_json_path=output_path,
            disease="Influenza",  # Disease not in test data
            location="CA",
            forecast_date=dt.date(2024, 9, 15),
            target="nssp",
        )


def test_custom_logger(sample_nhsn_data, tmp_path, caplog):
    """Test that custom logger is used when provided."""
    import logging

    output_path = tmp_path / "epiautogp_input.json"
    custom_logger = logging.getLogger("test_custom_logger")

    with caplog.at_level(logging.INFO, logger="test_custom_logger"):
        convert_to_epiautogp_json(
            combined_training_data_path=tmp_path / "dummy.tsv",
            nhsn_data_path=sample_nhsn_data,
            output_json_path=output_path,
            disease="COVID-19",
            location="CA",
            forecast_date=dt.date(2024, 9, 21),
            target="nhsn",
            logger=custom_logger,
        )

    # Verify custom logger was used
    assert "Extracting NHSN data" in caplog.text
    assert "Saved EpiAutoGP input JSON" in caplog.text
