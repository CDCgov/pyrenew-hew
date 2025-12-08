"""
Tests for EpiAutoGP data conversion functions.
"""

import datetime as dt
import json
import logging

import polars as pl
import pytest

from pipelines.epiautogp.prep_epiautogp_data import convert_to_epiautogp_json


class TestEpiAutoGPDataConversion:
    """Test suite for EpiAutoGP data conversion functions."""

    @pytest.fixture
    def sample_epiweekly_data(self, tmp_path):
        """Create a sample epiweekly_combined_training_data.tsv file with NSSP data."""
        # Create weekly NSSP data in long format
        data = {
            "date": [
                "2024-09-01",
                "2024-09-01",
                "2024-09-08",
                "2024-09-08",
                "2024-09-15",
                "2024-09-15",
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
            ".value": [50, 450, 60, 540, 70, 630],
            "lab_site_index": ["NA", "NA", "NA", "NA", "NA", "NA"],
        }
        df = pl.DataFrame(data)
        file_path = tmp_path / "epiweekly_combined_training_data.tsv"
        df.write_csv(file_path, separator="\t")
        return file_path

    @pytest.fixture
    def sample_data_for_model_fit_nssp(self, tmp_path):
        """Create a sample data_for_model_fit.json file with NHSN data."""
        data = {
            "loc_pop": 39512223,
            "right_truncation_offset": 0,
            "nwss_training_data": None,
            "nssp_training_data": {
                "date": ["2024-09-01", "2024-09-02", "2024-09-03"],
                "geo_value": ["CA", "CA", "CA"],
                "data_type": ["train", "train", "train"],
                "observed_ed_visits": [50, 60, 70],
                "other_ed_visits": [450, 540, 630],
            },
            "nhsn_training_data": {
                "jurisdiction": ["CA", "CA", "CA"],
                "weekendingdate": ["2024-09-07", "2024-09-14", "2024-09-21"],
                "hospital_admissions": [45, 52, 38],
                "data_type": ["train", "train", "train"],
            },
            "nhsn_step_size": 7,
            "nssp_step_size": 1,
            "nwss_step_size": 1,
        }
        file_path = tmp_path / "data_for_model_fit.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    @pytest.fixture
    def sample_data_for_model_fit_nhsn(self, tmp_path):
        """Create a sample data_for_model_fit.json file with NHSN data."""
        data = {
            "loc_pop": 39512223,
            "right_truncation_offset": 0,
            "nwss_training_data": None,
            "nssp_training_data": {
                "date": ["2024-09-01", "2024-09-08", "2024-09-15"],
                "geo_value": ["CA", "CA", "CA"],
                "data_type": ["train", "train", "train"],
                "observed_ed_visits": [50, 60, 70],
                "other_ed_visits": [450, 540, 630],
            },
            "nhsn_training_data": {
                "jurisdiction": ["CA", "CA", "CA"],
                "weekendingdate": ["2024-09-07", "2024-09-14", "2024-09-21"],
                "hospital_admissions": [45, 52, 38],
                "data_type": ["train", "train", "train"],
            },
            "nhsn_step_size": 7,
            "nssp_step_size": 1,
            "nwss_step_size": 1,
        }
        file_path = tmp_path / "data_for_model_fit.json"
        with open(file_path, "w") as f:
            json.dump(data, f)
        return file_path

    def test_convert_nssp_to_epiautogp_json(
        self, sample_epiweekly_data, sample_data_for_model_fit_nssp, tmp_path
    ):
        """Test conversion of NSSP data to EpiAutoGP JSON format."""
        output_path = tmp_path / "epiautogp_input.json"

        convert_to_epiautogp_json(
            target="nssp",
            data_for_model_fit_path=sample_data_for_model_fit_nssp,
            epiweekly_data_path=sample_epiweekly_data,
            output_json_path=output_path,
            disease="COVID-19",
            location="CA",
            forecast_date=dt.date(2024, 9, 15),
        )

        # Verify file was created
        assert output_path.exists()

        # Read and verify JSON content
        with open(output_path) as f:
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

    def test_convert_nhsn_to_epiautogp_json(
        self, sample_data_for_model_fit_nhsn, sample_epiweekly_data, tmp_path
    ):
        """Test conversion of NHSN data to EpiAutoGP JSON format."""
        output_path = tmp_path / "epiautogp_input.json"

        convert_to_epiautogp_json(
            target="nhsn",
            data_for_model_fit_path=sample_data_for_model_fit_nhsn,
            epiweekly_data_path=sample_epiweekly_data,
            output_json_path=output_path,
            disease="COVID-19",
            location="CA",
            forecast_date=dt.date(2024, 9, 21),
        )

        # Verify file was created
        assert output_path.exists()

        # Read and verify JSON content
        with open(output_path) as f:
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

    def test_convert_with_nowcast_data(
        self, sample_epiweekly_data, sample_data_for_model_fit_nssp, tmp_path
    ):
        """Test conversion with nowcast dates and reports."""
        output_path = tmp_path / "epiautogp_input.json"

        nowcast_dates = [dt.date(2024, 9, 8), dt.date(2024, 9, 15)]
        nowcast_reports = [[58.0, 60.0, 62.0], [68.0, 70.0, 72.0]]

        convert_to_epiautogp_json(
            target="nssp",
            data_for_model_fit_path=sample_data_for_model_fit_nssp,
            epiweekly_data_path=sample_epiweekly_data,
            output_json_path=output_path,
            disease="COVID-19",
            location="CA",
            forecast_date=dt.date(2024, 9, 15),
            nowcast_dates=nowcast_dates,
            nowcast_reports=nowcast_reports,
        )

        with open(output_path) as f:
            result = json.load(f)

        assert result["nowcast_dates"] == ["2024-09-08", "2024-09-15"]
        assert result["nowcast_reports"] == [[58.0, 60.0, 62.0], [68.0, 70.0, 72.0]]

    def test_invalid_target_raises_error(
        self, sample_epiweekly_data, sample_data_for_model_fit_nssp, tmp_path
    ):
        """Test that invalid target raises ValueError."""
        output_path = tmp_path / "epiautogp_input.json"

        with pytest.raises(ValueError, match="target must be 'nssp' or 'nhsn'"):
            convert_to_epiautogp_json(
                target="invalid_target",
                data_for_model_fit_path=sample_data_for_model_fit_nssp,
                epiweekly_data_path=sample_epiweekly_data,
                output_json_path=output_path,
                disease="COVID-19",
                location="CA",
                forecast_date=dt.date(2024, 9, 15),
            )

    def test_nhsn_without_data_raises_error(self, sample_epiweekly_data, tmp_path):
        """Test that NHSN target without NHSN data in JSON raises ValueError."""
        output_path = tmp_path / "epiautogp_input.json"

        # Create data_for_model_fit.json without NHSN data
        data = {
            "loc_pop": 39512223,
            "right_truncation_offset": 0,
            "nwss_training_data": None,
            "nssp_training_data": {
                "date": ["2024-09-01"],
                "geo_value": ["CA"],
                "data_type": ["train"],
                "observed_ed_visits": [50],
                "other_ed_visits": [450],
            },
            "nhsn_training_data": None,  # No NHSN data
            "nhsn_step_size": 7,
            "nssp_step_size": 1,
            "nwss_step_size": 1,
        }
        data_file = tmp_path / "data_for_model_fit.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="No NHSN training data found"):
            convert_to_epiautogp_json(
                target="nhsn",
                data_for_model_fit_path=data_file,
                epiweekly_data_path=sample_epiweekly_data,
                output_json_path=output_path,
                disease="COVID-19",
                location="CA",
                forecast_date=dt.date(2024, 9, 15),
            )

    def test_missing_nssp_data_raises_error(self, tmp_path):
        """Test that missing NSSP data raises ValueError."""
        output_path = tmp_path / "epiautogp_input.json"

        # Create empty epiweekly TSV (no NSSP data for the location/disease)
        empty_data = {
            "date": ["2024-09-01"],
            "geo_value": ["NY"],  # Different location
            "disease": ["COVID-19"],
            "data_type": ["train"],
            ".variable": ["observed_ed_visits"],
            ".value": [50],
            "lab_site_index": ["NA"],
        }
        df = pl.DataFrame(empty_data)
        epiweekly_file = tmp_path / "epiweekly_empty.tsv"
        df.write_csv(epiweekly_file, separator="\t")

        # Create data_for_model_fit.json
        data = {
            "loc_pop": 39512223,
            "right_truncation_offset": 0,
            "nwss_training_data": None,
            "nssp_training_data": None,
            "nhsn_training_data": {
                "jurisdiction": ["CA"],
                "weekendingdate": ["2024-09-07"],
                "hospital_admissions": [45],
                "data_type": ["train"],
            },
            "nhsn_step_size": 7,
            "nssp_step_size": 1,
            "nwss_step_size": 1,
        }
        data_file = tmp_path / "data_for_model_fit.json"
        with open(data_file, "w") as f:
            json.dump(data, f)

        with pytest.raises(ValueError, match="No NSSP data found"):
            convert_to_epiautogp_json(
                target="nssp",
                data_for_model_fit_path=data_file,
                epiweekly_data_path=epiweekly_file,
                output_json_path=output_path,
                disease="COVID-19",
                location="CA",
                forecast_date=dt.date(2024, 9, 15),
            )

    def test_custom_logger(
        self, sample_data_for_model_fit_nhsn, sample_epiweekly_data, tmp_path, caplog
    ):
        """Test that custom logger is used when provided."""
        output_path = tmp_path / "epiautogp_input.json"
        custom_logger = logging.getLogger("test_custom_logger")

        with caplog.at_level(logging.INFO, logger="test_custom_logger"):
            convert_to_epiautogp_json(
                target="nhsn",
                data_for_model_fit_path=sample_data_for_model_fit_nhsn,
                epiweekly_data_path=sample_epiweekly_data,
                output_json_path=output_path,
                disease="COVID-19",
                location="CA",
                forecast_date=dt.date(2024, 9, 21),
                logger=custom_logger,
            )

        # Verify custom logger was used
        assert "Extracting NHSN data" in caplog.text
        assert "Saved EpiAutoGP input JSON" in caplog.text
