"""Test that model fits create separate data directories to avoid race conditions."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import datetime as dt

from pipelines.prep_data import process_and_save_loc_data


@pytest.fixture
def mock_data_components():
    """Create mock data components for testing."""
    return {
        "loc_abb": "CA",
        "disease": "COVID-19",
        "report_date": dt.date(2023, 1, 15),
        "first_training_date": dt.date(2023, 1, 1),
        "last_training_date": dt.date(2023, 1, 14),
    }


def test_process_and_save_loc_data_creates_model_specific_directory(mock_data_components):
    """Test that process_and_save_loc_data creates a model-specific data directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_run_dir = Path(temp_dir)
        model_name = "test_model"
        
        # Mock the required data components to avoid dependencies
        mock_nssp_data = MagicMock()
        mock_nssp_data.to_dict.return_value = {"date": ["2023-01-01"], "geo_value": ["CA"]}
        mock_nssp_data.filter.return_value = mock_nssp_data
        mock_nssp_data.with_columns.return_value = mock_nssp_data
        mock_nssp_data.pivot.return_value = mock_nssp_data
        mock_nssp_data.rename.return_value = mock_nssp_data
        mock_nssp_data.sort.return_value = mock_nssp_data
        
        mock_nhsn_data = MagicMock()
        mock_nhsn_data.to_dict.return_value = {"weekendingdate": ["2023-01-01"], "jurisdiction": ["CA"]}
        mock_nhsn_data.filter.return_value = mock_nhsn_data
        mock_nhsn_data.with_columns.return_value = mock_nhsn_data
        
        # Mock the data loading functions to avoid external dependencies
        import pipelines.prep_data as prep_data
        original_get_facility_level_nssp = prep_data.get_facility_level_nssp
        original_get_loc_level_nssp = prep_data.get_loc_level_nssp
        original_get_nhsn = prep_data.get_nhsn
        original_get_loc_pop = prep_data.get_loc_pop
        original_combine_surveillance_data = prep_data.combine_surveillance_data
        original_right_truncation_offset = prep_data.right_truncation_offset
        
        prep_data.get_facility_level_nssp = MagicMock(return_value=mock_nssp_data)
        prep_data.get_loc_level_nssp = MagicMock(return_value=mock_nssp_data)
        prep_data.get_nhsn = MagicMock(return_value=mock_nhsn_data)
        prep_data.get_loc_pop = MagicMock(return_value=[100000])
        prep_data.combine_surveillance_data = MagicMock(return_value=mock_nssp_data)
        prep_data.right_truncation_offset = 10
        
        mock_combined_data = MagicMock()
        mock_combined_data.write_csv = MagicMock()
        prep_data.combine_surveillance_data.return_value = mock_combined_data
        
        try:
            # Call the function with a model name
            process_and_save_loc_data(
                model_name=model_name,
                model_run_dir=model_run_dir,
                **mock_data_components
            )
            
            # Check that the model-specific directory was created
            model_data_dir = model_run_dir / "data" / model_name
            assert model_data_dir.exists(), f"Model-specific data directory {model_data_dir} was not created"
            
            # Check that the data file was created in the model-specific directory
            data_file = model_data_dir / "data_for_model_fit.json"
            assert data_file.exists(), f"Data file {data_file} was not created"
            
            # Verify the data file contains expected structure
            with open(data_file, 'r') as f:
                data = json.load(f)
                assert "nssp_training_data" in data
                assert "nhsn_training_data" in data
                assert "loc_pop" in data
                
        finally:
            # Restore original functions
            prep_data.get_facility_level_nssp = original_get_facility_level_nssp
            prep_data.get_loc_level_nssp = original_get_loc_level_nssp
            prep_data.get_nhsn = original_get_nhsn
            prep_data.get_loc_pop = original_get_loc_pop
            prep_data.combine_surveillance_data = original_combine_surveillance_data
            prep_data.right_truncation_offset = original_right_truncation_offset


def test_process_and_save_loc_data_fallback_shared_directory(mock_data_components):
    """Test that process_and_save_loc_data falls back to shared directory when model_name is None."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_run_dir = Path(temp_dir)
        
        # Mock the required data components to avoid dependencies
        mock_nssp_data = MagicMock()
        mock_nssp_data.to_dict.return_value = {"date": ["2023-01-01"], "geo_value": ["CA"]}
        mock_nssp_data.filter.return_value = mock_nssp_data
        mock_nssp_data.with_columns.return_value = mock_nssp_data
        mock_nssp_data.pivot.return_value = mock_nssp_data
        mock_nssp_data.rename.return_value = mock_nssp_data
        mock_nssp_data.sort.return_value = mock_nssp_data
        
        mock_nhsn_data = MagicMock()
        mock_nhsn_data.to_dict.return_value = {"weekendingdate": ["2023-01-01"], "jurisdiction": ["CA"]}
        mock_nhsn_data.filter.return_value = mock_nhsn_data
        mock_nhsn_data.with_columns.return_value = mock_nhsn_data
        
        # Mock the data loading functions to avoid external dependencies
        import pipelines.prep_data as prep_data
        original_get_facility_level_nssp = prep_data.get_facility_level_nssp
        original_get_loc_level_nssp = prep_data.get_loc_level_nssp
        original_get_nhsn = prep_data.get_nhsn
        original_get_loc_pop = prep_data.get_loc_pop
        original_combine_surveillance_data = prep_data.combine_surveillance_data
        original_right_truncation_offset = prep_data.right_truncation_offset
        
        prep_data.get_facility_level_nssp = MagicMock(return_value=mock_nssp_data)
        prep_data.get_loc_level_nssp = MagicMock(return_value=mock_nssp_data)
        prep_data.get_nhsn = MagicMock(return_value=mock_nhsn_data)
        prep_data.get_loc_pop = MagicMock(return_value=[100000])
        prep_data.combine_surveillance_data = MagicMock(return_value=mock_nssp_data)
        prep_data.right_truncation_offset = 10
        
        mock_combined_data = MagicMock()
        mock_combined_data.write_csv = MagicMock()
        prep_data.combine_surveillance_data.return_value = mock_combined_data
        
        try:
            # Call the function without a model name (should use fallback)
            process_and_save_loc_data(
                model_name=None,
                model_run_dir=model_run_dir,
                **mock_data_components
            )
            
            # Check that the shared directory was created
            shared_data_dir = model_run_dir / "data"
            assert shared_data_dir.exists(), f"Shared data directory {shared_data_dir} was not created"
            
            # Check that the data file was created in the shared directory
            data_file = shared_data_dir / "data_for_model_fit.json"
            assert data_file.exists(), f"Data file {data_file} was not created"
            
        finally:
            # Restore original functions
            prep_data.get_facility_level_nssp = original_get_facility_level_nssp
            prep_data.get_loc_level_nssp = original_get_loc_level_nssp
            prep_data.get_nhsn = original_get_nhsn
            prep_data.get_loc_pop = original_get_loc_pop
            prep_data.combine_surveillance_data = original_combine_surveillance_data
            prep_data.right_truncation_offset = original_right_truncation_offset


def test_multiple_models_create_separate_directories(mock_data_components):
    """Test that different model names create separate data directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_run_dir = Path(temp_dir)
        model_names = ["model_a", "model_b", "model_c"]
        
        # Mock the required data components to avoid dependencies
        mock_nssp_data = MagicMock()
        mock_nssp_data.to_dict.return_value = {"date": ["2023-01-01"], "geo_value": ["CA"]}
        mock_nssp_data.filter.return_value = mock_nssp_data
        mock_nssp_data.with_columns.return_value = mock_nssp_data
        mock_nssp_data.pivot.return_value = mock_nssp_data
        mock_nssp_data.rename.return_value = mock_nssp_data
        mock_nssp_data.sort.return_value = mock_nssp_data
        
        mock_nhsn_data = MagicMock()
        mock_nhsn_data.to_dict.return_value = {"weekendingdate": ["2023-01-01"], "jurisdiction": ["CA"]}
        mock_nhsn_data.filter.return_value = mock_nhsn_data
        mock_nhsn_data.with_columns.return_value = mock_nhsn_data
        
        # Mock the data loading functions to avoid external dependencies
        import pipelines.prep_data as prep_data
        original_get_facility_level_nssp = prep_data.get_facility_level_nssp
        original_get_loc_level_nssp = prep_data.get_loc_level_nssp
        original_get_nhsn = prep_data.get_nhsn
        original_get_loc_pop = prep_data.get_loc_pop
        original_combine_surveillance_data = prep_data.combine_surveillance_data
        original_right_truncation_offset = prep_data.right_truncation_offset
        
        prep_data.get_facility_level_nssp = MagicMock(return_value=mock_nssp_data)
        prep_data.get_loc_level_nssp = MagicMock(return_value=mock_nssp_data)
        prep_data.get_nhsn = MagicMock(return_value=mock_nhsn_data)
        prep_data.get_loc_pop = MagicMock(return_value=[100000])
        prep_data.combine_surveillance_data = MagicMock(return_value=mock_nssp_data)
        prep_data.right_truncation_offset = 10
        
        mock_combined_data = MagicMock()
        mock_combined_data.write_csv = MagicMock()
        prep_data.combine_surveillance_data.return_value = mock_combined_data
        
        try:
            # Create data for multiple models
            for model_name in model_names:
                process_and_save_loc_data(
                    model_name=model_name,
                    model_run_dir=model_run_dir,
                    **mock_data_components
                )
            
            # Check that separate directories were created for each model
            for model_name in model_names:
                model_data_dir = model_run_dir / "data" / model_name
                assert model_data_dir.exists(), f"Model-specific data directory {model_data_dir} was not created"
                
                data_file = model_data_dir / "data_for_model_fit.json"
                assert data_file.exists(), f"Data file {data_file} was not created"
                
        finally:
            # Restore original functions
            prep_data.get_facility_level_nssp = original_get_facility_level_nssp
            prep_data.get_loc_level_nssp = original_get_loc_level_nssp
            prep_data.get_nhsn = original_get_nhsn
            prep_data.get_loc_pop = original_get_loc_pop
            prep_data.combine_surveillance_data = original_combine_surveillance_data
            prep_data.right_truncation_offset = original_right_truncation_offset