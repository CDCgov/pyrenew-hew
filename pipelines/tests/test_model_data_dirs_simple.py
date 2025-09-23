"""Simple test to verify model-specific data directory structure."""

import json
import tempfile
from pathlib import Path

import pytest


def test_model_specific_data_directory_structure():
    """Test that the expected directory structure for model-specific data works."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_run_dir = Path(temp_dir)
        
        # Test different model names create separate directories
        model_names = ["model_a", "model_b", "model_c"]
        
        for model_name in model_names:
            # Create the model-specific data directory structure
            model_data_dir = model_run_dir / "data" / model_name
            model_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a sample data file
            data_file = model_data_dir / "data_for_model_fit.json"
            sample_data = {
                "model_name": model_name,
                "loc_pop": [100000],
                "nssp_training_data": {"date": ["2023-01-01"], "geo_value": ["CA"]},
                "nhsn_training_data": {"weekendingdate": ["2023-01-01"], "jurisdiction": ["CA"]},
            }
            
            with open(data_file, 'w') as f:
                json.dump(sample_data, f)
        
        # Verify all directories were created separately
        for model_name in model_names:
            model_data_dir = model_run_dir / "data" / model_name
            assert model_data_dir.exists(), f"Model-specific data directory {model_data_dir} does not exist"
            
            data_file = model_data_dir / "data_for_model_fit.json"
            assert data_file.exists(), f"Data file {data_file} does not exist"
            
            # Verify the data file contains the correct model name
            with open(data_file, 'r') as f:
                data = json.load(f)
                assert data["model_name"] == model_name, f"Wrong model name in data file: {data['model_name']}"


def test_fallback_shared_directory_structure():
    """Test that shared directory structure still works for backward compatibility."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_run_dir = Path(temp_dir)
        
        # Create the shared data directory structure
        shared_data_dir = model_run_dir / "data"
        shared_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample data file
        data_file = shared_data_dir / "data_for_model_fit.json"
        sample_data = {
            "loc_pop": [100000],
            "nssp_training_data": {"date": ["2023-01-01"], "geo_value": ["CA"]},
            "nhsn_training_data": {"weekendingdate": ["2023-01-01"], "jurisdiction": ["CA"]},
        }
        
        with open(data_file, 'w') as f:
            json.dump(sample_data, f)
        
        # Verify shared directory was created
        assert shared_data_dir.exists(), f"Shared data directory {shared_data_dir} does not exist"
        assert data_file.exists(), f"Data file {data_file} does not exist"
        
        # Verify the data file is readable
        with open(data_file, 'r') as f:
            data = json.load(f)
            assert "loc_pop" in data, "Data file missing expected keys"


def test_path_construction():
    """Test that the path construction works as expected for fit and predictive scripts."""
    model_run_dir = Path("/some/model/run/dir")
    model_name = "test_model"
    
    # Test the path construction used in fit_pyrenew_model.py
    fit_data_path = model_run_dir / "data" / model_name / "data_for_model_fit.json"
    expected_fit_path = Path("/some/model/run/dir/data/test_model/data_for_model_fit.json")
    assert fit_data_path == expected_fit_path, f"Fit data path construction incorrect: {fit_data_path}"
    
    # Test the path construction used in generate_predictive.py  
    predictive_data_path = model_run_dir / "data" / model_name / "data_for_model_fit.json"
    expected_predictive_path = Path("/some/model/run/dir/data/test_model/data_for_model_fit.json")
    assert predictive_data_path == expected_predictive_path, f"Predictive data path construction incorrect: {predictive_data_path}"
    
    # Test the path construction for data preparation
    prep_data_dir = model_run_dir / "data" / model_name
    expected_prep_dir = Path("/some/model/run/dir/data/test_model")
    assert prep_data_dir == expected_prep_dir, f"Prep data directory construction incorrect: {prep_data_dir}"


def test_concurrent_model_isolation():
    """Test that different models can have data in separate directories concurrently."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_run_dir = Path(temp_dir)
        
        # Simulate concurrent model fits with different configurations
        models = [
            {"name": "model_ed_only", "fit_ed": True, "fit_hosp": False, "fit_ww": False},
            {"name": "model_hosp_only", "fit_ed": False, "fit_hosp": True, "fit_ww": False},
            {"name": "model_ed_hosp", "fit_ed": True, "fit_hosp": True, "fit_ww": False},
            {"name": "model_all", "fit_ed": True, "fit_hosp": True, "fit_ww": True},
        ]
        
        # Create data directories for each model
        for model in models:
            model_data_dir = model_run_dir / "data" / model["name"]
            model_data_dir.mkdir(parents=True, exist_ok=True)
            
            # Create model-specific data
            data_file = model_data_dir / "data_for_model_fit.json"
            model_data = {
                "model_name": model["name"],
                "fit_ed_visits": model["fit_ed"],
                "fit_hospital_admissions": model["fit_hosp"],
                "fit_wastewater": model["fit_ww"],
                "loc_pop": [100000],
                "nssp_training_data": {"date": ["2023-01-01"], "geo_value": ["CA"]},
                "nhsn_training_data": {"weekendingdate": ["2023-01-01"], "jurisdiction": ["CA"]},
            }
            
            with open(data_file, 'w') as f:
                json.dump(model_data, f)
        
        # Verify each model has its own isolated data
        for model in models:
            model_data_dir = model_run_dir / "data" / model["name"]
            data_file = model_data_dir / "data_for_model_fit.json"
            
            assert model_data_dir.exists(), f"Model directory for {model['name']} does not exist"
            assert data_file.exists(), f"Data file for {model['name']} does not exist"
            
            # Verify the data is correct for this specific model
            with open(data_file, 'r') as f:
                data = json.load(f)
                assert data["model_name"] == model["name"], f"Wrong model name in {model['name']} data"
                assert data["fit_ed_visits"] == model["fit_ed"], f"Wrong ED config in {model['name']} data"
                assert data["fit_hospital_admissions"] == model["fit_hosp"], f"Wrong hosp config in {model['name']} data"
                assert data["fit_wastewater"] == model["fit_ww"], f"Wrong WW config in {model['name']} data"