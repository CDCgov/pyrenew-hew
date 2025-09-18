import tempfile
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipelines.fit_pyrenew_model import fit_and_save_model
from pipelines.forecast_pyrenew import record_rng_key


def test_fit_and_save_model_default_rng_key():
    """Test that fit_and_save_model uses the default RNG key when none is provided."""
    # Mock the dependencies
    with (
        patch("pipelines.fit_pyrenew_model.PyrenewHEWData") as mock_data,
        patch(
            "pipelines.fit_pyrenew_model.build_pyrenew_hew_model_from_dir"
        ) as mock_build,
        patch("pipelines.fit_pyrenew_model.Path") as mock_path,
        patch("builtins.open", create=True) as mock_open,
        patch("pickle.dump") as mock_pickle,
    ):
        # Setup mocks
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        mock_data.from_json.return_value = MagicMock()
        mock_path.return_value.__truediv__.return_value.mkdir = MagicMock()

        # Call the function without rng_key
        fit_and_save_model(model_run_dir="test_dir", model_name="test_model")

        # Check that model.run was called with the default RNG key converted to JAX key
        mock_model.run.assert_called_once()
        call_args = mock_model.run.call_args

        # The rng_key should be a JAX PRNGKey created from the default seed
        import jax

        expected_rng_key = jax.random.key(12345)
        assert call_args[1]["rng_key"].dtype == expected_rng_key.dtype


def test_fit_and_save_model_custom_rng_key():
    """Test that fit_and_save_model uses a custom RNG key when provided."""
    # Mock the dependencies
    with (
        patch("pipelines.fit_pyrenew_model.PyrenewHEWData") as mock_data,
        patch(
            "pipelines.fit_pyrenew_model.build_pyrenew_hew_model_from_dir"
        ) as mock_build,
        patch("pipelines.fit_pyrenew_model.Path") as mock_path,
        patch("builtins.open", create=True) as mock_open,
        patch("pickle.dump") as mock_pickle,
    ):
        # Setup mocks
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        mock_data.from_json.return_value = MagicMock()
        mock_path.return_value.__truediv__.return_value.mkdir = MagicMock()

        # Call the function with custom rng_key
        custom_rng_key = 54321
        fit_and_save_model(
            model_run_dir="test_dir", model_name="test_model", rng_key=custom_rng_key
        )

        # Check that model.run was called with the custom RNG key converted to JAX key
        mock_model.run.assert_called_once()
        call_args = mock_model.run.call_args

        # The rng_key should be a JAX PRNGKey created from the custom seed
        import jax

        expected_rng_key = jax.random.key(custom_rng_key)
        assert call_args[1]["rng_key"].dtype == expected_rng_key.dtype


def test_fit_and_save_model_invalid_rng_key():
    """Test that fit_and_save_model raises ValueError for invalid RNG key types."""
    with pytest.raises(ValueError, match="rng_key must be an integer"):
        fit_and_save_model(
            model_run_dir="test_dir", model_name="test_model", rng_key="invalid"
        )


def test_record_rng_key():
    """Test that record_rng_key properly saves RNG key to metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_run_dir = Path(temp_dir)
        rng_key = 98765

        # Call the function
        record_rng_key(model_run_dir, rng_key)

        # Check that metadata file was created with correct content
        metadata_file = model_run_dir / "metadata.toml"
        assert metadata_file.exists()

        with open(metadata_file, "rb") as f:
            metadata = tomllib.load(f)

        assert metadata["rng_key"] == rng_key


def test_record_rng_key_updates_existing_metadata():
    """Test that record_rng_key updates existing metadata without overwriting other fields."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model_run_dir = Path(temp_dir)
        metadata_file = model_run_dir / "metadata.toml"

        # Create existing metadata
        existing_metadata = {"existing_field": "existing_value"}
        with open(metadata_file, "wb") as f:
            import tomli_w

            tomli_w.dump(existing_metadata, f)

        rng_key = 11111

        # Call the function
        record_rng_key(model_run_dir, rng_key)

        # Check that both existing field and new rng_key are present
        with open(metadata_file, "rb") as f:
            metadata = tomllib.load(f)

        assert metadata["rng_key"] == rng_key
        assert metadata["existing_field"] == "existing_value"
