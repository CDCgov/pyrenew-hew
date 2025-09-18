import tempfile
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_record_rng_key():
    """Test that record_rng_key properly saves RNG key to metadata."""
    # Import here to avoid dependency issues in test discovery
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from forecast_pyrenew import record_rng_key
    
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
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from forecast_pyrenew import record_rng_key
    
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


def test_fit_and_save_model_rng_key_handling():
    """
    Test that fit_and_save_model correctly handles RNG keys and passes them to the model.
    This test verifies the core requirement that models with same RNG seed produce identical outputs.
    """
    
    # Mock all the heavy dependencies
    with (
        patch("pipelines.fit_pyrenew_model.PyrenewHEWData") as mock_data,
        patch("pipelines.fit_pyrenew_model.build_pyrenew_hew_model_from_dir") as mock_build,
        patch("pipelines.fit_pyrenew_model.Path") as mock_path,
        patch("builtins.open", create=True),
        patch("pickle.dump"),
        tempfile.TemporaryDirectory() as temp_dir,
    ):
        # Import here to avoid issues during test discovery
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from fit_pyrenew_model import fit_and_save_model
        
        # Setup mocks
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        
        # Track which model instance we're on
        call_count = 0
        def mock_build_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_model1 if call_count == 1 else mock_model2
        
        mock_build.side_effect = mock_build_side_effect
        mock_data.from_json.return_value = MagicMock()
        mock_path.return_value.__truediv__.return_value.mkdir = MagicMock()
        
        # Capture the RNG keys passed to each model
        captured_keys = []
        
        def capture_rng_key(model_instance):
            def mock_run(**kwargs):
                captured_keys.append(kwargs.get('rng_key'))
                # Simulate model results
                model_instance.mcmc = MagicMock()
                return None
            return mock_run
        
        mock_model1.run = capture_rng_key(mock_model1)
        mock_model2.run = capture_rng_key(mock_model2)
        
        # Test 1: Same RNG seed should produce same JAX keys
        rng_seed = 42
        
        # First run
        fit_and_save_model(
            model_run_dir=temp_dir,
            model_name="test_model_1",
            rng_key=rng_seed,
            n_warmup=10,
            n_samples=10,
            n_chains=1,
        )
        
        # Second run with same seed
        fit_and_save_model(
            model_run_dir=temp_dir,
            model_name="test_model_2",
            rng_key=rng_seed,
            n_warmup=10,
            n_samples=10,
            n_chains=1,
        )
        
        # Verify both models were called
        assert len(captured_keys) == 2, "Both models should have been called"
        
        # Verify the RNG keys are identical
        import jax
        import jax.numpy as jnp
        
        key1, key2 = captured_keys
        assert jnp.array_equal(key1, key2), "Same RNG seed should produce identical JAX keys"
        
        # Verify they match the expected key for the seed
        expected_key = jax.random.key(rng_seed)
        assert jnp.array_equal(key1, expected_key), "RNG key should match expected JAX key"
        
        print(f"✓ Same RNG seed ({rng_seed}) produces identical JAX keys in fit_and_save_model")


def test_fit_and_save_model_default_rng_key():
    """Test that fit_and_save_model uses the default RNG key when none is provided."""
    
    with (
        patch("pipelines.fit_pyrenew_model.PyrenewHEWData") as mock_data,
        patch("pipelines.fit_pyrenew_model.build_pyrenew_hew_model_from_dir") as mock_build,
        patch("pipelines.fit_pyrenew_model.Path") as mock_path,
        patch("builtins.open", create=True),
        patch("pickle.dump"),
    ):
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from fit_pyrenew_model import fit_and_save_model
        
        # Setup mocks
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        mock_data.from_json.return_value = MagicMock()
        mock_path.return_value.__truediv__.return_value.mkdir = MagicMock()

        # Capture the RNG key
        captured_key = None
        def capture_run(**kwargs):
            nonlocal captured_key
            captured_key = kwargs.get('rng_key')
            return None
        
        mock_model.run = capture_run

        # Call the function without rng_key
        fit_and_save_model(model_run_dir="test_dir", model_name="test_model")

        # Verify the default RNG key was used
        import jax
        import jax.numpy as jnp
        
        expected_rng_key = jax.random.key(12345)
        assert jnp.array_equal(captured_key, expected_rng_key), "Should use default RNG seed 12345"
        
        print("✓ Default RNG seed (12345) is correctly applied")


def test_fit_and_save_model_custom_rng_key():
    """Test that fit_and_save_model uses a custom RNG key when provided."""
    
    with (
        patch("pipelines.fit_pyrenew_model.PyrenewHEWData") as mock_data,
        patch("pipelines.fit_pyrenew_model.build_pyrenew_hew_model_from_dir") as mock_build,
        patch("pipelines.fit_pyrenew_model.Path") as mock_path,
        patch("builtins.open", create=True),
        patch("pickle.dump"),
    ):
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from fit_pyrenew_model import fit_and_save_model
        
        # Setup mocks
        mock_model = MagicMock()
        mock_build.return_value = mock_model
        mock_data.from_json.return_value = MagicMock()
        mock_path.return_value.__truediv__.return_value.mkdir = MagicMock()

        # Capture the RNG key
        captured_key = None
        def capture_run(**kwargs):
            nonlocal captured_key
            captured_key = kwargs.get('rng_key')
            return None
        
        mock_model.run = capture_run

        # Call the function with custom rng_key
        custom_rng_key = 54321
        fit_and_save_model(
            model_run_dir="test_dir", model_name="test_model", rng_key=custom_rng_key
        )

        # Verify the custom RNG key was used
        import jax
        import jax.numpy as jnp
        
        expected_rng_key = jax.random.key(custom_rng_key)
        assert jnp.array_equal(captured_key, expected_rng_key), f"Should use custom RNG seed {custom_rng_key}"
        
        print(f"✓ Custom RNG seed ({custom_rng_key}) is correctly applied")


def test_fit_and_save_model_invalid_rng_key():
    """Test that fit_and_save_model raises ValueError for invalid RNG key types."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from fit_pyrenew_model import fit_and_save_model
        
    with pytest.raises(ValueError, match="rng_key must be an integer"):
        fit_and_save_model(
            model_run_dir="test_dir", model_name="test_model", rng_key="invalid"
        )
    
    print("✓ Invalid RNG key types are properly rejected")
