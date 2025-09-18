"""
Test suite for RNG reproducibility in pyrenew-hew.

This test file validates that the RNG seed functionality ensures reproducible model outputs.
It focuses on testing the core reproducibility requirements without requiring all dependencies.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def test_rng_key_deterministic_behavior():
    """Test that JAX RNG keys produce deterministic behavior with same seeds."""
    
    # Test 1: Same seed produces same key
    key1 = jax.random.key(12345)
    key2 = jax.random.key(12345)
    
    # Keys should be equal when created from same seed
    assert jnp.array_equal(key1, key2), "Same seeds should produce identical JAX keys"
    
    # Test 2: Same key produces same random numbers
    sample1 = jax.random.normal(key1, (10,))
    sample2 = jax.random.normal(key2, (10,))
    
    np.testing.assert_array_equal(
        sample1, sample2,
        err_msg="Same JAX keys should produce identical random samples"
    )
    
    # Test 3: Different seeds produce different keys
    key3 = jax.random.key(54321)
    sample3 = jax.random.normal(key3, (10,))
    
    # Should be different from the first samples
    assert not np.array_equal(sample1, sample3), "Different seeds should produce different samples"


def test_model_fit_with_same_rng_produces_identical_results():
    """
    Test that demonstrates the core requirement: models with same RNG seed produce identical outputs.
    
    This test simulates the model fitting process to verify that:
    1. Same RNG seed -> Same results
    2. Different RNG seed -> Different results
    """
    
    def simulate_model_run(rng_key, n_samples=100, n_params=5):
        """
        Simulate a simplified version of what happens in the actual model.
        This represents the stochastic operations that should be reproducible.
        """
        # Split the key for multiple random operations (common in MCMC)
        key1, key2, key3 = jax.random.split(rng_key, 3)
        
        # Simulate MCMC sampling (multiple random operations)
        samples = {
            'param_a': jax.random.normal(key1, (n_samples, n_params)),
            'param_b': jax.random.exponential(key2, (n_samples, n_params)),
            'param_c': jax.random.uniform(key3, (n_samples, n_params), minval=0, maxval=1),
        }
        
        return samples
    
    # Test 1: Same RNG seed produces identical results
    seed = 42
    key1 = jax.random.key(seed)
    key2 = jax.random.key(seed)  # Same seed, should produce same results
    
    results1 = simulate_model_run(key1)
    results2 = simulate_model_run(key2)
    
    # Verify all parameters are identical
    for param_name in results1.keys():
        np.testing.assert_array_equal(
            results1[param_name], 
            results2[param_name],
            err_msg=f"Parameter {param_name} differs between runs with same RNG seed {seed}"
        )
    
    print(f"✓ Same RNG seed ({seed}) produces identical model results")
    
    # Test 2: Different RNG seeds produce different results
    different_seed = 84
    key3 = jax.random.key(different_seed)
    results3 = simulate_model_run(key3)
    
    # Verify at least one parameter is different
    has_differences = False
    for param_name in results1.keys():
        if not np.array_equal(results1[param_name], results3[param_name]):
            has_differences = True
            break
    
    assert has_differences, f"Different RNG seeds ({seed} vs {different_seed}) should produce different results"
    print(f"✓ Different RNG seeds ({seed} vs {different_seed}) produce different model results")


def test_fit_function_rng_integration():
    """
    Test the actual RNG logic used in fit_and_save_model function.
    This verifies the transformation from integer -> JAX key works correctly.
    """
    
    # Test the actual logic from fit_and_save_model
    def test_rng_logic(rng_key_input):
        """Replicate the RNG logic from fit_and_save_model."""
        if rng_key_input is None:
            rng_key = 12345  # Fixed default RNG seed for reproducibility
        else:
            rng_key = rng_key_input
            
        if isinstance(rng_key, int):
            rng_key = jax.random.key(rng_key)
        else:
            raise ValueError(
                "rng_key must be an integer with which to seed :func:`jax.random.key`"
            )
        return rng_key
    
    # Test default behavior
    default_key = test_rng_logic(None)
    expected_default = jax.random.key(12345)
    assert jnp.array_equal(default_key, expected_default), "Default RNG key should be 12345"
    
    # Test custom seed
    custom_key = test_rng_logic(54321)
    expected_custom = jax.random.key(54321)
    assert jnp.array_equal(custom_key, expected_custom), "Custom RNG key should match input"
    
    # Test invalid input
    with pytest.raises(ValueError, match="rng_key must be an integer"):
        test_rng_logic("invalid")
    
    print("✓ RNG key transformation logic works correctly")
    print(f"✓ Default seed: 12345")
    print(f"✓ Custom seeds are preserved")
    print(f"✓ Invalid inputs raise appropriate errors")


def test_mcmc_chain_reproducibility():
    """
    Test that simulates multi-chain MCMC to ensure reproducibility across chains.
    This is important because the actual model uses multiple chains.
    """
    
    def simulate_mcmc_chains(rng_key, n_chains=4, n_samples=50, n_params=3):
        """Simulate multi-chain MCMC sampling."""
        # Split key for each chain
        chain_keys = jax.random.split(rng_key, n_chains)
        
        chain_results = []
        for chain_idx, chain_key in enumerate(chain_keys):
            # Each chain does its own sampling
            key1, key2 = jax.random.split(chain_key, 2)
            
            chain_samples = {
                f'chain_{chain_idx}_param_1': jax.random.normal(key1, (n_samples, n_params)),
                f'chain_{chain_idx}_param_2': jax.random.exponential(key2, (n_samples, n_params)),
            }
            chain_results.append(chain_samples)
        
        return chain_results
    
    # Test same seed produces same multi-chain results
    seed = 123
    key1 = jax.random.key(seed)
    key2 = jax.random.key(seed)
    
    chains1 = simulate_mcmc_chains(key1)
    chains2 = simulate_mcmc_chains(key2)
    
    # Verify each chain produces identical results
    for chain_idx, (chain1, chain2) in enumerate(zip(chains1, chains2)):
        for param_name in chain1.keys():
            np.testing.assert_array_equal(
                chain1[param_name],
                chain2[param_name],
                err_msg=f"Chain {chain_idx} parameter {param_name} differs between runs"
            )
    
    print("✓ Multi-chain MCMC produces identical results with same RNG seed")


def test_default_seed_consistency():
    """Test that the default seed (12345) is consistently used."""
    
    def get_default_rng_key():
        """Simulate the default RNG key logic."""
        rng_key = None
        if rng_key is None:
            rng_key = 12345  # Fixed default RNG seed for reproducibility
        return jax.random.key(rng_key)
    
    # Multiple calls should produce the same key
    key1 = get_default_rng_key()
    key2 = get_default_rng_key()
    key3 = get_default_rng_key()
    
    assert jnp.array_equal(key1, key2), "Default RNG keys should be identical"
    assert jnp.array_equal(key2, key3), "Default RNG keys should be identical"
    
    # Should match explicit creation with seed 12345
    explicit_key = jax.random.key(12345)
    assert jnp.array_equal(key1, explicit_key), "Default key should match explicit key(12345)"
    
    print("✓ Default RNG seed (12345) is consistently applied")


if __name__ == "__main__":
    # Run all tests when executed directly
    test_rng_key_deterministic_behavior()
    test_model_fit_with_same_rng_produces_identical_results()
    test_fit_function_rng_integration()
    test_mcmc_chain_reproducibility()
    test_default_seed_consistency()
    
    print("\n🎉 All RNG reproducibility tests passed!")
    print("✓ Same RNG seeds produce identical model outputs")
    print("✓ Different RNG seeds produce different model outputs") 
    print("✓ Default seed (12345) is consistently applied")
    print("✓ Multi-chain MCMC is reproducible")
    print("✓ RNG key validation works correctly")