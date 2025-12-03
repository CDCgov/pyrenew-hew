#!/usr/bin/env python3
"""
Integration test for EpiAutoGP Julia model execution.

This test validates the complete workflow:
1. Generate synthetic test data
2. Transform to PyRenew format
3. Convert to EpiAutoGP JSON format
4. Run Julia model
5. Verify outputs
"""

import datetime as dt
import shutil
from pathlib import Path

# Import library functions from generate_test_data_lib
from pipelines.generate_test_data_lib import (
    create_default_param_estimates,
    simulate_data_from_bootstrap,
)

# Configuration for minimal test
TEST_OUTPUT_DIR = Path("pipelines/epiautogp_tests/test_output")
N_TRAINING_WEEKS = 16
N_TRAINING_DAYS = N_TRAINING_WEEKS * 7
N_FORECAST_WEEKS = 4
N_FORECAST_DAYS = 7 * N_FORECAST_WEEKS
N_NSSP_SITES = 3
N_WW_SITES = 3
TEST_LOCATION = "CA"
TEST_DISEASE = "COVID-19"
MAX_TRAIN_DATE_STR = "2024-12-21"
MAX_TRAIN_DATE = dt.datetime.strptime(MAX_TRAIN_DATE_STR, "%Y-%m-%d").date()


# ============================================================================
# Test Functions
# ============================================================================


def setup_test_environment():
    """Clean up and recreate test output directory."""
    if TEST_OUTPUT_DIR.exists():
        print(f"Removing existing test directory: {TEST_OUTPUT_DIR}")
        shutil.rmtree(TEST_OUTPUT_DIR)

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created test directory: {TEST_OUTPUT_DIR}")


def create_test_param_estimates():
    """Create parameter estimates for test data generation using library function."""
    param_estimates = create_default_param_estimates(
        states_to_simulate=[TEST_LOCATION],
        diseases_to_simulate=[TEST_DISEASE],
        max_train_date_str=MAX_TRAIN_DATE_STR,
        max_train_date=MAX_TRAIN_DATE,
    )

    # Save parameter estimates
    param_estimates_dir = TEST_OUTPUT_DIR / "private_data" / "prod_param_estimates"
    param_estimates_dir.mkdir(parents=True, exist_ok=True)
    param_estimates.write_parquet(param_estimates_dir / "prod.parquet")

    print(f"Created parameter estimates: {param_estimates_dir / 'prod.parquet'}")
    return param_estimates


def generate_test_data(param_estimates):
    """Generate synthetic test data using library function simulate_data_from_bootstrap."""
    print("\n=== Step 1: Generating Test Data ===")

    bootstrap_dir = TEST_OUTPUT_DIR / "bootstrap_private_data"
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    print(f"Simulating data for {TEST_LOCATION}, {TEST_DISEASE}")
    print(f"Training period: {N_TRAINING_WEEKS} weeks ({N_TRAINING_DAYS} days)")
    print(f"Max training date: {MAX_TRAIN_DATE}")

    simulated_data = simulate_data_from_bootstrap(
        n_training_days=N_TRAINING_DAYS,
        max_train_date=MAX_TRAIN_DATE,
        n_nssp_sites=N_NSSP_SITES,
        n_training_weeks=N_TRAINING_WEEKS,
        bootstrap_private_data_dir=bootstrap_dir,
        param_estimates=param_estimates.lazy(),
        n_forecast_days=N_FORECAST_DAYS,
        n_ww_sites=N_WW_SITES,
        states_to_simulate=[TEST_LOCATION],
        diseases_to_simulate=[TEST_DISEASE],
    )

    print("✓ Test data generated successfully")
    print(f"  - ED visits data: {len(simulated_data['observed_ed_visits'])} rows")
    print(
        f"  - Hospital admissions: {len(simulated_data['observed_hospital_admissions'])} rows"
    )
    print(f"  - Wastewater data: {len(simulated_data['site_level_log_ww_conc'])} rows")

    # Check that data_for_model_fit.json was created
    data_json_path = bootstrap_dir / TEST_LOCATION / "data" / "data_for_model_fit.json"
    if data_json_path.exists():
        print(f"✓ PyRenew data file created: {data_json_path}")
    else:
        raise FileNotFoundError(f"Expected data file not found: {data_json_path}")

    return simulated_data, data_json_path


def main():
    """Run the integration test."""
    print("=" * 70)
    print("EpiAutoGP Integration Test - Step 1: Test Data Generation")
    print("=" * 70)

    try:
        # Setup
        setup_test_environment()

        # Create parameter estimates
        param_estimates = create_test_param_estimates()

        # Generate test data
        simulated_data, data_json_path = generate_test_data(param_estimates)

        print("\n" + "=" * 70)
        print("✓ Step 1 COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nTest output directory: {TEST_OUTPUT_DIR}")
        print(f"PyRenew data JSON: {data_json_path}")
        print("\nNext steps:")
        print("  2. Transform to EpiAutoGP format")
        print("  3. Run Julia model")
        print("  4. Verify outputs")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ Step 1 FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
