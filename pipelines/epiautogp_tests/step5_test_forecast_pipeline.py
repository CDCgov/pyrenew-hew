#!/usr/bin/env python3
"""
Step 5: Test the forecast_epiautogp.py pipeline components.

This script uses the test data generated in previous steps to test the
core functions from forecast_epiautogp.py (conversion, fitting, post-processing).

Usage:
    python pipelines/epiautogp_tests/step5_test_forecast_pipeline.py
"""

from pathlib import Path

import polars as pl

from pipelines.fit_epiautogp_model import fit_and_save_model
from pipelines.prep_epiautogp_data import convert_to_epiautogp_format

# Configuration - must match previous steps
TEST_OUTPUT_DIR = Path("pipelines/epiautogp_tests/test_output")
TEST_LOCATION = "CA"
TEST_DISEASE = "COVID-19"
MAX_TRAIN_DATE_STR = "2024-12-21"

# Paths
DATA_JSON_PATH = (
    TEST_OUTPUT_DIR
    / "bootstrap_private_data"
    / TEST_LOCATION
    / "data"
    / "data_for_model_fit.json"
)
EPIAUTOGP_INPUT_JSON = TEST_OUTPUT_DIR / "epiautogp_input.json"
MODEL_RUN_DIR = TEST_OUTPUT_DIR / "forecast_test"

# Model parameters (minimal for testing)
MODEL_PARAMS = {
    "n_forecast_weeks": 4,
    "n_particles": 12,
    "n_mcmc": 50,
    "n_hmc": 25,
    "n_forecast_draws": 1000,
    "nthreads": 1,
}


def test_data_conversion():
    """Test converting PyRenew data to EpiAutoGP format."""
    print("\n" + "=" * 70)
    print("Testing Data Conversion")
    print("=" * 70)

    try:
        epiautogp_data = convert_to_epiautogp_format(
            input_json_path=str(DATA_JSON_PATH),
            output_json_path=str(EPIAUTOGP_INPUT_JSON),
            target="nhsn",
            disease=TEST_DISEASE,
            location=TEST_LOCATION,
            forecast_date=MAX_TRAIN_DATE_STR,
            nowcast_reports_path=None,
        )

        print("✓ Conversion successful")
        print(f"  - Historical dates: {len(epiautogp_data['dates'])}")
        print(f"  - Reports: {len(epiautogp_data['reports'])}")
        print(f"  - Target: {epiautogp_data['target']}")
        return True

    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_fitting():
    """Test fitting the EpiAutoGP model."""
    print("\n" + "=" * 70)
    print("Testing Model Fitting")
    print("=" * 70)

    try:
        MODEL_RUN_DIR.mkdir(parents=True, exist_ok=True)

        fit_and_save_model(
            model_run_dir=MODEL_RUN_DIR,
            model_name="epiautogp",
            epiautogp_input_json=EPIAUTOGP_INPUT_JSON,
            **MODEL_PARAMS,
        )

        # Check that output was created
        model_output_dir = MODEL_RUN_DIR / "epiautogp"
        csv_files = list(model_output_dir.glob("*.csv"))

        if len(csv_files) == 0:
            print(f"✗ No CSV output found in {model_output_dir}")
            return False

        print("✓ Model fitting successful")
        print(f"  - Output directory: {model_output_dir}")
        print(f"  - Generated files: {len(csv_files)}")
        for f in csv_files:
            print(f"    - {f.name}")

        return True

    except Exception as e:
        print(f"✗ Model fitting failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_output_validation():
    """Test validating the model outputs."""
    print("\n" + "=" * 70)
    print("Testing Output Validation")
    print("=" * 70)

    try:
        model_output_dir = MODEL_RUN_DIR / "epiautogp"
        csv_files = list(model_output_dir.glob("*.csv"))

        if len(csv_files) == 0:
            print("✗ No CSV files to validate")
            return False

        forecast_csv = csv_files[0]
        print(f"Validating: {forecast_csv.name}")

        # Read and validate structure
        df = pl.read_csv(forecast_csv)

        required_cols = [
            "reference_date",
            "target",
            "horizon",
            "target_end_date",
            "location",
            "output_type",
            "output_type_id",
            "value",
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
            return False

        print("✓ Output validation successful")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {df.columns}")
        print(f"  - Reference date: {df['reference_date'].unique().to_list()}")
        print(f"  - Location: {df['location'].unique().to_list()}")
        print(f"  - Output types: {df['output_type'].unique().to_list()}")

        # Check quantile levels if present
        if "quantile" in df["output_type"].to_list():
            quantiles = df.filter(pl.col("output_type") == "quantile")[
                "output_type_id"
            ].unique()
            print(f"  - Quantile levels: {len(quantiles)}")

        return True

    except Exception as e:
        print(f"✗ Output validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_parquet_conversion():
    """Test converting CSV to parquet format."""
    print("\n" + "=" * 70)
    print("Testing Parquet Conversion")
    print("=" * 70)

    try:
        model_output_dir = MODEL_RUN_DIR / "epiautogp"
        csv_files = list(model_output_dir.glob("*.csv"))

        if len(csv_files) == 0:
            print("✗ No CSV files to convert")
            return False

        forecast_csv = csv_files[0]
        parquet_path = model_output_dir / "hubverse_table.parquet"

        # Convert CSV to parquet
        df = pl.read_csv(forecast_csv)
        df.write_parquet(parquet_path)

        # Verify parquet file
        df_check = pl.read_parquet(parquet_path)

        print("✓ Parquet conversion successful")
        print(f"  - Parquet file: {parquet_path.name}")
        print(f"  - Rows in parquet: {len(df_check)}")
        print(f"  - File size: {parquet_path.stat().st_size / 1024:.2f} KB")

        return True

    except Exception as e:
        print(f"✗ Parquet conversion failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all forecast pipeline tests."""
    print("=" * 70)
    print("EpiAutoGP Forecast Pipeline Test - Step 5")
    print("=" * 70)
    print("\nTest configuration:")
    print(f"  - Location: {TEST_LOCATION}")
    print(f"  - Disease: {TEST_DISEASE}")
    print(f"  - Forecast date: {MAX_TRAIN_DATE_STR}")
    print(f"  - Output directory: {TEST_OUTPUT_DIR}")

    # Check prerequisites
    if not DATA_JSON_PATH.exists():
        print(f"\n✗ ERROR: Input data not found: {DATA_JSON_PATH}")
        print("\nPlease run Step 1 first:")
        print("  python pipelines/epiautogp_tests/step1_generate_test_data.py")
        return 1

    # Run tests
    results = {
        "Data Conversion": test_data_conversion(),
        "Model Fitting": test_model_fitting(),
        "Output Validation": test_output_validation(),
        "Parquet Conversion": test_parquet_conversion(),
    }

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n" + "=" * 70)
        print("✓ Step 5 COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print("\nAll forecast pipeline components working correctly!")
        print(f"Test outputs saved to: {MODEL_RUN_DIR}")
        return 0
    else:
        print("\n" + "=" * 70)
        print("✗ Step 5 FAILED")
        print("=" * 70)
        print("\nSome tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit(main())
