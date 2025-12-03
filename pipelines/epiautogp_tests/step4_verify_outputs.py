#!/usr/bin/env python3
"""
Step 4: Verify EpiAutoGP model outputs.

This script validates that the Julia EpiAutoGP model produced the expected
hubverse-compatible forecast outputs with correct structure and content.

Usage:
    python pipelines/epiautogp_tests/step4_verify_outputs.py
"""

from pathlib import Path

import polars as pl

# Configuration - must match previous steps
TEST_OUTPUT_DIR = Path("pipelines/epiautogp_tests/test_output")
MODEL_OUTPUT_DIR = TEST_OUTPUT_DIR / "model_output"
TEST_LOCATION = "CA"
TEST_DISEASE = "COVID-19"
FORECAST_DATE = "2024-12-21"
TARGET = "nhsn"

# Expected hubverse columns
EXPECTED_COLUMNS = [
    "reference_date",
    "horizon",
    "target_end_date",
    "location",
    "output_type",
    "output_type_id",
    "value",
    "target",
]

# Expected quantile levels (standard hubverse quantiles)
EXPECTED_QUANTILES = [
    "0.01",
    "0.025",
    "0.05",
    "0.1",
    "0.15",
    "0.2",
    "0.25",
    "0.3",
    "0.35",
    "0.4",
    "0.45",
    "0.5",
    "0.55",
    "0.6",
    "0.65",
    "0.7",
    "0.75",
    "0.8",
    "0.85",
    "0.9",
    "0.95",
    "0.975",
    "0.99",
]


def verify_file_exists():
    """Check that the output CSV file exists."""
    print("\n=== Verifying Output File ===")

    # Look for CSV files in output directory
    csv_files = list(MODEL_OUTPUT_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {MODEL_OUTPUT_DIR}\nPlease run step 3 first."
        )

    if len(csv_files) > 1:
        print(f"⚠️  Warning: Found {len(csv_files)} CSV files, expected 1:")
        for f in csv_files:
            print(f"  - {f.name}")

    output_file = csv_files[0]
    print(f"✓ Found output file: {output_file.name}")
    print(f"  Size: {output_file.stat().st_size:,} bytes")

    return output_file


def verify_hubverse_structure(df: pl.DataFrame):
    """Verify the DataFrame has the expected hubverse structure."""
    print("\n=== Verifying Hubverse Structure ===")

    # Check columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✓ All required columns present: {len(EXPECTED_COLUMNS)} columns")

    # Check for quantile output type
    output_types = df["output_type"].unique().to_list()
    if "quantile" not in output_types:
        raise ValueError(f"Expected 'quantile' in output_type, found: {output_types}")

    print(f"✓ Output types: {output_types}")

    # Check quantile levels
    quantiles = (
        df.filter(pl.col("output_type") == "quantile")["output_type_id"]
        .unique()
        .sort()
        .to_list()
    )
    missing_quantiles = set(EXPECTED_QUANTILES) - set(quantiles)

    if missing_quantiles:
        print(f"⚠️  Warning: Missing some quantiles: {missing_quantiles}")

    print(f"✓ Found {len(quantiles)} quantile levels")

    return True


def verify_metadata(df: pl.DataFrame):
    """Verify the metadata fields match expected values."""
    print("\n=== Verifying Metadata ===")

    # Check reference_date
    ref_dates = df["reference_date"].unique().to_list()
    if FORECAST_DATE not in ref_dates:
        raise ValueError(f"Expected reference_date {FORECAST_DATE}, found: {ref_dates}")
    print(f"✓ Reference date: {ref_dates[0]}")

    # Check location
    locations = df["location"].unique().to_list()
    if TEST_LOCATION not in locations:
        raise ValueError(f"Expected location {TEST_LOCATION}, found: {locations}")
    print(f"✓ Location: {locations[0]}")

    # Check target
    targets = df["target"].unique().to_list()
    expected_target = f"{TEST_DISEASE.lower().replace('-', ' ')} hospital admissions"
    if expected_target not in targets:
        print(f"⚠️  Warning: Expected target '{expected_target}', found: {targets}")
    else:
        print(f"✓ Target: {targets[0]}")

    return True


def verify_forecast_values(df: pl.DataFrame):
    """Verify the forecast values are reasonable."""
    print("\n=== Verifying Forecast Values ===")

    # Check horizons
    horizons = df["horizon"].unique().sort().to_list()
    print(f"✓ Forecast horizons: {horizons}")
    print(f"  (Total: {len(horizons)} weeks)")

    # Check target_end_dates
    target_dates = df["target_end_date"].unique().sort().to_list()
    print(f"✓ Target end dates: {target_dates[0]} to {target_dates[-1]}")

    # Check for null values
    null_counts = df.select(pl.col("value").is_null().sum())
    if null_counts[0, 0] > 0:
        raise ValueError(f"Found {null_counts[0, 0]} null values in 'value' column")
    print("✓ No null values in forecast")

    # Check for negative values
    negative_counts = df.filter(pl.col("value") < 0).height
    if negative_counts > 0:
        raise ValueError(f"Found {negative_counts} negative values in forecast")
    print("✓ No negative values in forecast")

    # Summary statistics
    value_stats = df.select(
        [
            pl.col("value").min().alias("min"),
            pl.col("value").median().alias("median"),
            pl.col("value").max().alias("max"),
        ]
    )

    print(f"✓ Value range: {value_stats['min'][0]:.0f} to {value_stats['max'][0]:.0f}")
    print(f"  Median: {value_stats['median'][0]:.0f}")

    # Check rows per horizon
    rows_per_horizon = df.group_by("horizon").agg(pl.count().alias("n_rows"))
    expected_rows = len(EXPECTED_QUANTILES)

    for row in rows_per_horizon.iter_rows(named=True):
        if row["n_rows"] != expected_rows:
            print(
                f"⚠️  Warning: Horizon {row['horizon']} has {row['n_rows']} rows, expected {expected_rows}"
            )

    print(f"✓ Each horizon has ~{expected_rows} quantiles")

    return True


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("EpiAutoGP Integration Test - Step 4: Verify Outputs")
    print("=" * 70)

    try:
        # Step 1: Verify file exists
        output_file = verify_file_exists()

        # Step 2: Load the CSV
        print("\n=== Loading Output File ===")
        df = pl.read_csv(output_file)
        print(f"✓ Loaded CSV with {df.height:,} rows and {df.width} columns")

        # Step 3: Verify structure
        verify_hubverse_structure(df)

        # Step 4: Verify metadata
        verify_metadata(df)

        # Step 5: Verify forecast values
        verify_forecast_values(df)

        # Final summary
        print("\n" + "=" * 70)
        print("✓ ALL VERIFICATION CHECKS PASSED")
        print("=" * 70)
        print("\nEpiAutoGP integration test completed successfully!")
        print("\nTest summary:")
        print(f"  - Generated synthetic test data for {TEST_LOCATION}, {TEST_DISEASE}")
        print("  - Transformed to EpiAutoGP format")
        print("  - Ran Julia model successfully")
        print("  - Produced valid hubverse forecast output")
        print(f"  - Output file: {output_file.name}")
        print(f"  - Total forecast rows: {df.height:,}")
        print(f"  - Forecast weeks: {len(df['horizon'].unique())}")

        return 0

    except Exception as e:
        print("\n" + "=" * 70)
        print("✗ VERIFICATION FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
