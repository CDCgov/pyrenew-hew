#!/bin/bash

# Integration test for EpiAutoGP pipeline
# Tests the complete workflow following the same structure as test_end_to_end.sh
# 1. Generate test data
# 2. Run EpiAutoGP forecast pipeline
# 3. Verify outputs
#
# Usage:
#   bash pipelines/tests/test_epiautogp_prep.sh [--force]
#
# Options:
#   --force    Delete existing test output directory

set -e # Exit on any error

BASE_DIR=pipelines/tests/epiautogp_test_output
LOCATION=CA
DISEASE=COVID-19
TARGET=nhsn
FORECAST_DATE=2024-12-21
# Convert disease to lowercase for directory name
DISEASE_LOWER=$(echo "$DISEASE" | tr '[:upper:]' '[:lower:]')

echo "========================================"
echo "EpiAutoGP Integration Test"
echo "========================================"
echo "Location: $LOCATION"
echo "Disease: $DISEASE"
echo "Target: $TARGET"
echo "Forecast date: $FORECAST_DATE"
echo ""

# Step 0: Clean up previous run
if [ -d "$BASE_DIR" ]; then
	if [[ "$*" == *"--force"* ]]; then
		echo "[0/4] Cleaning previous test output..."
		rm -rf "$BASE_DIR"
	else
		echo "ERROR: Test output directory exists: $BASE_DIR"
		echo "Delete it or run with --force flag"
		exit 1
	fi
fi

# Step 1: Generate test data
echo "[1/4] Generating test data..."
uv run python pipelines/generate_test_data.py "$BASE_DIR"
echo "✓ Test data generated"
echo ""

# Step 2: Run data preparation (equivalent to forecast_epiautogp.py setup)
echo "[2/4] Running EpiAutoGP pipeline..."
uv run python -c "
from pathlib import Path
from pipelines.epiautogp.forecast_epiautogp import main

# Run the complete pipeline
# Output follows end-to-end test pattern: BASE_DIR/FORECAST_DATE_forecasts/
main(
    disease='$DISEASE',
    report_date='$FORECAST_DATE',
    loc='$LOCATION',
    facility_level_nssp_data_dir=Path('$BASE_DIR/private_data/nssp_etl_gold'),
    state_level_nssp_data_dir=Path('$BASE_DIR/private_data/nssp_state_level_gold'),
    param_data_dir=Path('$BASE_DIR/private_data/prod_param_estimates'),
    output_dir=Path('$BASE_DIR/${FORECAST_DATE}_forecasts'),
    n_training_days=90,
    n_forecast_days=28,
    target='$TARGET',
    frequency='epiweekly',
    use_percentage=False,
    eval_data_path=Path('$BASE_DIR/private_data/nssp_state_level_gold/$FORECAST_DATE.parquet'),
    n_forecast_weeks=4,
    n_particles=2,  # Small for testing
    n_mcmc=2,       # Small for testing
    n_hmc=2,        # Small for testing
    n_forecast_draws=100,  # Small for testing
)
"
if [ $? -ne 0 ]; then
	echo "ERROR: Pipeline failed"
	exit 1
fi
echo "✓ Pipeline complete"
echo ""

# Step 3: Verify outputs exist
echo "[3/4] Verifying pipeline outputs..."

# Debug: Show what was created
if [ ! -d "$BASE_DIR/${FORECAST_DATE}_forecasts" ]; then
	echo "ERROR: Forecasts directory does not exist: $BASE_DIR/${FORECAST_DATE}_forecasts"
	exit 1
fi

echo "Checking forecasts directory structure..."
ls -la "$BASE_DIR/${FORECAST_DATE}_forecasts/"

# Find the model run directory (lowercase disease name in directory)
DISEASE_LOWER=$(echo "$DISEASE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
MODEL_BATCH_DIR=$(find "$BASE_DIR/${FORECAST_DATE}_forecasts" -maxdepth 1 -type d -name "*${DISEASE_LOWER}*r_*" 2>/dev/null | head -1)
if [ -z "$MODEL_BATCH_DIR" ]; then
	echo "ERROR: Model batch directory not found"
	echo "Expected pattern containing: ${DISEASE_LOWER}"
	echo "In directory: $BASE_DIR/${FORECAST_DATE}_forecasts/"
	exit 1
fi

echo "Found model batch directory: $MODEL_BATCH_DIR"

MODEL_RUN_DIR="$MODEL_BATCH_DIR/model_runs/$LOCATION"
MODEL_NAME="epiautogp_$TARGET"
MODEL_DIR="$MODEL_RUN_DIR/$MODEL_NAME"

# Determine the target suffix for file naming
if [ "$TARGET" = "nhsn" ]; then
	TARGET_SUFFIX="h"
else
	TARGET_SUFFIX="e"
fi

# Check for required files
REQUIRED_FILES=(
	"$MODEL_DIR/data/combined_training_data.tsv"
	"$MODEL_DIR/data/epiweekly_combined_training_data.tsv"
	"$MODEL_DIR/epiweekly_epiautogp_samples_${TARGET_SUFFIX}.parquet"
	"$MODEL_DIR/samples.parquet"
	"$MODEL_DIR/ci.parquet"
	"$MODEL_DIR/hubverse_table.parquet"
)

for file in "${REQUIRED_FILES[@]}"; do
	if [ ! -f "$file" ]; then
		echo "ERROR: Missing required file: $file"
		exit 1
	fi
done

echo "✓ All expected output files exist"
echo ""

# Step 4: Summary
echo "[4/4] Test summary"
echo "========================================"
echo "Test output directory: $BASE_DIR"
echo "Model directory: $MODEL_DIR"
echo ""
echo "Generated files:"
ls -lh "$MODEL_DIR"/*.parquet 2>/dev/null || true
echo ""
echo "✓ Integration test PASSED"
echo "========================================"
