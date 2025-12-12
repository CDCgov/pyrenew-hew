#!/bin/bash

# End-to-end test for EpiAutoGP forecasting pipeline
# Tests multiple locations and targets (weekly NHSN and weekly percentage NSSP)
#
# Usage:
#   bash pipelines/tests/test_epiautogp_end_to_end.sh [--force]
#
# Options:
#   --force    Delete existing test output directory

set -e # Exit on any error

BASE_DIR=pipelines/tests/epiautogp_end_to_end_test_output
LOCATIONS=(US CA MT DC)
DISEASES=(COVID-19) # Start with COVID-19 for testing
FORECAST_DATE=2024-12-21

echo "========================================"
echo "EpiAutoGP End-to-End Test"
echo "========================================"
echo "Testing configurations:"
echo "  Locations: ${LOCATIONS[*]}"
echo "  Diseases: ${DISEASES[*]}"
echo "  Targets: NHSN (weekly), NSSP (weekly percentage)"
echo "  Forecast date: $FORECAST_DATE"
echo ""

# Step 0: Clean up previous run
if [ -d "$BASE_DIR" ]; then
	if [[ "$*" == *"--force"* ]]; then
		echo "Cleaning previous test output..."
		rm -rf "$BASE_DIR"
	else
		echo "ERROR: Test output directory exists: $BASE_DIR"
		echo "Delete it or run with --force flag"
		exit 1
	fi
fi

# Step 1: Generate test data
echo "========================================="
echo "Step 1: Generating test data"
echo "========================================="
uv run python pipelines/generate_test_data.py "$BASE_DIR"

if [ "$?" -ne 0 ]; then
	echo "TEST-MODE FAIL: Generating test data failed"
	exit 1
else
	echo "✓ Test data generated"
	echo ""
fi

# Step 2: Run EpiAutoGP forecasting pipeline for all locations and targets
echo "========================================="
echo "Step 2: Running EpiAutoGP forecasts"
echo "========================================="

for location in "${LOCATIONS[@]}"; do
	for disease in "${DISEASES[@]}"; do
		echo ""
		echo "Testing $disease, $location"
		echo "-----------------------------------------"

		# Test 1: Weekly NHSN (hospital admissions)
		echo "  [1/2] Running weekly NHSN forecast..."
		bash pipelines/tests/test_epiautogp_fit.sh \
			"$BASE_DIR" \
			"$disease" \
			"$location" \
			"nhsn" \
			"epiweekly" \
			"false"

		if [ "$?" -ne 0 ]; then
			echo "TEST-MODE FAIL: NHSN forecast failed for $disease, $location"
			exit 1
		else
			echo "  ✓ Weekly NHSN forecast complete"
		fi

		# Test 2: Weekly NSSP percentage (ED visits as percentage)
		echo "  [2/2] Running weekly NSSP percentage forecast..."
		bash pipelines/tests/test_epiautogp_fit.sh \
			"$BASE_DIR" \
			"$disease" \
			"$location" \
			"nssp" \
			"epiweekly" \
			"true"

		if [ "$?" -ne 0 ]; then
			echo "TEST-MODE FAIL: NSSP percentage forecast failed for $disease, $location"
			exit 1
		else
			echo "  ✓ Weekly NSSP percentage forecast complete"
		fi

		echo "✓ All forecasts complete for $disease, $location"
	done
done

echo ""
echo "========================================="
echo "Step 3: Verifying outputs"
echo "========================================="

# Count expected outputs (2 targets × number of locations × number of diseases)
EXPECTED_MODELS=$((2 * ${#LOCATIONS[@]} * ${#DISEASES[@]}))
ACTUAL_MODELS=$(find "$BASE_DIR/${FORECAST_DATE}_forecasts" -type d -name "epiautogp_*" | wc -l)

echo "Expected models: $EXPECTED_MODELS"
echo "Actual models: $ACTUAL_MODELS"

if [ "$ACTUAL_MODELS" -eq "$EXPECTED_MODELS" ]; then
	echo "✓ Output verification passed"
else
	echo "ERROR: Output count mismatch"
	echo "Directory structure:"
	find "$BASE_DIR/${FORECAST_DATE}_forecasts" -type d -name "epiautogp_*"
	exit 1
fi

echo ""
echo "========================================="
echo "All Tests Passed!"
echo "========================================="
echo "Test output saved to: $BASE_DIR"
echo ""
