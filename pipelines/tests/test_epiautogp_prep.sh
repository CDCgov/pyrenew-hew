#!/bin/bash

# Test script for EpiAutoGP data preparation
# This script tests the first step of the EpiAutoGP workflow:
# 1. Generate test data
# 2. Convert to EpiAutoGP JSON format

BASE_DIR=pipelines/tests/epiautogp_test_output
LOCATIONS=(US CA MT DC)
DISEASES=(Influenza COVID-19 RSV)
TARGETS=(nssp nhsn)

echo "TEST-MODE: Running EpiAutoGP data preparation test with base directory $BASE_DIR"

if [ -d "$BASE_DIR" ]; then
	if [ "$1" = "--force" ]; then
		rm -r "$BASE_DIR"
	else
		# make the user delete the directory, to avoid accidental deletes of
		# test output
		echo "TEST-MODE FAIL: test output directory $BASE_DIR already exists. Delete the directory and re-run the test, or run with the --force flag".
		echo "DETAILS: The test output directory persists after each run to allow the user to examine output. It must be deleted and recreated at the start of each new end-to-end test run to ensure that old output does not compromise test validity."
		exit 1
	fi
fi

echo "TEST-MODE: Generating test data..."
uv run python pipelines/generate_test_data.py "$BASE_DIR"

if [ "$?" -ne 0 ]; then
	echo "TEST-MODE FAIL: Generating test data failed"
	exit 1
else
	echo "TEST-MODE: Finished generating test data"
fi

# Create output directory for JSON files
FORECAST_DATE="2024-12-21"
OUTPUT_DIR="$BASE_DIR/${FORECAST_DATE}_epiautogp_inputs"
mkdir -p "$OUTPUT_DIR"

echo "TEST-MODE: Converting surveillance data to EpiAutoGP JSON format"

# Test conversion for each location, disease, and target combination
for location in "${LOCATIONS[@]}"; do
	for disease in "${DISEASES[@]}"; do
		for target in "${TARGETS[@]}"; do
			# Construct paths
			if [ "$target" = "nhsn" ]; then
				NHSN_PATH="$BASE_DIR/private_data/nhsn_test_data/${disease}_${location}.parquet"

				# Check if NHSN data exists for this combination
				if [ ! -f "$NHSN_PATH" ]; then
					echo "TEST-MODE: Skipping NHSN conversion for $disease, $location (no data file)"
					continue
				fi
			fi

			# Create location-specific output directory
			LOC_OUTPUT_DIR="$OUTPUT_DIR/${location}"
			mkdir -p "$LOC_OUTPUT_DIR"

			# For NSSP, test both counts and percentages with epiweekly data
			if [ "$target" = "nssp" ]; then
				# Test 1: NSSP epiweekly counts
				echo "TEST-MODE: Converting $target epiweekly counts for $disease, $location"
				OUTPUT_JSON="$LOC_OUTPUT_DIR/epiautogp_input_${target}_epiweekly_counts_${disease// /-}.json"
				uv run python pipelines/tests/test_epiautogp_prep_script.py \
					"$target" "$disease" "$location" "$BASE_DIR" "$OUTPUT_JSON" "epiweekly" "false"

				if [ "$?" -ne 0 ]; then
					echo "TEST-MODE FAIL: Conversion failed for $target epiweekly counts, $disease, $location"
					exit 1
				else
					echo "TEST-MODE: Successfully converted $target epiweekly counts for $disease, $location"
				fi

				# Test 2: NSSP epiweekly percentage
				echo "TEST-MODE: Converting $target epiweekly percentage for $disease, $location"
				OUTPUT_JSON="$LOC_OUTPUT_DIR/epiautogp_input_${target}_epiweekly_percentage_${disease// /-}.json"
				uv run python pipelines/tests/test_epiautogp_prep_script.py \
					"$target" "$disease" "$location" "$BASE_DIR" "$OUTPUT_JSON" "epiweekly" "true"

				if [ "$?" -ne 0 ]; then
					echo "TEST-MODE FAIL: Conversion failed for $target epiweekly percentage, $disease, $location"
					exit 1
				else
					echo "TEST-MODE: Successfully converted $target epiweekly percentage for $disease, $location"
				fi
			else
				# For NHSN, just test epiweekly counts (no percentage option)
				echo "TEST-MODE: Converting $target epiweekly data for $disease, $location"
				OUTPUT_JSON="$LOC_OUTPUT_DIR/epiautogp_input_${target}_epiweekly_${disease// /-}.json"
				uv run python pipelines/tests/test_epiautogp_prep_script.py \
					"$target" "$disease" "$location" "$BASE_DIR" "$OUTPUT_JSON" "epiweekly" "false"

				if [ "$?" -ne 0 ]; then
					echo "TEST-MODE FAIL: Conversion failed for $target data, $disease, $location"
					exit 1
				else
					echo "TEST-MODE: Successfully converted $target data for $disease, $location"
				fi
			fi
		done
	done
done

echo "TEST-MODE: All conversions complete."
echo "TEST-MODE: JSON files are in $OUTPUT_DIR"

# Verify some JSON files were created
JSON_COUNT=$(find "$OUTPUT_DIR" -name "*.json" | wc -l)
echo "TEST-MODE: Created $JSON_COUNT JSON files"

if [ "$JSON_COUNT" -eq 0 ]; then
	echo "TEST-MODE FAIL: No JSON files were created"
	exit 1
fi

echo "TEST-MODE: All finished successfully."
echo "TEST-MODE: You can examine the output files in $OUTPUT_DIR"
