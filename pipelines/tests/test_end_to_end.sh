#!/bin/bash

BASE_DIR=pipelines/tests/end_to_end_test_output
LOCATIONS=(US CA MT DC)
DISEASES=(Influenza COVID-19)

echo "TEST-MODE: Running forecast_pyrenew.py in test mode with base directory $BASE_DIR"

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

uv run python pipelines/generate_test_data.py "$BASE_DIR"

if [ "$?" -ne 0 ]; then
	echo "TEST-MODE FAIL: Generating test data failed"
	exit 1
else
	echo "TEST-MODE: Finished generating test data"
fi

echo "TEST-MODE: Running data preparation for all locations, and diseases"

for location in "${LOCATIONS[@]}"; do
	for disease in "${DISEASES[@]}"; do
		echo "TEST-MODE: Running data preparation for $disease, $location"
		bash pipelines/tests/test_prep_data.sh "$BASE_DIR" "$disease" "$location"
		if [ "$?" -ne 0 ]; then
			echo "TEST-MODE FAIL: Data preparation failed"
			exit 1
		else
			echo "TEST-MODE: Finished data preparation for location $location, disease $disease."
		fi
	done
done

echo "TEST-MODE: Finished data preparation for all locations and diseases."

echo "TEST-MODE: Running Timeseries forecasting pipeline for all locations, and diseases"

for location in "${LOCATIONS[@]}"; do
	for disease in "${DISEASES[@]}"; do
		echo "TEST-MODE: Running Timeseries forecasting pipeline for $disease, $location"
		bash pipelines/tests/test_ts_fit.sh "$BASE_DIR" "$disease" "$location" "e"
		if [ "$?" -ne 0 ]; then
			echo "TEST-MODE FAIL: Timeseries forecasting pipeline failed"
			echo "TEST-MODE: Cleanup: removing temporary directories"
			exit 1
		else
			echo "TEST-MODE: Finished Timeseries forecasting pipeline for location $location, disease $disease."
		fi
	done
done
echo "TEST-MODE: Finished Timeseries forecasting pipeline for all locations and diseases."

echo "TEST-MODE: Running Pyrenew forecasting pipelines for various signals, locations, and diseases"
for location in "${LOCATIONS[@]}"; do
	for model in {,h}{,e}{,w}; do
		for disease in "${DISEASES[@]}"; do

			if [[ ($model == *w* && ($disease == "Influenza" || $location == "US")) || $model == "w" ]]; then
				echo "TEST-MODE: Skipping forecasting pipeline for $model, $disease, $location. " \
					"W-only models, US-level wastewater models, and Influenza wastewater models " \
					"are not yet supported."
			else
				echo "TEST-MODE: Running forecasting pipeline for $model, $disease, $location"
				bash pipelines/tests/test_pyrenew_fit.sh "$BASE_DIR" "$disease" "$location" "$model"
			fi
			if [ "$?" -ne 0 ]; then
				echo "TEST-MODE FAIL: Forecasting/postprocessing/scoring pipeline failed"
				echo "TEST-MODE: Cleanup: removing temporary directories"
				exit 1
			else
				echo "TEST-MODE: Finished forecasting/postprocessing/scoring pipeline for location $location."
			fi
		done
	done
done

echo "TEST-MODE: All pipeline runs complete."

echo "TEST-MODE: Running batch postprocess..."

python pipelines/postprocess_forecast_batches.py \
	"$BASE_DIR/2024-12-21_forecasts"

if [ "$?" -ne 0 ]; then
	echo "TEST-MODE FAIL: Batch postprocess failed."
	exit 1
else
	echo "TEST-MODE: Batch postprocess succeeded."
fi

echo "TEST-MODE: All finished successfully."
