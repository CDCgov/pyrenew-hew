#!/bin/bash

BASE_DIR=pipelines/tests/end_to_end_test_output
echo "TEST-MODE: Running forecast_loc.py in test mode with base directory $BASE_DIR"

if [ -d $BASE_DIR ]; then
	if [ $1 = "--force" ]; then
		rm -r $BASE_DIR
	else
		# make the user delete the directory, to avoid accidental deletes of
		# test output
		echo "TEST-MODE FAIL: test output directory $BASE_DIR already exists. Delete the directory and re-run the test, or run with the --force flag".
		echo "DETAILS: The test output directory persists after each run to allow the user to examine output. It must be deleted and recreated at the start of each new end-to-end test run to ensure that old output does not compromise test validity."
		exit 1
	fi
fi

uv run python pipelines/generate_test_data.py

if [ $? -ne 0 ]; then
	echo "TEST-MODE FAIL: Generating test data failed"
	exit 1
else
	echo "TEST-MODE: Finished generating test data"
fi
echo "TEST-MODE: Running forecasting pipelines for various signals, locations, and diseases"
for location in US CA MT; do
	for model in {,h}{,e}{,w}; do
		for disease in Influenza COVID-19; do

			if [[ ($model == *w* && ($disease == "Influenza" || $location == "US")) || $model == "w" ]]; then
				echo "TEST-MODE: Skipping forecasting pipeline for $model, $disease, $location. " \
					"W-only models, US-level wastewater models, and Influenza wastewater models " \
					"are not yet supported."
			else
				echo "TEST-MODE: Running forecasting pipeline for $model, $disease, $location"
				bash pipelines/tests/test_fit.sh $BASE_DIR $disease $location $model
			fi
			if [ $? -ne 0 ]; then
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
	$BASE_DIR/2024-12-21_forecasts \
	$BASE_DIR/private_data/nssp-etl/latest_comprehensive.parquet

if [ $? -ne 0 ]; then
	echo "TEST-MODE FAIL: Batch postprocess failed."
	exit 1
else
	echo "TEST-MODE: Batch postprocess succeeded."
fi

echo "TEST-MODE: All finished successfully."
