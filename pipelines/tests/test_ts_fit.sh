#!/bin/bash

# Check if the base directory is provided as an argument
if [[ $# -ne 3 ]]; then
	echo "Usage: $0 <base_dir> <disease> <location>"
	exit 1
fi

BASE_DIR="$1"
disease="$2"
location="$3"

python pipelines/fable/forecast_timeseries.py \
	--disease "$disease" \
	--loc "$location" \
	--facility-level-nssp-data-dir "$BASE_DIR/private_data/nssp_etl_gold" \
	--param-data-dir "$BASE_DIR/private_data/prod_param_estimates" \
	--output-dir "$BASE_DIR/2024-12-21_forecasts" \
	--n-training-days 90 \
	--n-samples 500 \
	--nhsn-data-path "$BASE_DIR/private_data/nhsn_test_data/${disease}_${location}.parquet"
if [ "$?" -ne 0 ]; then
	echo "TEST-MODE FAIL: Forecasting/postprocessing pipeline failed"
	exit 1
fi
