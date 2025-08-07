#!/bin/bash

# Check if the base directory is provided as an argument
if [[ $# -ne 4 ]]; then
	echo "Usage: $0 <base_dir> <disease> <location> <model_letters>"
	exit 1
fi

BASE_DIR="$1"
disease="$2"
location="$3"
model_letters="$4"

python pipelines/forecast_pyrenew.py \
	--loc "$location" \
	--model-run-dir \
	"$BASE_DIR/2024-12-21_forecasts/${disease,,}_r_2024-12-21_f_2024-09-22_t_2024-12-20/model_runs/${location}" \
	--n-forecast-days 28 \
	--exclude-last-n-days 0 \
	--n-chains 2 \
	--n-samples 250 \
	--n-warmup 250 \
	--model-letters "$model_letters" \
	--additional-forecast-letters "$model_letters"
if [ "$?" -ne 0 ]; then
	echo "TEST-MODE FAIL: Forecasting/postprocessing pipeline failed"
	exit 1
fi
