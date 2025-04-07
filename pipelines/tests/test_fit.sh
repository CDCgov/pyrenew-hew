#!/bin/bash

# Check if the base directory is provided as an argument
if [[ $# -lt 4 ]]; then
	echo "Usage: $0 <base_dir> <disease> <location> <model_letters>"
	exit 1
fi

BASE_DIR="$1"
disease="$2"
location="$3"
model_letters="$4"

python pipelines/forecast_state.py \
	--disease $disease \
	--state $location \
	--facility-level-nssp-data-dir "$BASE_DIR/private_data/nssp_etl_gold" \
	--state-level-nssp-data-dir "$BASE_DIR/private_data/nssp_state_level_gold" \
	--priors-path pipelines/priors/prod_priors.py \
	--param-data-dir "$BASE_DIR/private_data/prod_param_estimates" \
	--nwss-data-dir "$BASE_DIR/private_data/nwss_vintages" \
	--output-dir "$BASE_DIR/private_data" \
	--n-training-days 60 \
	--n-chains 2 \
	--n-samples 250 \
	--n-warmup 250 \
	--model-letters $model_letters \
	--no-score \
	--eval-data-path "$BASE_DIR/private_data/nssp-etl"
if [ $? -ne 0 ]; then
	echo "TEST-MODE FAIL: Forecasting/postprocessing/scoring pipeline failed"
	exit 1
fi
