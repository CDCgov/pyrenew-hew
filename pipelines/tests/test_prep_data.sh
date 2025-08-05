#!/bin/bash

# Check if the base directory is provided as an argument
if [[ $# -ne 3 ]]; then
	echo "Usage: $0 <base_dir> <disease> <location>"
	exit 1
fi

BASE_DIR="$1"
disease="$2"
location="$3"

python pipelines/prep_data.py \
	--disease "$disease" \
	--loc "$location" \
	--report-date "2024-12-21" \
	--last-training-date "2024-12-20" \
	--first-training-date "2024-09-22" \
	--facility-level-nssp-data-dir "$BASE_DIR/private_data/nssp_etl_gold" \
	--state-level-nssp-data-dir "$BASE_DIR/private_data/nssp_state_level_gold" \
	--nwss-data-dir "$BASE_DIR/private_data/nwss_vintages" \
	--param-data-dir "$BASE_DIR/private_data/prod_param_estimates" \
	--priors-path pipelines/priors/prod_priors.py \
	--model-run-dir \
	"$BASE_DIR/2024-12-21_forecasts/${disease,,}_r_2024-12-21_f_2024-09-22_t_2024-12-20/model_runs/${location}" \
	--nhsn-data-path "$BASE_DIR/private_data/nhsn_test_data/${disease}_${location}.parquet"
if [ "$?" -ne 0 ]; then
	echo "TEST-MODE FAIL: Data preparation pipeline failed"
	exit 1
fi
