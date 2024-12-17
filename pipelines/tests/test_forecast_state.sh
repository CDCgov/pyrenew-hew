#!/bin/bash

# Check if the base directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

BASE_DIR="$1"

Rscript pipelines/generate_test_data.R "$BASE_DIR/private_data"
python pipelines/forecast_state.py \
    --disease "COVID-19" \
    --state "CA" \
    --facility-level-nssp-data-dir "$BASE_DIR/private_data/nssp_etl_gold" \
    --state-level-nssp-data-dir "$BASE_DIR/private_data/nssp_state_level_gold" \
    --priors-path "$BASE_DIR/test_output/priors.py" \
    --param-data-dir "$BASE_DIR/private_data/prod_param_estimates" \
    --output-dir "$BASE_DIR/private_data" \
    --n-training-days 90 \
    --n-chains 1 \
    --n-samples 500 \
    --n-warmup 500 \
    --score \
    --eval-data-path "$BASE_DIR/private_data/nssp-archival-vintages"
