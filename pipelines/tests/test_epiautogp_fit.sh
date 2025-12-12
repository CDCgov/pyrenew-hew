#!/bin/bash

# Run a single EpiAutoGP forecast
# Called by test_epiautogp_end_to_end.sh
#
# Usage:
#   bash test_epiautogp_fit.sh <base_dir> <disease> <location> <target> <frequency> <use_percentage>

if [[ $# -ne 6 ]]; then
	echo "Usage: $0 <base_dir> <disease> <location> <target> <frequency> <use_percentage>"
	echo "  target: nhsn or nssp"
	echo "  frequency: daily or epiweekly"
	echo "  use_percentage: true or false"
	exit 1
fi

BASE_DIR="$1"
disease="$2"
location="$3"
target="$4"
frequency="$5"
use_percentage="$6"

# Build command arguments
cmd_args=(
	--disease "$disease"
	--loc "$location"
	--facility-level-nssp-data-dir "$BASE_DIR/private_data/nssp_etl_gold"
	--state-level-nssp-data-dir "$BASE_DIR/private_data/nssp_state_level_gold"
	--param-data-dir "$BASE_DIR/private_data/prod_param_estimates"
	--output-dir "$BASE_DIR/2024-12-21_forecasts"
	--n-training-days 90
	--target "$target"
	--frequency "$frequency"
	--eval-data-path "$BASE_DIR/private_data/nssp-etl"
	--nhsn-data-path "$BASE_DIR/private_data/nhsn_test_data/${disease}_${location}.parquet"
	--n-forecast-weeks 4
	--n-particles 2
	--n-mcmc 2
	--n-hmc 2
	--n-forecast-draws 100
	--smc-data-proportion 0.1
)

# Add percentage flag if needed
if [ "$use_percentage" = "true" ]; then
	cmd_args+=(--use-percentage)
fi

uv run python pipelines/epiautogp/forecast_epiautogp.py "${cmd_args[@]}"

if [ "$?" -ne 0 ]; then
	echo "TEST-MODE FAIL: EpiAutoGP forecast pipeline failed"
	exit 1
fi
