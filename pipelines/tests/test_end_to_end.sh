#!/bin/bash

# Check if the base directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

BASE_DIR="$1"
echo "TEST-MODE: Running forecast_state.py in test mode with base directory $BASE_DIR"
Rscript pipelines/generate_test_data.R "$BASE_DIR/private_data"

if [ $? -ne 0 ]; then
    echo "TEST-MODE FAIL: Generating test data failed"
    exit 1
else
    echo "TEST-MODE: Finished generating test data"
fi
echo "TEST-MODE: Running forecasting pipeline for COVID-19 in multiple states"
for state in CA MT
do
	python pipelines/forecast_state.py \
	       --disease COVID-19 \
	       --state $state \
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
	       --fit-ed-visits \
	       --fit-hospital-admissions \
	       --fit-wastewater \
	       --forecast-ed-visits \
	       --forecast-hospital-admissions \
	       --forecast-wastewater \
	       --score \
	       --eval-data-path "$BASE_DIR/private_data/nssp-etl"
	if [ $? -ne 0 ]; then
	    echo "TEST-MODE FAIL: Forecasting/postprocessing/scoring pipeline failed"
	    exit 1
	else
	    echo "TEST-MODE: Finished forecasting/postprocessing/scoring pipeline for COVID-19 in location" $state"."
	fi
done

echo "TEST-MODE: Running forecasting pipeline for Influenza in multiple states"
for state in CA MT US
do
	python pipelines/forecast_state.py \
	       --disease Influenza \
	       --state $state \
	       --facility-level-nssp-data-dir "$BASE_DIR/private_data/nssp_etl_gold" \
	       --state-level-nssp-data-dir "$BASE_DIR/private_data/nssp_state_level_gold" \
	       --priors-path pipelines/priors/prod_priors.py \
	       --param-data-dir "$BASE_DIR/private_data/prod_param_estimates" \
	       --output-dir "$BASE_DIR/private_data" \
	       --n-training-days 60 \
	       --n-chains 2 \
	       --n-samples 250 \
	       --n-warmup 250 \
	       --fit-ed-visits \
	       --no-fit-hospital-admissions \
	       --no-fit-wastewater \
	       --forecast-ed-visits \
	       --forecast-hospital-admissions \
	       --no-forecast-wastewater \
	       --score \
	       --eval-data-path "$BASE_DIR/private_data/nssp-etl"
	if [ $? -ne 0 ]; then
	    echo "TEST-MODE FAIL: Forecasting/postprocessing/scoring pipeline failed"
	    exit 1
	else
	    echo "TEST-MODE: Finished forecasting/postprocessing/scoring pipeline for Influenza in location" $state"."
	fi
done

echo "TEST-MODE: pipeline runs complete for all location/disease pairs."

echo "TEST-MODE: Extending tests for H and HE models..."
Rscript pipelines/tests/create_more_model_test_data.R

if [ $? -ne 0 ]; then
	echo "TEST-MODE FAIL: Creating more model test data failed"
	exit 1
else
	echo "TEST-MODE: Finished creating more model test data"
fi

echo "TEST-MODE: Running batch postprocess..."

python pipelines/postprocess_forecast_batches.py \
       $BASE_DIR/private_data \
       $BASE_DIR/private_data/nssp-etl/latest_comprehensive.parquet

if [ $? -ne 0 ]; then
    echo "TEST-MODE FAIL: Batch postprocess failed."
    exit 1
else
    echo "TEST-MODE: Batch postprocess succeeded."
fi


echo "TEST-MODE: All finished successfully."
