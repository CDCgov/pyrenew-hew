#!/bin/bash

# Check if the base directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

# Base directory containing subdirectories
BASE_DIR="$1"
N_SAMPLES=$2
N_AHEAD=$3

# Iterate over each subdirectory in the base directory
echo "TEST-MODE: Running loop over subdirectories in $BASE_DIR"
echo "For $N_SAMPLES samples on 1 chain, and $N_AHEAD forecast points"
echo ""
for SUBDIR in "$BASE_DIR"/*/; do
    echo "TEST-MODE: Inference for $SUBDIR"
    python fit_model.py "$SUBDIR" --n-chains 1 --n-samples $N_SAMPLES
    echo "TEST-MODE: Finished inference"
    echo ""
    echo "TEST-MODE: Generating posterior predictions for $SUBDIR"
    python generate_predictive.py "$SUBDIR" --n-forecast-points $N_AHEAD
    echo "TEST-MODE: Finished generating posterior predictions"
    echo ""
    echo "TEST-MODE: Converting inferencedata to parquet for $SUBDIR"
    Rscript convert_inferencedata_to_parquet.R "$SUBDIR"
    echo "TEST-MODE: Finished converting inferencedata to parquet"
    echo ""
    echo "TEST-MODE: Generate epiweekly data for $SUBDIR"
    Rscript generate_epiweekly.R "$SUBDIR"
    echo "TEST-MODE: Finished generating epiweekly data"
    echo ""
    echo "TEST-MODE: Forecasting baseline models for $SUBDIR"
    Rscript timeseries_forecasts.R "$SUBDIR" --n-forecast-days $N_AHEAD --n-samples $N_SAMPLES
    echo "TEST-MODE: Finished forecasting baseline models"
    echo ""
    echo "TEST-MODE: Postprocessing state forecast for $SUBDIR"
    Rscript postprocess_state_forecast.R "$SUBDIR"
    echo "TEST-MODE: Finished postprocessing state forecast"
    echo ""
    echo "TEST-MODE: Rendering webpage for $SUBDIR"
    Rscript render_webpage.R "$SUBDIR"
    echo "TEST-MODE: Finished rendering webpage"
    echo ""
    echo "TEST-MODE: Scoring forecast for $SUBDIR"
    Rscript score_forecast.R "$SUBDIR"
    echo "TEST-MODE: Finished scoring forecast"
done
