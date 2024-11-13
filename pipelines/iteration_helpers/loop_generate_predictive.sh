#!/bin/bash

# Check if the base directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <base_dir>"
    exit 1
fi

# Base directory containing subdirectories
BASE_DIR="$1"


# Iterate over each subdirectory in the base directory
for SUBDIR in "$BASE_DIR"/*/; do
    # Run the Python script with the current subdirectory as the model_dir argument
    echo "$SUBDIR"
    python generate_predictive.py --model_dir "$SUBDIR" --n_forecast_points 28
done
