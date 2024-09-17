#!/bin/bash

# Base directory containing subdirectories
BASE_DIR="/path/to/base/directory"

# Iterate over each subdirectory in the base directory
for SUBDIR in "$BASE_DIR"/*/; do
    # Run the Python script with the current subdirectory as the model_dir argument
    python fit_model.py --model_dir "$SUBDIR"
done
