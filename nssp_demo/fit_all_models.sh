#!/bin/bash

# Base directory containing subdirectories
BASE_DIR="private_data/r_2024-09-19_f_2024-03-22_l_2024-09-18_t_2024-09-15/"

# Iterate over each subdirectory in the base directory
for SUBDIR in "$BASE_DIR"/*/; do
    # Run the Python script with the current subdirectory as the model_dir argument
    echo "$SUBDIR"
    python fit_model.py --model_dir "$SUBDIR"
done
