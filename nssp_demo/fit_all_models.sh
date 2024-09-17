#!/bin/bash

# Base directory containing subdirectories
BASE_DIR="private_data/r_2024-09-10_f_2024-03-13_l_2024-09-09_t_2024-08-14/"

# Iterate over each subdirectory in the base directory
for SUBDIR in "$BASE_DIR"/*/; do
    # Run the Python script with the current subdirectory as the model_dir argument
    echo "$SUBDIR"
    python fit_model.py --model_dir "$SUBDIR"
done
