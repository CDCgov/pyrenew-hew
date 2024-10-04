#!/bin/bash

# Base directory containing subdirectories

BASE_DIR="private_data/influenza_r_2024-10-01_f_2024-04-03_l_2024-09-30_t_2024-09-25"


# Iterate over each subdirectory in the base directory
for SUBDIR in "$BASE_DIR"/*/; do
    # Run the Python script with the current subdirectory as the model_dir argument
    echo "$SUBDIR"
    python fit_model.py --model_dir "$SUBDIR"
done
