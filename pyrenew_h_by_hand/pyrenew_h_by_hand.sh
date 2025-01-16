#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

SUPER_DIR=$1

# Run the R scripts with the provided argument
echo "Running 01_create_h_data.R..."
Rscript pyrenew_h_by_hand/01_create_h_data.R "$SUPER_DIR"
echo "Finished running 01_create_h_data.R"

echo "Running 02_process_h_data.R..."
Rscript pyrenew_h_by_hand/02_process_h_data.R "$SUPER_DIR"
echo "Finished running 02_process_h_data.R"

echo "Running 03_create_hubverse_table.R..."
Rscript pyrenew_h_by_hand/03_create_hubverse_table.R "$SUPER_DIR"
echo "Finished running 03_create_hubverse_table.R"

echo "Running 04_make_figures.R..."
Rscript pyrenew_h_by_hand/04_make_figures.R "$SUPER_DIR"
echo "Finished running 04_make_figures.R"
