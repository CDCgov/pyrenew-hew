#!/bin/bash

# Check if the argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <argument>"
    exit 1
fi

SUPER_DIR=$1

# Run the R scripts with the provided argument
Rscript 01_create_h_data.R "$SUPER_DIR"
Rscript 02_process_h_data.R "$SUPER_DIR"
Rscript 03_create_hubverse_table.R "$SUPER_DIR"
