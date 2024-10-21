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
    # Run the R script with the current subdirectory as the model_dir argument
    echo "$SUBDIR"
    # will work once  https://github.com/rstudio/renv/pull/2018 is merged
    Rscript -e "renv::run(\"post_process.R\", project = \"..\", args = c(\"--model_dir ${SUBDIR}\"))"
done


# # Get the name of the current directory (base_dir)
base_dir_name=$(basename "$(pwd)")

# Find all forecast_plot.pdf files and combine them using pdfunite
find . -name "forecast_plot.pdf" | sort | xargs pdfunite - "${BASE_DIR}/${base_dir_name}_all_forecasts.pdf"

echo "Combined PDF created: ${base_dir_name}_all_forecasts.pdf"
