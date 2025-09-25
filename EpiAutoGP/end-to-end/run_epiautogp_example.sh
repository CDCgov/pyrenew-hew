#!/bin/bash

# EpiAutoGP End-to-End Example Script
#
# This script demonstrates how to run the EpiAutoGP model using the generated
# JSON input data from the vintaged NHSN dataset for report_date 2025-08-16.
#
# Usage: ./run_epiautogp_example.sh [--threads N]
#
# Arguments:
#   --threads N    Number of Julia threads to use (default: 4)
#
# The script will:
# 1. Use the generated JSON input file
# 2. Run the EpiAutoGP model with reasonable parameters for demonstration
# 3. Save outputs to the end-to-end directory
# 4. Display results and summary

set -e # Exit on any error

# Parse command line arguments
THREADS=4 # Default number of threads

while [[ $# -gt 0 ]]; do
	case $1 in
	--threads)
		THREADS="$2"
		shift 2
		;;
	-h | --help)
		echo "Usage: $0 [--threads N]"
		echo ""
		echo "Arguments:"
		echo "  --threads N    Number of Julia threads to use (default: 4)"
		echo "  -h, --help     Show this help message"
		exit 0
		;;
	*)
		echo "Unknown argument: $1"
		echo "Use --help for usage information"
		exit 1
		;;
	esac
done

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Handle running from different directories
if [[ "$(basename "$(pwd)")" == "EpiAutoGP" ]]; then
	# Running from /EpiAutoGP directory
	END_TO_END_DIR="$PWD/end-to-end"
else
	# Running from end-to-end directory or elsewhere
	END_TO_END_DIR="$SCRIPT_DIR"
fi

JSON_INPUT="$END_TO_END_DIR/epiautogp_input_2025-08-16.json"
OUTPUT_DIR="$END_TO_END_DIR/output"
RUN_SCRIPT="$PROJECT_DIR/run.jl"

echo "=== EpiAutoGP End-to-End Example ==="
echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo "End-to-end directory: $END_TO_END_DIR"
echo "JSON input file: $JSON_INPUT"
echo "Output directory: $OUTPUT_DIR"
echo "Run script: $RUN_SCRIPT"
echo "Julia threads: $THREADS"
echo ""

# Check that required files exist
if [[ ! -f "$JSON_INPUT" ]]; then
	echo "❌ Error: JSON input file not found: $JSON_INPUT"
	echo "Please run the Python script first from the end-to-end directory:"
	echo "  cd $END_TO_END_DIR && uv run python create_epiautogp_input.py"
	exit 1
fi

if [[ ! -f "$RUN_SCRIPT" ]]; then
	echo "❌ Error: Run script not found: $RUN_SCRIPT"
	exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "✅ All required files found. Starting EpiAutoGP model run..."
echo ""

# Run the EpiAutoGP model with demonstration parameters
# Using smaller parameter values for faster execution in this example and fewer particles
echo "🚀 Running EpiAutoGP model..."
echo "Command: julia --project=$PROJECT_DIR --threads=$THREADS $RUN_SCRIPT \\"
echo "    --json-input $JSON_INPUT \\"
echo "    --output-dir $OUTPUT_DIR \\"
echo "    --n-forecast-weeks 3 \\"
echo "    --n-particles 4 \\"
echo "    --n-mcmc 50 \\"
echo "    --n-hmc 25 \\"
echo "    --n-forecast-draws 100 \\"
echo "    --transformation positive \\"
echo "    --smc-data-proportion 0.2"
echo ""

julia --project="$PROJECT_DIR" --threads="$THREADS" "$RUN_SCRIPT" \
	--json-input "$JSON_INPUT" \
	--output-dir "$OUTPUT_DIR" \
	--n-forecast-weeks 3 \
	--n-particles 4 \
	--n-mcmc 50 \
	--n-hmc 25 \
	--n-forecast-draws 100 \
	--transformation positive \
	--smc-data-proportion 0.2

echo ""
echo "✅ EpiAutoGP model run completed!"
echo ""

# Display results summary
echo "=== Results Summary ==="
echo "Output directory: $OUTPUT_DIR"

if [[ -d "$OUTPUT_DIR" ]]; then
	echo "Files created:"
	ls -la "$OUTPUT_DIR"
	echo ""

	# Look for CSV files (hubverse output)
	CSV_FILES=$(find "$OUTPUT_DIR" -name "*.csv" -type f 2>/dev/null || true)
	if [[ -n "$CSV_FILES" ]]; then
		echo "📊 Hubverse forecast files:"
		echo "$CSV_FILES"
		echo ""

		# Show first few lines of first CSV file
		FIRST_CSV=$(echo "$CSV_FILES" | head -1)
		if [[ -f "$FIRST_CSV" ]]; then
			echo "Preview of forecast output ($FIRST_CSV):"
			echo "----------------------------------------"
			head -10 "$FIRST_CSV"
			echo "----------------------------------------"
			echo "Total rows: $(wc -l <"$FIRST_CSV")"

			# Generate plots using forecasttools
			echo ""
			echo "🎨 Generating forecast plots using forecasttools..."
			PLOT_OUTPUT_DIR="$OUTPUT_DIR/plots"
			R_SCRIPT="$END_TO_END_DIR/plot_forecast.R"

			if [[ -f "$R_SCRIPT" ]]; then
				Rscript "$R_SCRIPT" "$FIRST_CSV" "$PLOT_OUTPUT_DIR"

				if [[ -d "$PLOT_OUTPUT_DIR" ]]; then
					echo ""
					echo "📊 Plot files created:"
					ls -la "$PLOT_OUTPUT_DIR"
				fi
			else
				echo "⚠️  R plotting script not found: $R_SCRIPT"
				echo "   Plots will not be generated"
			fi
		fi
	else
		echo "⚠️  No CSV files found in output directory"
		echo "   Cannot generate plots without forecast data"
	fi
else
	echo "⚠️  Output directory not found or empty"
fi

echo ""
echo "=== End-to-End Example Complete ==="
echo ""
echo "💡 Tips:"
echo "  - Use --threads N to control Julia threading (current: $THREADS)"
echo "  - Increase --n-particles, --n-mcmc, --n-hmc, and --n-forecast-draws for more robust results"
echo "  - Try different --transformation options: boxcox, positive, percentage"
echo "  - Adjust --n-forecast-weeks to change forecast horizon"
echo "  - Check output CSV files for hubverse-compatible forecast tables"
echo "  - View generated plots in the plots/ subdirectory"
echo ""
echo "📁 All outputs saved to: $OUTPUT_DIR"
echo "🎨 Forecast plots saved to: $OUTPUT_DIR/plots"
