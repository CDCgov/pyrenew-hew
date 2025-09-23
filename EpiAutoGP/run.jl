#!/usr/bin/env julia

"""
EpiAutoGP model runner for PyRenew-HEW pipeline

This script serves as the entry point for running EpiAutoGP models in the PyRenew-HEW
forecasting pipeline. It accepts JSON input data and produces hubverse-compatible outputs.

Usage:
    julia run.jl --json-input path/to/input.json --output-dir path/to/output --disease COVID-19 --location CA --forecast-date 2024-12-21

Arguments:
    --json-input: Path to JSON file containing model input data
    --output-dir: Directory for saving model outputs
    --disease: Disease name (COVID-19, Influenza, RSV)
    --location: Two-letter state/territory abbreviation
    --forecast-date: Reference date for forecasting (YYYY-MM-DD format)
    --n-forecast-days: Number of days to forecast (default: 28)
    --n-warmup: Number of warmup samples (default: 1000)
    --n-samples: Number of posterior samples (default: 1000)
    --n-chains: Number of MCMC chains (default: 4)
"""

using Pkg
Pkg.activate(@__DIR__)

using NowcastAutoGP
using JSON3
using CSV
using TidierData
using Dates
using ArgParse
using Logging

# Configure logging
logger = SimpleLogger()
global_logger(logger)

include("parse_arguments.jl") # Function to parse command line arguments
include("input.jl")           # Functions to load and process input JSON data



function create_hubverse_table(
    results::Dict,
    disease::String,
    location::String,
    reference_date::String,
    output_dir::String
)
    """
    Convert EpiAutoGP results to hubverse-compatible quantile table format
    """
    @info "Creating hubverse table for $disease in $location"

    posterior_samples = results["posterior_samples"]
    dates = results["dates"]
    n_training = results["n_training_points"]

    # Focus on forecast period only
    forecast_dates = dates[(n_training+1):end]
    forecast_samples = posterior_samples[:, (n_training+1):end]

    # Calculate quantiles
    quantile_levels = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

    hubverse_rows = []

    for (day_idx, forecast_date) in enumerate(forecast_dates)
        day_samples = forecast_samples[:, day_idx]

        for q_level in quantile_levels
            q_value = quantile(day_samples, q_level)

            # Calculate horizon (days from reference date)
            ref_date = Date(reference_date)
            horizon = (forecast_date - ref_date).value

            row = (
                model_id = "epiautogp_daily",
                model = "epiautogp",
                output_type = "quantile",
                output_type_id = round(q_level, digits = 4),
                value = round(q_value, digits = 2),
                reference_date = reference_date,
                target = "inc $(lowercase(disease)) ed visits",
                horizon = horizon,
                horizon_timescale = "days",
                resolution = "daily",
                target_end_date = string(forecast_date),
                location = location,
                disease = disease,
                aggregated_numerator = false,
                aggregated_denominator = missing
            )

            push!(hubverse_rows, row)
        end
    end

    # Convert to DataFrame
    hubverse_df = DataFrame(hubverse_rows)

    # Save as parquet file
    hubverse_path = joinpath(output_dir, "hubverse_table.parquet")
    mkpath(dirname(hubverse_path))

    # For now, save as CSV since Parquet.jl might not be available
    # In production, you'd want to use Arrow.jl or similar for parquet format
    csv_path = joinpath(output_dir, "hubverse_table.csv")
    CSV.write(csv_path, hubverse_df)

    @info "Saved hubverse table to $csv_path"
    @info "Table contains $(nrow(hubverse_df)) rows"

    return hubverse_df
end

function save_model_outputs(
    results::Dict,
    disease::String,
    location::String,
    reference_date::String,
    output_dir::String
)
    """
    Save additional model outputs (samples, diagnostics, etc.)
    """
    @info "Saving model outputs to $output_dir"

    # Create output directories
    mcmc_dir = joinpath(output_dir, "mcmc_output")
    mkpath(mcmc_dir)

    # Save posterior samples using TidierData
    posterior_df = DataFrame(results["posterior_samples"], :auto)
    posterior_df.date = repeat(results["dates"], inner=size(results["posterior_samples"], 1))
    posterior_df.chain = repeat(1:4, outer=div(size(results["posterior_samples"], 1), 4) * length(results["dates"]))
    posterior_df.draw = repeat(1:div(size(results["posterior_samples"], 1), 4), outer=4 * length(results["dates"]))

    # Reshape to tidy format using TidierData syntax
    tidy_df = DataFrame(
        date = posterior_df.date,
        chain = posterior_df.chain,
        draw = posterior_df.draw,
        observed_ed_visits = vec(Matrix(posterior_df[:, Not([:date, :chain, :draw])]))
    )

    tidy_path = joinpath(mcmc_dir, "tidy_posterior_predictive.csv")
    CSV.write(tidy_path, tidy_df)

    @info "Saved tidy posterior samples to $tidy_path"

    # Save summary statistics using TidierData
    summary_stats = DataFrame(
        variable = ["observed_ed_visits"],
        mean = [mean(results["posterior_samples"])],
        std = [std(results["posterior_samples"])],
        q05 = [quantile(vec(results["posterior_samples"]), 0.05)],
        q50 = [quantile(vec(results["posterior_samples"]), 0.5)],
        q95 = [quantile(vec(results["posterior_samples"]), 0.95)]
    )

    summary_path = joinpath(mcmc_dir, "summary_stats.csv")
    CSV.write(summary_path, summary_stats)

    @info "Saved summary statistics to $summary_path"
end

function main()
    """
    Main execution function
    """
    try
        @info "Starting EpiAutoGP model run"

        # Parse command line arguments
        args = parse_arguments()
        @info "Parsed arguments: $args"

        # Load input data
        json_data = load_json_data(args["json-input"])

        # Prepare data for EpiAutoGP
        time_series_data = prepare_epiautogp_data(
            json_data,
            args["disease"],
            args["location"]
        )

        # Run the model
        results = run_epiautogp_model(
            time_series_data,
            args["n-forecast-days"],
            args["n-warmup"],
            args["n-samples"],
            args["n-chains"]
        )

        # Create hubverse output
        hubverse_table = create_hubverse_table(
            results,
            args["disease"],
            args["location"],
            args["forecast-date"],
            args["output-dir"]
        )

        # Save additional outputs
        save_model_outputs(
            results,
            args["disease"],
            args["location"],
            args["forecast-date"],
            args["output-dir"]
        )

        @info "EpiAutoGP model run completed successfully"
        @info "Results saved to $(args["output-dir"])"

    catch e
        @error "EpiAutoGP model run failed: $e"
        rethrow(e)
    end
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
