function create_hubverse_table(
        results::NamedTuple,
        output_dir::String
)
    """
    Convert EpiAutoGP results to hubverse-compatible quantile table format.

    Expected format matches CFA-EpiAutoGP CSV structure:
    target_end_date, value, output_type_id, horizon, output_type, reference_date, location, target
    """
    @info "Creating hubverse table for $(results.disease) in $(results.location)"

    forecast_dates = results.forecast_dates
    forecast_samples = results.forecasts
    forecast_date = results.forecast_date
    location = results.location
    disease = results.disease

    # Define quantile levels to match CDC hub format
    quantile_levels = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]

    hubverse_rows = []

    # Create target name based on disease
    disease_abbr = disease == "COVID-19" ? "covid" :
                  disease == "Influenza" ? "flu" :
                  lowercase(disease)
    target = "wk inc $disease_abbr hosp"

    for (date_idx, target_end_date) in enumerate(forecast_dates)
        date_samples = forecast_samples[date_idx, :]

        # Calculate horizon as weeks from forecast_date to target_end_date
        # This handles both positive (forecasts) and negative (nowcasts) horizons
        horizon = Int(Dates.value(target_end_date - forecast_date) ÷ 7)

        for q_level in quantile_levels
            # Calculate quantile value
            q_value = quantile(date_samples, q_level)

            # Create row matching exact hubverse format
            row = (
                target_end_date = string(target_end_date),
                value = round(q_value, digits=6),
                output_type_id = q_level,
                horizon = horizon,
                output_type = "quantile",
                reference_date = string(forecast_date),
                location = location,  # Keep as provided (2-letter abbreviation for now)
                target = target
            )

            push!(hubverse_rows, row)
        end
    end

    # Convert to DataFrame
    hubverse_df = DataFrame(hubverse_rows)

    # Save as CSV to match expected format
    csv_path = joinpath(output_dir, "hubverse_table.csv")
    mkpath(dirname(csv_path))
    CSV.write(csv_path, hubverse_df)

    @info "Saved hubverse table to $csv_path"
    @info "Table contains $(nrow(hubverse_df)) rows"

    return hubverse_df
end

function save_model_outputs(
        results::NamedTuple,
        output_dir::String
)
    """
    Save additional model outputs (samples, diagnostics, etc.)
    """
    @info "Saving model outputs to $output_dir"

    # Create output directories
    mcmc_dir = joinpath(output_dir, "mcmc_output")
    mkpath(mcmc_dir)

    forecast_dates = results.forecast_dates
    forecast_samples = results.forecasts
    n_dates, n_samples = size(forecast_samples)

    # Create a tidy DataFrame with samples
    # Using TidierData approach - reshape to long format
    sample_rows = []
    for (date_idx, date) in enumerate(forecast_dates)
        for sample_idx in 1:n_samples
            row = (
                date = date,
                sample = sample_idx,
                chain = ((sample_idx - 1) % 4) + 1,  # Assume 4 chains
                draw = (sample_idx - 1) ÷ 4 + 1,
                forecast_value = forecast_samples[date_idx, sample_idx],
                disease = results.disease,
                location = results.location,
                forecast_date = results.forecast_date
            )
            push!(sample_rows, row)
        end
    end

    tidy_df = DataFrame(sample_rows)
    tidy_path = joinpath(mcmc_dir, "tidy_forecast_samples.csv")
    CSV.write(tidy_path, tidy_df)

    @info "Saved tidy forecast samples to $tidy_path"

    # Save summary statistics for each forecast date
    summary_rows = []
    for (date_idx, date) in enumerate(forecast_dates)
        date_samples = forecast_samples[date_idx, :]

        row = (
            date = date,
            variable = "forecast_value",
            mean = mean(date_samples),
            std = std(date_samples),
            q01 = quantile(date_samples, 0.01),
            q05 = quantile(date_samples, 0.05),
            q25 = quantile(date_samples, 0.25),
            q50 = quantile(date_samples, 0.5),
            q75 = quantile(date_samples, 0.75),
            q95 = quantile(date_samples, 0.95),
            q99 = quantile(date_samples, 0.99),
            disease = results.disease,
            location = results.location,
            forecast_date = results.forecast_date
        )
        push!(summary_rows, row)
    end

    summary_df = DataFrame(summary_rows)
    summary_path = joinpath(mcmc_dir, "forecast_summary_stats.csv")
    CSV.write(summary_path, summary_df)

    @info "Saved forecast summary statistics to $summary_path"

    # Save metadata as JSON
    metadata = Dict(
        "disease" => results.disease,
        "location" => results.location,
        "forecast_date" => string(results.forecast_date),
        "forecast_dates" => string.(forecast_dates),
        "n_samples" => n_samples,
        "n_forecast_dates" => n_dates,
        "generated_at" => string(now())
    )

    metadata_path = joinpath(output_dir, "forecast_metadata.json")
    open(metadata_path, "w") do f
        JSON3.write(f, metadata)
    end

    @info "Saved forecast metadata to $metadata_path"
end
