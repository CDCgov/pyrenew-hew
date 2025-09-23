function run_epiautogp_model(
    data::DataFrame,
    n_forecast_days::Int,
    n_warmup::Int,
    n_samples::Int,
    n_chains::Int
)
    @info "Running EpiAutoGP model with $n_chains chains, $n_warmup warmup, $n_samples samples"
    @info "Forecasting $n_forecast_days days ahead"

    try
        # Extract time series values and dates
        y = Float64.(data.count)
        dates = data.date

        @info "Preparing data transformation for $(length(y)) observations"

        # Get appropriate data transformation based on data characteristics
        # Use "positive" transformation for strictly positive data (common for counts)
        # Could also use "boxcox" for automatic parameter fitting
        transform_type = all(y .> 0) ? "positive" : "boxcox"
        transformation, inv_transformation = get_transformations(transform_type, y)

        # Create transformed data structure
        training_data = create_transformed_data(dates, y; transformation = transformation)

        @info "Fitting Gaussian Process model using Sequential Monte Carlo"

        # Fit the GP model using Sequential Monte Carlo
        # Use n_chains as n_particles (particle count for SMC ensemble)
        # Map warmup/samples to MCMC parameters
        model = make_and_fit_model(
            training_data;
            n_particles = n_chains,
            smc_data_proportion = 0.1,  # Use 10% of data per SMC step
            n_mcmc = n_warmup,          # Use warmup as structure MCMC steps
            n_hmc = n_samples ÷ 4       # Use reduced HMC steps for efficiency
        )

        @info "Generating forecasts for $n_forecast_days days ahead"

        # Create forecast dates
        forecast_dates = [dates[end] + Day(i) for i in 1:n_forecast_days]

        # Generate forecasts using the fitted model
        # Use n_samples * n_chains for total forecast draws
        n_forecast_draws = n_samples * n_chains
        forecasts = forecast(
            model,
            forecast_dates,
            n_forecast_draws;
            inv_transformation = inv_transformation
        )

        @info "Model fitting and forecasting completed successfully"

        # Package results in dictionary format compatible with PyRenew-HEW pipeline
        all_dates = vcat(dates, forecast_dates)

        results = Dict(
            "posterior_samples" => forecasts',  # Transpose to match expected format
            "dates" => all_dates,
            "forecast_dates" => forecast_dates,
            "training_dates" => dates,
            "n_training_points" => length(y),
            "n_forecast_points" => n_forecast_days,
            "training_data" => y,
            "transformation" => transform_type,
            "model" => model,  # Include fitted model for potential reuse
            "n_draws" => n_forecast_draws
        )

        return results

    catch e
        @error "Failed to run EpiAutoGP model: $e"
        rethrow(e)
    end
end
