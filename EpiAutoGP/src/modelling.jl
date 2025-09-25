"""
    prepare_for_modelling(input::EpiAutoGPInput, transformation_name::String, n_forecast_weeks::Int, n_forecasts::Int) -> NamedTuple

Prepare all data and configuration needed for NowcastAutoGP modeling.

This function extracts stable training data (excluding nowcast dates), sets up data transformations,
formats nowcast data for the modeling pipeline, and calculates forecast dates and sample sizes.

# Arguments
- `input::EpiAutoGPInput`: The input data structure containing dates, reports, and nowcast information
- `transformation_name::String`: Name of transformation to apply ("boxcox", "positive", "percentage")
- `n_forecast_weeks::Int`: Number of weeks to forecast into the future
- `n_forecasts::Int`: Total number of forecast samples desired

# Returns
A NamedTuple containing:
- `stable_data_dates::Vector{Date}`: Dates for confirmed/stable data (excluding nowcast dates)
- `stable_data_values::Vector{<:Real}`: Values for confirmed/stable data
- `nowcast_data`: Formatted nowcast data for NowcastAutoGP (empty if no nowcasts)
- `forecast_dates::Vector{Date}`: Dates for which forecasts will be generated
- `n_forecasts_per_nowcast::Int`: Number of forecast samples per nowcast scenario
- `transformation::Function`: Forward transformation function
- `inv_transformation::Function`: Inverse transformation function

# Examples
```julia
input = EpiAutoGPInput(...)
model_setup = prepare_for_modelling(input, "boxcox", 4, 1000)
```
"""
function prepare_for_modelling(input::EpiAutoGPInput, transformation_name::String,
        n_forecast_weeks::Int, n_forecasts::Int)
    # Extract stable confirmed data, excluding recent uncertain dates with nowcasts
    stable_data_idxs = findall(d -> !(d in input.nowcast_dates), input.dates)
    stable_data_dates = input.dates[stable_data_idxs]
    stable_data_values = input.reports[stable_data_idxs]

    # Get transformation functions
    transformation,
    inv_transformation = get_transformations(transformation_name, input.reports)

    # Format nowcast data (only if nowcasts exist)
    nowcast_data = isempty(input.nowcast_dates) ?
                   # Return empty vector when no nowcasts
                   TData[] :
                   # Create nowcast data structure
                   create_nowcast_data(input.nowcast_reports, input.nowcast_dates; transformation)

    # Calculate forecasting dates
    forecast_dates = [input.forecast_date + Week(i) for i in -1:1:n_forecast_weeks]

    # Calculate number of forecasts per nowcast sample
    n_forecasts_per_nowcast = length(nowcast_data) > 0 ?
                              max(1, n_forecasts ÷ length(nowcast_data)) :
                              n_forecasts

    return (; stable_data_dates, stable_data_values, nowcast_data, forecast_dates,
        n_forecasts_per_nowcast, transformation, inv_transformation)
end

"""
    fit_base_model(dates::Vector{Date}, values::Vector{<:Real};
                   transformation::Function,
                   n_particles::Int=24,
                   smc_data_proportion::Float64=0.1,
                   n_mcmc::Int=50,
                   n_hmc::Int=50) -> AutoGP.Model

Fit a base Gaussian Process model using NowcastAutoGP on confirmed/stable data.

This function creates transformed data and fits a GP model using Sequential Monte Carlo (SMC)
sampling. The model will be used as the foundation for forecasting, either directly or in
combination with nowcast scenarios.

# Arguments
- `dates::Vector{Date}`: Vector of observation dates in chronological order
- `values::Vector{<:Real}`: Vector of corresponding observation values
- `transformation::Function`: Data transformation function (from get_transformations)
- `n_particles::Int=24`: Number of SMC particles for model fitting
- `smc_data_proportion::Float64=0.1`: Proportion of data used in each SMC step
- `n_mcmc::Int=50`: Number of MCMC samples for structure exploration
- `n_hmc::Int=50`: Number of HMC samples for parameter updates

# Returns
- Fitted AutoGP model ready for forecasting

# Examples
```julia
dates = [Date(2024,1,1), Date(2024,1,8), Date(2024,1,15)]
values = [100.0, 120.0, 95.0]
transform_func, _ = get_transformations("boxcox", values)
model = fit_base_model(dates, values; transformation=transform_func)
```
"""
function fit_base_model(dates::Vector{Date}, values::Vector{<:Real};
        transformation::Function,
        n_particles::Int = 24,
        smc_data_proportion::Float64 = 0.1,
        n_mcmc::Int = 50,
        n_hmc::Int = 50)

    # Create transformed data
    transformed_data = create_transformed_data(dates, values; transformation)

    # Fit the model
    model = make_and_fit_model(transformed_data;
        n_particles = n_particles,
        smc_data_proportion = smc_data_proportion,
        n_mcmc = n_mcmc,
        n_hmc = n_hmc)

    return model
end

"""
    forecast_with_epiautogp(input::EpiAutoGPInput;
                           n_forecast_weeks::Int=8,
                           n_forecasts::Int=20,
                           transformation_name::String="boxcox",
                           n_particles::Int=24,
                           smc_data_proportion::Float64=0.1,
                           n_mcmc::Int=50,
                           n_hmc::Int=50) -> NamedTuple

Main forecasting function that combines EpiAutoGP input with NowcastAutoGP modeling.

This function implements the complete nowcasting and forecasting workflow:
1. Prepares stable training data and nowcast scenarios from EpiAutoGPInput
2. Fits a base GP model on confirmed data
3. Generates forecasts either directly (if no nowcasts) or incorporating nowcast uncertainty

# Arguments
- `input::EpiAutoGPInput`: The input data structure with dates, reports, and nowcast information
- `n_forecast_weeks::Int=8`: Number of weeks to forecast ahead from forecast_date
- `n_forecasts::Int=20`: Total number of forecast samples to generate
- `transformation_name::String="boxcox"`: Data transformation type ("boxcox", "positive", "percentage")
- `n_particles::Int=24`: Number of SMC particles for GP model fitting
- `smc_data_proportion::Float64=0.1`: Proportion of data used in each SMC step
- `n_mcmc::Int=50`: Number of MCMC samples for GP structure exploration
- `n_hmc::Int=50`: Number of HMC samples for GP parameter updates

# Returns
A NamedTuple containing:
- `forecast_dates::Vector{Date}`: Dates for which forecasts were generated
- `forecasts::Matrix`: Forecast samples matrix (dates × samples)
- `forecast_date::Date`: The reference date for forecasting (from input.forecast_date)
- `location::String`: The location identifier (from input.location)
- `disease::String`: The disease name (from input.disease)

# Examples
```julia
# Basic forecasting
input = EpiAutoGPInput(...)
results = forecast_with_epiautogp(input)
forecast_dates, forecasts = results.forecast_dates, results.forecasts

# Custom parameters
results = forecast_with_epiautogp(input;
                                 n_forecast_weeks=4,
                                 n_forecasts=1000,
                                 transformation_name="positive")
```
"""
function forecast_with_epiautogp(input::EpiAutoGPInput;
        n_forecast_weeks::Int = 8,
        n_forecasts::Int = 20,
        transformation_name::String = "boxcox",
        n_particles::Int = 24,
        smc_data_proportion::Float64 = 0.1,
        n_mcmc::Int = 50,
        n_hmc::Int = 50)

    # Prepare training data, nowcasting data and forecasting dates
    model_info = prepare_for_modelling(input, transformation_name, n_forecast_weeks, n_forecasts)

    # Fit base model on confirmed/stable data
    base_model = fit_base_model(
        model_info.stable_data_dates, model_info.stable_data_values;
        transformation = model_info.transformation,
        n_particles = n_particles,
        smc_data_proportion = smc_data_proportion,
        n_mcmc = n_mcmc,
        n_hmc = n_hmc
    )

    forecasts = isempty(model_info.nowcast_data) ?
                # Direct forecast when no nowcasts
                forecast(base_model, model_info.forecast_dates,
        model_info.n_forecasts_per_nowcast;
        inv_transformation = model_info.inv_transformation) :
                forecast_with_nowcasts(
        base_model, model_info.nowcast_data, model_info.forecast_dates,
        model_info.n_forecasts_per_nowcast;
        inv_transformation = model_info.inv_transformation)

    return (;
        forecast_dates = model_info.forecast_dates,
        forecasts = forecasts
    )
end

"""
    forecast_with_epiautogp(input::EpiAutoGPInput, args::Dict{String, Any}) -> NamedTuple

Run the complete EpiAutoGP modeling pipeline using parsed command-line arguments.

This is the main entry point for command-line usage that combines EpiAutoGPInput data
with parsed command-line arguments to execute the full nowcasting and forecasting workflow.

# Arguments
- `input::EpiAutoGPInput`: The input data structure with epidemiological time series
- `args::Dict{String, Any}`: Parsed command-line arguments from parse_arguments()

# Returns
- Same as forecast_with_epiautogp(): NamedTuple with forecast results and metadata

# Expected command-line arguments
- `"n-forecast-weeks"`: Number of weeks to forecast
- `"n-forecast-draws"`: Total number of forecast samples
- `"transformation"`: Data transformation type
- `"n-particles"`: Number of SMC particles
- `"smc-data-proportion"`: SMC data proportion
- `"n-mcmc"`: Number of MCMC samples
- `"n-hmc"`: Number of HMC samples

# Examples
```julia
# Typical usage pattern
args = parse_arguments()
input_data = read_and_validate_data(args["json-input"])
results = forecast_with_epiautogp(input_data, args)
forecast_dates, forecasts = results.forecast_dates, results.forecasts
```
"""
function forecast_with_epiautogp(input::EpiAutoGPInput, args::Dict{String, Any})
    return forecast_with_epiautogp(input;
        n_forecast_weeks = args["n-forecast-weeks"],
        n_forecasts = args["n-forecast-draws"],
        transformation_name = args["transformation"],
        n_particles = args["n-particles"],
        smc_data_proportion = args["smc-data-proportion"],
        n_mcmc = args["n-mcmc"],
        n_hmc = args["n-hmc"]
    )
end
