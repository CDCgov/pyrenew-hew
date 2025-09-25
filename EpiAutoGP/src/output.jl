abstract type AbstractForecastOutput end
abstract type AbstractHubverseOutput <: AbstractForecastOutput end

@kwdef struct QuantileOutput <: AbstractHubverseOutput
    quantile_levels::Vector{Float64} = [
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
end

function _make_horizon_col(target_end_dates::Vector{Date}, reference_date::Date)
    return [Dates.value(d - reference_date) ÷ 7 for d in target_end_dates]
end

function create_forecast_df(results::NamedTuple, output_type::QuantileOutput)
    # Extract relevant data
    forecast_dates = results.forecast_dates
    forecasts = results.forecasts
    # Create a DataFrame with columns: output_type_id, value, target_end_date, output_type
    forecast_df = DataFrame(output_type_id = Float64[], value = Float64[], target_end_date = Date[])
    # Populate the DataFrame row by row
    for (date_idx, target_end_date) in enumerate(forecast_dates)
        date_samples = forecasts[date_idx, :]
        for q_level in output_type.quantile_levels
            q_value = quantile(date_samples, q_level)
            push!(forecast_df,
                (output_type_id = q_level,
                    value = q_value,
                    target_end_date = target_end_date))
        end
    end
    # Add constant column for output_type, this method is specifically for quantiles
    forecast_df[!, "output_type"] .= "quantile"
    return forecast_df
end

function create_forecast_output(
        input::EpiAutoGPInput,
        results::NamedTuple,
        output_dir::String,
        output_type::AbstractHubverseOutput;
        save_output::Bool,
        disease_abbr::Dict{String, String} = DEFAULT_PATHOGEN_DICT,
        target_abbr::Dict{String, String} = DEFAULT_TARGET_DICT,
        group_name::String = DEFAULT_GROUP_NAME,
        model_name::String = DEFAULT_MODEL_NAME
)
    @info "Creating hubverse table for $(results.disease) with target $(input.target) in $(results.location)"

    # Extract relevant data
    forecast_date = input.forecast_date
    location = input.location
    pathogen = input.pathogen
    target = input.target
    target_col_string = "wk inc $(disease_abbr[pathogen]) $(target_abbr[target])"

    # Create basic forecast DataFrame
    forecast_df = create_forecast_df(results, output_type)

    # Add additional required columns
    forecast_df[!, "reference_date"] .= forecast_date
    forecast_df[!, "location"] .= location
    forecast_df[!, "target"] .= target_col_string

    # Add horizon column
    @transform!(forecast_df, :horizon = _make_horizon_col(:target_end_date, forecast_date))
    # Reorder columns and check all required columns are present
    @select!(forecast_df,
        :output_type, :output_type_id, :value, :reference_date, :target, :horizon, :target_end_date,
        :location)

    # Save as CSV to match expected format
    if save_output
        outputfilename = "$(string(forecast_date))-$(group_name)-$(model_name)-$(location)-$(disease_abbr[pathogen])-$(target).csv"
        csv_path = joinpath(output_dir, outputfilename)
        mkpath(dirname(csv_path))
        CSV.write(csv_path, forecast_df)

        @info "Saved hubverse forecast table to $csv_path"
    end
    
    return forecast_df
end

