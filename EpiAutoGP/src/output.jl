abstract type AbstractForecastOutput end
abstract type AbstractHubverseOutput <: AbstractForecastOutput end

@kwdef struct QuantileOutput <: AbstractHubverseOutput
    quantile_levels::Vector{Float64} = [
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
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

# function create_forecast_output(
#         input::EpiAutoGPInput,
#         results::NamedTuple,
#         output_dir::String,
#         output_type::AbstractHubverseOutput;
#         disease_abbr::Dict{String, String} = DEFAULT_PATHOGEN_DICT,
#         target_abbr::Dict{String, String} = DEFAULT_TARGET_DICT
# )
#     @info "Creating hubverse table for $(results.disease) with target $(input.target) in $(results.location)"

#     # Extract relevant data    
#     forecast_date = input.forecast_date
#     location = input.location
#     pathogen = input.pathogen
#     target = input.target
#     target_col_string = "wk inc $(disease_abbr[pathogen]) $(target_abbr[target])"

#     # Create basic forecast DataFrame
#     forecast_df = create_forecast_df(results, output_type)

#     # Add additional required columns
#     forecast_df[!, "reference_date"] .= string(forecast_date)
#     forecast_df[!, "horizon"] .= [Dates.value(d - forecast_date) ÷ 7 for d in forecast_df.target_end_date]
#     forecast_df[!, "location"] .= location
#     forecast_df[!, "target"] .= target_col_string

#     # Reorder columns and check all required columns are present
#     forecast_df = @select!(forecast_df, [output_type, :location, :target, :target_end_date, :horizon, :output_type, :output_type_id, :value])
    


#     # Save as CSV to match expected format
#     csv_path = joinpath(output_dir, "hubverse_table.csv")
#     mkpath(dirname(csv_path))
#     CSV.write(csv_path, hubverse_df)

#     @info "Saved hubverse table to $csv_path"
#     @info "Table contains $(nrow(hubverse_df)) rows"

#     return hubverse_df
# end
