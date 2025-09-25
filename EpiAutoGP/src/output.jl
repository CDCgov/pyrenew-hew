abstract type AbstractForecastOutput end
abstract type AbstractHubverseOutput <: AbstractForecastOutput end

@kwdef struct QuantileOutput <: AbstractHubverseOutput
    quantile_levels::Vector{Float64} = [
        0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
end

function create_df(forecast_dates, forecasts, output_type::QuantileOutput)
    df = DataFrame(output_type_id = Float64[], value = Float64[], target_end_date = String[])
    for (date_idx, target_end_date) in enumerate(forecast_dates)
        date_samples = forecasts[date_idx, :]
        for q_level in output_type.quantile_levels
            q_value = quantile(date_samples, q_level)
            push!(df,
                (output_type_id = q_level,
                    value = q_value,
                    target_end_date = string(target_end_date)))
        end
    end
    df[!, "output_type"] .= "quantile"
    return df
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
#     forecast_dates = results.forecast_dates
#     forecasts = results.forecasts
#     forecast_date = input.forecast_date
#     location = input.location
#     pathogen = input.pathogen
#     target = input.target
#     target_col_string = "wk inc $(disease_abbr[pathogen]) $(target_abbr[target])"

#     output_df = _format_output()

#     for (date_idx, target_end_date) in enumerate(forecast_dates)
#         date_samples = forecast_samples[date_idx, :]

#         # Calculate horizon as weeks from forecast_date to target_end_date
#         # This handles both positive (forecasts) and negative (nowcasts) horizons
#         horizon = Int(Dates.value(target_end_date - forecast_date) ÷ 7)

#         for q_level in quantile_levels
#             # Calculate quantile value
#             q_value = quantile(date_samples, q_level)

#             # Create row matching exact hubverse format
#             row = (
#                 target_end_date = string(target_end_date),
#                 value = round(q_value, digits = 6),
#                 output_type_id = q_level,
#                 horizon = horizon,
#                 output_type = "quantile",
#                 reference_date = string(forecast_date),
#                 location = location,  # Keep as provided (2-letter abbreviation for now)
#                 target = target
#             )

#             push!(hubverse_rows, row)
#         end
#     end

#     # Convert to DataFrame
#     hubverse_df = DataFrame(hubverse_rows)

#     # Save as CSV to match expected format
#     csv_path = joinpath(output_dir, "hubverse_table.csv")
#     mkpath(dirname(csv_path))
#     CSV.write(csv_path, hubverse_df)

#     @info "Saved hubverse table to $csv_path"
#     @info "Table contains $(nrow(hubverse_df)) rows"

#     return hubverse_df
# end
