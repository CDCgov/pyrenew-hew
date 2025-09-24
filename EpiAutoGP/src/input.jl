"""
EpiAutoGPInput

A structured input data type for EpiAutoGP epidemiological modeling.

# Fields
- `dates`: Vector of observation dates
- `reports`: Vector of case counts/measurements
- `pathogen`: Disease identifier (e.g., "COVID-19")
- `location`: Geographic location (e.g., "CA", "NY")
- `forecast_date`: Reference date for forecasting
- `nowcast_dates`: Dates requiring nowcasting
- `nowcast_reports`: Uncertainty bounds for nowcast dates
"""
struct EpiAutoGPInput
    dates::Vector{Date}
    reports::Vector{Real}
    pathogen::String
    location::String
    forecast_date::Date
    nowcast_dates::Vector{Date}
    nowcast_reports::Vector{Vector{Real}}
end

# Enable JSON3 serialization
StructTypes.StructType(::Type{EpiAutoGPInput}) = StructTypes.Struct()

"""
validate_input(data::EpiAutoGPInput) -> Bool

Validate EpiAutoGPInput data structure for consistency and correctness.
Returns true if valid, throws ArgumentError if invalid.
"""
function validate_input(data::EpiAutoGPInput)
    # Check array length consistency
    if length(data.dates) != length(data.reports)
        throw(ArgumentError("Length mismatch: dates ($(length(data.dates))) and reports ($(length(data.reports))) must have the same length"))
    end

    # Check for non-empty essential data
    if length(data.dates) == 0
        throw(ArgumentError("Empty data: dates and reports cannot be empty"))
    end

    # Check date ordering
    if !issorted(data.dates)
        throw(ArgumentError("Date ordering: dates must be sorted chronologically"))
    end

    # Check nowcast data consistency
    if length(data.nowcast_dates) != length(data.nowcast_reports)
        throw(ArgumentError("Nowcast length mismatch: nowcast_dates ($(length(data.nowcast_dates))) and nowcast_reports ($(length(data.nowcast_reports))) must have the same length"))
    end

    # Check nowcast date ordering
    if !isempty(data.nowcast_dates) && !issorted(data.nowcast_dates)
        throw(ArgumentError("Nowcast date ordering: nowcast_dates must be sorted chronologically"))
    end

    # Check string identifiers
    if isempty(strip(data.pathogen))
        throw(ArgumentError("Invalid pathogen: pathogen cannot be empty or whitespace"))
    end

    if isempty(strip(data.location))
        throw(ArgumentError("Invalid location: location cannot be empty or whitespace"))
    end

    # Check numerical validity
    for (i, report) in enumerate(data.reports)
        if !isfinite(report) || report < 0
            throw(ArgumentError("Invalid report value at index $i: reports must be non-negative finite numbers (got $report)"))
        end
    end

    # Check nowcast reports validity
    for (i, report_vec) in enumerate(data.nowcast_reports)
        if isempty(report_vec)
            throw(ArgumentError("Empty nowcast reports at index $i: each nowcast_reports entry must contain at least one value"))
        end
        for (j, report) in enumerate(report_vec)
            if !isfinite(report) || report < 0
                throw(ArgumentError("Invalid nowcast report value at index [$i][$j]: must be non-negative finite number (got $report)"))
            end
        end
    end

    # Check forecast date reasonableness
    if !isempty(data.dates)
        date_range = maximum(data.dates) - minimum(data.dates)
        days_buffer = max(30, Int(ceil(date_range.value / 10)))

        if data.forecast_date < minimum(data.dates) - Day(days_buffer)
            throw(ArgumentError("Forecast date ($(data.forecast_date)) is too far before the data range ($(minimum(data.dates)) to $(maximum(data.dates)))"))
        end

        if data.forecast_date > maximum(data.dates) + Day(days_buffer)
            throw(ArgumentError("Forecast date ($(data.forecast_date)) is too far after the data range ($(minimum(data.dates)) to $(maximum(data.dates)))"))
        end
    end

    return true
end

"""
read_data(path_to_json::String) -> EpiAutoGPInput

Read and parse epidemiological input data from a JSON file.
"""
function read_data(path_to_json::String)
    json_string = read(path_to_json, String)
    data = JSON3.read(json_string, EpiAutoGPInput)
    return data
end

"""
read_and_validate_data(path_to_json::String) -> EpiAutoGPInput

Read epidemiological data from JSON file with automatic validation.
Combines read_data and validate_input for production use.
"""
function read_and_validate_data(path_to_json::String)
    data = read_data(path_to_json)
    validate_input(data)
    return data
end

"""
safe_read_data(path_to_json::String) -> Union{EpiAutoGPInput, Nothing}

Safely read data from JSON file, returning Nothing on any error.
Useful for optional data loading or when errors should be handled silently.
"""
function safe_read_data(path_to_json::String)
    try
        return read_and_validate_data(path_to_json)
    catch e
        @warn "Failed to read data from $path_to_json: $(e)"
        return nothing
    end
end

"""
validate_and_report(data::EpiAutoGPInput) -> Tuple{Bool, String}

Validate data and return both success status and detailed message.
Returns (true, "Validation passed") on success or (false, error_message) on failure.
"""
function validate_and_report(data::EpiAutoGPInput)
    try
        validate_input(data)
        return (true, "Validation passed")
    catch e
        return (false, string(e))
    end
end

"""
create_sample_input(;
    n_days::Int=30,
    pathogen::String="COVID-19",
    location::String="CA"
) -> EpiAutoGPInput

Create a sample EpiAutoGPInput for testing and demonstration purposes.
"""
function create_sample_input(; n_days::Int=30, pathogen::String="COVID-19", location::String="CA")
    start_date = Date("2024-01-01")
    dates = [start_date + Day(i) for i in 0:(n_days-1)]
    reports = [rand(20:100) + 10*sin(2π*i/7) + rand() * 5 for i in 1:n_days]  # Weekly pattern with noise

    forecast_date = dates[end]
    nowcast_dates = dates[max(1, end-2):end]  # Last 3 days
    nowcast_reports = [[r + rand(-5:5) for _ in 1:3] for r in reports[max(1, end-2):end]]

    return EpiAutoGPInput(
        dates, reports, pathogen, location,
        forecast_date, nowcast_dates, nowcast_reports
    )
end
