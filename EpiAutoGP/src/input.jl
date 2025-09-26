"""
    struct EpiAutoGPInput

A structured input data type for EpiAutoGP epidemiological modeling.

This struct represents the complete input dataset required for running epidemiological
forecasting models in combination with nowcasting using `NowcastAutoGP.jl`. It combines historical observation data
with nowcasting requirements and forecast parameters.

# Fields
- `dates::Vector{Date}`: Vector of observation dates in chronological order
- `reports::Vector{Real}`: Vector of case counts/measurements corresponding to each date
- `pathogen::String`: Disease identifier (e.g., "COVID-19", "Influenza", "RSV")
- `location::String`: Geographic location identifier (e.g., "CA", "NY", "US")
- `forecast_date::Date`: Reference date from which forecasting begins
- `nowcast_dates::Vector{Date}`: Dates requiring nowcasting (typically recent dates with incomplete data)
- `nowcast_reports::Vector{Vector{Real}}`: Uncertainty bounds or samples for nowcast dates

# Examples
```julia
# Create a simple input dataset
data = EpiAutoGPInput(
    [Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03")],
    [45.0, 52.0, 38.0],
    "COVID-19",
    "CA",
    Date("2024-01-03"),
    [Date("2024-01-02"), Date("2024-01-03")],
    [[50.0, 52.0, 54.0], [36.0, 38.0, 40.0]]
)

# Validate the input
validate_input(data)  # returns true if valid
```
"""
struct EpiAutoGPInput
    dates::Vector{Date}
    reports::Vector{Real}
    pathogen::String
    location::String
    target::String
    forecast_date::Date
    nowcast_dates::Vector{Date}
    nowcast_reports::Vector{Vector{Real}}
end

# Enable JSON3 serialization
StructTypes.StructType(::Type{EpiAutoGPInput}) = StructTypes.Struct()

"""
    function validate_input(data::EpiAutoGPInput)

Validate an `EpiAutoGPInput` data structure for consistency and correctness.

Performs comprehensive validation including:
- Array length consistency between dates and reports
- Chronological ordering of dates and nowcast dates
- Non-negative finite values for all reports
- Non-empty string identifiers for pathogen and location
- Reasonable forecast date relative to data range
- Proper structure of nowcast data

# Arguments
- `data::EpiAutoGPInput`: The input data structure to validate

# Returns
- `Bool`: Returns `true` if validation passes

# Throws
- `ArgumentError`: If any validation check fails, with descriptive error message

# Examples
```julia
# Valid data passes validation
valid_data = EpiAutoGPInput(
    [Date("2024-01-01"), Date("2024-01-02")],
    [45.0, 52.0],
    "COVID-19", "CA", Date("2024-01-02"),
    Date[], Vector{Real}[]
)
validate_input(valid_data)  # returns true

# Invalid data throws ArgumentError
invalid_data = EpiAutoGPInput(
    [Date("2024-01-01")],
    [-5.0],  # negative values not allowed
    "COVID-19", "CA", Date("2024-01-01"),
    Date[], Vector{Real}[]
)
validate_input(invalid_data)  # throws ArgumentError
```
"""
function validate_input(data::EpiAutoGPInput; valid_targets = ["nhsn", "nssp"])
    @assert data.target in valid_targets "Target must be one of $(valid_targets), got '$(data.target)'"
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
    # If no nowcast dates, should have no nowcast reports (pure forecasting)
    if isempty(data.nowcast_dates) && !isempty(data.nowcast_reports)
        throw(ArgumentError("Nowcast consistency error: no nowcast_dates provided but nowcast_reports is not empty"))
    end

    # If nowcast dates exist, each vector in nowcast_reports should have length equal to number of nowcast_dates
    # (each vector represents one realization across all nowcast dates)
    if !isempty(data.nowcast_dates)
        for (i, report_vec) in enumerate(data.nowcast_reports)
            if length(report_vec) != length(data.nowcast_dates)
                throw(ArgumentError("Nowcast vector length mismatch at index $i: nowcast_reports[$i] has length $(length(report_vec)) but should have length $(length(data.nowcast_dates)) to match nowcast_dates"))
            end
        end
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
    function read_data(path_to_json::String)

Read and parse epidemiological input data from a JSON file.

This function reads a JSON file and deserializes it into an `EpiAutoGPInput` struct
using JSON3.jl. The JSON file should contain all required fields matching the
struct definition.

# Arguments
- `path_to_json::String`: Path to the JSON file containing input data

# Returns
- `EpiAutoGPInput`: Parsed data structure ready for model input

# Throws
- `SystemError`: If the file cannot be read (e.g., file not found)
- `JSON3.StructuralError`: If JSON structure doesn't match expected format
- `ArgumentError`: If date parsing fails or data types are incompatible

# Examples
```julia
# Read data from a JSON file
data = read_data("path/to/input_data.json")

# The JSON file should have structure like:
# {
#   "dates": ["2024-01-01", "2024-01-02"],
#   "reports": [45.0, 52.0],
#   "pathogen": "COVID-19",
#   "location": "CA",
#   "forecast_date": "2024-01-02",
#   "nowcast_dates": [],
#   "nowcast_reports": []
# }
```

!!! note
    This function does not validate the data. Use [`read_and_validate_data`](@ref)
    for automatic validation, or call [`validate_input`](@ref) separately.
"""
function read_data(path_to_json::String)
    json_string = read(path_to_json, String)
    data = JSON3.read(json_string, EpiAutoGPInput)
    return data
end

"""
    read_and_validate_data(path_to_json::String) -> EpiAutoGPInput

Read epidemiological data from JSON file with automatic validation.

This is the recommended function for loading input data in production workflows.
It combines [`read_data`](@ref) and [`validate_input`](@ref) to ensure that
loaded data is both structurally correct and passes all validation checks.

# Arguments
- `path_to_json::String`: Path to the JSON file containing input data

# Returns
- `EpiAutoGPInput`: Validated data structure ready for modeling

# Throws
- `SystemError`: If the file cannot be read
- `JSON3.StructuralError`: If JSON structure is invalid
- `ArgumentError`: If data fails validation checks

# Examples
```julia
# Load and validate data in one step
data = read_and_validate_data("epidata.json")

# This is equivalent to:
data = read_data("epidata.json")
validate_input(data)

# Use in a try-catch block for error handling
try
    data = read_and_validate_data("uncertain_data.json")
    println("Data loaded successfully")
catch e
    @error "Failed to load data" exception=e
end
```

!!! tip "Production Usage"
    This function is preferred over [`read_data`](@ref) for production workflows
    as it ensures data integrity before model execution.

See also: [`read_data`](@ref), [`validate_input`](@ref), [`EpiAutoGPInput`](@ref)
"""
function read_and_validate_data(path_to_json::String)
    data = read_data(path_to_json)
    validate_input(data)
    return data
end
