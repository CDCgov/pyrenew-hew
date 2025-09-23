using TidierData
using JSON3
using Dates

# Valid data keys for surveillance systems
const VALID_DATA_KEYS = ["nssp_training_data", "nhsn_training_data", "nwss_training_data"]

# Valid nowcast data keys
const VALID_NOWCAST_KEYS = ["nowcast_samples", "nowcast_dates"]

"""
    infer_location_from_json(json_data)

Infer location from JSON data by checking common location fields in surveillance data.

# Arguments
- `json_data`: Parsed JSON data object from PyRenew-HEW

# Returns
- `String`: Location identifier (e.g., state abbreviation) or throws error if not found

# Notes
Searches for location information in the following priority order:
1. `nwss_training_data.location` field
2. `nssp_training_data.geo_value` field  
3. `nhsn_training_data.jurisdiction` field
"""
function infer_location_from_json(json_data)
    # Try NWSS data first (most likely to have clean location field)
    if haskey(json_data, "nwss_training_data") && !isnothing(json_data["nwss_training_data"])
        nwss_data = json_data["nwss_training_data"]
        if haskey(nwss_data, "location") && !isempty(nwss_data["location"])
            locations = unique(nwss_data["location"])
            if length(locations) == 1
                @info "Inferred location from NWSS data: $(locations[1])"
                return String(locations[1])
            elseif length(locations) > 1
                @warn "Multiple locations found in NWSS data: $locations, using first: $(locations[1])"
                return String(locations[1])
            end
        end
    end
    
    # Try NSSP data (ED visits)
    if haskey(json_data, "nssp_training_data") && !isnothing(json_data["nssp_training_data"])
        nssp_data = json_data["nssp_training_data"]
        if haskey(nssp_data, "geo_value") && !isempty(nssp_data["geo_value"])
            geo_values = unique(nssp_data["geo_value"])
            if length(geo_values) == 1
                @info "Inferred location from NSSP data: $(geo_values[1])"
                return String(geo_values[1])
            elseif length(geo_values) > 1
                @warn "Multiple geo_values found in NSSP data: $geo_values, using first: $(geo_values[1])"
                return String(geo_values[1])
            end
        end
    end
    
    # Try NHSN data (hospital admissions)
    if haskey(json_data, "nhsn_training_data") && !isnothing(json_data["nhsn_training_data"])
        nhsn_data = json_data["nhsn_training_data"]
        if haskey(nhsn_data, "jurisdiction") && !isempty(nhsn_data["jurisdiction"])
            jurisdictions = unique(nhsn_data["jurisdiction"])
            if length(jurisdictions) == 1
                @info "Inferred location from NHSN data: $(jurisdictions[1])"
                return String(jurisdictions[1])
            elseif length(jurisdictions) > 1
                @warn "Multiple jurisdictions found in NHSN data: $jurisdictions, using first: $(jurisdictions[1])"
                return String(jurisdictions[1])
            end
        end
    end
    
    error("Could not infer location from JSON data. No valid location information found in any surveillance data stream.")
end

"""
    infer_disease_from_json(json_data)

Infer disease from JSON data by checking disease-specific fields in surveillance data.

# Arguments
- `json_data`: Parsed JSON data object from PyRenew-HEW

# Returns
- `String`: Disease identifier (e.g., "COVID-19", "influenza") or throws error if not found

# Notes
Currently returns a placeholder since disease inference logic needs to be developed
based on the specific data structure patterns in the PyRenew-HEW pipeline.
For now, suggests using a default disease or implementing custom logic.
"""
function infer_disease_from_json(json_data)
    # For now, we'll return a placeholder since disease inference 
    # depends on specific conventions in the PyRenew-HEW pipeline
    # This could be enhanced to look for disease-specific indicators
    @warn "Disease inference from JSON data not yet implemented. Please specify --disease explicitly or implement custom inference logic."
    return "COVID-19"  # Default fallback
end

"""
    load_json_data(json_file_path::String)

Load and parse JSON input data compatible with PyRenew-HEW format.

# Arguments
- `json_file_path::String`: Path to the JSON input file

# Returns
- Parsed JSON data object

# Throws
- `SystemError`: If the JSON file is not found
- `Error`: If JSON parsing fails
"""
function load_json_data(json_file_path::String)
    @info "Loading JSON data from $json_file_path"
    
    if !isfile(json_file_path)
        error("JSON input file not found: $json_file_path")
    end
    
    try
        data = JSON3.read(read(json_file_path, String))
        @info "Successfully loaded JSON data"
        return data
    catch e
        error("Failed to parse JSON file: $e")
    end
end

"""
    extract_time_series_data(json_data, data_key::String)

Extract time series data from JSON input for the specified data key.

# Arguments
- `json_data`: Parsed JSON data object
- `data_key::String`: Key for data to extract ("nssp_training_data", "nhsn_training_data", or "nwss_training_data")

# Returns
- `DataFrame`: Extracted time series data, or `nothing` if data is not available
"""
function extract_time_series_data(json_data, data_key::String)
    @assert data_key ∈ VALID_DATA_KEYS "Invalid data_key: $data_key. Must be one of $VALID_DATA_KEYS"
    
    if !haskey(json_data, data_key) || isnothing(json_data[data_key])
        @warn "No data found for $data_key"
        return nothing
    end
    
    data_dict = json_data[data_key]
    if isempty(data_dict)
        @warn "Empty data dictionary for $data_key"
        return nothing
    end
    
    # Convert dictionary format (keys=columns, values=lists) to DataFrame
    # This matches the Polars .to_dict(as_series=False) format
    df = DataFrame(data_dict)
    @info "Extracted $(nrow(df)) rows for $data_key with columns: $(names(df))"
    
    return df
end

"""
    prepare_epiautogp_data(json_data, disease::String, location::String)

Transform PyRenew-HEW JSON data format into EpiAutoGP-compatible format.

# Arguments
- `json_data`: Parsed JSON data object from PyRenew-HEW
- `disease::String`: Disease identifier (e.g., "COVID-19", "influenza")
- `location::String`: Location identifier (e.g., state abbreviation)

# Returns
- `DataFrame`: Standardized time series data with columns: date, count, location, disease

# Notes
Currently focuses on ED visit data as the primary signal for EpiAutoGP modeling.
Requires "nssp_training_data" with "observed_ed_visits" column to be present.
"""
function prepare_epiautogp_data(json_data, disease::String, location::String)
    @info "Preparing EpiAutoGP data for $disease in $location"
    
    # Extract different data streams
    ed_data = extract_time_series_data(json_data, "nssp_training_data")
    hosp_data = extract_time_series_data(json_data, "nhsn_training_data") 
    ww_data = extract_time_series_data(json_data, "nwss_training_data")
    
    # For EpiAutoGP, we'll focus on ED visit data as the primary signal
    if isnothing(ed_data)
        error("No ED visit data available - required for EpiAutoGP model")
    end
    
    # Convert date column if present
    if "date" in names(ed_data)
        ed_data.date = Date.(ed_data.date)
    end
    
    # Filter for disease-specific ED visits if available
    if "observed_ed_visits" in names(ed_data)
        # Create standardized format expected by EpiAutoGP using TidierData
        time_series_df = DataFrame(
            date = ed_data.date,
            count = ed_data.observed_ed_visits,
            location = fill(location, nrow(ed_data)),
            disease = fill(disease, nrow(ed_data))
        )
    else
        error("No 'observed_ed_visits' column found in ED data")
    end
    
    # Sort by date
    sort!(time_series_df, :date)
    
    @info "Prepared time series with $(nrow(time_series_df)) observations from $(minimum(time_series_df.date)) to $(maximum(time_series_df.date))"
    
    return time_series_df
end

"""
    extract_nowcast_data(json_data)

Extract nowcast data from JSON input for use with forecast_with_nowcasts.

# Arguments
- `json_data`: Parsed JSON data object from PyRenew-HEW

# Returns
- `NamedTuple`: Contains `(samples, dates)` where:
  - `samples`: Vector of vectors of nowcast samples
  - `dates`: Vector of Date objects for nowcast time points
- `nothing`: If no nowcast data is available

# Notes
Expects JSON structure with vector of vectors format:
```json
{
  "nowcast_samples": [[scenario1_val1, scenario1_val2], [scenario2_val1, scenario2_val2], ...],
  "nowcast_dates": ["2024-01-01", "2024-01-02", ...]
}
```

Each inner vector represents one nowcast scenario across time points.
"""
function extract_nowcast_data(json_data)
    # Check if nowcast data is present using same pattern as other functions
    @assert all(key -> haskey(json_data, key), VALID_NOWCAST_KEYS) "Missing nowcast data keys. Required: $VALID_NOWCAST_KEYS"
    
    raw_samples = json_data["nowcast_samples"]
    raw_dates = json_data["nowcast_dates"]
    
    if isnothing(raw_samples) || isnothing(raw_dates) || isempty(raw_samples) || isempty(raw_dates)
        @info "Empty nowcast data in JSON input"
        return nothing
    end
    
    # Convert dates
    dates = Date.(raw_dates)
    
    # Validate vector of vectors format
    if !(raw_samples isa AbstractVector) || !all(x -> x isa AbstractVector, raw_samples)
        error("nowcast_samples must be a vector of vectors format: [[scenario1], [scenario2], ...]")
    end
    
    # Convert to proper vector of vectors with Float64 values
    samples = [Vector{Float64}(scenario) for scenario in raw_samples]
    n_scenarios = length(samples)
    n_timepoints = length(samples[1])
    
    # Validate all scenarios have same length
    if !all(length(scenario) == n_timepoints for scenario in samples)
        error("All nowcast scenarios must have the same number of time points")
    end
    
    # Validate dates match sample dimensions
    if length(dates) != n_timepoints
        error("Number of nowcast dates ($(length(dates))) must match number of time points per scenario ($n_timepoints)")
    end
    
    @info "Extracted $n_scenarios nowcast scenarios with $n_timepoints time points each"
    @info "Successfully extracted nowcast data for dates from $(minimum(dates)) to $(maximum(dates))"
    
    return (samples = samples, dates = dates)
end
