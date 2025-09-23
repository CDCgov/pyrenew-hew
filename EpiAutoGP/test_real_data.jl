#!/usr/bin/env julia

# Simple test to verify our implementation works with real PyRenew-HEW data
using JSON3
using DataFrames

# Load our implementation
include("../input.jl")

# Test with real PyRenew-HEW data  
test_data_file = "test/data/bootstrap_private_data/MT/data/data_for_model_fit.json"

println("Testing with real PyRenew-HEW data from: $test_data_file")

# Load and process the data
data = load_json_data(test_data_file)
time_series_data = extract_time_series_data(data)
epiautogp_data = prepare_epiautogp_data(time_series_data)

# Display results
println("\nSuccessfully processed real PyRenew-HEW data!")
println("Time series data shape: $(size(time_series_data))")
println("Available columns: $(names(time_series_data))")
println("Date range: $(minimum(time_series_data.date)) to $(maximum(time_series_data.date))")

# Test data types and structure
println("\nData types:")
println("- NHSN data rows: $(sum(time_series_data.data_source .== "nhsn"))")
println("- NSSP data rows: $(sum(time_series_data.data_source .== "nssp"))")  
println("- NWSS data rows: $(sum(time_series_data.data_source .== "nwss"))")

println("\nNWSS data sample (first 5 rows):")
nwss_data = time_series_data[time_series_data.data_source .== "nwss", :]
if nrow(nwss_data) > 0
    println(first(nwss_data, 5))
end

println("\nTest completed successfully!")