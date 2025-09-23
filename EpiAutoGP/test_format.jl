#!/usr/bin/env julia

# Minimal test to verify real PyRenew-HEW data format compatibility
using JSON

test_data_file = "test/data/bootstrap_private_data/MT/data/data_for_model_fit.json"

println("Testing PyRenew-HEW data format compatibility...")
println("Loading: $test_data_file")

# Load the data
data = JSON.parsefile(test_data_file)

# Check the structure
println("\nTop-level keys:")
for key in sort(collect(keys(data)))
    println("  - $key")
end

# Check NWSS data structure (this is the tricky one)
nwss_data = data["nwss_training_data"]
println("\nNWSS data structure:")
println("  Type: $(typeof(nwss_data))")
for key in sort(collect(keys(nwss_data)))
    value = nwss_data[key]
    println("  - $key: $(typeof(value)) with $(length(value)) elements")
    if key == "date"
        println("    Sample dates: $(value[1]) to $(value[end])")
    elseif key == "log_genome_copies_per_ml"
        unique_vals = unique(value)
        println("    Unique values: $(length(unique_vals)) (sample: $(unique_vals[1]))")
    end
end

# Test that we can create a basic "DataFrame-like" structure
println("\nTesting DataFrame construction...")
try
    # Simulate what our extract_time_series_data function does
    dates = nwss_data["date"]
    values = nwss_data["log_genome_copies_per_ml"]
    below_lod = nwss_data["below_lod"]
    
    println("  Successfully accessed arrays:")
    println("    Dates: $(length(dates)) elements")
    println("    Values: $(length(values)) elements") 
    println("    Below LOD: $(length(below_lod)) elements")
    
    # Check consistency
    if length(dates) == length(values) == length(below_lod)
        println("  ✓ All arrays have consistent lengths")
    else
        println("  ✗ Array length mismatch!")
    end
    
catch e
    println("  ✗ Error: $e")
end

println("\nFormat compatibility test completed!")