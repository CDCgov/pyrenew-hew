#!/usr/bin/env julia

"""
Test script to validate the generated EpiAutoGP JSON input.

This script loads and validates the JSON file created by the Python script
to ensure it matches the EpiAutoGPInput format expected by the Julia model.
"""

using EpiAutoGP
using JSON3
using Dates

function test_json_input(json_path::String)
    """Test that the JSON file can be loaded and validated by EpiAutoGP."""

    println("=== Testing EpiAutoGP JSON Input ===")
    println("Loading JSON file: $json_path")

    try
        # Load the JSON data
        data = read_and_validate_data(json_path)

        println("✅ Successfully loaded and validated JSON data!")
        println()
        println("=== Data Summary ===")
        println("Pathogen: $(data.pathogen)")
        println("Location: $(data.location)")
        println("Target: $(data.target)")
        println("Forecast date: $(data.forecast_date)")
        println("Number of dates: $(length(data.dates))")
        println("Date range: $(minimum(data.dates)) to $(maximum(data.dates))")
        println("Number of reports: $(length(data.reports))")
        println("Report range: $(minimum(data.reports)) to $(maximum(data.reports))")
        println("Number of nowcast dates: $(length(data.nowcast_dates))")
        println("Number of nowcast reports: $(length(data.nowcast_reports))")

        # Check data quality
        println()
        println("=== Data Quality Checks ===")

        # Check for missing values
        if any(ismissing, data.reports)
            println("⚠️  Warning: Found missing values in reports")
        else
            println("✅ No missing values in reports")
        end

        # Check for negative values
        if any(r -> r < 0, data.reports)
            println("⚠️  Warning: Found negative values in reports")
        else
            println("✅ All report values are non-negative")
        end

        # Check date ordering
        if issorted(data.dates)
            println("✅ Dates are properly sorted")
        else
            println("⚠️  Warning: Dates are not sorted")
        end

        # Check for reasonable forecast date
        if data.forecast_date >= minimum(data.dates) &&
           data.forecast_date <= maximum(data.dates) + Day(30)
            println("✅ Forecast date is reasonable")
        else
            println("⚠️  Warning: Forecast date seems unreasonable")
        end

        println()
        println("=== Test Results ===")
        println("✅ JSON file successfully loaded and validated!")
        println("✅ Data structure matches EpiAutoGPInput format")
        println("✅ Ready for use with EpiAutoGP modeling")

        return true

    catch e
        println("❌ Error loading or validating JSON data:")
        println("   $(typeof(e)): $e")

        # Print stack trace for debugging
        println()
        println("Stack trace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end

        return false
    end
end

function main()
    """Main function to test the JSON input."""
    json_file = joinpath(@__DIR__(), "epiautogp_input_2025-08-16.json")

    if !isfile(json_file)
        println("❌ JSON file not found: $json_file")
        println("Please run the Python script first to create the input file.")
        return false
    end

    success = test_json_input(json_file)

    if success
        println("\n🎉 All tests passed! The JSON file is ready for EpiAutoGP modeling.")
    else
        println("\n💥 Tests failed. Please check the JSON file format.")
    end

    return success
end

# Run the test if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
