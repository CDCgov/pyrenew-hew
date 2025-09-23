#!/usr/bin/env julia

# Test with real PyRenew-HEW data using the existing project setup
using Test

# Load our implementation
include("../input.jl")

@testset "Real PyRenew-HEW Data Tests" begin
    # Test with MT data
    mt_data_file = "test/data/bootstrap_private_data/MT/data/data_for_model_fit.json"
    if isfile(mt_data_file)
        @info "Testing with real MT data: $mt_data_file"
        
        # Load the data
        data = load_json_data(mt_data_file)
        @test !isnothing(data)
        
        # Test each data source separately with correct function signature
        @info "Testing NHSN data extraction..."
        nhsn_data = extract_time_series_data(data, "nhsn_training_data")
        @test nrow(nhsn_data) > 0
        @info "  NHSN rows: $(nrow(nhsn_data))"
        
        @info "Testing NSSP data extraction..."
        nssp_data = extract_time_series_data(data, "nssp_training_data")
        @test nrow(nssp_data) > 0
        @info "  NSSP rows: $(nrow(nssp_data))"
        
        @info "Testing NWSS data extraction..."
        nwss_data = extract_time_series_data(data, "nwss_training_data")
        @test nrow(nwss_data) > 0
        @info "  NWSS rows: $(nrow(nwss_data))"
        
        # Test EpiAutoGP data preparation with correct function signature
        @info "Testing EpiAutoGP data preparation..."
        epiautogp_data = prepare_epiautogp_data(data, "COVID-19", "MT")
        @test !isnothing(epiautogp_data)
        @info "✓ Successfully prepared EpiAutoGP data for MT"
        
    else
        @warn "MT data file not found: $mt_data_file"
    end
    
    # Test with DC data
    dc_data_file = "test/data/bootstrap_private_data/DC/data/data_for_model_fit.json"
    if isfile(dc_data_file)
        @info "Testing with real DC data: $dc_data_file"
        
        # Load the data
        data = load_json_data(dc_data_file)
        @test !isnothing(data)
        
        # Test each data source separately with correct function signature
        @info "Testing DC data sources..."
        nhsn_data = extract_time_series_data(data, "nhsn_training_data")
        @test nrow(nhsn_data) > 0
        
        nssp_data = extract_time_series_data(data, "nssp_training_data")
        @test nrow(nssp_data) > 0
        
        nwss_data = extract_time_series_data(data, "nwss_training_data")
        @test nrow(nwss_data) > 0
        
        # Test EpiAutoGP data preparation with correct function signature
        epiautogp_data = prepare_epiautogp_data(data, "COVID-19", "DC")
        @test !isnothing(epiautogp_data)
        @info "✓ Successfully prepared EpiAutoGP data for DC"
        
    else
        @warn "DC data file not found: $dc_data_file"
    end
end

@info "Real data compatibility tests completed!"