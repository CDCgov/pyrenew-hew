using Test
using TidierData
using JSON3
using Dates
using Logging

# Include the input module 
include("../input.jl")

@testset "Input Functions Tests" begin
    
    # Setup: Create temporary directory and test JSON files
    temp_dir = mktempdir()
    
    try
        # Define test data sets - matching Polars .to_dict(as_series=False) format
        comprehensive_data = Dict(
            "nssp_training_data" => Dict(
                "date" => ["2024-01-01", "2024-01-02", "2024-01-03"],
                "observed_ed_visits" => [100, 110, 120],
                "other_ed_visits" => [50, 55, 60],
                "geo_value" => ["NY", "NY", "NY"],
                "data_type" => ["train", "train", "train"]
            ),
            "nhsn_training_data" => Dict(
                "weekendingdate" => ["2024-01-01"],
                "hospital_admissions" => [50],
                "jurisdiction" => ["NY"],
                "data_type" => ["train"]
            ),
            "nwss_training_data" => Dict(
                "date" => ["2024-01-01"],
                "site_level_log_ww_conc" => [3.0],
                "location" => ["NY"],
                "data_type" => ["train"]
            )
        )
        
        partial_data = Dict("other_data" => Dict("value" => [1]))
        null_data = Dict("nssp_training_data" => nothing)
        empty_data = Dict("nssp_training_data" => Dict())
        no_ed_data = Dict("nhsn_training_data" => Dict(
            "weekendingdate" => ["2024-01-01"], 
            "hospital_admissions" => [50],
            "jurisdiction" => ["NY"],
            "data_type" => ["train"]
        ))
        wrong_ed_column = Dict("nssp_training_data" => Dict("date" => ["2024-01-01"], "other_column" => [100]))
        single_point_data = Dict("nssp_training_data" => Dict("date" => ["2024-01-01"], "observed_ed_visits" => [100]))
        simple_valid_data = Dict("test_key" => "test_value", "number" => 42)
        
        # Create JSON files using JSON3.write
        comprehensive_file = joinpath(temp_dir, "comprehensive.json")
        partial_file = joinpath(temp_dir, "partial.json")
        null_file = joinpath(temp_dir, "null.json")
        empty_file = joinpath(temp_dir, "empty.json")
        no_ed_file = joinpath(temp_dir, "no_ed.json")
        wrong_column_file = joinpath(temp_dir, "wrong_column.json")
        single_point_file = joinpath(temp_dir, "single_point.json")
        valid_json_file = joinpath(temp_dir, "valid.json")
        invalid_json_file = joinpath(temp_dir, "invalid.json")
        
        # Write all JSON files with proper JSON3 serialization
        open(comprehensive_file, "w") do io; JSON3.write(io, comprehensive_data); end
        open(partial_file, "w") do io; JSON3.write(io, partial_data); end
        open(null_file, "w") do io; JSON3.write(io, null_data); end
        open(empty_file, "w") do io; JSON3.write(io, empty_data); end
        open(no_ed_file, "w") do io; JSON3.write(io, no_ed_data); end
        open(wrong_column_file, "w") do io; JSON3.write(io, wrong_ed_column); end
        open(single_point_file, "w") do io; JSON3.write(io, single_point_data); end
        open(valid_json_file, "w") do io; JSON3.write(io, simple_valid_data); end
        
        # Create invalid JSON file (malformed JSON)
        write(invalid_json_file, "{ invalid json content")
        
        @testset "Data Key Validation" begin
            @testset "Valid data keys" begin
                @test "nssp_training_data" ∈ VALID_DATA_KEYS
                @test "nhsn_training_data" ∈ VALID_DATA_KEYS
                @test "nwss_training_data" ∈ VALID_DATA_KEYS
            end
            
            @testset "Invalid data key assertion" begin
                json_data = load_json_data(comprehensive_file)
                @test_throws AssertionError extract_time_series_data(json_data, "invalid_key")
            end
        end
        
        @testset "load_json_data" begin
            @testset "Valid JSON file" begin
                result = load_json_data(valid_json_file)
                @test result.test_key == "test_value"
                @test result.number == 42
            end
            
            @testset "Nonexistent file" begin
                nonexistent_path = joinpath(temp_dir, "nonexistent.json")
                @test_throws ErrorException load_json_data(nonexistent_path)
            end
            
            @testset "Invalid JSON file" begin
                @test_throws ErrorException load_json_data(invalid_json_file)
            end
        end
        
        @testset "extract_time_series_data" begin
            @testset "Valid data extraction with JSON3 serialization" begin
                json_data = load_json_data(comprehensive_file)
                
                # Test NSSP data extraction
                nssp_result = extract_time_series_data(json_data, "nssp_training_data")
                @test nssp_result !== nothing
                @test size(nssp_result, 1) == 3
                @test "date" in names(nssp_result)
                @test "observed_ed_visits" in names(nssp_result)
                
                # Test NHSN data extraction
                nhsn_result = extract_time_series_data(json_data, "nhsn_training_data")
                @test nhsn_result !== nothing
                @test size(nhsn_result, 1) == 1
                @test "hospital_admissions" in names(nhsn_result)
            end
            
            @testset "Missing data key with real JSON" begin
                json_data = load_json_data(partial_file)
                result = extract_time_series_data(json_data, "nwss_training_data")
                @test result === nothing
            end
            
            @testset "Null data with JSON3" begin
                json_data = load_json_data(null_file)
                result = extract_time_series_data(json_data, "nssp_training_data")
                @test result === nothing
            end
            
            @testset "Empty data array with JSON3" begin
                json_data = load_json_data(empty_file)
                result = extract_time_series_data(json_data, "nssp_training_data")
                @test result === nothing
            end
        end
        
        @testset "prepare_epiautogp_data" begin
            @testset "Valid preparation with JSON3 round-trip" begin
                json_data = load_json_data(comprehensive_file)
                result = prepare_epiautogp_data(json_data, "COVID-19", "ny")
                
                # Check structure
                @test result !== nothing
                @test size(result, 1) == 3
                @test names(result) == ["date", "count", "location", "disease"]
                
                # Check content
                @test all(result.location .== "ny")
                @test all(result.disease .== "COVID-19")
                @test all(result.count .∈ Ref([100, 110, 120]))
                
                # Check sorting (should be sorted by date)
                @test issorted(result.date)
                @test result.date == [Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03")]
            end
            
            @testset "Missing ED data with JSON3" begin
                json_data = load_json_data(no_ed_file)
                @test_throws ErrorException prepare_epiautogp_data(json_data, "COVID-19", "ny")
            end
            
            @testset "Missing observed_ed_visits column with JSON3" begin
                json_data = load_json_data(wrong_column_file)
                @test_throws ErrorException prepare_epiautogp_data(json_data, "COVID-19", "ny")
            end
            
            @testset "Date handling with JSON3" begin
                json_data = load_json_data(single_point_file)
                result = prepare_epiautogp_data(json_data, "influenza", "ca")
                @test result.date[1] isa Date
                @test result.date[1] == Date("2024-01-01")
            end
            
            @testset "Edge cases with JSON3" begin
                json_data = load_json_data(single_point_file)
                result = prepare_epiautogp_data(json_data, "COVID-19", "tx")
                @test size(result, 1) == 1
                @test result.count[1] == 100
                @test result.location[1] == "tx"
                @test result.disease[1] == "COVID-19"
            end
        end
        
        @testset "Integration Tests" begin
            @testset "Full pipeline with file I/O" begin
                # Test full pipeline using existing comprehensive file
                json_data = load_json_data(comprehensive_file)
                result = prepare_epiautogp_data(json_data, "COVID-19", "fl")
                
                @test size(result, 1) == 3
                @test all(result.location .== "fl")
                @test all(result.disease .== "COVID-19")
                @test minimum(result.count) == 100
                @test maximum(result.count) == 120
            end
        end
        
        @testset "extract_nowcast_data" begin
            # Create nowcast test data files
            valid_nowcast_data = Dict(
                "nowcast_samples" => [
                    [10.5, 11.2, 12.1],
                    [9.8, 10.9, 11.5],
                    [10.2, 11.1, 12.0]
                ],
                "nowcast_dates" => ["2024-01-01", "2024-01-02", "2024-01-03"]
            )
            
            missing_samples_data = Dict("nowcast_dates" => ["2024-01-01"])
            missing_dates_data = Dict("nowcast_samples" => [[10.5]])
            empty_samples_data = Dict("nowcast_samples" => [], "nowcast_dates" => [])
            null_samples_data = Dict("nowcast_samples" => nothing, "nowcast_dates" => ["2024-01-01"])
            
            invalid_format_data = Dict(
                "nowcast_samples" => [1.0, 2.0, 3.0],  # Not vector of vectors
                "nowcast_dates" => ["2024-01-01", "2024-01-02", "2024-01-03"]
            )
            
            mismatched_length_data = Dict(
                "nowcast_samples" => [[10.5, 11.2], [9.8]],  # Different lengths
                "nowcast_dates" => ["2024-01-01", "2024-01-02"]
            )
            
            date_mismatch_data = Dict(
                "nowcast_samples" => [[10.5, 11.2], [9.8, 10.9]],
                "nowcast_dates" => ["2024-01-01"]  # Wrong number of dates
            )
            
            # Write test files
            valid_nowcast_file = joinpath(temp_dir, "valid_nowcast.json")
            missing_samples_file = joinpath(temp_dir, "missing_samples.json")
            missing_dates_file = joinpath(temp_dir, "missing_dates.json")
            empty_nowcast_file = joinpath(temp_dir, "empty_nowcast.json")
            null_nowcast_file = joinpath(temp_dir, "null_nowcast.json")
            invalid_format_file = joinpath(temp_dir, "invalid_format.json")
            mismatched_length_file = joinpath(temp_dir, "mismatched_length.json")
            date_mismatch_file = joinpath(temp_dir, "date_mismatch.json")
            
            open(valid_nowcast_file, "w") do io; JSON3.write(io, valid_nowcast_data); end
            open(missing_samples_file, "w") do io; JSON3.write(io, missing_samples_data); end
            open(missing_dates_file, "w") do io; JSON3.write(io, missing_dates_data); end
            open(empty_nowcast_file, "w") do io; JSON3.write(io, empty_samples_data); end
            open(null_nowcast_file, "w") do io; JSON3.write(io, null_samples_data); end
            open(invalid_format_file, "w") do io; JSON3.write(io, invalid_format_data); end
            open(mismatched_length_file, "w") do io; JSON3.write(io, mismatched_length_data); end
            open(date_mismatch_file, "w") do io; JSON3.write(io, date_mismatch_data); end
            
            @testset "Valid nowcast data extraction" begin
                json_data = load_json_data(valid_nowcast_file)
                result = extract_nowcast_data(json_data)
                
                @test result !== nothing
                @test haskey(result, :samples)
                @test haskey(result, :dates)
                
                # Check samples structure
                @test length(result.samples) == 3  # 3 scenarios
                @test length(result.samples[1]) == 3  # 3 time points each
                @test all(length(scenario) == 3 for scenario in result.samples)
                
                # Check date conversion
                @test length(result.dates) == 3
                @test result.dates[1] == Date("2024-01-01")
                @test result.dates[3] == Date("2024-01-03")
                
                # Check values
                @test result.samples[1] == [10.5, 11.2, 12.1]
                @test result.samples[2] == [9.8, 10.9, 11.5]
                @test result.samples[3] == [10.2, 11.1, 12.0]
            end
            
            @testset "Missing nowcast keys" begin
                json_data_missing_samples = load_json_data(missing_samples_file)
                @test_throws AssertionError extract_nowcast_data(json_data_missing_samples)
                
                json_data_missing_dates = load_json_data(missing_dates_file)
                @test_throws AssertionError extract_nowcast_data(json_data_missing_dates)
            end
            
            @testset "Empty or null nowcast data" begin
                json_data_empty = load_json_data(empty_nowcast_file)
                result_empty = extract_nowcast_data(json_data_empty)
                @test result_empty === nothing
                
                json_data_null = load_json_data(null_nowcast_file)
                result_null = extract_nowcast_data(json_data_null)
                @test result_null === nothing
            end
            
            @testset "Invalid nowcast format" begin
                json_data_invalid = load_json_data(invalid_format_file)
                @test_throws ErrorException extract_nowcast_data(json_data_invalid)
            end
            
            @testset "Dimension validation" begin
                json_data_mismatched = load_json_data(mismatched_length_file)
                @test_throws ErrorException extract_nowcast_data(json_data_mismatched)
                
                json_data_date_mismatch = load_json_data(date_mismatch_file)
                @test_throws ErrorException extract_nowcast_data(json_data_date_mismatch)
            end
        end
        
    finally
        # Cleanup: Remove temporary directory and all files
        rm(temp_dir, recursive=true)
    end
end
