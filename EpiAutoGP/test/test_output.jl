"""
Test suite for output.jl - testing structs and create_df function

Tests cover:
- AbstractForecastOutput and AbstractHubverseOutput abstract types
- QuantileOutput struct construction and default values
- create_df function with various inputs and edge cases
"""

using Random
using Statistics

@testset "Output Types and Structures Tests" begin
    @testset "Abstract Type Hierarchy" begin
        # Test that our concrete type inherits from the correct abstract types
        @test QuantileOutput <: AbstractHubverseOutput
        @test AbstractHubverseOutput <: AbstractForecastOutput
        
        # Test that we can create instances of concrete types
        output = QuantileOutput()
        @test isa(output, AbstractForecastOutput)
        @test isa(output, AbstractHubverseOutput)
        @test isa(output, QuantileOutput)
    end

    @testset "QuantileOutput Construction" begin
        # Test default construction
        default_output = QuantileOutput()
        expected_quantiles = [
            0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
        ]
        
        @test default_output.quantile_levels == expected_quantiles
        @test length(default_output.quantile_levels) == 23
        @test all(0.0 .<= default_output.quantile_levels .<= 1.0)
        @test issorted(default_output.quantile_levels)
        
        # Test that median (0.5) is included
        @test 0.5 in default_output.quantile_levels
        
        # Test custom construction
        custom_quantiles = [0.1, 0.5, 0.9]
        custom_output = QuantileOutput(quantile_levels=custom_quantiles)
        @test custom_output.quantile_levels == custom_quantiles
        
        # Test empty quantile levels (edge case)
        empty_output = QuantileOutput(quantile_levels=Float64[])
        @test empty_output.quantile_levels == Float64[]
        @test length(empty_output.quantile_levels) == 0
        
        # Test single quantile level
        single_output = QuantileOutput(quantile_levels=[0.5])
        @test single_output.quantile_levels == [0.5]
        @test length(single_output.quantile_levels) == 1
    end

    @testset "QuantileOutput Edge Cases" begin
        # Test quantile levels at boundaries
        boundary_output = QuantileOutput(quantile_levels=[0.0, 1.0])
        @test boundary_output.quantile_levels == [0.0, 1.0]
        
        # Test unsorted quantile levels (should be allowed)
        unsorted_output = QuantileOutput(quantile_levels=[0.9, 0.1, 0.5])
        @test unsorted_output.quantile_levels == [0.9, 0.1, 0.5]
        
        # Test duplicate quantile levels (should be allowed)
        duplicate_output = QuantileOutput(quantile_levels=[0.5, 0.5, 0.9])
        @test duplicate_output.quantile_levels == [0.5, 0.5, 0.9]
        
        # Test high precision quantile levels
        precision_output = QuantileOutput(quantile_levels=[0.123456789, 0.987654321])
        @test precision_output.quantile_levels[1] ≈ 0.123456789
        @test precision_output.quantile_levels[2] ≈ 0.987654321
    end
end

@testset "create_df Function Tests" begin
    @testset "Basic Functionality" begin
        # Test basic case with simple data
        forecast_dates = [Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03")]
        
        # Create sample forecast data (3 dates, 100 samples each)
        forecasts = rand(3, 100) .* 50 .+ 25  # Random values between 25-75
        
        output_type = QuantileOutput(quantile_levels=[0.25, 0.5, 0.75])
        
        result_df = create_df(forecast_dates, forecasts, output_type)
        
        # Test basic structure
        @test isa(result_df, DataFrame)
        @test size(result_df, 1) == 3 * 3  # 3 dates × 3 quantiles = 9 rows
        @test size(result_df, 2) == 4  # output_type_id, value, target_end_date, output_type
        
        # Test column names
        expected_columns = ["output_type_id", "value", "target_end_date", "output_type"]
        @test names(result_df) == expected_columns
        
        # Test column types
        @test eltype(result_df.output_type_id) == Float64
        @test eltype(result_df.value) == Float64
        @test eltype(result_df.target_end_date) == String
        @test eltype(result_df.output_type) == String
        
        # Test output_type column content
        @test all(result_df.output_type .== "quantile")
        
        # Test that all quantile levels are present
        unique_quantiles = unique(result_df.output_type_id)
        @test Set(unique_quantiles) == Set([0.25, 0.5, 0.75])
        
        # Test that all dates are present
        unique_dates = unique(result_df.target_end_date)
        expected_date_strings = ["2024-01-01", "2024-01-02", "2024-01-03"]
        @test Set(unique_dates) == Set(expected_date_strings)
        
        # Test that quantile values are reasonable (median should be between quartiles)
        for date_str in expected_date_strings
            date_rows = result_df[result_df.target_end_date .== date_str, :]
            q25_row = date_rows[date_rows.output_type_id .== 0.25, :]
            q50_row = date_rows[date_rows.output_type_id .== 0.5, :]
            q75_row = date_rows[date_rows.output_type_id .== 0.75, :]
            
            q25_val = q25_row.value[1]
            q50_val = q50_row.value[1]
            q75_val = q75_row.value[1]
            
            @test q25_val <= q50_val <= q75_val
        end
    end

    @testset "Single Date and Single Quantile" begin
        # Test with single date
        single_date = [Date("2024-01-01")]
        single_forecasts = rand(1, 50) .* 100  # 1 date, 50 samples
        output_type = QuantileOutput(quantile_levels=[0.5])
        
        result_df = create_df(single_date, single_forecasts, output_type)
        
        @test size(result_df, 1) == 1
        @test result_df.output_type_id[1] == 0.5
        @test result_df.target_end_date[1] == "2024-01-01"
        @test isa(result_df.value[1], Float64)
        
        # Verify quantile calculation is reasonable
        expected_median = quantile(single_forecasts[1, :], 0.5)
        @test result_df.value[1] ≈ expected_median
    end

    @testset "Multiple Quantiles with Known Data" begin
        # Test with known data to verify quantile calculations
        forecast_dates = [Date("2024-01-01")]
        
        # Create predictable data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        known_data = reshape(collect(1.0:10.0), 1, 10)
        
        output_type = QuantileOutput(quantile_levels=[0.0, 0.5, 1.0])
        result_df = create_df(forecast_dates, known_data, output_type)
        
        @test size(result_df, 1) == 3
        
        # Find rows for each quantile
        min_row = result_df[result_df.output_type_id .== 0.0, :]
        median_row = result_df[result_df.output_type_id .== 0.5, :]
        max_row = result_df[result_df.output_type_id .== 1.0, :]
        
        # Test quantile values
        @test min_row.value[1] == 1.0  # minimum
        @test median_row.value[1] == 5.5  # median of 1:10
        @test max_row.value[1] == 10.0  # maximum
    end

    @testset "Default Quantile Levels" begin
        # Test with default QuantileOutput (23 quantile levels)
        forecast_dates = [Date("2024-01-01"), Date("2024-01-02")]
        forecasts = rand(2, 1000) .* 100  # Large sample for stable quantiles
        
        default_output = QuantileOutput()  # Uses default quantile levels
        result_df = create_df(forecast_dates, forecasts, default_output)
        
        @test size(result_df, 1) == 2 * 23  # 2 dates × 23 quantiles = 46 rows
        
        # Test that all default quantile levels are present
        unique_quantiles = sort(unique(result_df.output_type_id))
        expected_quantiles = sort(default_output.quantile_levels)
        @test unique_quantiles ≈ expected_quantiles
        
        # Test that quantiles are monotonically increasing for each date
        for date_str in unique(result_df.target_end_date)
            date_data = result_df[result_df.target_end_date .== date_str, :]
            sorted_data = sort(date_data, :output_type_id)
            
            # Values should be non-decreasing as quantile levels increase
            for i in 2:size(sorted_data, 1)
                @test sorted_data.value[i] >= sorted_data.value[i-1]
            end
        end
    end

    @testset "Edge Cases and Error Conditions" begin
        # Test with empty forecast data
        empty_dates = Date[]
        empty_forecasts = reshape(Float64[], 0, 0)
        output_type = QuantileOutput(quantile_levels=[0.5])
        
        empty_result = create_df(empty_dates, empty_forecasts, output_type)
        @test size(empty_result, 1) == 0
        @test size(empty_result, 2) == 4
        @test names(empty_result) == ["output_type_id", "value", "target_end_date", "output_type"]
        
        # Test with empty quantile levels
        forecast_dates = [Date("2024-01-01")]
        forecasts = rand(1, 10)
        empty_quantiles = QuantileOutput(quantile_levels=Float64[])
        
        empty_quantile_result = create_df(forecast_dates, forecasts, empty_quantiles)
        @test size(empty_quantile_result, 1) == 0
        @test size(empty_quantile_result, 2) == 4
        
        # Test with single sample (edge case for quantile calculation)
        single_sample = reshape([42.0], 1, 1)
        single_date = [Date("2024-01-01")]
        output_type = QuantileOutput(quantile_levels=[0.0, 0.5, 1.0])
        
        single_result = create_df(single_date, single_sample, output_type)
        @test size(single_result, 1) == 3
        @test all(single_result.value .== 42.0)  # All quantiles should equal the single value
    end

    @testset "Data Type and Precision Tests" begin
        # Test with integer-like data
        forecast_dates = [Date("2024-01-01")]
        integer_data = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 1, 5)
        output_type = QuantileOutput(quantile_levels=[0.5])
        
        result_df = create_df(forecast_dates, integer_data, output_type)
        @test result_df.value[1] == 3.0  # Median of 1,2,3,4,5
        
        # Test with very small numbers
        small_data = reshape([1e-10, 2e-10, 3e-10], 1, 3)
        small_result = create_df(forecast_dates, small_data, output_type)
        @test small_result.value[1] ≈ 2e-10
        
        # Test with very large numbers
        large_data = reshape([1e10, 2e10, 3e10], 1, 3)
        large_result = create_df(forecast_dates, large_data, output_type)
        @test large_result.value[1] ≈ 2e10
        
        # Test precision preservation
        precise_data = reshape([1.123456789, 1.123456790, 1.123456791], 1, 3)
        precise_result = create_df(forecast_dates, precise_data, output_type)
        @test precise_result.value[1] ≈ 1.123456790 atol=1e-9
    end

    @testset "Date Handling and String Conversion" begin
        # Test various date formats and their string conversion
        diverse_dates = [
            Date("2024-01-01"),
            Date("2024-12-31"),
            Date("2023-02-28"),
            Date("2024-02-29"),  # Leap year
            Date("2025-07-04")
        ]
        
        forecasts = rand(5, 10)
        output_type = QuantileOutput(quantile_levels=[0.5])
        
        result_df = create_df(diverse_dates, forecasts, output_type)
        
        expected_date_strings = [
            "2024-01-01",
            "2024-12-31", 
            "2023-02-28",
            "2024-02-29",
            "2025-07-04"
        ]
        
        @test result_df.target_end_date == expected_date_strings
        
        # Test that dates maintain their order
        for i in 1:length(diverse_dates)
            @test result_df.target_end_date[i] == string(diverse_dates[i])
        end
    end

    @testset "Statistical Properties" begin
        # Test with known statistical properties
        forecast_dates = [Date("2024-01-01")]
        
        # Generate data with known properties: normal distribution
        Random.seed!(42)  # For reproducible tests
        normal_samples = randn(1000)
        normal_data = reshape(normal_samples, 1, 1000)
        
        output_type = QuantileOutput(quantile_levels=[0.5])
        result_df = create_df(forecast_dates, normal_data, output_type)
        
        # Median of standard normal should be close to 0
        @test abs(result_df.value[1]) < 0.1  # Should be close to 0 with 1000 samples
        
        # Test extreme quantiles
        extreme_output = QuantileOutput(quantile_levels=[0.01, 0.99])
        extreme_result = create_df(forecast_dates, normal_data, extreme_output)
        
        # 1st and 99th percentiles should be far apart for normal distribution
        q01_val = extreme_result[extreme_result.output_type_id .== 0.01, :].value[1]
        q99_val = extreme_result[extreme_result.output_type_id .== 0.99, :].value[1]
        
        @test q01_val < q99_val
        @test abs(q99_val - q01_val) > 3  # Should span roughly 3+ standard deviations
    end
end

@testset "Integration Tests" begin
    @testset "Realistic Forecasting Scenario" begin
        # Simulate a realistic epidemiological forecasting scenario
        
        # Generate 4 weeks of forecast dates
        start_date = Date("2024-01-01")
        forecast_dates = [start_date + Day(7*i) for i in 0:3]  # Weekly forecasts
        
        # Generate realistic forecast samples (cases declining over time)
        n_samples = 500
        forecasts = Matrix{Float64}(undef, 4, n_samples)
        
        Random.seed!(123)
        for week in 1:4
            # Declining trend with uncertainty
            base_level = 100 * exp(-0.1 * (week - 1))  # Exponential decline
            # Add lognormal noise for realism (cases can't be negative)
            forecasts[week, :] = [base_level * exp(randn() * 0.3) for _ in 1:n_samples]
        end
        
        # Use realistic quantile levels
        realistic_output = QuantileOutput(quantile_levels=[
            0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975
        ])
        
        result_df = create_df(forecast_dates, forecasts, realistic_output)
        
        # Test realistic properties
        @test size(result_df, 1) == 4 * 7  # 4 weeks × 7 quantiles
        
        # Test that forecast values are positive (epidemiological constraint)
        @test all(result_df.value .>= 0)
        
        # Test declining trend (medians should generally decrease)
        median_rows = result_df[result_df.output_type_id .== 0.5, :]
        median_values = median_rows.value
        
        # Allow some noise but expect general decline
        @test median_values[1] > median_values[4]  # First week > last week
        
        # Test uncertainty bounds (wider intervals for later weeks)
        for week_idx in 1:4
            week_data = result_df[
                result_df.target_end_date .== string(forecast_dates[week_idx]), :
            ]
            
            q025 = week_data[week_data.output_type_id .== 0.025, :].value[1]
            q975 = week_data[week_data.output_type_id .== 0.975, :].value[1]
            
            # Confidence interval should be positive and meaningful
            @test q025 < q975
            @test q025 >= 0
            @test (q975 - q025) > 0  # Should have meaningful uncertainty
        end
    end

    @testset "Cross-validation with Manual Calculations" begin
        # Create simple test case we can verify manually
        forecast_dates = [Date("2024-01-01"), Date("2024-01-02")]
        
        # Simple data: first date has values [10, 20, 30], second has [40, 50, 60]
        simple_forecasts = [10.0 20.0 30.0; 40.0 50.0 60.0]  # 2x3 matrix
        
        output_type = QuantileOutput(quantile_levels=[0.0, 0.5, 1.0])
        result_df = create_df(forecast_dates, simple_forecasts, output_type)
        
        # Manually verify results
        @test size(result_df, 1) == 6  # 2 dates × 3 quantiles
        
        # Check first date results
        date1_data = result_df[result_df.target_end_date .== "2024-01-01", :]
        date1_min = date1_data[date1_data.output_type_id .== 0.0, :].value[1]
        date1_med = date1_data[date1_data.output_type_id .== 0.5, :].value[1]
        date1_max = date1_data[date1_data.output_type_id .== 1.0, :].value[1]
        
        @test date1_min == 10.0
        @test date1_med == 20.0  # Median of [10, 20, 30]
        @test date1_max == 30.0
        
        # Check second date results
        date2_data = result_df[result_df.target_end_date .== "2024-01-02", :]
        date2_min = date2_data[date2_data.output_type_id .== 0.0, :].value[1]
        date2_med = date2_data[date2_data.output_type_id .== 0.5, :].value[1]
        date2_max = date2_data[date2_data.output_type_id .== 1.0, :].value[1]
        
        @test date2_min == 40.0
        @test date2_med == 50.0  # Median of [40, 50, 60]
        @test date2_max == 60.0
    end
end