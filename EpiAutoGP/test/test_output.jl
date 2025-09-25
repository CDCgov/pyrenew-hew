"""
Test suite for output.jl - testing structs and create_forecast_df function

Tests cover:
- AbstractForecastOutput and AbstractHubverseOutput abstract types
- QuantileOutput struct construction and default values
- create_forecast_df function with various inputs and edge cases
"""

using Random
using Statistics
using CSV

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
        custom_output = QuantileOutput(quantile_levels = custom_quantiles)
        @test custom_output.quantile_levels == custom_quantiles

        # Test empty quantile levels (edge case)
        empty_output = QuantileOutput(quantile_levels = Float64[])
        @test empty_output.quantile_levels == Float64[]
        @test length(empty_output.quantile_levels) == 0

        # Test single quantile level
        single_output = QuantileOutput(quantile_levels = [0.5])
        @test single_output.quantile_levels == [0.5]
        @test length(single_output.quantile_levels) == 1
    end

    @testset "QuantileOutput Edge Cases" begin
        # Test quantile levels at boundaries
        boundary_output = QuantileOutput(quantile_levels = [0.0, 1.0])
        @test boundary_output.quantile_levels == [0.0, 1.0]

        # Test unsorted quantile levels (should be allowed)
        unsorted_output = QuantileOutput(quantile_levels = [0.9, 0.1, 0.5])
        @test unsorted_output.quantile_levels == [0.9, 0.1, 0.5]

        # Test duplicate quantile levels (should be allowed)
        duplicate_output = QuantileOutput(quantile_levels = [0.5, 0.5, 0.9])
        @test duplicate_output.quantile_levels == [0.5, 0.5, 0.9]

        # Test high precision quantile levels
        precision_output = QuantileOutput(quantile_levels = [0.123456789, 0.987654321])
        @test precision_output.quantile_levels[1] ≈ 0.123456789
        @test precision_output.quantile_levels[2] ≈ 0.987654321
    end
end

@testset "create_forecast_df Function Tests" begin
    @testset "Basic Functionality" begin
        # Test basic case with simple data
        forecast_dates = [Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03")]

        # Create sample forecast data (3 dates, 100 samples each)
        forecasts = rand(3, 100) .* 50 .+ 25  # Random values between 25-75

        output_type = QuantileOutput(quantile_levels = [0.25, 0.5, 0.75])

        result_df = create_forecast_df(
            (
                forecast_dates = forecast_dates, forecasts = forecasts), output_type)

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
        @test eltype(result_df.target_end_date) == Date
        @test eltype(result_df.output_type) == String

        # Test output_type column content
        @test all(result_df.output_type .== "quantile")

        # Test that all quantile levels are present
        unique_quantiles = unique(result_df.output_type_id)
        @test Set(unique_quantiles) == Set([0.25, 0.5, 0.75])

        # Test that all dates are present
        unique_dates = unique(result_df.target_end_date)
        expected_dates = [Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03")]
        @test Set(unique_dates) == Set(expected_dates)

        # Test that quantile values are reasonable (median should be between quartiles)
        for date_obj in expected_dates
            date_rows = result_df[result_df.target_end_date .== date_obj, :]
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
        output_type = QuantileOutput(quantile_levels = [0.5])

        result_df = create_forecast_df(
            (forecast_dates = single_date, forecasts = single_forecasts), output_type)

        @test size(result_df, 1) == 1
        @test result_df.output_type_id[1] == 0.5
        @test result_df.target_end_date[1] == Date("2024-01-01")
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

        output_type = QuantileOutput(quantile_levels = [0.0, 0.5, 1.0])
        result_df = create_forecast_df(
            (
                forecast_dates = forecast_dates, forecasts = known_data), output_type)

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
        result_df = create_forecast_df(
            (
                forecast_dates = forecast_dates, forecasts = forecasts), default_output)

        @test size(result_df, 1) == 2 * 23  # 2 dates × 23 quantiles = 46 rows

        # Test that all default quantile levels are present
        unique_quantiles = sort(unique(result_df.output_type_id))
        expected_quantiles = sort(default_output.quantile_levels)
        @test unique_quantiles ≈ expected_quantiles

        # Test that quantiles are monotonically increasing for each date
        for date_obj in unique(result_df.target_end_date)
            date_data = result_df[result_df.target_end_date .== date_obj, :]
            sorted_data = sort(date_data, :output_type_id)

            # Values should be non-decreasing as quantile levels increase
            for i in 2:size(sorted_data, 1)
                @test sorted_data.value[i] >= sorted_data.value[i - 1]
            end
        end
    end

    @testset "Edge Cases and Error Conditions" begin
        # Test with empty forecast data
        empty_dates = Date[]
        empty_forecasts = reshape(Float64[], 0, 0)
        output_type = QuantileOutput(quantile_levels = [0.5])

        empty_result = create_forecast_df(
            (forecast_dates = empty_dates, forecasts = empty_forecasts), output_type)
        @test size(empty_result, 1) == 0
        @test size(empty_result, 2) == 4
        @test names(empty_result) ==
              ["output_type_id", "value", "target_end_date", "output_type"]

        # Test with empty quantile levels
        forecast_dates = [Date("2024-01-01")]
        forecasts = rand(1, 10)
        empty_quantiles = QuantileOutput(quantile_levels = Float64[])

        empty_quantile_result = create_forecast_df(
            (forecast_dates = forecast_dates, forecasts = forecasts), empty_quantiles)
        @test size(empty_quantile_result, 1) == 0
        @test size(empty_quantile_result, 2) == 4

        # Test with single sample (edge case for quantile calculation)
        single_sample = reshape([42.0], 1, 1)
        single_date = [Date("2024-01-01")]
        output_type = QuantileOutput(quantile_levels = [0.0, 0.5, 1.0])

        single_result = create_forecast_df(
            (forecast_dates = single_date, forecasts = single_sample), output_type)
        @test size(single_result, 1) == 3
        @test all(single_result.value .== 42.0)  # All quantiles should equal the single value
    end

    @testset "Data Type and Precision Tests" begin
        # Test with integer-like data
        forecast_dates = [Date("2024-01-01")]
        integer_data = reshape([1.0, 2.0, 3.0, 4.0, 5.0], 1, 5)
        output_type = QuantileOutput(quantile_levels = [0.5])

        result_df = create_forecast_df(
            (
                forecast_dates = forecast_dates, forecasts = integer_data), output_type)
        @test result_df.value[1] == 3.0  # Median of 1,2,3,4,5

        # Test with very small numbers
        small_data = reshape([1e-10, 2e-10, 3e-10], 1, 3)
        small_result = create_forecast_df(
            (forecast_dates = forecast_dates, forecasts = small_data), output_type)
        @test small_result.value[1] ≈ 2e-10

        # Test with very large numbers
        large_data = reshape([1e10, 2e10, 3e10], 1, 3)
        large_result = create_forecast_df(
            (forecast_dates = forecast_dates, forecasts = large_data), output_type)
        @test large_result.value[1] ≈ 2e10

        # Test precision preservation
        precise_data = reshape([1.123456789, 1.123456790, 1.123456791], 1, 3)
        precise_result = create_forecast_df(
            (forecast_dates = forecast_dates, forecasts = precise_data), output_type)
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
        output_type = QuantileOutput(quantile_levels = [0.5])

        result_df = create_forecast_df(
            (
                forecast_dates = diverse_dates, forecasts = forecasts), output_type)

        expected_dates = [
            Date("2024-01-01"),
            Date("2024-12-31"),
            Date("2023-02-28"),
            Date("2024-02-29"),
            Date("2025-07-04")
        ]

        @test result_df.target_end_date == expected_dates

        # Test that dates maintain their order
        for i in 1:length(diverse_dates)
            @test result_df.target_end_date[i] == diverse_dates[i]
        end
    end

    @testset "Statistical Properties" begin
        # Test with known statistical properties
        forecast_dates = [Date("2024-01-01")]

        # Generate data with known properties: normal distribution
        Random.seed!(42)  # For reproducible tests
        normal_samples = randn(1000)
        normal_data = reshape(normal_samples, 1, 1000)

        output_type = QuantileOutput(quantile_levels = [0.5])
        result_df = create_forecast_df(
            (
                forecast_dates = forecast_dates, forecasts = normal_data), output_type)

        # Median of standard normal should be close to 0
        @test abs(result_df.value[1]) < 0.1  # Should be close to 0 with 1000 samples

        # Test extreme quantiles
        extreme_output = QuantileOutput(quantile_levels = [0.01, 0.99])
        extreme_result = create_forecast_df(
            (forecast_dates = forecast_dates, forecasts = normal_data), extreme_output)

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
        realistic_output = QuantileOutput(quantile_levels = [
            0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975
        ])

        result_df = create_forecast_df(
            (forecast_dates = forecast_dates, forecasts = forecasts), realistic_output)

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
            result_df.target_end_date .== forecast_dates[week_idx], :
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

        output_type = QuantileOutput(quantile_levels = [0.0, 0.5, 1.0])
        result_df = create_forecast_df(
            (forecast_dates = forecast_dates, forecasts = simple_forecasts), output_type)

        # Manually verify results
        @test size(result_df, 1) == 6  # 2 dates × 3 quantiles

        # Check first date results
        date1_data = result_df[result_df.target_end_date .== Date("2024-01-01"), :]
        date1_min = date1_data[date1_data.output_type_id .== 0.0, :].value[1]
        date1_med = date1_data[date1_data.output_type_id .== 0.5, :].value[1]
        date1_max = date1_data[date1_data.output_type_id .== 1.0, :].value[1]

        @test date1_min == 10.0
        @test date1_med == 20.0  # Median of [10, 20, 30]
        @test date1_max == 30.0

        # Check second date results
        date2_data = result_df[result_df.target_end_date .== Date("2024-01-02"), :]
        date2_min = date2_data[date2_data.output_type_id .== 0.0, :].value[1]
        date2_med = date2_data[date2_data.output_type_id .== 0.5, :].value[1]
        date2_max = date2_data[date2_data.output_type_id .== 1.0, :].value[1]

        @test date2_min == 40.0
        @test date2_med == 50.0  # Median of [40, 50, 60]
        @test date2_max == 60.0
    end
end

@testset "create_forecast_output Function Tests" begin
    @testset "Basic Functionality" begin
        # Create test input data
        forecast_date = Date("2024-01-01")
        location = "US"
        pathogen = "COVID-19"
        target = "nhsn"

        input = EpiAutoGPInput(
            [Date("2023-12-01"), Date("2023-12-08")],  # dates
            [100.0, 90.0],                                   # reports
            pathogen,                                         # pathogen
            location,                                         # location
            target,                                           # target
            forecast_date,                                    # forecast_date
            Date[],                                           # nowcast_dates
            Vector{Real}[]                                    # nowcast_reports
        )

        # Create test results data
        forecast_dates = [Date("2024-01-08"), Date("2024-01-15"), Date("2024-01-22")]
        forecasts = [80.0 85.0 75.0; 70.0 75.0 65.0; 60.0 65.0 55.0]  # 3 dates × 3 samples

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        output_type = QuantileOutput(quantile_levels = [0.25, 0.5, 0.75])

        # Create temporary directory for output
        temp_dir = mktempdir()

        try
            # Test the function with save_output=true
            result_df = create_forecast_output(
                input, results, temp_dir, output_type, save_output = true)

            # Test basic structure
            @test isa(result_df, DataFrame)
            @test size(result_df, 1) == 3 * 3  # 3 dates × 3 quantiles = 9 rows
            @test size(result_df, 2) == 8  # All required columns

            # Test required columns are present
            expected_columns = [
                "output_type", "output_type_id", "value", "reference_date",
                "target", "horizon", "target_end_date", "location"
            ]
            @test Set(names(result_df)) == Set(expected_columns)

            # Test column values
            @test all(result_df.output_type .== "quantile")
            @test all(result_df.reference_date .== forecast_date)
            @test all(result_df.location .== location)
            @test all(result_df.target .== "wk inc covid hosp")

            # Test quantile levels
            unique_quantiles = sort(unique(result_df.output_type_id))
            @test unique_quantiles == [0.25, 0.5, 0.75]

            # Test dates
            unique_dates = sort(unique(result_df.target_end_date))
            @test unique_dates == sort(forecast_dates)

            # Test horizon calculation (should be weeks from reference date)
            expected_horizons = [1, 2, 3]  # 1, 2, 3 weeks after forecast_date
            unique_horizons = sort(unique(result_df.horizon))
            @test unique_horizons == expected_horizons

            # Test file was created with correct filename format
            expected_filename = "2024-01-01-CFA-EpiAutoGP-US-covid-nhsn.csv"
            csv_path = joinpath(temp_dir, expected_filename)
            @test isfile(csv_path)

            # Test file content matches returned DataFrame
            file_df = CSV.read(csv_path, DataFrame)
            @test nrow(file_df) == nrow(result_df)
            @test names(file_df) == names(result_df)

        finally
            # Clean up temporary directory
            rm(temp_dir, recursive = true)
        end
    end

    @testset "Different Pathogens and Targets" begin
        # Test different combinations of pathogen and target
        test_cases = [
            ("COVID-19", "nhsn", "wk inc covid hosp"),
            ("Influenza", "nhsn", "wk inc flu hosp"),
            ("RSV", "nssp", "wk inc rsv prop ed visits"),
            ("COVID-19", "nssp", "wk inc covid prop ed visits")
        ]

        forecast_date = Date("2024-01-01")
        location = "CA"
        forecast_dates = [Date("2024-01-08")]
        forecasts = reshape([50.0, 60.0, 70.0], 1, 3)  # 1 date × 3 samples

        output_type = QuantileOutput(quantile_levels = [0.5])

        for (pathogen, target, expected_target_string) in test_cases
            input = EpiAutoGPInput(
                [Date("2023-12-01")],  # dates
                [100.0],               # reports
                pathogen,              # pathogen
                location,              # location
                target,                # target
                forecast_date,         # forecast_date
                Date[],                # nowcast_dates
                Vector{Real}[]         # nowcast_reports
            )

            results = (
                forecast_dates = forecast_dates,
                forecasts = forecasts,
                disease = pathogen,
                location = location
            )

            temp_dir = mktempdir()

            try
                result_df = create_forecast_output(
                    input, results, temp_dir, output_type, save_output = false)

                @test all(result_df.target .== expected_target_string)
                @test all(result_df.location .== location)
                @test size(result_df, 1) == 1  # 1 date × 1 quantile

            finally
                rm(temp_dir, recursive = true)
            end
        end
    end

    @testset "Custom Disease and Target Dictionaries" begin
        # Test with custom abbreviation dictionaries
        custom_disease_abbr = Dict("TestDisease" => "test")
        custom_target_abbr = Dict("test_target" => "test visits")

        forecast_date = Date("2024-01-01")
        location = "NY"
        pathogen = "TestDisease"
        target = "test_target"

        input = EpiAutoGPInput(
            [Date("2023-12-01")],  # dates
            [100.0],               # reports
            pathogen,              # pathogen
            location,              # location
            target,                # target
            forecast_date,         # forecast_date
            Date[],                # nowcast_dates
            Vector{Real}[]         # nowcast_reports
        )

        forecast_dates = [Date("2024-01-08")]
        forecasts = reshape([30.0, 40.0, 50.0], 1, 3)

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        output_type = QuantileOutput(quantile_levels = [0.5])
        temp_dir = mktempdir()

        try
            result_df = create_forecast_output(
                input, results, temp_dir, output_type;
                save_output = false,
                disease_abbr = custom_disease_abbr,
                target_abbr = custom_target_abbr
            )

            @test all(result_df.target .== "wk inc test test visits")

        finally
            rm(temp_dir, recursive = true)
        end
    end

    @testset "Multiple Forecast Dates and Quantiles" begin
        # Test with realistic scenario: 4 weeks of forecasts, full quantile set
        forecast_date = Date("2024-01-01")
        location = "TX"
        pathogen = "Influenza"
        target = "nhsn"

        input = EpiAutoGPInput(
            [Date("2023-12-01"), Date("2023-12-08"), Date("2023-12-15")],  # dates
            [200.0, 180.0, 160.0],                                           # reports
            pathogen,                                                         # pathogen
            location,                                                         # location
            target,                                                           # target
            forecast_date,                                                    # forecast_date
            Date[],                                                           # nowcast_dates
            Vector{Real}[]                                                    # nowcast_reports
        )

        # 4 weeks of forecasts with declining trend
        forecast_dates = [
            Date("2024-01-08"), Date("2024-01-15"),
            Date("2024-01-22"), Date("2024-01-29")
        ]

        # Generate realistic forecast samples
        Random.seed!(42)
        n_samples = 100
        forecasts = Matrix{Float64}(undef, 4, n_samples)
        base_values = [140.0, 120.0, 100.0, 85.0]  # Declining trend

        for (i, base) in enumerate(base_values)
            forecasts[i, :] = base .+ randn(n_samples) .* 10  # Add noise
        end

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        # Use default quantiles (23 levels)
        output_type = QuantileOutput()
        temp_dir = mktempdir()

        try
            result_df = create_forecast_output(
                input, results, temp_dir, output_type, save_output = false)

            # Test dimensions
            @test size(result_df, 1) == 4 * 23  # 4 dates × 23 quantiles = 92 rows
            @test size(result_df, 2) == 8

            # Test horizon values
            expected_horizons = [1, 2, 3, 4]
            unique_horizons = sort(unique(result_df.horizon))
            @test unique_horizons == expected_horizons

            # Test that values are reasonable (declining trend in medians)
            median_rows = result_df[result_df.output_type_id .== 0.5, :]
            median_values = [median_rows[median_rows.target_end_date .== d, :].value[1]
                             for d in forecast_dates]

            # Should generally decline (allowing some noise)
            @test median_values[1] > median_values[4]

            # Test quantile ordering for each date
            for date in forecast_dates
                date_data = result_df[result_df.target_end_date .== date, :]
                sorted_data = sort(date_data, :output_type_id)

                # Values should be non-decreasing as quantile levels increase
                for i in 2:nrow(sorted_data)
                    @test sorted_data.value[i] >= sorted_data.value[i - 1]
                end
            end

        finally
            rm(temp_dir, recursive = true)
        end
    end

    @testset "Directory Creation and File Handling" begin
        # Test that function creates nested directories as needed
        forecast_date = Date("2024-01-01")
        location = "FL"
        pathogen = "RSV"
        target = "nhsn"

        input = EpiAutoGPInput(
            [Date("2023-12-01")],  # dates
            [50.0],                # reports
            pathogen,              # pathogen
            location,              # location
            target,                # target
            forecast_date,         # forecast_date
            Date[],                # nowcast_dates
            Vector{Real}[]         # nowcast_reports
        )

        forecast_dates = [Date("2024-01-08")]
        forecasts = reshape([25.0, 30.0, 35.0], 1, 3)

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        output_type = QuantileOutput(quantile_levels = [0.5])

        # Test with nested directory that doesn't exist
        base_temp_dir = mktempdir()
        nested_output_dir = joinpath(base_temp_dir, "forecast_outputs", "2024", "january")

        try
            @test !isdir(nested_output_dir)  # Directory doesn't exist yet

            result_df = create_forecast_output(
                input, results, nested_output_dir, output_type, save_output = true)

            # Test that directory was created
            @test isdir(nested_output_dir)

            # Test that file exists in the nested directory with correct filename
            expected_filename = "2024-01-01-CFA-EpiAutoGP-FL-rsv-nhsn.csv"
            csv_path = joinpath(nested_output_dir, expected_filename)
            @test isfile(csv_path)

        finally
            rm(base_temp_dir, recursive = true)
        end
    end

    @testset "Edge Cases and Error Handling" begin
        # Test with single forecast date and single sample
        forecast_date = Date("2024-01-01")
        location = "AK"
        pathogen = "COVID-19"
        target = "nssp"

        input = EpiAutoGPInput(
            [Date("2023-12-01")],  # dates
            [10.0],                # reports
            pathogen,              # pathogen
            location,              # location
            target,                # target
            forecast_date,         # forecast_date
            Date[],                # nowcast_dates
            Vector{Real}[]         # nowcast_reports
        )

        # Single date, single sample
        forecast_dates = [Date("2024-01-08")]
        forecasts = reshape([42.0], 1, 1)  # 1 date × 1 sample

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        output_type = QuantileOutput(quantile_levels = [0.0, 0.5, 1.0])
        temp_dir = mktempdir()

        try
            result_df = create_forecast_output(
                input, results, temp_dir, output_type, save_output = false)

            @test size(result_df, 1) == 3  # 1 date × 3 quantiles
            @test all(result_df.value .== 42.0)  # All quantiles should equal the single value
            @test result_df.horizon[1] == 1  # One week ahead

        finally
            rm(temp_dir, recursive = true)
        end
    end

    @testset "Data Type Consistency" begin
        # Test that all columns have expected data types
        forecast_date = Date("2024-01-01")
        location = "WA"
        pathogen = "Influenza"
        target = "nhsn"

        input = EpiAutoGPInput(
            [Date("2023-12-01")],  # dates
            [75.0],                # reports
            pathogen,              # pathogen
            location,              # location
            target,                # target
            forecast_date,         # forecast_date
            Date[],                # nowcast_dates
            Vector{Real}[]         # nowcast_reports
        )

        forecast_dates = [Date("2024-01-08"), Date("2024-01-15")]
        forecasts = [80.0 85.0 90.0; 70.0 75.0 80.0]  # 2 dates × 3 samples

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        output_type = QuantileOutput(quantile_levels = [0.25, 0.5, 0.75])
        temp_dir = mktempdir()

        try
            result_df = create_forecast_output(
                input, results, temp_dir, output_type, save_output = false)

            # Test column types
            @test eltype(result_df.output_type) == String
            @test eltype(result_df.output_type_id) == Float64
            @test eltype(result_df.value) == Float64
            @test eltype(result_df.reference_date) == Date
            @test eltype(result_df.target) == String
            @test eltype(result_df.horizon) == Int64
            @test eltype(result_df.target_end_date) == Date
            @test eltype(result_df.location) == String

            # Test value ranges are reasonable
            @test all(r -> r >= 0, result_df.value)  # All values should be non-negative
            @test all(r -> r > 0, result_df.horizon)  # All horizons should be positive

        finally
            rm(temp_dir, recursive = true)
        end
    end

    @testset "Horizon Calculation Edge Cases" begin
        # Test horizon calculation with different forecast dates and target dates
        forecast_date = Date("2024-02-15")  # Thursday
        location = "OR"
        pathogen = "RSV"
        target = "nssp"

        input = EpiAutoGPInput(
            [Date("2024-02-01")],  # dates
            [25.0],                # reports
            pathogen,              # pathogen
            location,              # location
            target,                # target
            forecast_date,         # forecast_date
            Date[],                # nowcast_dates
            Vector{Real}[]         # nowcast_reports
        )

        # Test various target dates relative to forecast date
        forecast_dates = [
            Date("2024-02-22"),  # 1 week later
            Date("2024-03-07"),  # 3 weeks later
            Date("2024-03-28")   # 6 weeks later
        ]
        forecasts = [20.0 25.0; 18.0 22.0; 15.0 18.0]  # 3 dates × 2 samples

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        output_type = QuantileOutput(quantile_levels = [0.5])
        temp_dir = mktempdir()

        try
            result_df = create_forecast_output(
                input, results, temp_dir, output_type, save_output = false)

            # Check horizon calculations
            horizons_by_date = Dict()
            for (i, date) in enumerate(forecast_dates)
                date_rows = result_df[result_df.target_end_date .== date, :]
                horizons_by_date[date] = date_rows.horizon[1]
            end

            @test horizons_by_date[Date("2024-02-22")] == 1
            @test horizons_by_date[Date("2024-03-07")] == 3
            @test horizons_by_date[Date("2024-03-28")] == 6

        finally
            rm(temp_dir, recursive = true)
        end
    end

    @testset "save_output Parameter Tests" begin
        # Test save_output=false (no file should be created)
        forecast_date = Date("2024-03-15")
        location = "MA"
        pathogen = "COVID-19"
        target = "nhsn"

        input = EpiAutoGPInput(
            [Date("2024-03-01")],  # dates
            [50.0],                # reports
            pathogen,              # pathogen
            location,              # location
            target,                # target
            forecast_date,         # forecast_date
            Date[],                # nowcast_dates
            Vector{Real}[]         # nowcast_reports
        )

        forecast_dates = [Date("2024-03-22")]
        forecasts = reshape([45.0, 50.0, 55.0], 1, 3)

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        output_type = QuantileOutput(quantile_levels = [0.5])
        temp_dir = mktempdir()

        try
            # Test with save_output=false
            result_df = create_forecast_output(
                input, results, temp_dir, output_type, save_output = false)

            # DataFrame should still be returned
            @test isa(result_df, DataFrame)
            @test size(result_df, 1) == 1

            # No files should be created in the directory
            @test isempty(readdir(temp_dir))

            # Test with save_output=true
            result_df_saved = create_forecast_output(
                input, results, temp_dir, output_type, save_output = true)

            # DataFrame should be identical
            @test result_df == result_df_saved

            # File should now exist with correct name
            expected_filename = "2024-03-15-CFA-EpiAutoGP-MA-covid-nhsn.csv"
            csv_path = joinpath(temp_dir, expected_filename)
            @test isfile(csv_path)
            @test length(readdir(temp_dir)) == 1

        finally
            rm(temp_dir, recursive = true)
        end
    end

    @testset "Custom Group and Model Names" begin
        # Test custom group_name and model_name parameters
        forecast_date = Date("2024-06-01")
        location = "TX"
        pathogen = "Influenza"
        target = "nssp"

        input = EpiAutoGPInput(
            [Date("2024-05-15")],  # dates
            [75.0],                # reports
            pathogen,              # pathogen
            location,              # location
            target,                # target
            forecast_date,         # forecast_date
            Date[],                # nowcast_dates
            Vector{Real}[]         # nowcast_reports
        )

        forecast_dates = [Date("2024-06-08")]
        forecasts = reshape([70.0, 75.0, 80.0], 1, 3)

        results = (
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            disease = pathogen,
            location = location
        )

        output_type = QuantileOutput(quantile_levels = [0.5])
        temp_dir = mktempdir()

        try
            # Test with custom group and model names
            result_df = create_forecast_output(
                input, results, temp_dir, output_type;
                save_output = true,
                group_name = "TestGroup",
                model_name = "TestModel"
            )

            # Check filename includes custom names
            expected_filename = "2024-06-01-TestGroup-TestModel-TX-flu-nssp.csv"
            csv_path = joinpath(temp_dir, expected_filename)
            @test isfile(csv_path)

        finally
            rm(temp_dir, recursive = true)
        end
    end

    @testset "Filename Format Validation" begin
        # Test that filenames are generated correctly for different combinations
        test_cases = [
            (Date("2024-01-01"), "US", "COVID-19", "nhsn", "CFA",
                "EpiAutoGP", "2024-01-01-CFA-EpiAutoGP-US-covid-nhsn.csv"),
            (Date("2024-12-31"), "CA", "Influenza", "nssp", "CDC",
                "TestModel", "2024-12-31-CDC-TestModel-CA-flu-nssp.csv"),
            (Date("2024-07-04"), "NY", "RSV", "nhsn", "FDA",
                "Model-v2", "2024-07-04-FDA-Model-v2-NY-rsv-nhsn.csv")
        ]

        for (forecast_date, location, pathogen, target,
            group_name, model_name, expected_filename) in test_cases

            input = EpiAutoGPInput(
                [forecast_date - Day(7)],  # dates
                [100.0],                    # reports
                pathogen,                   # pathogen
                location,                   # location
                target,                     # target
                forecast_date,              # forecast_date
                Date[],                     # nowcast_dates
                Vector{Real}[]              # nowcast_reports
            )

            forecast_dates = [forecast_date + Day(7)]
            forecasts = reshape([90.0, 95.0, 100.0], 1, 3)

            results = (
                forecast_dates = forecast_dates,
                forecasts = forecasts,
                disease = pathogen,
                location = location
            )

            output_type = QuantileOutput(quantile_levels = [0.5])
            temp_dir = mktempdir()

            try
                result_df = create_forecast_output(
                    input, results, temp_dir, output_type;
                    save_output = true,
                    group_name = group_name,
                    model_name = model_name
                )

                csv_path = joinpath(temp_dir, expected_filename)
                @test isfile(csv_path)

            finally
                rm(temp_dir, recursive = true)
            end
        end
    end
end
