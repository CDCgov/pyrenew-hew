"""
    create_sample_in       nowcast_dates = dates[max(1, end - 2):end]  # Last 3 days
    nowcast_reports = [[reports[max(1, end - 2) + j - 1] + rand(-5:5) for j in 1:3] for _ in 1:10]  # 10 realizations, each with 3 values

    input_data = EpiAutoGPInput(owcast_dates = dates[max(1, end - 2):end]  # Last 3 days
    nowcast_reports = [[reports[max(1, end - 2) + j - 1] + rand(-5:5) for j in 1:3] for _ in 1:10]  # 10 realizations, each with 3 values

    input_data = EpiAutoGPInput(t(output_path::String; n_weeks::Int=30, pathogen::String="COVID-19", location::String="CA")

Create a sample EpiAutoGPInput for testing and write it to a JSON file.

Creates realistic epidemiological data with weekly observations, seasonal patterns,
and nowcasting requirements, then serializes it to JSON format.

# Arguments
- `output_path::String`: Path where the JSON file will be written
- `n_weeks::Int=30`: Number of weeks of data to generate
- `pathogen::String="COVID-19"`: Disease identifier
- `location::String="CA"`: Geographic location

# Returns
- `EpiAutoGPInput`: The created data structure (also written to file)
"""
function create_sample_input(output_path::String; n_weeks::Int = 30,
        pathogen::String = "COVID-19", location::String = "CA")
    start_date = Date("2024-01-01")
    dates = [start_date + Week(i) for i in 0:(n_weeks - 1)]
    reports = [rand(20:100) + 10*sin(2π*i/7) + rand() * 5 for i in 1:n_weeks]  # Weekly pattern with noise

    forecast_date = dates[end]
    nowcast_dates = dates[max(1, end - 2):end]  # Last 3 days
    nowcast_reports = [[reports[max(1, end - 2) + j - 1] + rand(-5:5) for j in 1:3]
                       for _ in 1:10]  # 10 realizations, each with 3 values

    input_data = EpiAutoGPInput(
        dates, reports, pathogen, location, "nhsn",
        forecast_date, nowcast_dates, nowcast_reports
    )

    # Write to JSON file
    open(output_path, "w") do f
        JSON3.write(f, input_data)
    end

    return input_data
end

@testset "EpiAutoGPInput Tests" begin
    @testset "EpiAutoGPInput Construction" begin
        # Test valid construction
        dates = [Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03")]
        reports = [45.0, 52.0, 38.0]
        pathogen = "COVID-19"
        location = "CA"
        forecast_date = Date("2024-01-03")
        nowcast_dates = [Date("2024-01-02"), Date("2024-01-03")]
        nowcast_reports = [[50.0, 36.0], [52.0, 38.0], [54.0, 40.0]]  # 3 realizations, each with 2 values for 2 dates

        input_data = EpiAutoGPInput(
            dates, reports, pathogen, location, "nhsn",
            forecast_date, nowcast_dates, nowcast_reports
        )

        @test input_data.dates == dates
        @test input_data.reports == reports
        @test input_data.pathogen == pathogen
        @test input_data.location == location
        @test input_data.forecast_date == forecast_date
        @test input_data.nowcast_dates == nowcast_dates
        @test input_data.nowcast_reports == nowcast_reports
    end

    @testset "JSON Serialization/Deserialization" begin
        # Create test data
        input_data = EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-02")],
            [45.0, 52.0],
            "Influenza",
            "NY",
            "nssp",
            Date("2024-01-02"),
            [Date("2024-01-02")],
            [[50.0], [55.0]]  # 2 realizations, each with 1 value
        )

        # Test JSON serialization
        json_string = JSON3.write(input_data)
        @test isa(json_string, String)
        @test occursin("Influenza", json_string)
        @test occursin("NY", json_string)

        # Test JSON deserialization
        parsed_data = JSON3.read(json_string, EpiAutoGPInput)
        @test parsed_data.dates == input_data.dates
        @test parsed_data.reports == input_data.reports
        @test parsed_data.pathogen == input_data.pathogen
        @test parsed_data.location == input_data.location
        @test parsed_data.forecast_date == input_data.forecast_date
        @test parsed_data.nowcast_dates == input_data.nowcast_dates
        @test parsed_data.nowcast_reports == input_data.nowcast_reports
    end

    @testset "Data Validation - Valid Cases" begin
        # Test minimal valid data
        valid_data = EpiAutoGPInput(
            [Date("2024-01-01")],
            [10.0],
            "COVID-19",
            "TX",
            "nhsn",
            Date("2024-01-01"),
            Date[],
            Vector{Real}[]
        )
        @test validate_input(valid_data) == true

        # Test typical valid data
        typical_data = EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-02"), Date("2024-01-03")],
            [45.0, 52.0, 38.0],
            "Influenza",
            "CA",
            "nhsn",
            Date("2024-01-03"),
            [Date("2024-01-02"), Date("2024-01-03")],
            [[50.0, 36.0], [52.0, 40.0]]  # 2 realizations, each with 2 values
        )
        @test validate_input(typical_data) == true

        # Test zero values (should be valid)
        zero_data = EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-02")],
            [0.0, 0.0],
            "RSV",
            "FL",
            "nssp",
            Date("2024-01-02"),
            Date[],
            Vector{Real}[]
        )
        @test validate_input(zero_data) == true
    end

    @testset "Data Validation - Invalid Cases" begin
        # Test mismatched dates and reports length
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-02")],
            [45.0],  # Wrong length
            "COVID-19", "CA", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        ))

        # Test empty essential data
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            Date[], Real[], "COVID-19", "CA", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        ))

        # Test unsorted dates
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-02"), Date("2024-01-01")],  # Wrong order
            [45.0, 52.0],
            "COVID-19", "CA", "nhsn", Date("2024-01-02"), Date[], Vector{Real}[]
        ))

        # Test mismatched nowcast vector lengths - each vector should have length equal to nowcast_dates
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [45.0], "COVID-19", "CA", "nhsn", Date("2024-01-01"),
            [Date("2024-01-01")], [[50.0, 55.0]]  # Vector has length 2 but only 1 nowcast date
        ))

        # Test correct nowcast structure - 100 vectors each with 1 value for 1 nowcast date
        correct_nowcast = EpiAutoGPInput(
            [Date("2024-01-01")], [45.0], "COVID-19", "CA", "nhsn", Date("2024-01-01"),
            [Date("2024-01-01")], [[50.0 + i] for i in 1:100]  # 100 vectors, each with 1 value
        )
        @test validate_input(correct_nowcast) == true

        # Test correct nowcast structure - 50 vectors each with 2 values for 2 nowcast dates
        correct_nowcast_2dates = EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-02")], [45.0, 52.0], "COVID-19", "CA", "nhsn", Date("2024-01-02"),
            [Date("2024-01-01"), Date("2024-01-02")], [[40.0 + i, 50.0 + i] for i in 1:50]  # 50 vectors, each with 2 values
        )
        @test validate_input(correct_nowcast_2dates) == true

        # Test unsorted nowcast dates
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-02")], [45.0, 52.0], "COVID-19", "CA", "nhsn", Date("2024-01-02"),
            [Date("2024-01-02"), Date("2024-01-01")],  # Wrong order
            [[50.0, 45.0]]  # 1 realization with 2 values
        ))

        # Test empty pathogen
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [45.0], "", "CA", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        ))

        # Test empty location
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [45.0], "COVID-19", "  ", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        ))

        # Test negative reports
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [-5.0], "COVID-19", "CA", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        ))

        # Test infinite reports
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [Inf], "COVID-19", "CA", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        ))

        # Test NaN reports
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [NaN], "COVID-19", "CA", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        ))

        # Test empty nowcast - no nowcast dates and no nowcast reports (pure forecasting)
        @test validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [45.0], "COVID-19", "CA", "nhsn", Date("2024-01-01"),
            Date[], Vector{Real}[]  # Empty nowcast arrays for pure forecasting
        )) == true

        # Test nowcast dates with empty reports (0 realizations/samples)
        @test validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [45.0], "COVID-19", "CA", "nhsn", Date("2024-01-01"),
            [Date("2024-01-01")], Vector{Real}[]  # Nowcast date but no samples yet
        )) == true

        # Test invalid nowcast report values
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01")], [45.0], "COVID-19", "CA", "nhsn", Date("2024-01-01"),
            [Date("2024-01-01")], [[-5.0]]  # 1 realization with negative value
        ))

        # Test forecast date too far in past
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-31")], [45.0, 52.0],
            "COVID-19", "CA", "nhsn", Date("2023-01-01"), Date[], Vector{Real}[]  # Way too early
        ))

        # Test forecast date too far in future
        @test_throws ArgumentError validate_input(EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-31")], [45.0, 52.0],
            "COVID-19", "CA", "nhsn", Date("2025-01-01"), Date[], Vector{Real}[]  # Way too late
        ))
    end

    @testset "File I/O Functions" begin
        # Test read_data with temporary file
        test_data = Dict(
            "dates" => ["2024-01-01", "2024-01-02"],
            "reports" => [45.0, 52.0],
            "pathogen" => "COVID-19",
            "location" => "CA",
            "target" => "nhsn",
            "forecast_date" => "2024-01-02",
            "nowcast_dates" => ["2024-01-02"],
            "nowcast_reports" => [[50.0], [55.0]]
        )

        tmpfile = tempname() * ".json"
        try
            # Write test JSON
            open(tmpfile, "w") do f
                JSON3.write(f, test_data)
            end

            # Test read_data
            loaded_data = read_data(tmpfile)
            @test loaded_data.dates == [Date("2024-01-01"), Date("2024-01-02")]
            @test loaded_data.reports == [45.0, 52.0]
            @test loaded_data.pathogen == "COVID-19"
            @test loaded_data.location == "CA"
            @test loaded_data.forecast_date == Date("2024-01-02")
            @test loaded_data.nowcast_dates == [Date("2024-01-02")]
            @test loaded_data.nowcast_reports == [[50.0], [55.0]]

            # Test read_and_validate_data
            validated_data = read_and_validate_data(tmpfile)
            @test validated_data.dates == loaded_data.dates
            @test validated_data.reports == loaded_data.reports

        finally
            # Clean up
            isfile(tmpfile) && rm(tmpfile)
        end

        # Test file not found error
        @test_throws SystemError read_data("nonexistent_file.json")
        @test_throws SystemError read_and_validate_data("nonexistent_file.json")
    end

    @testset "Edge Cases and Special Values" begin
        # Test with integer reports (should work as Real)
        int_data = EpiAutoGPInput(
            [Date("2024-01-01")], [45], "COVID-19", "CA", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        )
        @test validate_input(int_data) == true

        # Test with mixed integer/float reports
        mixed_data = EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-02")], [45, 52.5],
            "COVID-19", "CA", "nhsn", Date("2024-01-02"), Date[], Vector{Real}[]
        )
        @test validate_input(mixed_data) == true

        # Test with very large valid numbers
        large_data = EpiAutoGPInput(
            [Date("2024-01-01")], [1e6], "COVID-19", "CA", "nhsn", Date("2024-01-01"), Date[], Vector{Real}[]
        )
        @test validate_input(large_data) == true

        # Test forecast date edge cases (boundary conditions)
        boundary_data = EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-31")], [45.0, 52.0],
            "COVID-19", "CA", "nhsn", Date("2024-01-31"), Date[], Vector{Real}[]  # Exactly at max date
        )
        @test validate_input(boundary_data) == true

        # Test single nowcast entry
        single_nowcast = EpiAutoGPInput(
            [Date("2024-01-01"), Date("2024-01-02")], [45.0, 52.0],
            "COVID-19", "CA", "nhsn", Date("2024-01-02"),
            [Date("2024-01-02")], [[50.0]]  # 1 realization with 1 value
        )
        @test validate_input(single_nowcast) == true
    end

    @testset "Real-World Data Patterns" begin
        # Test data pattern similar to PyRenew-HEW usage
        realistic_dates = [Date("2024-01-01") + Day(i) for i in 0:29]  # 30 days
        realistic_reports = [rand(20:100) for _ in 1:30]  # Random case counts

        realistic_data = EpiAutoGPInput(
            realistic_dates,
            realistic_reports,
            "COVID-19",
            "NY",
            "nhsn",
            Date("2024-01-30"),
            realistic_dates[(end - 2):end],  # Last 3 days for nowcasting
            [[realistic_reports[end - 2] + rand(),
                 realistic_reports[end - 1] + rand(), realistic_reports[end] + rand()]
             for _ in 1:5]  # 5 realizations with 3 values each
        )

        @test validate_input(realistic_data) == true

        # Test JSON round-trip with realistic data
        json_str = JSON3.write(realistic_data)
        reconstructed = JSON3.read(json_str, EpiAutoGPInput)
        @test validate_input(reconstructed) == true
        @test reconstructed.pathogen == realistic_data.pathogen
        @test reconstructed.location == realistic_data.location
        @test length(reconstructed.dates) == length(realistic_data.dates)
    end

    @testset "Sample Input Creation and JSON Round-trip" begin
        # Create temporary directory for test files
        tmpdir = mktempdir()

        try
            # Test create_sample_input with default parameters
            default_json_path = joinpath(tmpdir, "default_sample.json")
            default_sample = create_sample_input(default_json_path)

            @test validate_input(default_sample) == true
            @test default_sample.pathogen == "COVID-19"
            @test default_sample.location == "CA"
            @test length(default_sample.dates) == 30
            @test isfile(default_json_path)

            # Test loading the written JSON file
            loaded_default = read_and_validate_data(default_json_path)
            @test loaded_default.dates == default_sample.dates
            @test loaded_default.reports == default_sample.reports
            @test loaded_default.pathogen == default_sample.pathogen
            @test loaded_default.location == default_sample.location
            @test loaded_default.forecast_date == default_sample.forecast_date
            @test loaded_default.nowcast_dates == default_sample.nowcast_dates
            @test loaded_default.nowcast_reports == default_sample.nowcast_reports

            # Test create_sample_input with custom parameters
            custom_json_path = joinpath(tmpdir, "custom_sample.json")
            custom_sample = create_sample_input(
                custom_json_path; n_weeks = 14, pathogen = "Influenza", location = "NY")

            @test validate_input(custom_sample) == true
            @test custom_sample.pathogen == "Influenza"
            @test custom_sample.location == "NY"
            @test length(custom_sample.dates) == 14
            @test length(custom_sample.reports) == 14
            @test isfile(custom_json_path)

            # Test loading the custom JSON file
            loaded_custom = read_and_validate_data(custom_json_path)
            @test loaded_custom.dates == custom_sample.dates
            @test loaded_custom.reports == custom_sample.reports
            @test loaded_custom.pathogen == custom_sample.pathogen
            @test loaded_custom.location == custom_sample.location

            # Test that nowcast dates are properly set (last 3 days)
            @test length(custom_sample.nowcast_dates) == 3
            @test custom_sample.nowcast_dates == custom_sample.dates[(end - 2):end]
            @test length(custom_sample.nowcast_reports) == 10  # 10 realizations as set in create_sample_input

            # Verify nowcast data round-trip
            @test loaded_custom.nowcast_dates == custom_sample.nowcast_dates
            @test loaded_custom.nowcast_reports == custom_sample.nowcast_reports

        finally
            # Clean up temporary directory
            rm(tmpdir, recursive = true)
        end
    end
end
