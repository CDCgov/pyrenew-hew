using Test
using Dates
using EpiAutoGP
using NowcastAutoGP

@testset "Modelling Functions Tests" begin

    # Helper function to create test input data with proper Float64 types
    function create_test_input(; include_nowcasts = true)
        dates = [Date(2024, 1, 1) + Week(i-1) for i in 1:10]
        # Use explicit Float64 for all values to avoid Box-Cox integer conversion issues
        reports = Float64[1000, 1100, 1050, 1150, 1200, 1250, 1300, 1350, 1400, 1450]

        if include_nowcasts
            nowcast_dates = dates[(end - 1):end]  # Last 2 weeks
            # Each nowcast sample must have same length as nowcast_dates (2 values each)
            # Ensure all values are Float64
            nowcast_reports = [
                Float64[1350, 1400], Float64[1400, 1450], Float64[1300, 1350]]
        else
            nowcast_dates = Date[]
            nowcast_reports = Vector{Float64}[]
        end

        return EpiAutoGPInput(
            dates,
            reports,
            "COVID-19",
            "US",
            "nhsn",
            dates[end],  # forecast from last date
            nowcast_dates,
            nowcast_reports
        )
    end

    @testset "prepare_for_modelling function" begin
        input = create_test_input(include_nowcasts = true)
        result = prepare_for_modelling(input, "boxcox", 4, 100)

        # Check that all expected fields are present
        @test haskey(result, :stable_data_dates)
        @test haskey(result, :stable_data_values)
        @test haskey(result, :nowcast_data)
        @test haskey(result, :forecast_dates)
        @test haskey(result, :n_forecasts_per_nowcast)
        @test haskey(result, :transformation)
        @test haskey(result, :inv_transformation)

        # Check that stable data excludes nowcast dates
        @test length(result.stable_data_dates) == 8  # 10 total - 2 nowcast dates
        @test length(result.stable_data_values) == 8

        # Check forecast dates
        @test length(result.forecast_dates) == 4
        @test result.forecast_dates[1] == input.forecast_date + Week(1)
        @test result.forecast_dates[end] == input.forecast_date + Week(4)

        # Check that transformation functions are callable
        @test isa(result.transformation, Function)
        @test isa(result.inv_transformation, Function)
    end

    @testset "fit_base_model function" begin
        input = create_test_input(include_nowcasts = true)
        prep_result = prepare_for_modelling(input, "positive", 2, 50)

        # Use minimal parameters for faster testing
        # fit_base_model expects dates and values directly
        model = fit_base_model(
            prep_result.stable_data_dates, prep_result.stable_data_values;
            transformation = prep_result.transformation,
            n_particles = 1,
            smc_data_proportion = 0.5,
            n_mcmc = 3,
            n_hmc = 3
        )

        # Check that we get a valid model
        @test model !== nothing

        # Check that the model has been fitted (should have some internal state)
        @test typeof(model) != Nothing
    end

    @testset "forecast_with_epiautogp function" begin
        @testset "Forecasting without nowcasts" begin
            input = create_test_input(include_nowcasts = false)

            # Test without nowcasts using positive transformation to avoid Box-Cox issues
            forecast_dates,
            forecasts = forecast_with_epiautogp(input;
                n_forecast_weeks = 2,
                n_forecasts = 10,
                transformation_name = "positive",
                n_particles = 1,
                smc_data_proportion = 0.5,
                n_mcmc = 3,
                n_hmc = 3
            )

            # Check forecast dates
            @test length(forecast_dates) == 2
            @test forecast_dates[1] == input.forecast_date + Week(1)

            # Check forecasts dimensions
            @test size(forecasts, 1) == 2
            @test size(forecasts, 2) == 10

            # Check that all forecasts are positive
            @test all(forecasts .> 0)
        end

        @testset "Forecasting with nowcasts" begin
            input = create_test_input(include_nowcasts = true)

            # Test with nowcasts using positive transformation
            forecast_dates,
            forecasts = forecast_with_epiautogp(input;
                n_forecast_weeks = 2,
                n_forecasts = 20,
                transformation_name = "positive",
                n_particles = 1,
                smc_data_proportion = 0.5,
                n_mcmc = 3,
                n_hmc = 3
            )

            # Check forecast dates
            @test length(forecast_dates) == 2
            @test forecast_dates[1] == input.forecast_date + Week(1)
            @test forecast_dates[2] == input.forecast_date + Week(2)

            # Check forecasts matrix dimensions
            @test size(forecasts, 1) == 2  # forecast weeks
            # With nowcasts, the actual number of forecasts may be less than requested
            @test size(forecasts, 2) > 0  # should have some forecasts

            # Check that forecasts are positive numbers
            @test all(forecasts .> 0)
        end
    end

    @testset "forecast_with_epiautogp function" begin
        input = create_test_input(include_nowcasts = false)

        args = Dict(
            "n-forecast-weeks" => 2,
            "n-forecast-draws" => 30,
            "transformation" => "positive",
            "n-particles" => 1,
            "smc-data-proportion" => 0.5,
            "n-mcmc" => 3,
            "n-hmc" => 3
        )

        forecast_dates, forecasts = forecast_with_epiautogp(input, args)

        # Check results
        @test length(forecast_dates) == 2
        @test size(forecasts, 1) == 2
        @test size(forecasts, 2) == 30
        @test all(forecasts .> 0)
    end
end
