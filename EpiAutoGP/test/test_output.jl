using Test
using Dates
using EpiAutoGP
using CSV
using TidierData
using JSON3

@testset "Output Functions Tests" begin

    # Helper function to create test forecast results
    function create_test_results()
        forecast_dates = [Date(2024, 1, 15), Date(2024, 1, 22), Date(2024, 1, 29)]
        # Create forecast matrix: 3 dates × 5 samples
        forecasts = [100.0 120.0 95.0 110.0 105.0;  # Week 1
                    110.0 130.0 105.0 115.0 112.0;  # Week 2
                    120.0 135.0 115.0 125.0 118.0]  # Week 3

        return (;
            forecast_dates = forecast_dates,
            forecasts = forecasts,
            forecast_date = Date(2024, 1, 8),  # Reference date
            location = "CA",
            disease = "COVID-19"
        )
    end

    @testset "create_hubverse_table function" begin
        results = create_test_results()
        output_dir = mktempdir()

        # Test basic hubverse table creation
        hubverse_df = create_hubverse_table(results, output_dir)

        # Check that we get a DataFrame
        @test hubverse_df isa DataFrame

        # Check required columns exist
        required_cols = ["reference_date", "target", "horizon", "target_end_date",
                        "location", "output_type", "output_type_id", "value"]
        for col in required_cols
            @test col in names(hubverse_df)
        end

        # Check reference_date is correct
        @test all(Date.(hubverse_df.reference_date) .== Date(2024, 1, 8))

        # Check location and disease
        @test all(hubverse_df.location .== "CA")
        @test all(hubverse_df.target .== "wk inc covid hosp")

        # Check target_end_dates match forecast_dates
        unique_targets = sort(unique(Date.(hubverse_df.target_end_date)))
        @test unique_targets == results.forecast_dates

        # Check horizon calculation (weeks from forecast_date)
        for (i, date) in enumerate(results.forecast_dates)
            expected_horizon = Int(Dates.value(date - results.forecast_date) ÷ 7)
            horizon_rows = filter(row -> row.target_end_date == date, hubverse_df)
            @test all(horizon_rows.horizon .== expected_horizon)
        end

        # Check output types
        unique_output_types = unique(hubverse_df.output_type)
        @test "quantile" in unique_output_types
        # Note: Sample outputs not currently implemented

        # Check quantile levels for quantile outputs
        quantile_rows = filter(row -> row.output_type == "quantile", hubverse_df)
        if nrow(quantile_rows) > 0
            expected_quantiles = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
                                 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
            actual_quantiles = sort(unique(quantile_rows.output_type_id))
            @test actual_quantiles == expected_quantiles
        end

        # Sample outputs not currently implemented - skip validation

        # Check that values are positive
        @test all(hubverse_df.value .> 0)

        # Check that CSV file was created
        csv_path = joinpath(output_dir, "hubverse_table.csv")
        @test isfile(csv_path)

        # Check that saved CSV can be read back
        saved_df = CSV.read(csv_path, DataFrame)
        @test nrow(saved_df) == nrow(hubverse_df)
        @test names(saved_df) == names(hubverse_df)

        # Clean up
        rm(output_dir, recursive=true)
    end

    @testset "save_model_outputs function" begin
        results = create_test_results()
        output_dir = mktempdir()

        # Test model outputs saving
        save_model_outputs(results, output_dir)

        # Check that forecast samples CSV was created
        samples_path = joinpath(output_dir, "mcmc_output", "tidy_forecast_samples.csv")
        @test isfile(samples_path)

        # Read and validate forecast samples
        samples_df = CSV.read(samples_path, DataFrame)
        @test "date" in names(samples_df)
        @test "sample" in names(samples_df)
        @test "forecast_value" in names(samples_df)

        # Should have one row per forecast_date × sample combination
        expected_rows = length(results.forecast_dates) * size(results.forecasts, 2)
        @test nrow(samples_df) == expected_rows

        # Check forecast dates are correct
        unique_dates = sort(unique(samples_df.date))
        @test unique_dates == results.forecast_dates

        # Check sample IDs
        unique_samples = sort(unique(samples_df.sample))
        @test unique_samples == collect(1:size(results.forecasts, 2))

        # Check summary statistics CSV was created
        summary_path = joinpath(output_dir, "mcmc_output", "forecast_summary_stats.csv")
        @test isfile(summary_path)

        # Read and validate summary statistics
        summary_df = CSV.read(summary_path, DataFrame)
        @test "date" in names(summary_df)
        @test "mean" in names(summary_df)
        @test "q50" in names(summary_df)  # median
        @test "q25" in names(summary_df)
        @test "q75" in names(summary_df)

        # Should have one row per forecast date
        @test nrow(summary_df) == length(results.forecast_dates)

        # Check metadata JSON was created
        metadata_path = joinpath(output_dir, "forecast_metadata.json")
        @test isfile(metadata_path)

        # Read and validate metadata
        metadata = JSON3.read(read(metadata_path, String))
        @test metadata["location"] == "CA"
        @test metadata["disease"] == "COVID-19"
        @test Date(metadata["forecast_date"]) == Date(2024, 1, 8)
        @test metadata["n_forecast_dates"] == 3
        @test metadata["n_samples"] == 5

        # Clean up
        rm(output_dir, recursive=true)
    end

    @testset "Edge cases and error handling" begin
        @testset "Empty forecasts" begin
            # Test with empty forecast matrix
            empty_results = (;
                forecast_dates = Date[],
                forecasts = Matrix{Float64}(undef, 0, 0),
                forecast_date = Date(2024, 1, 8),
                location = "CA",
                disease = "COVID-19"
            )

            output_dir = mktempdir()

            # Should handle empty results gracefully
            hubverse_df = create_hubverse_table(empty_results, output_dir)
            @test nrow(hubverse_df) == 0

            save_model_outputs(empty_results, output_dir)

            # Files should still be created but with minimal content
            @test isfile(joinpath(output_dir, "hubverse_table.csv"))
            @test isfile(joinpath(output_dir, "mcmc_output", "tidy_forecast_samples.csv"))
            @test isfile(joinpath(output_dir, "forecast_metadata.json"))

            rm(output_dir, recursive=true)
        end

        @testset "Negative horizons" begin
            # Test with forecast dates before reference date (nowcasts)
            nowcast_results = (;
                forecast_dates = [Date(2024, 1, 1), Date(2024, 1, 8), Date(2024, 1, 15)],
                forecasts = [90.0 95.0; 100.0 105.0; 110.0 115.0],  # 3 dates × 2 samples
                forecast_date = Date(2024, 1, 8),  # Middle date as reference
                location = "NY",
                disease = "Influenza"
            )

            output_dir = mktempdir()
            hubverse_df = create_hubverse_table(nowcast_results, output_dir)

            # Check horizon calculations include negative values
            horizons = sort(unique(hubverse_df.horizon))
            @test -1 in horizons  # Date(2024, 1, 1) is 1 week before reference
            @test 0 in horizons   # Date(2024, 1, 8) is same as reference
            @test 1 in horizons   # Date(2024, 1, 15) is 1 week after reference

            rm(output_dir, recursive=true)
        end
    end
end
