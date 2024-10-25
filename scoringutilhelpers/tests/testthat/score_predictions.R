examplepreds <- exampleprediction(savedata = TRUE)
exampledata <- exampletruthdata(savedata = TRUE)
forecast_source <- "scoringutilhelpers/assets/examplepredictions"
truthdata_file <- "scoringutilhelpers/assets/exampletruthdata.tsv"
forecast_unit <- c("area", "reference_date", "target_end_date", "model")
observed <- "truthdata"
predicted <- "prediction"

test_that("score_forecasts returns a data frame", {
    scorable_data <- join_forecast_and_data(forecast_source, truthdata_file,
        join_key = join_by(area, target_end_date == date)) |>
        collect()
    result <- score_forecasts(scorable_data,
        forecast_unit = forecast_unit,
        observed = observed,
        predicted = predicted)
    expect_s3_class(result, "data.frame")
})

test_that("score_forecasts works with different sample_id", {
    scorable_data <- join_forecast_and_data(forecast_source, truthdata_file,
        join_key = join_by(area, target_end_date == date)) |>
        collect()
    scorable_data <- scorable_data %>% rename(sample_id = .draw)
    result <- score_forecasts(scorable_data,
        forecast_unit = forecast_unit,
        observed = observed,
        predicted = predicted,
        sample_id = "sample_id"
    )
    expect_s3_class(result, "data.frame")
})
# Clean up temporary files
unlink(forecast_source, recursive = TRUE)
unlink(truthdata_file)