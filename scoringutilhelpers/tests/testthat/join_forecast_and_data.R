# Test join_forecast_and_data function
test_that("join_forecast_and_data correctly joins forecast and actual data", {
  # Use testing data functions to test the join_forecast_and_data function
  examplepreds <- example_prediction(save_data = TRUE)
  exampledata <- example_truthdata(save_data = TRUE)
  forecast_source <- "scoringutilhelpers/assets/example_predictions"
  truthdata_file <- "scoringutilhelpers/assets/example_truthdata.tsv"
  # Write mock data to temporary files
  predictions <-
    arrow::open_dataset("scoringutilhelpers/assets/example_predictions")
  actual_data <-
    readr::read_tsv("scoringutilhelpers/assets/example_truthdata.tsv")
  scoringdataset <- join_forecast_and_data(forecast_source, truthdata_file,
    join_key = join_by(area, target_end_date == date)
  ) |> collect()
  expect_true(all(c(
    "area", "reference_date", "target_end_date", "truthdata",
    "prediction", ".chain", ".iteration", ".draw", "model"
  )
  %in% colnames(scoringdataset)))
  # Clean up temporary files
  unlink(forecast_source, recursive = TRUE)
  unlink(truthdata_file)
})
