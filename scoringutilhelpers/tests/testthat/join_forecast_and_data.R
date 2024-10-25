

# Test join_forecast_and_data function
test_that("join_forecast_and_data correctly joins forecast and actual data", {
    # Use testing data functions to test the join_forecast_and_data function
    examplepreds <- exampleprediction(savedata = TRUE)
    exampledata <- exampletruthdata(savedata = TRUE)
    forecast_source <- "scoringutilhelpers/assets/examplepredictions"
    truthdata_file <- "scoringutilhelpers/assets/exampletruthdata.tsv"
    # Write mock data to temporary files
    predictions <-
        arrow::open_dataset("scoringutilhelpers/assets/examplepredictions")
    actual_data <- 
        readr::read_tsv("scoringutilhelpers/assets/exampletruthdata.tsv")
    scoringdataset <- join_forecast_and_data(forecast_source, truthdata_file,
        join_key = c("area", "date")) |> collect()
   expect_true(all(c("area", "date", "truthdata", "prediction", ".chain",
   ".iteration", ".draw") %in% colnames(scoringdataset)))
   # Clean up temporary files
    unlink(forecast_source, recursive = TRUE)
    unlink(truthdata_file)
})