# generate data for testing
test_data <- generate_example_data(target = "covid", est_non_resp = 2.6e6, savedata = FALSE)

# Test the forecast_counts function
test_that("forecast_counts returns correct structure", {
  result <- forecast_counts(test_data, count_col = "other_ed_visits", date_col = "date", h = "3 weeks")

  # Check if the result is a list
  expect_type(result, "list")

  # Check if the list contains the correct elements
  expect_named(result, c("predictive_samples", "fc"))

  # Check if predictive_samples is a tsibble
  expect_s3_class(result$predictive_samples, "tbl_ts")

  # Check if fc is a fable object
  expect_s3_class(result$fc, "fbl_ts")
})

test_that("forecast_counts handles different forecast horizons", {
  result_1_week <- forecast_counts(test_data, count_col = "other_ed_visits", date_col = "date", h = "1 week")
  result_4_weeks <- forecast_counts(test_data, count_col = "other_ed_visits", date_col = "date", h = "4 weeks")

  # Check if the forecast horizon affects the number of rows in the forecast
  expect_true(nrow(result_1_week$fc) < nrow(result_4_weeks$fc))
})

test_that("forecast_counts handles missing values gracefully", {
  test_data_with_na <- test_data
  test_data_with_na$other_ed_visits[1:5] <- NA

  result <- forecast_counts(test_data_with_na, count_col = "other_ed_visits", date_col = "date", h = "3 weeks")

  # Check if the function still returns a list with the correct elements
  expect_named(result, c("predictive_samples", "fc"))
})

test_that("forecast_counts handles different count columns", {
  test_data_renamed <- test_data %>%
    rename(new_count = other_ed_visits)

  result <- forecast_counts(test_data_renamed, count_col = "new_count", date_col = "date", h = "3 weeks")

  # Check if the function still returns a list with the correct elements
  expect_named(result, c("predictive_samples", "fc"))
})

test_that("forecast_counts samples the correct number of times", {
  result <- forecast_counts(test_data, count_col = "other_ed_visits", date_col = "date", h = "3 weeks", times = 2000)

  # Check if the number of samples is correct
  # divide by 3 because of the 3 models used
  expected_samples <- 2000 # This should match the 'times' parameter in the generate function
  actual_samples <- nrow(result$predictive_samples) / (3 * length(unique(result$predictive_samples$date)))

  expect_equal(actual_samples, expected_samples)
})
