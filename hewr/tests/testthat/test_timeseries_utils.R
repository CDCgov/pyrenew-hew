# Common test data fixtures
common_required_columns <- c(
  ".draw",
  "date",
  "geo_value",
  "disease",
  ".variable",
  ".value",
  "resolution",
  "aggregated_numerator",
  "aggregated_denominator"
)

base_date <- as.Date("2024-01-01")

test_that("format_timeseries_output formats forecast data correctly", {
  # Create minimal test data
  forecast_data <- tibble::tibble(
    date = base_date + 0:2,
    .draw = c(1, 1, 1),
    observed_ed_visits = c(10, 15, 20),
    other_ed_visits = c(5, 7, 9)
  )

  result <- format_timeseries_output(
    forecast_data = forecast_data,
    geo_value = "US",
    disease = "COVID-19",
    resolution = "daily",
    output_type_id = ".draw"
  )

  # Check that output has expected structure
  expect_s3_class(result, "data.frame")
  expect_true(all(
    c(
      "date",
      "geo_value",
      "disease",
      "resolution",
      "aggregated_numerator",
      "aggregated_denominator",
      ".variable",
      ".draw",
      ".value"
    ) %in%
      colnames(result)
  ))

  # Check that geo_value and disease are set correctly
  expect_true(all(result$geo_value == "US"))
  expect_true(all(result$disease == "COVID-19"))
  expect_true(all(result$resolution == "daily"))

  # Check aggregation flags
  expect_true(!any(result$aggregated_numerator))

  # Check that data was pivoted (should have 2 variables x 3 dates = 6 rows)
  expect_equal(nrow(result), 6)
})

test_that("format_timeseries_output handles proportion variables", {
  forecast_data <- tibble::tibble(
    date = base_date,
    .draw = 1,
    prop_disease_ed_visits = 0.5
  )

  result <- format_timeseries_output(
    forecast_data = forecast_data,
    geo_value = "CA",
    disease = "Influenza",
    resolution = "epiweekly",
    output_type_id = ".draw"
  )

  # Proportion variables should have aggregated_denominator = FALSE
  prop_row <- result[result$.variable == "prop_disease_ed_visits", ]
  expect_equal(prop_row$aggregated_denominator, FALSE)
})

test_that("prop_from_timeseries calculates proportions correctly", {
  e_denominator_samples <- tibble::tibble(
    resolution = "daily",
    .draw = 1:3,
    date = base_date,
    geo_value = "US",
    disease = "COVID-19",
    other_ed_visits = c(5, 10, 15)
  )

  e_numerator_samples <- tibble::tibble(
    resolution = "daily",
    .draw = 1:3,
    date = base_date,
    geo_value = "US",
    disease = "COVID-19",
    observed_ed_visits = c(10, 20, 30),
    aggregated_numerator = FALSE
  )

  result <- prop_from_timeseries(
    e_denominator_samples,
    e_numerator_samples,
    common_required_columns
  )

  # Check that proportions are calculated correctly
  # prop = {observed \over observed + other}
  # For draw 1: 10 / (10 + 5) = 0.6666...
  # For draw 2: 20 / (20 + 10) = 0.6666...
  # For draw 3: 30 / (30 + 15) = 0.6666...
  expect_equal(nrow(result), 3)
  expect_true(all(result$.variable == "prop_disease_ed_visits"))
  expect_equal(result$.value, rep(2 / 3, 3), tolerance = 1e-10)
})

test_that("epiweekly_samples_from_daily aggregates correctly", {
  # Create test data - simple smoke test
  # Use dates that align with epiweeks (Sunday to Saturday)
  daily_samples <- tibble::tibble(
    .draw = rep(1, 7),
    date = as.Date("2024-01-07") + 0:6, # One complete epiweek
    geo_value = "US",
    disease = "COVID-19",
    .variable = "observed_ed_visits",
    .value = rep(10, 7),
    resolution = "daily",
    aggregated_numerator = FALSE,
    aggregated_denominator = NA
  )

  result <- epiweekly_samples_from_daily(
    daily_samples = daily_samples,
    variables_to_aggregate = "observed_ed_visits",
    required_columns = common_required_columns
  )

  # Basic smoke test - should aggregate successfully
  expect_s3_class(result, "data.frame")
  expect_true(nrow(result) >= 1)
  expect_true(all(result$resolution == "epiweekly"))
  expect_true(all(result$aggregated_numerator))
  expect_true(all(result$.variable == "observed_ed_visits"))
})

test_that("augment_timeseries_draws_with_obs combines forecast and observed", {
  # Create minimal forecast data - 3 forecast dates x 2 draws = 6 rows
  tidy_forecast <- tibble::tibble(
    date = rep(as.Date("2024-01-08") + 0:2, each = 2),
    .draw = rep(1:2, 3),
    .variable = rep("observed_ed_visits", 6),
    .value = c(15, 16, 17, 18, 19, 20),
    resolution = "daily"
  )

  observed <- tibble::tibble(
    date = base_date + 0:6,
    .variable = "observed_ed_visits",
    .value = 10:16
  )

  result <- augment_timeseries_draws_with_obs(
    tidy_forecast = tidy_forecast,
    observed = observed
  )

  # Should combine observed (7 days * 2 draws = 14) + forecast (6 rows)
  expect_equal(nrow(result), 20)

  # Should have .draw column first
  expect_equal(colnames(result)[1], ".draw")

  # All rows should have .draw values
  expect_false(anyNA(result$.draw))

  # Check observed dates all have same value across draws
  obs_dates <- observed$date
  for (d in obs_dates) {
    date_rows <- result[result$date == d, ]
    expect_equal(length(unique(date_rows$.value)), 1)
  }
})
