testthat::test_that("score_hubverse works as expected with valid inputs", {
  forecast <- create_hubverse_table(
    date_cols = seq(
      lubridate::ymd("2023-11-01"), lubridate::ymd("2024-01-29"),
      by = "day"
    ),
    horizon = c(0, 1, 2),
    location = c("loc1", "loc2"),
    output_type = "quantile",
    output_type_id = c(0.01, 0.025, seq(0.05, 0.95, 0.05), 0.975, 0.99)
  )

  observed <- create_observation_data(
    date_cols = seq(
      lubridate::ymd("2023-11-01"), lubridate::ymd("2024-01-29"),
      by = "day"
    ),
    location = c("loc1", "loc2")
  )

  scored <- score_hubverse(forecast, observed)
  expect_setequal(forecast$location, scored$location)
  expect_setequal(scored$horizon, c(0, 1))

  scored_all_horizon <- score_hubverse(
    forecast, observed,
    horizons = c(0, 1, 2)
  )
  expect_setequal(forecast$location, scored_all_horizon$location)
  expect_setequal(forecast$horizon, scored_all_horizon$horizon)
})


testthat::test_that("score_hubverse handles missing location data", {
  forecast <- create_hubverse_table(
    date_cols = seq(
      lubridate::ymd("2024-11-01"), lubridate::ymd("2024-11-29"),
      by = "day"
    ),
    horizon = c(0, 1),
    location = c("loc1", "loc2"),
    output_type = "quantile",
    output_type_id = seq(0.05, 0.95, 0.05)
  )

  observed <- create_observation_data(
    date_cols = seq(
      lubridate::ymd("2024-11-01"), lubridate::ymd("2024-11-29"),
      by = "day"
    ),
    location = c("loc1")
  )

  result <- score_hubverse(forecast, observed)
  expect_false("loc2" %in% result$location)
  expect_setequal(observed$location, result$location)
})
