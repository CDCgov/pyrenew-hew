hubverse_quantiles <- c(0.01, 0.025, seq(0.05, 0.95, 0.05), 0.975, 0.99)

testthat::test_that("score_hubverse works as expected with valid inputs", {
  date_range <- seq(
    lubridate::ymd("2023-11-01"), lubridate::ymd("2024-01-29"),
    by = "day"
  )

  horizons <- c(0, 1, 2)
  locations <- c("loc1", "loc2")

  forecast <- create_hubverse_table(
    date_range = date_range,
    horizons = horizons,
    locations = locations,
    output_type = "quantile",
    output_type_id = hubverse_quantiles
  )

  expect_equal(
    nrow(forecast),
    length(date_range) *
      length(horizons) *
      length(locations) *
      length(hubverse_quantiles)
  )

  observed <- create_observation_data(
    date_range = date_range,
    locations = locations
  )

  expect_equal(
    nrow(observed),
    length(date_range) * length(locations)
  )

  scored <- score_hewr(forecast, observed)
  expect_setequal(forecast$location, scored$location)
  expect_setequal(scored$horizon, c(0, 1))

  scored_all_horizon <- score_hewr(
    forecast, observed,
    horizons = c(0, 1, 2)
  )
  expect_setequal(forecast$location, scored_all_horizon$location)
  expect_setequal(forecast$horizon, scored_all_horizon$horizon)
})


testthat::test_that("score_hubverse handles missing location data", {
  forecast <- create_hubverse_table(
    date_range = seq(
      lubridate::ymd("2024-11-01"), lubridate::ymd("2024-11-29"),
      by = "day"
    ),
    horizons = c(0, 1),
    locations = c("loc1", "loc2"),
    output_type = "quantile",
    output_type_id = hubverse_quantiles
  )

  observed <- create_observation_data(
    date_range = seq(
      lubridate::ymd("2024-11-01"), lubridate::ymd("2024-11-29"),
      by = "day"
    ),
    locations = c("loc1")
  )

  result <- score_hewr(forecast, observed)
  expect_false("loc2" %in% result$location)
  expect_setequal(observed$location, result$location)
})


testthat::test_that("score_hubverse handles zero length forecast table", {
  forecast <- tibble::tibble(
    reference_date = as.Date(character(0)),
    horizon = integer(0),
    output_type_id = numeric(0),
    location = character(0),
    value = numeric(0),
    target = character(0),
    output_type = character(0),
    target_end_date = as.Date(character(0))
  )

  observed <- create_observation_data(
    date_range = seq(
      lubridate::ymd("2024-11-01"), lubridate::ymd("2024-11-02"),
      by = "day"
    ),
    location = c("loc1")
  )

  expect_error(
    result <- score_hewr(forecast, observed),
    "Assertion on 'data' failed: Must have at least 1 rows, but has 0 rows."
  )
})
