create_forecast_data <- function(
    directory, filename, date_cols, disease_cols, n_draw) {
  data <- tibble::tibble(
    date = rep(date_cols, times = length(disease_cols) * n_draw),
    disease = rep(disease_cols, each = length(date_cols) * n_draw),
    .value = sample(
      1:100, length(disease_cols) * length(date_cols) * n_draw,
      replace = TRUE
    ),
    .draw = rep(
      rep(1:20, each = length(date_cols)),
      times = length(disease_cols)
    )
  )
  if (length(disease_cols) == 1) {
    data <- data |>
      dplyr::rename(!!disease_cols := ".value") |>
      dplyr::select(-disease)
  }
  arrow::write_parquet(data, fs::path(directory, filename))
}

testthat::test_that("to_epiweekly_quantiles works as expected", {
  # create temporary directories and forecast files for tests
  temp_dir <- fs::path_temp("CA")
  fs::dir_create(fs::path(temp_dir, "pyrenew_e"))
  fs::dir_create(fs::path(temp_dir, "timeseries_e"))

  create_forecast_data(
    fs::path(temp_dir, "pyrenew_e"),
    "forecast_samples.parquet",
    seq(lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"), by = "day"),
    c("Disease", "Other", "prop_disease_ed_visits"),
    20
  )

  create_forecast_data(
    fs::path(temp_dir, "timeseries_e"),
    "epiweekly_other_ed_visits_forecast.parquet",
    seq(lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"), by = "day"),
    "other_ed_visits",
    20
  )

  check <- function(epiweekly_other_bool) {
    result <- to_epiweekly_quantiles(
      model_run_dir = temp_dir,
      report_date = "2024-12-14",
      max_lookback_days = 8,
      epiweekly_other = epiweekly_other_bool
    )

    testthat::expect_s3_class(result, "tbl_df")
    testthat::expect_true(all(c(
      "epiweek", "epiyear", "quantile_value", "quantile_level", "location"
    ) %in% colnames(result)))
    testthat::expect_gt(nrow(result), 0)
  }

  check(epiweekly_other_bool = FALSE)
  check(epiweekly_other_bool = TRUE)

  fs::dir_delete(temp_dir)
})



testthat::test_that("to_epiweekly_quantiles handles missing forecast files", {
  temp_dir <- fs::path_temp("CA")
  fs::dir_create(fs::path(temp_dir, "pyrenew_e"))

  expect_error(
    result <- to_epiweekly_quantiles(
      model_run_dir = temp_dir,
      report_date = "2024-12-14",
      max_lookback_days = 8
    ),
    "Failed to open local file"
  )

  fs::dir_delete(temp_dir)
})
