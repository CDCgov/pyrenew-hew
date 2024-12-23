create_forecast_data <- function(
    directory, filename, date_cols, disease_cols, n_draw) {
  data <- tidyr::expand_grid(
    date = date_cols,
    disease = disease_cols,
    .draw = 1:n_draw
  ) |>
    dplyr::mutate(.value = sample(1:100, dplyr::n(), replace = TRUE))
  if (length(disease_cols) == 1) {
    data <- data |>
      dplyr::rename(!!disease_cols := ".value") |>
      dplyr::select(-disease)
  }
  arrow::write_parquet(data, fs::path(directory, filename))
}

# tests for `to_epiweekly_quantile`
test_that("to_epiweekly_quantiles works as expected", {
  # create temporary directories and forecast files for tests
  temp_dir <- withr::local_tempdir("CA")
  fs::dir_create(fs::path(temp_dir, "pyrenew_e"))
  fs::dir_create(fs::path(temp_dir, "timeseries_e"))

  create_forecast_data(
    directory = fs::path(temp_dir, "pyrenew_e"),
    filename = "forecast_samples.parquet",
    date_cols = seq(
      lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
      by = "day"
    ),
    disease_cols = c("Disease", "Other", "prop_disease_ed_visits"),
    n_draw = 20
  )

  create_forecast_data(
    directory = fs::path(temp_dir, "timeseries_e"),
    filename = "epiweekly_other_ed_visits_forecast.parquet",
    date_cols = seq(
      lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
      by = "day"
    ),
    disease_cols = "other_ed_visits",
    n_draw = 20
  )

  check_epiweekly_quantiles <- function(epiweekly_other_bool) {
    result <- to_epiweekly_quantiles(
      model_run_dir = temp_dir,
      report_date = "2024-12-14",
      max_lookback_days = 8,
      epiweekly_other = epiweekly_other_bool
    )

    expect_s3_class(result, "tbl_df")
    expect_setequal(c(
      "epiweek", "epiyear", "quantile_value", "quantile_level", "location"
    ), colnames(result))
    expect_gt(nrow(result), 0)
  }

  check_epiweekly_quantiles(epiweekly_other_bool = FALSE)
  check_epiweekly_quantiles(epiweekly_other_bool = TRUE)
})


test_that("to_epiweekly_quantiles calculates quantiles accurately", {
  temp_dir <- withr::local_tempdir("test")
  fs::dir_create(fs::path(temp_dir, "pyrenew_e"))

  create_forecast_data(
    directory = fs::path(temp_dir, "pyrenew_e"),
    filename = "forecast_samples.parquet",
    date_cols = seq(
      lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
      by = "day"
    ),
    disease_cols = c("Disease", "Other", "prop_disease_ed_visits"),
    n_draw = 20
  )

  result <- to_epiweekly_quantiles(
    model_run_dir = temp_dir,
    report_date = "2024-12-14",
    max_lookback_days = 8,
    epiweekly_other = FALSE
  )

  forecast_path <- fs::path(
    temp_dir,
    "pyrenew_e",
    "forecast_samples",
    ext = "parquet"
  )

  quantiles <- c(0.01, 0.025, seq(0.05, 0.95, 0.05), 0.975, 0.99)
  check_quantiles <- arrow::read_parquet(forecast_path) |>
    dplyr::group_by(.draw, disease) |>
    dplyr::summarise(
      epiweekly_val = sum(.data$.value),
      .groups = "drop"
    ) |>
    tidyr::pivot_wider(
      names_from = disease,
      values_from = epiweekly_val
    ) |>
    dplyr::mutate(
      epiweekly_proportion = Disease / (Disease + Other)
    ) |>
    dplyr::summarise(
      quantile_value = list(quantile(epiweekly_proportion, probs = quantiles))
    ) |>
    tidyr::unnest(cols = c(quantile_value)) |>
    dplyr::mutate(
      quantile_level = quantiles
    )

  expect_equal(
    result$quantile_value,
    check_quantiles$quantile_value,
    tolerance = 1e-6
  )

  expect_equal(
    result$quantile_level,
    check_quantiles$quantile_level,
    tolerance = 1e-6
  )
})


test_that("to_epiweekly_quantiles handles missing forecast files", {
  temp_dir <- withr::local_tempdir("CA")
  fs::dir_create(fs::path(temp_dir, "pyrenew_e"))

  expect_error(
    result <- to_epiweekly_quantiles(
      model_run_dir = temp_dir,
      report_date = "2024-12-14",
      max_lookback_days = 8
    ),
    "Failed to open local file"
  )
})


# tests for `to_epiweekly_quantile_table`
test_that("to_epiweekly_quantile_table handles multiple locations", {
  temp_batch_dir <- withr::local_tempdir("test_model_batch_dir")

  locations <- c("loc1", "loc2")
  purrr::walk(locations, function(loc) {
    loc_dir <- fs::path(temp_batch_dir, "model_runs", loc, "pyrenew_e")
    fs::dir_create(loc_dir)

    create_forecast_data(
      directory = loc_dir,
      filename = "forecast_samples.parquet",
      date_cols = seq(
        lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
        by = "day"
      ),
      disease_cols = c("Disease", "Other", "prop_disease_ed_visits"),
      n_draw = 20
    )
  })

  # Rename using expected format
  batch_dir_name <- "covid-19_r_2024-12-14_f_2024-12-08_t_2024-12-14"
  fs::file_move(temp_batch_dir, fs::path_temp(batch_dir_name))
  renamed_path <- fs::path_temp(batch_dir_name)

  result_w_both_locations <- to_epiweekly_quantile_table(renamed_path)

  expect_s3_class(result_w_both_locations, "tbl_df")
  expect_gt(nrow(result_w_both_locations), 0)
  expect_setequal(c(
    "reference_date", "target", "horizon", "target_end_date",
    "location", "output_type", "output_type_id", "value"
  ), colnames(result_w_both_locations))
  expect_setequal(locations, result_w_both_locations$location)

  result_w_one_location <- to_epiweekly_quantile_table(
    model_batch_dir = renamed_path,
    exclude = "loc1"
  )
  expect_true("loc2" %in% result_w_one_location$location)
  expect_false("loc1" %in% result_w_one_location$location)
})
