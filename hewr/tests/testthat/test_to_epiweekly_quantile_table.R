# tests for `to_epiweekly_quantile`
test_that("to_epiweekly_quantiles works as expected", {
  # create temporary directories and forecast files for tests
  temp_dir <- withr::local_tempdir("CA")
  fs::dir_create(fs::path(temp_dir, "pyrenew_e"))
  fs::dir_create(fs::path(temp_dir, "timeseries_e"))

  create_tidy_forecast_data(
    directory = fs::path(temp_dir, "pyrenew_e"),
    filename = "epiweekly_samples.parquet",
    date_cols = seq(
      lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
      by = "week"
    ),
    disease_cols = c("Disease", "Other", "prop_disease_ed_visits"),
    n_draw = 20,
    with_epiweek = TRUE
  )

  create_tidy_forecast_data(
    directory = fs::path(temp_dir, "pyrenew_e"),
    filename = "epiweekly_with_epiweekly_other_samples.parquet",
    date_cols = seq(
      lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
      by = "week"
    ),
    disease_cols = c("Disease", "Other", "prop_disease_ed_visits"),
    n_draw = 25,
    with_epiweek = TRUE
  )


  check_epiweekly_quantiles <- function(draws_file_name) {
    result <- to_epiweekly_quantiles(
      model_run_dir = temp_dir,
      report_date = "2024-12-14",
      max_lookback_days = 8,
      draws_file_name = draws_file_name,
    ) |> suppressMessages()

    expect_s3_class(result, "tbl_df")
    expect_setequal(c(
      "epiweek", "epiyear", "quantile_value", "quantile_level", "location"
    ), colnames(result))
    expect_gt(nrow(result), 0)
  }

  check_epiweekly_quantiles(draws_file_name = "epiweekly_samples")
  check_epiweekly_quantiles(
    draws_file_name = "epiweekly_with_epiweekly_other_samples"
  )
})


test_that("to_epiweekly_quantiles calculates quantiles accurately", {
  temp_dir <- withr::local_tempdir("test")
  fs::dir_create(fs::path(temp_dir, "pyrenew_e"))

  create_tidy_forecast_data(
    directory = fs::path(temp_dir, "pyrenew_e"),
    filename = "epiweekly_samples.parquet",
    date_cols = seq(
      lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
      by = "week"
    ),
    disease_cols = c("Disease", "Other", "prop_disease_ed_visits"),
    n_draw = 20,
    with_epiweek = TRUE
  )

  result <- to_epiweekly_quantiles(
    model_run_dir = temp_dir,
    report_date = "2024-12-14",
    max_lookback_days = 8,
  ) |> suppressMessages()

  forecast_path <- fs::path(
    temp_dir,
    "pyrenew_e",
    "epiweekly_samples",
    ext = "parquet"
  )

  quantiles <- c(0.01, 0.025, seq(0.05, 0.95, 0.05), 0.975, 0.99)
  check_quantiles <- arrow::read_parquet(forecast_path) |>
    tidyr::pivot_wider(
      names_from = "disease",
      values_from = ".value"
    ) |>
    dplyr::summarise(
      quantile_value = list(quantile(prop_disease_ed_visits,
        probs = quantiles
      ))
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
    ) |> suppressMessages(),
    "Failed to open local file"
  )
})


# tests for `to_epiweekly_quantile_table`
test_that("to_epiweekly_quantile_table handles multiple locations", {
  batch_dir_name <- "covid-19_r_2024-12-14_f_2024-12-08_t_2024-12-14"
  tempdir <- withr::local_tempdir()

  temp_batch_dir <- fs::dir_create(fs::path(tempdir, batch_dir_name))

  locations <- c("loc1", "loc2", "loc3")
  purrr::walk(locations, function(loc) {
    loc_dir <- fs::path(temp_batch_dir, "model_runs", loc, "pyrenew_e")
    fs::dir_create(loc_dir)

    disease_cols <- c("Other", "Disease")
    if (loc != "loc3") {
      disease_cols <- c(disease_cols, "prop_disease_ed_visits")
    }

    create_tidy_forecast_data(
      directory = loc_dir,
      filename = "epiweekly_samples.parquet",
      date_cols = seq(
        lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
        by = "week"
      ),
      disease_cols = disease_cols,
      n_draw = 20,
      with_epiweek = TRUE
    )
  })

  ## should succeed despite loc3 not having valid draws with strict = FALSE
  result_w_both_locations <- to_epiweekly_quantile_table(temp_batch_dir) |>
    suppressMessages()

  ## should error if strict = TRUE because loc3 does not have
  ## valid draws.
  expect_error(
    to_epiweekly_quantile_table(temp_batch_dir, strict = TRUE) |>
      suppressMessages(),
    "did not find valid draws"
  )

  expect_s3_class(result_w_both_locations, "tbl_df")
  expect_gt(nrow(result_w_both_locations), 0)
  expect_setequal(c(
    "reference_date", "target", "horizon", "target_end_date",
    "location", "output_type", "output_type_id", "value"
  ), colnames(result_w_both_locations))
  expect_setequal(
    c("loc1", "loc2"),
    result_w_both_locations$location
  )
  expect_false("loc3" %in% result_w_both_locations$location)

  result_w_one_location <- to_epiweekly_quantile_table(
    model_batch_dir = temp_batch_dir,
    exclude = "loc1"
  ) |>
    suppressMessages()
  expect_true("loc2" %in% result_w_one_location$location)
  expect_false("loc1" %in% result_w_one_location$location)
})
