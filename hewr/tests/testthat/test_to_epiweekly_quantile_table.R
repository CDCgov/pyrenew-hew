# tests for `to_epiweekly_quantile_table`
test_that(paste0(
  "to_epiweekly_quantile_table ",
  "handles multiple locations ",
  "and multiple source files"
), {
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
      filename = "epiweekly_with_epiweekly_other_samples.parquet",
      date_cols = seq(
        lubridate::ymd("2024-12-08"), lubridate::ymd("2024-12-14"),
        by = "week"
      ),
      disease_cols = disease_cols,
      n_draw = 25,
      with_epiweek = TRUE
    )

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
  result_w_both_locations <-
    to_epiweekly_quantile_table(temp_batch_dir,
      epiweekly_other_locations = "loc1"
    ) |>
    suppressMessages()

  ## should error if strict = TRUE because loc3 does not have
  ## valid draws.
  expect_error(
    to_epiweekly_quantile_table(temp_batch_dir, strict = TRUE) |>
      suppressMessages(),
    "did not find valid draws"
  )

  ## should succeed with strict = TRUE if loc3 is excluded
  alt_result_w_both_locations <- (
    to_epiweekly_quantile_table(temp_batch_dir,
      strict = TRUE,
      exclude = "loc3"
  )) |>
    suppressMessages()

  ## results should be equivalent for loc2,
  ## but not for loc1
  expect_equal(
    result_w_both_locations |>
      dplyr::filter(location == "loc2"),
    alt_result_w_both_locations |>
      dplyr::filter(location == "loc2")
  )

  ## check that one used epiweekly
  ## other for loc1 while other used
  ## default, resulting in different values
  loc1_a <- result_w_both_locations |>
    dplyr::filter(location == "loc1") |>
    dplyr::pull(.data$value)
  loc1_b <- alt_result_w_both_locations |>
    dplyr::filter(location == "loc1") |>
    dplyr::pull(.data$value)

  ## length checks ensure that the
  ## number of allowed equalities _could_
  ## be reached if the vectors were mostly
  ## or entirely identical
  expect_gt(length(loc1_a), 10)
  expect_gt(length(loc1_b), 10)
  expect_lt(
    sum(loc1_a == loc1_b),
    5
  )

  expect_s3_class(result_w_both_locations, "tbl_df")
  expect_gt(nrow(result_w_both_locations), 0)
  checkmate::expect_names(
    colnames(result_w_both_locations),
    identical.to = c(
      "reference_date",
      "target",
      "horizon",
      "target_end_date",
      "location",
      "output_type",
      "output_type_id",
      "value",
      "source_samples"
    )
  )
  expect_setequal(
    result_w_both_locations$location,
    c("loc1", "loc2")
  )
  expect_setequal(
    alt_result_w_both_locations$location,
    c("loc1", "loc2")
  )

  expect_setequal(
    result_w_both_locations$source_samples,
    c(
      "epiweekly_samples",
      "epiweekly_with_epiweekly_other_samples"
    )
  )

  expect_setequal(
    alt_result_w_both_locations$source_samples,
    "epiweekly_samples"
  )


  expect_false("loc3" %in% result_w_both_locations$location)
  expect_false("loc3" %in% alt_result_w_both_locations$location)
})
