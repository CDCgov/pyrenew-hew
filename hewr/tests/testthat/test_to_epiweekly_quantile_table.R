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

    date_options <- as.Date(c("2024-12-13", "2024-12-14", "2024-12-15"))
    disease_options <- "COVID-19"
    variable_options <- c("observed_ed_visits", "other_ed_visits")
    n_draw <- 4

    if (loc != "loc3") {
      variable_options <- c(variable_options, "prop_disease_ed_visits")
    }

    create_model_results(
      file = fs::path(loc_dir,
        "epiweekly_samples",
        ext = "parquet"
      ),
      variable_options = variable_options,
      disease_options = disease_options,
      geo_value_options = loc,
      date_options = date_options,
      n_draw = n_draw
    )

    create_model_results(
      file = fs::path(loc_dir,
        "epiweekly_with_epiweekly_other_samples",
        ext = "parquet"
      ),
      variable_options = variable_options,
      disease_options = disease_options,
      geo_value_options = loc,
      date_options = date_options,
      n_draw = n_draw
    )
  })

  ## should succeed despite loc3 not having valid draws with strict = FALSE
  result_w_both_locations <-
    to_epiweekly_quantile_table(temp_batch_dir) |>
    suppressMessages()




  ## check that one used epiweekly
  ## other for loc1 while other used
  ## default, resulting in different values
  loc1_a <- result_w_both_locations |>
    dplyr::filter(location == "loc1") |>
    dplyr::pull(.data$value)


  ## length checks ensure that the
  ## number of allowed equalities _could_
  ## be reached if the vectors were mostly
  ## or entirely identical
  expect_gt(length(loc1_a), 10)


  expect_s3_class(result_w_both_locations, "tbl_df")
  expect_gt(nrow(result_w_both_locations), 0)
  checkmate::expect_names(
    colnames(result_w_both_locations),
    identical.to = c(
      "model",
      "forecast_type",
      "resolution",
      "reference_date",
      "target",
      "horizon",
      "target_end_date",
      "location",
      "output_type",
      "output_type_id",
      "value"
    )
  )
  expect_setequal(
    result_w_both_locations$location,
    c("loc1", "loc2")
  )

  expect_false("loc3" %in% result_w_both_locations$location)
})
