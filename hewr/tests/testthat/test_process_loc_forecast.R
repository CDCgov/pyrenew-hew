# Common test fixtures
fake_dir <- "/fake/dir"
minimal_required_columns <- c("date", ".value")
full_required_columns <- c(
  ".chain",
  ".iteration",
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

example_train_dat <- tibble::tibble(
  geo_value = "CA",
  disease = "COVID-19",
  data_type = "train",
  .variable = c(
    "observed_ed_visits",
    "other_ed_visits",
    "observed_hospital_admissions",
    "site_level_log_ww_conc"
  ),
  lab_site_index = c(NA, NA, NA, 1)
) |>
  tidyr::expand_grid(
    date = seq.Date(as.Date("2024-10-22"), as.Date("2024-10-24"), by = "day")
  ) |>
  dplyr::mutate(.value = rpois(dplyr::n(), 100))

example_eval_dat <- tibble::tibble(
  geo_value = "CA",
  disease = "COVID-19",
  data_type = "eval",
  .variable = c(
    "observed_ed_visits",
    "other_ed_visits",
    "observed_hospital_admissions",
    "site_level_log_ww_conc"
  ),
  lab_site_index = c(NA, NA, NA, 1)
) |>
  tidyr::expand_grid(
    date = seq.Date(as.Date("2024-10-22"), as.Date("2024-10-24"), by = "day")
  ) |>
  dplyr::mutate(.value = rpois(dplyr::n(), 100))

test_that("to_tidy_draws_timeseries() works as expected", {
  forecast <- tibble::tibble(
    date = as.Date(c("2024-12-21", "2024-12-22")),
    resolution = "daily",
    .draw = c(1L, 1L),
    geo_value = c("CA", "CA"),
    disease = c("COVID-19", "COVID-19"),
    .variable = c("other_ed_visits", "other_ed_visits"),
    .value = c(20641.1242073179819, 25812.84128089781),
  )

  obs <- tibble::tibble(
    date = as.Date(c("2024-12-17", "2024-12-18", "2024-12-19", "2024-12-20")),
    geo_value = rep("CA", 4L),
    disease = rep("COVID-19", 4L),
    .variable = rep("other_ed_visits", 4L),
    .value = c(11037, 12898, 15172, 17716),
  )
  result <- to_tidy_draws_timeseries(
    forecast,
    obs
  )

  expected <- tibble::tibble(
    .draw = rep(1L, 6L),
    date = as.Date(c(
      "2024-12-17",
      "2024-12-18",
      "2024-12-19",
      "2024-12-20",
      "2024-12-21",
      "2024-12-22"
    )),
    geo_value = rep("CA", 6L),
    disease = rep("COVID-19", 6L),
    .variable = rep("other_ed_visits", 6L),
    .value = c(
      11037,
      12898,
      15172,
      17716,
      20641.1242073179819,
      25812.84128089781
    ),
    resolution = "daily"
  )

  expect_equal(result, expected)
})

test_that("detect_model_type() correctly identifies model types", {
  # Test timeseries detection with ts_ prefix
  expect_equal(detect_model_type("ts_model_v1"), "timeseries")
  expect_equal(detect_model_type("ts_ensemble"), "timeseries")

  # Test timeseries detection with name containing "timeseries"
  expect_equal(detect_model_type("timeseries_model"), "timeseries")
  expect_equal(detect_model_type("my_TimeSeries_model"), "timeseries")

  # Test epiautogp detection
  expect_equal(detect_model_type("epiautogp_model"), "epiautogp")
  expect_equal(detect_model_type("EpiAutoGP_v2"), "epiautogp")

  # Test default to pyrenew
  expect_equal(detect_model_type("pyrenew_hew"), "pyrenew")
  expect_equal(detect_model_type("pyrenew_e"), "pyrenew")
  expect_equal(detect_model_type("some_other_model"), "pyrenew")
})

test_that("process_model_samples S3 dispatch works correctly", {
  # Test that the generic function exists and has the right class
  expect_true(is.function(process_model_samples))

  # Test that methods exist for expected classes
  expect_true(
    "process_model_samples.pyrenew" %in%
      methods("process_model_samples")
  )
  expect_true(
    "process_model_samples.timeseries" %in%
      methods("process_model_samples")
  )
})

test_that("process_model_samples.timeseries validates ts_samples", {
  # Should error when ts_samples is NULL
  expect_error(
    process_model_samples.timeseries(
      model_type = "timeseries",
      model_run_dir = fake_dir,
      model_name = "ts_model",
      ts_samples = NULL,
      required_columns_e = minimal_required_columns,
      n_forecast_days = 7
    ),
    "ts_samples must be provided for timeseries model type"
  )
})

test_that("process_model_samples.timeseries returns ts_samples", {
  # Create mock ts_samples
  mock_ts_samples <- tibble::tibble(
    .chain = 1,
    .iteration = 1,
    .draw = 1,
    date = as.Date("2024-01-01"),
    geo_value = "US",
    disease = "COVID-19",
    .variable = "other_ed_visits",
    .value = 100,
    resolution = "daily",
    aggregated_numerator = FALSE,
    aggregated_denominator = NA
  )

  result <- process_model_samples.timeseries(
    model_type = "timeseries",
    model_run_dir = fake_dir,
    model_name = "ts_model",
    ts_samples = mock_ts_samples,
    required_columns_e = minimal_required_columns,
    n_forecast_days = 7
  )

  # Should return the ts_samples unchanged
  expect_equal(result, mock_ts_samples)
})

test_that("process_model_samples.pyrenew dispatches correctly", {
  # This test just verifies the S3 method exists and dispatches
  # We expect it to error since we're not providing real data/files
  # The key is that it calls the method (for code coverage)

  expect_error(
    process_model_samples.pyrenew(
      model_type = "pyrenew",
      model_run_dir = "any_path",
      model_name = "pyrenew_h",
      ts_samples = NULL,
      required_columns_e = minimal_required_columns,
      n_forecast_days = 7
    )
  )

  # Verify the method exists
  expect_true(
    "process_model_samples.pyrenew" %in% methods("process_model_samples")
  )
})

test_that("process_loc_forecast delegates correctly", {
  # Test that process_loc_forecast calls process_forecast when
  # model_name is provided by checking that it doesn't use the
  # legacy code path

  # Create a simple test: when model_name is provided, function
  # should attempt to call process_forecast, which will try to
  # read files. When model_name is NA, it uses the legacy path
  # with different error message

  # Test with model_name provided (new interface)
  expect_error(
    process_loc_forecast(
      model_run_dir = fake_dir,
      n_forecast_days = 7,
      model_name = "test_model",
      save = FALSE
    ),
    # This error comes from process_forecast trying to read
    # training data
    "does not exist"
  )

  # Test with legacy interface - should give different error
  expect_error(
    process_loc_forecast(
      model_run_dir = fake_dir,
      n_forecast_days = 7,
      model_name = NA,
      pyrenew_model_name = NA,
      timeseries_model_name = NA
    ),
    "At least one of"
  )
})

test_that("process_loc_forecast validates legacy interface parameters", {
  # Should error when neither model_name nor pyrenew/timeseries
  # names provided
  expect_error(
    process_loc_forecast(
      model_run_dir = fake_dir,
      n_forecast_days = 7,
      model_name = NA,
      pyrenew_model_name = NA,
      timeseries_model_name = NA
    ),
    "At least one of"
  )
})
