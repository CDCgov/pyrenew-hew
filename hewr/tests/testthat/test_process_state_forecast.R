example_train_dat <- tibble::tibble(
  geo_value = "CA",
  disease = "COVID-19",
  data_type = "train",
  .variable = c(
    "observed_ed_visits", "other_ed_visits",
    "observed_hospital_admissions", "site_level_log_ww_conc"
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
    "observed_ed_visits", "other_ed_visits",
    "observed_hospital_admissions", "site_level_log_ww_conc"
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
      "2024-12-17", "2024-12-18", "2024-12-19", "2024-12-20",
      "2024-12-21", "2024-12-22"
    )),
    geo_value = rep("CA", 6L),
    disease = rep("COVID-19", 6L),
    .variable = rep("other_ed_visits", 6L),
    .value = c(
      11037, 12898, 15172, 17716, 20641.1242073179819,
      25812.84128089781
    ),
  )

  expect_equal(result, expected)
})
