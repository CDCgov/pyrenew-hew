test_that("example_prediction generates data with correct dimensions", {
  n_days <- 21
  reps <- 100
  n_chains <- 4
  n_areas <- 3
  nlookaheads <- 2
  result <- example_prediction(
    n_days = n_days, reps = reps, n_chains = n_chains,
    n_areas = n_areas
  )
  expect_equal(nrow(result), n_days * reps * n_chains * n_areas * nlookaheads)
  expect_equal(ncol(result), 8)
})

test_that("example_prediction generates data with correct columns", {
  result <- example_prediction()
  expect_true(all(c(
    "area", "reference_date", "target_end_date", "prediction",
    ".chain", ".iteration", ".draw"
  ) %in% colnames(result)))
})

test_that("example_prediction generates data with correct date range", {
  n_days <- 24
  start_date <- ymd("2024-10-24")
  end_date <- start_date + days(n_days - 1)
  result <- example_prediction(n_days = n_days)
  expect_true(
    all(result$reference_date >= start_date & result$reference_date <= end_date)
  )
})

test_that("example_prediction generates data with correct areas", {
  n_areas <- 2
  result <- example_prediction(n_areas = n_areas)
  expect_true(setequal(result$area, LETTERS[1:n_areas]))
})

test_that("example_prediction generates data with correct number of chains", {
  n_chains <- 2
  result <- example_prediction(n_chains = n_chains)
  expect_equal(length(unique(result$.chain)), n_chains)
})

test_that("example_prediction saves data when save_data is TRUE", {
  temp_dir <- tempdir()
  result <- example_prediction(save_path = temp_dir, save_data = TRUE)
  expect_true(file.exists(file.path(temp_dir, "part-0.parquet")))
})
