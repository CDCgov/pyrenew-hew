test_that("example_prediction generates data with correct dimensions", {
  ndays <- 21
  reps <- 100
  nchains <- 4
  nareas <- 3
  nlookaheads <- 2
  result <- example_prediction(
    ndays = ndays, reps = reps, nchains = nchains,
    nareas = nareas
  )
  expect_equal(nrow(result), ndays * reps * nchains * nareas * nlookaheads)
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
  ndays <- 24
  start_date <- ymd("2024-10-24")
  end_date <- start_date + days(ndays - 1)
  result <- example_prediction(ndays = ndays)
  expect_true(
    all(result$reference_date >= start_date & result$reference_date <= end_date)
  )
})

test_that("example_prediction generates data with correct areas", {
  nareas <- 2
  result <- example_prediction(nareas = nareas)
  expect_true(all(result$area %in% LETTERS[1:nareas]))
})

test_that("example_prediction generates data with correct number of chains", {
  nchains <- 2
  result <- example_prediction(nchains = nchains)
  expect_equal(length(unique(result$.chain)), nchains)
})

test_that("example_prediction saves data when save_data is TRUE", {
  temp_dir <- tempdir()
  result <- example_prediction(save_path = temp_dir, save_data = TRUE)
  expect_true(file.exists(file.path(temp_dir, "part-0.parquet")))
})
