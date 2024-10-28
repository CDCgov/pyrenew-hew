test_that("example_truthdata generates correct number of rows and columns", {
  n_days <- 10
  n_areas <- 2
  result <- example_truthdata(n_days = n_days, n_areas = n_areas)
  expect_equal(nrow(result), n_days * n_areas)
  expect_equal(ncol(result), 3)
})

test_that("example_truthdata generates correct column names", {
  result <- example_truthdata()
  expect_true(all(c("area", "date", "truthdata") %in% colnames(result)))
})

test_that("example_truthdata generates correct date range", {
  n_days <- 17
  start_date <- ymd("2024-10-24")
  result <- example_truthdata(n_days = n_days)
  expected_dates <- seq.Date(
    from = start_date, by = "day",
    length.out = n_days
  )
  expect_equal(unique(result$date), expected_dates)
})

test_that("example_truthdata generates correct number of unique areas", {
  n_areas <- 5
  result <- example_truthdata(n_areas = n_areas)
  expect_equal(length(unique(result$area)), n_areas)
})

test_that("example_truthdata saves data when save_data is TRUE", {
  save_path <- tempdir()
  filename <- file.path(save_path, "example_truthdata.tsv")
  result <- example_truthdata(save_path = save_path, save_data = TRUE)
  expect_true(file.exists(filename))
  saved_data <- readr::read_tsv(filename)
  expect_equal(nrow(saved_data), nrow(result))
  expect_equal(ncol(saved_data), ncol(result))
  expect_equal(colnames(saved_data), colnames(result))
  unlink(save_path) # Clean up the temporary file
})
