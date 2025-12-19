test_that("fetch_reporting_data works with epidatr_source", {
  skip_on_cran()

  # Recent Saturdays
  report_dates <- seq(as.Date("2024-11-30"), as.Date("2024-12-14"), by = "week")
  reference_dates <- seq(as.Date("2024-11-23"), as.Date("2024-12-07"), by = "week")

  # Create epidatr source
  src <- epidata_source(
    signal = "confirmed_admissions_covid_ew_prelim",
    geo_types = "state"
  )

  # Fetch data
  result <- fetch_reporting_data(
    source = src,
    reference_dates = reference_dates,
    report_dates = report_dates,
    locations = "ca"
  )

  # Check structure
  expect_s3_class(result, "data.frame")
  expect_true(nrow(result) > 0)
  expect_true(all(c("reference_date", "report_date", "location", "count") %in% names(result)))

  # Check date ranges
  expect_true(all(result$reference_date >= min(reference_dates)))
  expect_true(all(result$reference_date <= max(reference_dates)))
  expect_true(all(result$report_date %in% report_dates))
})

test_that("fetch_reporting_data works with github_source", {
  skip_on_cran()
  skip_if_offline()

  # Recent Saturdays
  report_dates <- seq(as.Date("2024-11-30"), as.Date("2024-12-14"), by = "week")
  reference_dates <- seq(as.Date("2024-11-23"), as.Date("2024-12-07"), by = "week")

  # Create github source
  src <- github_source(
    signal = "covid"
  )

  # Fetch data
  result <- fetch_reporting_data(
    source = src,
    reference_dates = reference_dates,
    report_dates = report_dates,
    locations = "CA"
  )

  # Check structure
  expect_s3_class(result, "data.frame")
  expect_true(nrow(result) > 0)
  expect_true(all(c("reference_date", "report_date", "location", "count") %in% names(result)))

  # Check date ranges
  expect_true(all(result$reference_date >= min(reference_dates)))
  expect_true(all(result$reference_date <= max(reference_dates)))
  expect_true(all(result$report_date %in% report_dates))

  # Check location
  expect_true(all(tolower(result$location) == "ca"))
})

test_that("both methods return compatible data structures", {
  skip_on_cran()
  skip_if_offline()

  # Use same dates for both
  report_dates <- seq(as.Date("2024-12-07"), as.Date("2024-12-14"), by = "week")
  reference_dates <- seq(as.Date("2024-11-30"), as.Date("2024-12-07"), by = "week")

  # GitHub source
  src_gh <- github_source(signal = "covid")
  data_gh <- fetch_reporting_data(
    source = src_gh,
    reference_dates = reference_dates,
    report_dates = report_dates,
    locations = "CA"
  )

  # Epidata source
  src_epi <- epidata_source(
    signal = "confirmed_admissions_covid_ew_prelim",
    geo_types = "state"
  )
  data_epi <- fetch_reporting_data(
    source = src_epi,
    reference_dates = reference_dates,
    report_dates = report_dates,
    locations = "ca"
  )

  # Both should have same column structure (modulo column order)
  expect_true(all(c("reference_date", "report_date", "location", "count") %in% names(data_gh)))
  expect_true(all(c("reference_date", "report_date", "location", "count") %in% names(data_epi)))

  # Both should have data
  expect_true(nrow(data_gh) > 0)
  expect_true(nrow(data_epi) > 0)
})
