valid_model_batch_dirs <- list(
  list(
    dirname = "covid-19_r_2024-02-03_f_2021-04-01_t_2024-01-23",
    expected = tibble::tibble(
      disease = "COVID-19",
      report_date = lubridate::ymd("2024-02-03"),
      first_training_date = lubridate::ymd("2021-04-1"),
      last_training_date = lubridate::ymd("2024-01-23")
    )
  ),
  list(
    dirname = "influenza_r_2022-12-11_f_2021-02-05_t_2027-12-30",
    expected = tibble::tibble(
      disease = "Influenza",
      report_date = lubridate::ymd("2022-12-11"),
      first_training_date = lubridate::ymd("2021-02-5"),
      last_training_date = lubridate::ymd("2027-12-30")
    )
  )
)

invalid_model_batch_dirs <- c(
  "qcovid-19_r_2024-02-03_f_2021-04-01_t_2024-01-23",
  "influenza_r_2022-12-33_f_2021-02-05_t_2027-12-30"
)

to_valid_run_dir <- function(valid_batch_dir_entry, location) {
  x <- valid_batch_dir_entry
  x$dirname <- fs::path(x$dirname, "model_runs", location)
  x$expected$location <- location
  return(x)
}

valid_model_run_dirs <- c(
  lapply(
    valid_model_batch_dirs, to_valid_run_dir,
    location = "ME"
  ),
  lapply(
    valid_model_batch_dirs, to_valid_run_dir,
    location = "US"
  )
)


test_that("parse_model_batch_dir_path() works as expected.", {
  for (valid_pair in valid_model_batch_dirs) {
    ## should work with base dirnames that are valid
    expect_equal(
      parse_model_batch_dir_path(valid_pair$dirname),
      valid_pair$expected
    )

    ## should work identically with a full path rather
    ## than just base dir
    also_valid <- fs::path("this", "is", "a", "test", valid_pair$dirname)
    expect_equal(
      parse_model_batch_dir_path(also_valid),
      valid_pair$expected
    )

    ## should error if the terminal directory is not
    ## what is to be parsed
    not_valid <- fs::path(valid_pair$dirname, "test")
    expect_error(
      {
        parse_model_batch_dir_path(not_valid)
      },
      regex = "Invalid format for model batch directory name"
    )
  }

  ## should error if entries cannot be parsed as what is expected

  for (invalid_entry in invalid_model_batch_dirs) {
    expect_error(
      {
        parse_model_batch_dir_path(invalid_entry)
      },
      regex = "Could not parse extracted disease and/or date values"
    )
  }
})

test_that("parse_model_run_dir_path() works as expected.", {
  for (valid_pair in valid_model_run_dirs) {
    expect_equal(
      parse_model_run_dir_path(valid_pair$dirname),
      valid_pair$expected
    )

    ## should work identically with a longer path
    expect_equal(
      parse_model_run_dir_path(fs::path(
        "this", "is", "a", "test",
        valid_pair$dirname
      )),
      valid_pair$expected
    )

    ## should fail if there is additional terminal pathing
    expect_error(
      {
        parse_model_run_dir_path(fs::path(valid_pair$dirname, "test"))
      },
      regex = "Invalid format for model batch directory name"
    )
  }
})

test_that("get_all_model_batch_dirs() returns expected output.", {
  withr::with_tempdir({
    ## create some directories
    valid_covid <- c(
      "covid-19_r_2024-02-01_f_2021-01-01_t_2024-01-31",
      "covid-19_r"
    )
    valid_flu <- c(
      "influenza_r_2022-11-12_f_2022-11-01_t_2022_11_10",
      "influenza_r"
    )
    valid_dirs <- c(valid_flu, valid_covid)

    invalid_dirs <- c(
      "this_is_not_valid",
      "covid19_r",
      "covid-19-r",
      "influenza-r",
      "influnza_r",
      "covid-19",
      "influenza"
    )

    invalid_files <- c(
      "covid-19_r.txt",
      "influenza_r.txt"
    )
    fs::dir_create(c(valid_dirs, invalid_dirs))
    fs::file_create(invalid_files)
    expected_all_files <- c(
      valid_dirs,
      invalid_dirs,
      invalid_files
    )

    result_all <- fs::dir_ls(".") |> fs::path_file()

    result_valid <- get_all_model_batch_dirs(
      ".",
      c("COVID-19", "Influenza")
    )

    result_valid_alt <- get_all_model_batch_dirs(
      ".",
      c("Influenza", "COVID-19")
    )

    result_valid_covid <- get_all_model_batch_dirs(
      ".",
      "COVID-19"
    )

    result_valid_flu <- get_all_model_batch_dirs(
      ".",
      "Influenza"
    )

    expect_setequal(result_all, expected_all_files)
    expect_setequal(result_valid, c(valid_flu, valid_covid))
    expect_setequal(result_valid_alt, c(valid_flu, valid_covid))
    expect_setequal(result_valid_covid, valid_covid)
    expect_setequal(result_valid_flu, valid_flu)
  })
})
