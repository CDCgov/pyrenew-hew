test_that("get_all_model_batch_dirs returns expected output", {
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
