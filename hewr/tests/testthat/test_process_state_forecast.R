example_train_dat <-
  tibble::tibble(
    disease = c(
      "COVID-19", "Influenza",
      "Total", "Not A Value"
    ),
    data_type = c(
      "train", "train",
      "train", "Not a type"
    ),
    date = as.Date(c(
      "2024-01-01",
      "2024-01-01",
      "2024-01-01",
      "2025-05-05"
    )),
    ed_visits = c(5, 6, 50, NA)
  )
example_eval_dat <-
  tibble::tibble(
    disease = c(
      "Total",
      "Influenza",
      "COVID-19"
    ),
    data_type = c(
      "eval",
      "eval",
      "eval"
    ),
    date = as.Date(c(
      "2024-01-05",
      "2024-01-05",
      "2024-01-05"
    )),
    ed_visits = c(10, 8, 60)
  )


test_that("combine_training_and_eval_data works as expected", {
  check_disease <- function(disease_name) {
    result <- combine_training_and_eval_data(
      example_train_dat,
      example_eval_dat,
      disease_name
    )

    checkmate::assert_names(names(result),
      permutation.of = c(
        "time",
        "date",
        "data_type",
        "disease",
        ".value"
      )
    )
    checkmate::expect_names(
      result$disease,
      permutation.of = c(
        "Disease",
        "Other",
        "prop_disease_ed_visits"
      )
    )
    checkmate::expect_names(
      result$data_type,
      permutation.of = c("eval", "train")
    )

    expect_equal(nrow(result), 6) # 2 dates x 3 diseases
  }

  check_disease("COVID-19")
  check_disease("Influenza")
})


test_that("to_tidy_draws_timeseries() works as expected", {
  forecast <- tibble::tibble(
    date = c(
      "2024-02-02",
      "2024-02-03",
      "2024-02-02",
      "2024-02-03"
    ),
    Test = c(5, 6, 10, 11),
    .draw = c(1, 1, 2, 2)
  )
  obs <- tibble::tibble(
    date = c(
      "2024-02-01",
      "2024-02-02"
    ),
    .value = c(4, 8),
    disease = c("Test", "Test")
  )
  result <- to_tidy_draws_timeseries(
    forecast,
    obs,
    "Test"
  )

  expected <- tibble::tibble(
    date = c(
      "2024-02-01",
      "2024-02-01",
      "2024-02-02",
      "2024-02-03",
      "2024-02-02",
      "2024-02-03"
    ),
    Test = c(4, 4, 5, 6, 10, 11),
    .draw = c(1, 2, 1, 1, 2, 2)
  )

  expect_equal(result, expected)
})
