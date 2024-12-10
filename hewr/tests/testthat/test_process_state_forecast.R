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
