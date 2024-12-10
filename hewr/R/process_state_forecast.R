#' Combine training and evaluation data for
#' postprocessing.
#'
#' @param train_dat Training data, as a [`tibble`][tibble::tibble()].
#' @param eval_dat Evaluation data, as a [`tibble`][tibble::tibble()].
#' @param disease_name Name of the target disease.
#' One of `"COVID-19"` or `"Influenza"`.
#' @return The combined data, as a [`tibble`][tibble::tibble()].
#' @export
combine_training_and_eval_data <- function(train_dat,
                                           eval_dat,
                                           disease_name) {
  combined_dat <- dplyr::bind_rows(
    train_dat |>
      dplyr::filter(.data$data_type == "train"),
    eval_dat
  ) |>
    dplyr::mutate(
      disease = dplyr::if_else(
        .data$disease == !!disease_name,
        "Disease", # assign a common name for
        # use in plotting functions
        .data$disease
      )
    ) |>
    dplyr::filter(.data$disease %in% c("Total", "Disease")) |>
    tidyr::pivot_wider(names_from = "disease", values_from = "ed_visits") |>
    dplyr::mutate(
      Other = .data$Total - .data$Disease,
      prop_disease_ed_visits = .data$Disease / .data$Total
    ) |>
    dplyr::select(-"Total") |>
    dplyr::mutate(time = dplyr::dense_rank(.data$date)) |>
    tidyr::pivot_longer(
      c("Disease", "Other", "prop_disease_ed_visits"),
      names_to = "disease",
      values_to = ".value"
    )

  return(combined_dat)
}


#' Process state forecast
#'
#' @param model_run_dir Model run directory
#' @param pyrenew_model_name Name of directory containing pyrenew
#' model outputs
#' @param timeseries_model_name Name of directory containing timeseries
#' model outputs
#' @param save Logical indicating whether or not to save
#'
#' @return a list with four tibbles: `combined_dat`,
#' `forecast_samples`, `epiweekly_forecast_samples`,
#' and `forecast_ci`
#' @export
process_state_forecast <- function(model_run_dir,
                                   pyrenew_model_name,
                                   timeseries_model_name,
                                   save = TRUE) {
  pyrenew_model_dir <- fs::path(model_run_dir, pyrenew_model_name)
  timeseries_model_dir <- fs::path(model_run_dir, timeseries_model_name)
  disease_name <- parse_model_run_dir_path(model_run_dir)$disease

  train_data_path <- fs::path(model_run_dir, "data", "data", ext = "tsv")
  train_dat <- readr::read_tsv(train_data_path, show_col_types = FALSE)

  eval_data_path <- fs::path(model_run_dir, "data", "eval_data", ext = "tsv")
  eval_dat <- readr::read_tsv(eval_data_path, show_col_types = FALSE) |>
    dplyr::mutate(data_type = "eval")

  posterior_predictive_path <- fs::path(pyrenew_model_dir, "mcmc_tidy",
    "pyrenew_posterior_predictive",
    ext = "parquet"
  )
  posterior_predictive <- arrow::read_parquet(posterior_predictive_path)

  other_ed_visits_path <- fs::path(timeseries_model_dir,
    "other_ed_visits_forecast",
    ext = "parquet"
  )
  other_ed_visits_forecast <- arrow::read_parquet(other_ed_visits_path) |>
    dplyr::rename(Other = "other_ed_visits")

  combined_dat <- combine_training_and_eval_data(
    train_dat,
    eval_dat,
    disease_name
  )

  last_training_date <- combined_dat |>
    dplyr::filter(.data$data_type == "train") |>
    dplyr::pull("date") |>
    max()

  other_ed_visits_samples <-
    dplyr::bind_rows(
      combined_dat |>
        dplyr::filter(
          .data$data_type == "train",
          .data$disease == "Other",
          .data$date <= !!last_training_date
        ) |>
        dplyr::select("date", Other = ".value") |>
        tidyr::expand_grid(
          .draw = 1:max(other_ed_visits_forecast$.draw)
        ),
      other_ed_visits_forecast
    )


  forecast_samples <-
    posterior_predictive |>
    tidybayes::gather_draws(observed_ed_visits[time]) |>
    tidyr::pivot_wider(
      names_from = ".variable",
      values_from = ".value"
    ) |>
    dplyr::rename(Disease = "observed_ed_visits") |>
    dplyr::ungroup() |>
    dplyr::mutate(date = min(combined_dat$date) + .data$time) |>
    dplyr::left_join(other_ed_visits_samples,
      by = c(".draw", "date")
    ) |>
    dplyr::mutate(prop_disease_ed_visits = .data$Disease /
      (.data$Disease + .data$Other)) |>
    tidyr::pivot_longer(
      c(
        "Other",
        "Disease",
        "prop_disease_ed_visits"
      ),
      names_to = "disease",
      values_to = ".value"
    )

  epiweekly_forecast_samples <- forecast_samples |>
    dplyr::filter(.data$disease != "prop_disease_ed_visits") |>
    dplyr::group_by(.data$disease) |>
    dplyr::group_modify(~ forecasttools::daily_to_epiweekly(.x,
      value_col = ".value", weekly_value_name = ".value",
      strict = TRUE
    )) |>
    dplyr::ungroup() |>
    tidyr::pivot_wider(
      names_from = "disease",
      values_from = ".value"
    ) |>
    dplyr::mutate(prop_disease_ed_visits = .data$Disease /
      (.data$Disease + .data$Other)) |>
    tidyr::pivot_longer(
      c(
        "Disease",
        "Other",
        "prop_disease_ed_visits"
      ),
      names_to = "disease",
      values_to = ".value"
    ) |>
    dplyr::mutate(date = forecasttools::epiweek_to_date(
      .data$epiweek,
      .data$epiyear,
      day_of_week = 7
    ))

  forecast_ci <-
    forecast_samples |>
    dplyr::select("date", "disease", ".value") |>
    dplyr::group_by(.data$date, .data$disease) |>
    ggdist::median_qi(.width = c(0.5, 0.8, 0.95))

  # Save data
  if (save) {
    arrow::write_parquet(
      combined_dat,
      fs::path(pyrenew_model_dir,
        "combined_training_eval_data",
        ext = "parquet"
      )
    )

    arrow::write_parquet(
      forecast_samples,
      fs::path(pyrenew_model_dir, "forecast_samples",
        ext = "parquet"
      )
    )
    arrow::write_parquet(
      epiweekly_forecast_samples,
      fs::path(pyrenew_model_dir, "epiweekly_forecast_samples",
        ext = "parquet"
      )
    )

    arrow::write_parquet(
      forecast_ci,
      fs::path(pyrenew_model_dir, "forecast_ci",
        ext = "parquet"
      )
    )
  }
  return(list(
    combined_dat = combined_dat,
    forecast_samples = forecast_samples,
    epiweekly_forecast_samples = epiweekly_forecast_samples,
    forecast_ci = forecast_ci
  ))
}
