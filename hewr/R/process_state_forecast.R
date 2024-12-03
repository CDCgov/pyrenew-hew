#' Process state forecast
#'
#' @param model_run_dir Model run directory
#' @param save Logical indicating whether or not to save
#'
#' @return a list with three tibbles: combined_dat, forecast_samples,
#' and forecast_ci
#' @export
process_state_forecast <- function(model_run_dir, save = TRUE) {
  disease_name_nssp <- parse_model_run_dir_path(model_run_dir)$disease

  train_data_path <- fs::path(model_run_dir, "data", ext = "csv")
  train_dat <- readr::read_csv(train_data_path, show_col_types = FALSE)

  eval_data_path <- fs::path(model_run_dir, "eval_data", ext = "tsv")
  eval_dat <- readr::read_tsv(eval_data_path, show_col_types = FALSE) |>
    dplyr::mutate(data_type = "eval")

  posterior_predictive_path <- fs::path(model_run_dir, "mcmc_tidy",
    "pyrenew_posterior_predictive",
    ext = "parquet"
  )
  posterior_predictive <- arrow::read_parquet(posterior_predictive_path)


  other_ed_visits_path <- fs::path(model_run_dir, "other_ed_visits_forecast",
    ext = "parquet"
  )
  other_ed_visits_forecast <- arrow::read_parquet(other_ed_visits_path) |>
    dplyr::rename(Other = other_ed_visits)

  combined_dat <-
    dplyr::bind_rows(
      train_dat |>
        dplyr::filter(data_type == "train"),
      eval_dat
    ) |>
    dplyr::mutate(
      disease = dplyr::if_else(
        disease == disease_name_nssp,
        "Disease", # assign a common name for
        # use in plotting functions
        disease
      )
    ) |>
    tidyr::pivot_wider(names_from = disease, values_from = ed_visits) |>
    dplyr::mutate(
      Other = Total - Disease,
      prop_disease_ed_visits = Disease / Total
    ) |>
    dplyr::select(-Total) |>
    dplyr::mutate(time = dplyr::dense_rank(date)) |>
    tidyr::pivot_longer(c(Disease, Other, prop_disease_ed_visits),
      names_to = "disease",
      values_to = ".value"
    )

  last_training_date <- combined_dat |>
    dplyr::filter(data_type == "train") |>
    dplyr::pull(date) |>
    max()

  other_ed_visits_samples <-
    dplyr::bind_rows(
      combined_dat |>
        dplyr::filter(
          data_type == "train",
          disease == "Other",
          date <= last_training_date
        ) |>
        dplyr::select(date, Other = .value) |>
        tidyr::expand_grid(.draw = 1:max(other_ed_visits_forecast$.draw)),
      other_ed_visits_forecast
    )


  forecast_samples <-
    posterior_predictive |>
    tidybayes::gather_draws(observed_hospital_admissions[time]) |>
    tidyr::pivot_wider(names_from = .variable, values_from = .value) |>
    dplyr::rename(Disease = observed_hospital_admissions) |>
    dplyr::ungroup() |>
    dplyr::mutate(date = min(combined_dat$date) + time) |>
    dplyr::left_join(other_ed_visits_samples,
      by = c(".draw", "date")
    ) |>
    dplyr::mutate(prop_disease_ed_visits = Disease / (Disease + Other)) |>
    tidyr::pivot_longer(c(Other, Disease, prop_disease_ed_visits),
      names_to = "disease",
      values_to = ".value"
    )

  epiweekly_forecast_samples <- forecast_samples |>
    dplyr::filter(disease != "prop_disease_ed_visits") |>
    dplyr::group_by(disease) |>
    dplyr::group_modify(~ forecasttools::daily_to_epiweekly(.x,
      value_col = ".value", weekly_value_name = ".value",
      strict = TRUE
    )) |>
    dplyr::ungroup() |>
    tidyr::pivot_wider(
      names_from = disease,
      values_from = .value
    ) |>
    dplyr::mutate(prop_disease_ed_visits = Disease /
      (Disease + Other)) |>
    tidyr::pivot_longer(c(Disease, Other, prop_disease_ed_visits),
      names_to = "disease",
      values_to = ".value"
    ) |>
    dplyr::mutate(date = forecasttools::epiweek_to_date(
      epiweek,
      epiyear,
      day_of_week = 7
    ))

  forecast_ci <-
    forecast_samples |>
    dplyr::select(date, disease, .value) |>
    dplyr::group_by(date, disease) |>
    ggdist::median_qi(.width = c(0.5, 0.8, 0.95))

  # Save data
  if (save) {
    arrow::write_parquet(
      combined_dat,
      fs::path(model_run_dir,
        "combined_training_eval_data",
        ext = "parquet"
      )
    )

    arrow::write_parquet(
      forecast_samples,
      fs::path(model_run_dir, "forecast_samples",
        ext = "parquet"
      )
    )
    arrow::write_parquet(
      epiweekly_forecast_samples,
      fs::path(model_run_dir, "epiweekly_forecast_samples",
        ext = "parquet"
      )
    )

    arrow::write_parquet(
      forecast_ci,
      fs::path(model_run_dir, "forecast_ci",
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
