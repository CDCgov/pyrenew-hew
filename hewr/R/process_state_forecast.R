#' Annotate a dataframe of ED visits data with the
#' proportion of visits due to a target disease.
#'
#' @param df dataframe to annotate, with columns
#' `"Disease"` and `"Other"`.
#' @return the dataframe with an additional column
#' `prop_disease_ed_visits`.
#' @export
with_prop_disease_ed_visits <- function(df) {
  return(
    df |>
      dplyr::mutate(prop_disease_ed_visits = .data$Disease /
        (.data$Disease + .data$Other))
  )
}

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
  combined_dat <- dplyr::bind_rows(train_dat, eval_dat) |>
    dplyr::mutate(
      disease = dplyr::if_else(
        .data$disease == !!disease_name,
        "Disease", # assign a common name for
        # use in plotting functions
        .data$disease
      )
    ) |>
    dplyr::filter(.data$disease %in% c("Total", "Disease")) |>
    tidyr::pivot_wider(
      names_from = "disease",
      values_from = "ed_visits"
    ) |>
    dplyr::mutate(
      Other = .data$Total - .data$Disease
    ) |>
    with_prop_disease_ed_visits() |>
    dplyr::select(-"Total") |>
    dplyr::mutate(time = dplyr::dense_rank(.data$date)) |>
    tidyr::pivot_longer(
      c("Disease", "Other", "prop_disease_ed_visits"),
      names_to = "disease",
      values_to = ".value"
    )

  return(combined_dat)
}


#' Read in and combine training and evaluation
#' data from a model run directory.
#'
#' @param model_run_dir model run directoryh in which to look
#' for data.
#' @param disease_name name of the disease for which to get
#' combined training and evaluation data.
#' @param epiweekly Get epiweekly data instead of daily data?
#' Boolean, default `FALSE`.
read_and_combine_data <- function(model_run_dir,
                                  disease_name,
                                  epiweekly = FALSE) {
  data_cols <- readr::cols(
    date = readr::col_date(),
    ed_visits = readr::col_double()
  )

  prefix <- if (epiweekly) "epiweekly_" else ""

  train_data_path <- fs::path(model_run_dir,
    "data",
    glue::glue("{prefix}data"),
    ext = "tsv"
  )
  train_dat <- readr::read_tsv(train_data_path,
    col_types = data_cols
  )

  eval_data_path <- fs::path(model_run_dir,
    "data",
    glue::glue("{prefix}eval_data"),
    ext = "tsv"
  )
  eval_dat <- readr::read_tsv(eval_data_path, col_types = data_cols) |>
    dplyr::mutate(data_type = "eval")

  combined_dat <- combine_training_and_eval_data(
    train_dat,
    eval_dat,
    disease_name
  )
}

#' Combine a forecast in tidy draws based format
#' with observed values to create a synthetic set
#' of tidy posterior "samples".
#'
#' Observed timepoints have the observed value as
#' the sampled value for all sample ids.
#'
#' @param tidy_forecast Forecast in tidy format, with
#' a sample id column and a value column.
#' @param observed observed data to join with the forecast.
#' @param disease_name name of the disease in `tidy_forecast`,
#' for downsampling the observed data if it contains data
#' for multiple diseases.
#' @param date_colname Name of the column in `tidy_forecast`
#' and `observed` that identifies dates. Default `"date"`.
#' @param sample_id_colname Name of the column in
#' `tidy_forecast` that uniquely identifies individual
#' posterior samples / draws. Default `".draw"`.
#' @param value_colname Name of the column in
#' `tidy_forecast` for the sampled values.
#' Default `".value"`.
to_tidy_draws_timeseries <- function(tidy_forecast,
                                     observed,
                                     disease_name,
                                     date_colname = "date",
                                     sample_id_colname = ".draw",
                                     value_colname = ".value") {
  first_forecast_date <- min(tidy_forecast[[date_colname]])
  n_draws <- max(tidy_forecast[[sample_id_colname]])
  transformed_obs <- observed |>
    dplyr::filter(
      .data$disease == !!disease_name,
      .data[[date_colname]] < !!first_forecast_date
    ) |>
    dplyr::select(
      !!date_colname,
      !!disease_name := !!value_colname
    ) |>
    tidyr::expand_grid(!!sample_id_colname := 1:n_draws)

  stopifnot(
    max(as.Date(transformed_obs[[date_colname]])) +
      lubridate::ddays(1) == first_forecast_date
  )

  dplyr::bind_rows(
    transformed_obs,
    tidy_forecast
  )
}


#' Pivot a data table of counts and proportions of
#' ED visits to long format.
#'
#' @param df data frame to pivot. Should have columns
#' `"Disease"`, `"Other"`, and `"prop_disease_ed_visits"`.
#' @return the pivoted data frame, with disease names in
#' a column named `disease` and counts / proportions in
#' a column named `.value`.
#' @export
pivot_ed_visit_df_longer <- function(df) {
  return(tidyr::pivot_longer(
    df,
    c(
      "Disease",
      "Other",
      "prop_disease_ed_visits"
    ),
    names_to = "disease",
    values_to = ".value"
  ))
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
#' @return a list of 8 tibbles:
#' `combined_dat`,
#' `epiweekly_combined_dat`,
#' `forecast_samples`,
#' `epiweekly_forecast_samples`,
#' `forecast_with_epiweekly_other`,
#' `forecast_ci`,
#' `epiweekly_forecast_ci`,
#' `forecast_with_epiweekly_other_ci`
#' @export
process_state_forecast <- function(model_run_dir,
                                   pyrenew_model_name,
                                   timeseries_model_name,
                                   save = TRUE) {
  pyrenew_model_dir <- fs::path(model_run_dir, pyrenew_model_name)
  timeseries_model_dir <- fs::path(model_run_dir, timeseries_model_name)
  disease_name <- parse_model_run_dir_path(model_run_dir)$disease

  combined_dat <- read_and_combine_data(
    model_run_dir, disease_name,
    epiweekly = FALSE
  )
  epiweekly_combined_dat <- read_and_combine_data(
    model_run_dir, disease_name,
    epiweekly = TRUE
  )

  posterior_predictive_path <- fs::path(pyrenew_model_dir, "mcmc_tidy",
    "pyrenew_posterior_predictive",
    ext = "parquet"
  )

  posterior_predictive <- arrow::read_parquet(posterior_predictive_path)

  other_ed_visits_forecast <-
    arrow::read_parquet(
      fs::path(timeseries_model_dir,
        "other_ed_visits_forecast",
        ext = "parquet"
      )
    ) |>
    dplyr::rename(Other = "other_ed_visits")

  epiweekly_other_forecast <-
    arrow::read_parquet(
      fs::path(timeseries_model_dir,
        "epiweekly_other_ed_visits_forecast",
        ext = "parquet"
      )
    ) |>
    dplyr::rename(Other = "other_ed_visits")


  ## augment other ed visits forecast with "sample"
  ## format observed data
  other_ed_visits_samples <- to_tidy_draws_timeseries(
    other_ed_visits_forecast,
    combined_dat |> dplyr::filter(.data$data_type == "train"),
    ## use this rather than train_dat since it has "Other"
    ## (as opposed to "Total") pre-computed
    disease_name = "Other"
  )
  epiweekly_other_samples <- to_tidy_draws_timeseries(
    epiweekly_other_forecast,
    epiweekly_combined_dat |> dplyr::filter(.data$data_type == "train"),
    ## use this rather than epiweekly_train_dat since it has "Other"
    ## (as opposed to "Total") pre-computed
    disease_name = "Other"
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
    with_prop_disease_ed_visits() |>
    pivot_ed_visit_df_longer()

  epiweekly_forecast_samples_raw <- forecast_samples |>
    dplyr::filter(.data$disease != "prop_disease_ed_visits") |>
    forecasttools::daily_to_epiweekly(
      value_col = ".value",
      weekly_value_name = ".value",
      id_cols = c(".draw", "disease"),
      strict = TRUE
    ) |>
    tidyr::pivot_wider(
      names_from = "disease",
      values_from = ".value"
    ) |>
    dplyr::mutate(date = forecasttools::epiweek_to_date(
      .data$epiweek,
      .data$epiyear,
      day_of_week = 7
    ))

  epiweekly_forecast_samples <- epiweekly_forecast_samples_raw |>
    with_prop_disease_ed_visits() |>
    pivot_ed_visit_df_longer()

  forecast_with_epiweekly_other <- epiweekly_forecast_samples_raw |>
    dplyr::select(-"Other") |>
    dpylr::left_join(epiweekly_other_samples,
      by = c(".draw", "date")
    ) |>
    with_prop_disease_ed_visits() |>
    pivot_ed_visit_df_longer()

  forecast_ci <-
    forecast_samples |>
    dplyr::select("date", "disease", ".value") |>
    dplyr::group_by(.data$date, .data$disease) |>
    ggdist::median_qi(.width = c(0.5, 0.8, 0.95))

  epiweekly_forecast_ci <-
    epiweekly_forecast_samples |>
    dplyr::select("date", "disease", ".value") |>
    dplyr::group_by(.data$date, .data$disease) |>
    ggdist::median_qi(.width = c(0.5, 0.8, 0.95))

  forecast_w_epiweekly_other_ci <-
    forecast_with_epiweekly_other |>
    dplyr::select("date", "disease", ".value") |>
    dplyr::group_by(.data$date, .data$disease) |>
    ggdist::median_qi(.width = c(0.5, 0.8, 0.95))

  # Optionally save data to parquet
  if (save) {
    to_save <- list(
      list(
        table = combined_dat,
        save_name = "combined_training_eval_data"
      ),
      list(
        table = forecast_samples,
        save_name = "forecast_samples"
      ),
      list(
        table = epiweekly_forecast_samples,
        save_name = "epiweekly_forecast_samples"
      ),
      list(
        table = forecast_with_epiweekly_other,
        save_name = "forecast_with_epiweekly_other"
      ),
      list(
        table = forecast_ci,
        save_name = "forecast_ci"
      ),
      list(
        table = epiweekly_forecast_ci,
        save_name = "epiweekly_forecast_ci"
      ),
      list(
        table = forecast_w_epiweekly_other_ci,
        save_name = "forecast_with_epiweekly_other_ci"
      )
    )


    purrr::walk(
      to_save,
      \(x) {
        arrow::write_parquet(
          x$table,
          fs::path(pyrenew_model_dir,
            x$save_name,
            ext = "parquet"
          )
        )
      }
    )
  }

  return(list(
    combined_dat = combined_dat,
    epiweekly_combined_dat = epiweekly_combined_dat,
    forecast_samples = forecast_samples,
    epiweekly_forecast_samples = epiweekly_forecast_samples,
    forecast_with_epiweekly_other = forecast_with_epiweekly_other,
    forecast_ci = forecast_ci,
    epiweekly_forecast_ci = epiweekly_forecast_ci,
    forecast_with_epiweekly_other_ci = forecast_w_epiweekly_other_ci
  ))
}
