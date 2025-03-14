prop_from_timeseries <- function(timeseries_model_dir,
                                 e_numerator_samples,
                                 required_columns,
                                 daily_training_dat,
                                 epiweekly_training_dat) {
  samples_file_names <- c(
    "daily_baseline_ts_forecast_samples",
    "epiweekly_baseline_ts_forecast_samples"
  )

  unaggregated_ts_samples <- tibble::tibble(
    samples = fs::path(timeseries_model_dir,
      samples_file_names,
      ext = "parquet"
    ) |>
      purrr::map(\(x) {
        arrow::read_parquet(x) |>
          dplyr::filter(.data$.variable == "other_ed_visits")
      }),
    resolution = c("daily", "epiweekly"),
    observed = list(daily_training_dat, epiweekly_training_dat) |>
      purrr::map(\(x) {
        x |>
          dplyr::filter(.data$.variable == "other_ed_visits") |>
          dplyr::select(-"data_type")
      }),
    aggregated_denominator = FALSE
  ) |>
    dplyr::mutate(data = purrr::pmap(
      list(.data$samples, .data$observed, .data$resolution == "epiweekly"),
      function(samples, observed, epiweekly) {
        to_tidy_draws_timeseries(
          tidy_forecast = samples,
          observed = observed,
          epiweekly = epiweekly
        )
      }
    )) |>
    dplyr::select("resolution", "aggregated_denominator", "data") |>
    tidyr::unnest("data")

  aggregated_ts_samples <-
    unaggregated_ts_samples |>
    dplyr::filter(.data$resolution == "daily") |>
    forecasttools::daily_to_epiweekly(
      value_col = ".value",
      weekly_value_name = ".value",
      id_cols = setdiff(colnames(unaggregated_ts_samples), c("date", ".value")),
      strict = TRUE,
      with_epiweek_end_date = TRUE,
      epiweek_end_date_name = "date"
    ) |>
    dplyr::mutate(
      resolution = "epiweekly",
      aggregated_denominator = TRUE
    ) |>
    dplyr::select(tidyselect::any_of(required_columns))

  e_denominator_samples <- dplyr::bind_rows(
    unaggregated_ts_samples,
    aggregated_ts_samples
  ) |>
    dplyr::rename("other_ed_visits" = ".value") |>
    dplyr::select(-".variable")

  prop_disease_ed_visits_tbl <-
    dplyr::left_join(e_denominator_samples, e_numerator_samples,
      by = c("resolution", ".draw", "date", "geo_value", "disease")
    ) |>
    dplyr::mutate(prop_disease_ed_visits = .data$observed_ed_visits /
      (.data$observed_ed_visits + .data$other_ed_visits)) |>
    dplyr::rename(.value = "prop_disease_ed_visits") |>
    dplyr::mutate(.variable = "prop_disease_ed_visits") |>
    dplyr::select(tidyselect::all_of(required_columns))

  return(prop_disease_ed_visits_tbl)
}

epiweekly_samples_from_daily <- function(daily_samples,
                                         variables_to_aggregate =
                                           "observed_ed_visits",
                                         required_columns) {
  aggregated_samples <-
    daily_samples |>
    dplyr::filter(.data$.variable %in% variables_to_aggregate) |>
    forecasttools::daily_to_epiweekly(
      value_col = ".value",
      weekly_value_name = ".value",
      id_cols = setdiff(required_columns, c("date", ".value")),
      strict = TRUE,
      with_epiweek_end_date = TRUE,
      epiweek_end_date_name = "date"
    ) |>
    dplyr::mutate(
      resolution = "epiweekly",
      aggregated_numerator = TRUE
    ) |>
    dplyr::select(tidyselect::all_of(required_columns))

  return(aggregated_samples)
}

#' Combine training and evaluation data for
#' postprocessing.
#'
#' @param train_dat Training data, as a [`tibble`][tibble::tibble()].
#' @param eval_dat Evaluation data, as a [`tibble`][tibble::tibble()].
#' @return The combined data, as a [`tibble`][tibble::tibble()].
#' @export
combine_training_and_eval_data <- function(train_dat,
                                           eval_dat) {
  combined_dat <-
    dplyr::bind_rows(train_dat, eval_dat) |>
    tidyr::pivot_wider(names_from = ".variable", values_from = ".value") |>
    dplyr::mutate(prop_disease_ed_visits = .data$observed_ed_visits /
      (.data$observed_ed_visits + .data$other_ed_visits)) |>
    tidyr::pivot_longer(
      cols = -c("date", "geo_value", "disease", "data_type"),
      names_to = ".variable", values_to = ".value"
    ) |>
    tidyr::drop_na()

  return(combined_dat)
}

#' Read in and combine training and evaluation
#' data from a model run directory.
#'
#' @param model_run_dir model run directoryh in which to look
#' for data.
#' @param epiweekly Get epiweekly data instead of daily data?
#' Boolean, default `FALSE`.
#' @export
read_and_combine_data <- function(model_run_dir,
                                  epiweekly = FALSE) {
  prefix <- ifelse(epiweekly, "epiweekly_", "")

  data_cols <- readr::cols(
    date = readr::col_date(),
    geo_value = readr::col_character(),
    disease = readr::col_character(),
    data_type = readr::col_character(),
    .variable = readr::col_character(),
    .value = readr::col_double()
  )

  train_data_path <- fs::path(model_run_dir,
    "data",
    glue::glue("{prefix}combined_training_data"),
    ext = "tsv"
  )
  train_dat <- readr::read_tsv(train_data_path, col_types = data_cols)

  eval_data_path <- fs::path(model_run_dir,
    "data",
    glue::glue("{prefix}combined_eval_data"),
    ext = "tsv"
  )
  eval_dat <- readr::read_tsv(eval_data_path, col_types = data_cols)

  combined_dat <- combine_training_and_eval_data(train_dat, eval_dat)

  return(combined_dat)
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
#' @param date_colname Name of the column in `tidy_forecast`
#' and `observed` that identifies dates. Default `"date"`.
#' @param sample_id_colname Name of the column in
#' `tidy_forecast` that uniquely identifies individual
#' posterior samples / draws. Default `".draw"`.
#' @param value_colname Name of the column in
#' `tidy_forecast` for the sampled values.
#' Default `".value"`.
#' @param epiweekly Is the timeseries epiweekly (as opposed
#' to daily)? Boolean, default `FALSE` (i.e. daily timeseries).
to_tidy_draws_timeseries <- function(tidy_forecast,
                                     observed,
                                     date_colname = "date",
                                     sample_id_colname = ".draw",
                                     value_colname = ".value",
                                     epiweekly = FALSE) {
  first_forecast_date <- min(tidy_forecast[[date_colname]])
  day_count <- ifelse(epiweekly, 7, 1)
  n_draws <- max(tidy_forecast[[sample_id_colname]])

  transformed_obs <- observed |>
    dplyr::filter(
      .data[[date_colname]] < !!first_forecast_date
    ) |>
    tidyr::expand_grid(!!sample_id_colname := 1:n_draws)

  stopifnot(
    max(as.Date(transformed_obs[[date_colname]])) +
      lubridate::ddays(day_count) == first_forecast_date
  )

  dplyr::bind_rows(
    transformed_obs,
    tidy_forecast
  ) |>
    dplyr::select(!!sample_id_colname, tidyselect::everything())
}


#' Convert group time index to date
#'
#' @param group_time_index integer vector of group time indices
#' @param variable variable name
#' @param first_nssp_date first date in the nssp training data
#' @param first_nhsn_date first date in the nhsn training data
#' @param nhsn_step_size step size for nhsn data
#'
#' @returns a vector of dates
#' @export
#'
#' @examples group_time_index_to_date(
#'   3,
#'   "observed_hospital_admissions", "2024-01-01", "2024-01-01", 7
#' )
group_time_index_to_date <- function(group_time_index,
                                     variable,
                                     first_nssp_date,
                                     first_nhsn_date,
                                     nhsn_step_size) {
  first_date_key <- c(
    observed_hospital_admissions = first_nhsn_date,
    observed_ed_visits = first_nssp_date
  ) |>
    purrr::map_vec(as.Date)

  step_size_key <- c(
    observed_hospital_admissions = nhsn_step_size,
    observed_ed_visits = 1
  )

  first_date_key[variable] + lubridate::days(step_size_key[variable]) *
    group_time_index
}

#' Process state forecast
#'
#' @param model_run_dir Model run directory
#' @param pyrenew_model_name Name of directory containing pyrenew
#' model outputs
#' @param timeseries_model_name Name of directory containing timeseries
#' model outputs
#' @param ci_widths Vector of probabilities indicating one or more
#' central credible intervals to compute. Passed as the `.width`
#' argument to [ggdist::median_qi()]. Default `c(0.5, 0.8, 0.95)`.
#' @param save Boolean indicating whether or not to save the output
#' to parquet files. Default `TRUE`.
#' @return a list of 8 tibbles:
#' `daily_combined_training_eval_data`,
#' `epiweekly_combined_training_eval_data`,
#' `daily_samples`,
#' `epiweekly_samples`,
#' `epiweekly_with_epiweekly_other_samples`,
#' `daily_ci`,
#' `epiweekly_ci`,
#' `epiweekly_with_epiweekly_other_ci`
#' @export
process_state_forecast <- function(model_run_dir,
                                   pyrenew_model_name,
                                   timeseries_model_name = NULL,
                                   ci_widths = c(0.5, 0.8, 0.95),
                                   save = TRUE) {
  # an older version of this function may have processed and saved "numerators"
  # from the timeseries model. I'm not sure. If it did, we need to add
  # functionality to accept a NULL pyrenew_model_name.
  variable_resolution_key <-
    c(
      "observed_ed_visits" = "daily",
      "observed_hospital_admissions" = "epiweekly"
    )

  required_columns <- c(
    ".chain", ".iteration", ".draw", "date", "geo_value", "disease",
    ".variable", ".value", "resolution", "aggregated_numerator",
    "aggregated_denominator"
  )

  data_col_types <- readr::cols(
    date = readr::col_date(),
    geo_value = readr::col_character(),
    disease = readr::col_character(),
    data_type = readr::col_character(),
    .variable = readr::col_character(),
    .value = readr::col_double()
  )
  model_info <- parse_model_run_dir_path(model_run_dir)
  pyrenew_model_components <- parse_pyrenew_model_name(pyrenew_model_name)

  ## Process data
  data_for_model_fit <- jsonlite::read_json(
    fs::path(model_run_dir, "data", "data_for_model_fit", ext = "json")
  )

  first_nhsn_date <- data_for_model_fit$nhsn_training_dates[[1]]
  first_nssp_date <- data_for_model_fit$nssp_training_dates[[1]]
  nhsn_step_size <- data_for_model_fit$nhsn_step_size

  # Used for augmenting denominator forecasts with training period denominator
  daily_training_dat <- readr::read_tsv(fs::path(
    model_run_dir, "data", "combined_training_data",
    ext = "tsv"
  ), col_types = data_col_types)


  # Used for augmenting denominator forecasts with training period denominator
  epiweekly_training_dat <- readr::read_tsv(fs::path(
    model_run_dir, "data", "epiweekly_combined_training_data",
    ext = "tsv"
  ), col_types = data_col_types)

  ## Process PyRenew posterior
  pyrenew_model_dir <- fs::path(
    model_run_dir,
    pyrenew_model_name
  )

  pyrenew_posterior_predictive <-
    arrow::read_parquet(
      fs::path(pyrenew_model_dir,
        "mcmc_tidy",
        "pyrenew_posterior_predictive",
        ext = "parquet"
      )
    )

  # posterior predictive variables are expected to be of the form
  # "observed_zzzzz[n]". This creates tidybayes::gather_draws()
  # compatible expression for each variable.
  post_pred_vars_exp <-
    pyrenew_posterior_predictive |>
    colnames() |>
    stringr::str_remove("\\[.+\\]$") |>
    unique() |>
    purrr::keep(~ stringr::str_starts(., "observed_")) |>
    stringr::str_c("[group_time_index]") |>
    purrr::map(rlang::parse_expr)


  # must use gather_draws
  # use of spread_draws results in indices being dropped
  pyrenew_samples_tidy <-
    pyrenew_posterior_predictive |>
    tidybayes::gather_draws(!!!post_pred_vars_exp) |>
    dplyr::ungroup() |>
    dplyr::mutate(date = group_time_index_to_date(
      group_time_index = .data$group_time_index,
      variable = .data$.variable,
      first_nssp_date = first_nssp_date,
      first_nhsn_date = first_nhsn_date,
      nhsn_step_size = nhsn_step_size
    )) |>
    dplyr::select(-"group_time_index") |>
    dplyr::mutate(
      geo_value = model_info$location,
      disease = model_info$disease,
      resolution = variable_resolution_key[.data$.variable],
      aggregated_numerator = FALSE,
      aggregated_denominator = NA,
    ) |>
    dplyr::select(tidyselect::all_of(required_columns))

  # For the E model, do epiweekly and process denominator
  if (pyrenew_model_components["e"]) {
    epiweekly_e_numerator_samples <- epiweekly_samples_from_daily(
      daily_samples = pyrenew_samples_tidy,
      variables_to_aggregate = "observed_ed_visits",
      required_columns
    )

    e_numerator_samples <- dplyr::bind_rows(
      pyrenew_samples_tidy |>
        dplyr::filter(.data$.variable == "observed_ed_visits"),
      epiweekly_e_numerator_samples
    ) |>
      dplyr::rename("observed_ed_visits" = ".value") |>
      dplyr::select(-c(".variable", "aggregated_denominator"))


    pyrenew_samples_tidy <- dplyr::bind_rows(
      pyrenew_samples_tidy,
      epiweekly_e_numerator_samples
    )


    ## Process timeseries posterior
    if (!is.null(timeseries_model_name)) {
      timeseries_model_dir <- fs::path(
        model_run_dir,
        timeseries_model_name
      )

      prop_e_samples <- prop_from_timeseries(
        timeseries_model_dir,
        e_numerator_samples,
        required_columns,
        daily_training_dat,
        epiweekly_training_dat
      )

      pyrenew_samples_tidy <- dplyr::bind_rows(
        pyrenew_samples_tidy,
        prop_e_samples
      )
    }
  }

  ci <- pyrenew_samples_tidy |>
    dplyr::select(-c(".chain", ".iteration", ".draw")) |>
    dplyr::group_by(
      .data$date,
      .data$geo_value,
      .data$disease,
      .data$.variable,
      .data$resolution,
      .data$aggregated_numerator,
      .data$aggregated_denominator,
    ) |>
    ggdist::median_qi(.width = ci_widths)

  result <- list(
    "samples" = pyrenew_samples_tidy,
    "ci" = ci
  )

  if (save) {
    purrr::iwalk(result, \(tab, name) {
      arrow::write_parquet(
        tab,
        fs::path(pyrenew_model_dir, name, ext = "parquet")
      )
    })
  }

  return(result)
}
