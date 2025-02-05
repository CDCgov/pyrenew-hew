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

join_and_calc_prop <- function(model_1, model_2) {
  dplyr::inner_join(
    tidyr::pivot_wider(model_1,
      names_from = ".variable",
      values_from = ".value"
    ),
    tidyr::pivot_wider(model_2,
      names_from = ".variable",
      values_from = ".value"
    ),
    by = dplyr::join_by(".draw", "date", "geo_value", "disease")
  ) |>
    dplyr::mutate(prop_disease_ed_visits = .data$observed_ed_visits /
      (.data$observed_ed_visits + .data$other_ed_visits)) |>
    tidyr::pivot_longer(
      -c(
        tidyselect::starts_with("."),
        "date", "geo_value", "disease"
      ),
      names_to = ".variable", values_to = ".value"
    ) |>
    tidyr::drop_na()
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
  required_columns <- c(
    ".chain", ".iteration", ".draw", "date", "geo_value",
    "disease", ".variable", ".value"
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
  posterior_predictive_variables <-
    pyrenew_posterior_predictive |>
    colnames() |>
    stringr::str_remove("\\[.+\\]$") |>
    unique() |>
    purrr::keep(~ stringr::str_starts(., "observed_")) |>
    stringr::str_c("[group_time_index]") |>
    purrr::map(rlang::parse_expr)

  # must use gather_draws
  # use of spread_draws results in indices being dropped
  daily_samples <-
    pyrenew_posterior_predictive |>
    tidybayes::gather_draws(!!!posterior_predictive_variables) |>
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
      disease = model_info$disease
    ) |>
    dplyr::select(tidyselect::all_of(required_columns))

  samples_list <- list(daily_samples = daily_samples)

  # For the E model, do epiweekly
  if (pyrenew_model_components["e"]) {
    epiweekly_obs_ed_samples <-
      daily_samples |>
      dplyr::filter(.data$.variable == "observed_ed_visits") |>
      forecasttools::daily_to_epiweekly(
        value_col = ".value",
        weekly_value_name = ".value",
        id_cols = c(
          ".chain", ".iteration", ".draw", "geo_value", "disease",
          ".variable"
        ),
        strict = TRUE,
        with_epiweek_end_date = TRUE,
        epiweek_end_date_name = "date"
      ) |>
      dplyr::select(tidyselect::all_of(required_columns))

    epiweekly_samples <-
      daily_samples |>
      dplyr::filter(.data$.variable != "observed_ed_visits") |>
      dplyr::bind_rows(epiweekly_obs_ed_samples) |>
      dplyr::select(tidyselect::all_of(required_columns))

    samples_list$epiweekly_samples <- epiweekly_samples

    ## Process timeseries posterior
    if (!is.null(timeseries_model_name)) {
      timeseries_model_dir <- fs::path(
        model_run_dir,
        timeseries_model_name
      )

      # augment daily and epiweekly other ed visits forecast
      # with "sample" format observed data

      ## ts model, daily denominator
      daily_ts_denom_samples <- arrow::read_parquet(
        fs::path(timeseries_model_dir,
          "daily_baseline_ts_forecast_samples",
          ext = "parquet"
        )
      ) |>
        dplyr::filter(.data$.variable == "other_ed_visits") |>
        to_tidy_draws_timeseries(
          observed = daily_training_dat |>
            dplyr::filter(.data$.variable == "other_ed_visits") |>
            dplyr::select(-"data_type"),
          epiweekly = FALSE
        ) |>
        dplyr::select(tidyselect::any_of(required_columns))

      ## ts model, daily denominator aggregated to epiweekly
      agg_ewkly_ts_denom_samples <-
        daily_ts_denom_samples |>
        forecasttools::daily_to_epiweekly(
          value_col = ".value",
          weekly_value_name = ".value",
          id_cols = c(".draw", "geo_value", "disease", ".variable"),
          strict = TRUE,
          with_epiweek_end_date = TRUE,
          epiweek_end_date_name = "date"
        ) |>
        dplyr::select(tidyselect::any_of(required_columns))

      ## ts model, epiweekly denominator
      ewkly_ts_denom_samples <- arrow::read_parquet(
        fs::path(timeseries_model_dir,
          "epiweekly_baseline_ts_forecast_samples",
          ext = "parquet"
        )
      ) |>
        dplyr::filter(.data$.variable == "other_ed_visits") |>
        to_tidy_draws_timeseries(
          observed = epiweekly_training_dat |>
            dplyr::filter(.data$.variable == "other_ed_visits") |>
            dplyr::select(-"data_type"),
          epiweekly = TRUE
        ) |>
        dplyr::select(tidyselect::any_of(required_columns))

      # Daily Numerator, Daily Denominator
      daily_samples_daily_n_daily_d <- join_and_calc_prop(
        daily_samples,
        daily_ts_denom_samples
      )

      samples_list$daily_samples <- daily_samples_daily_n_daily_d

      # Epiweekly Aggregated Numerator, Epiweekly Aggregated Denominator
      ewkly_samples_agg_n_agg_d <- join_and_calc_prop(
        epiweekly_samples,
        agg_ewkly_ts_denom_samples
      )

      samples_list$epiweekly_samples <- ewkly_samples_agg_n_agg_d


      # Epiweekly Aggregated Numerator, Epiweekly Denominator
      ewkly_samples_agg_n_ewkly_d <- join_and_calc_prop(
        epiweekly_samples,
        ewkly_ts_denom_samples
      )

      samples_list$epiweekly_with_epiweekly_other_samples <-
        ewkly_samples_agg_n_ewkly_d
    }
  }


  ci_list <- purrr::map(
    samples_list |>
      purrr::set_names(~ stringr::str_replace(., "samples", "ci")),
    \(x) {
      dplyr::select(x, -c(".chain", ".iteration", ".draw")) |>
        dplyr::group_by(
          .data$date, .data$geo_value, .data$disease,
          .data$.variable
        ) |>
        ggdist::median_qi(.width = ci_widths)
    }
  )

  result <- c(
    samples_list,
    ci_list
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
