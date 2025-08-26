variable_resolution_key <-
  c(
    "observed_ed_visits" = "daily",
    "other_ed_visits" = "daily",
    "observed_hospital_admissions" = "epiweekly",
    "site_level_log_ww_conc" = "daily"
  )

load_and_aggregate_ts <- function(
  model_run_dir,
  timeseries_model_name,
  daily_training_dat,
  epiweekly_training_dat,
  required_columns
) {
  timeseries_model_dir <- fs::path(model_run_dir, timeseries_model_name)

  samples_file_names <- c(
    "daily_ts_ensemble_samples_e",
    "epiweekly_ts_ensemble_samples_e"
  )

  unaggregated_ts_samples <- tibble::tibble(
    samples = fs::path(
      timeseries_model_dir,
      samples_file_names,
      ext = "parquet"
    ) |>
      purrr::map(forecasttools::read_tabular),
    observed = list(daily_training_dat, epiweekly_training_dat) |>
      purrr::map(\(x) dplyr::select(x, -"data_type", -"lab_site_index"))
  ) |>
    dplyr::mutate(
      data = purrr::pmap(
        list(.data$samples, .data$observed),
        function(samples, observed, epiweekly) {
          to_tidy_draws_timeseries(
            tidy_forecast = samples,
            observed = observed
          )
        }
      )
    ) |>
    dplyr::select("data") |>
    tidyr::unnest("data") |>
    dplyr::mutate(
      aggregated_numerator = FALSE,
      aggregated_denominator = dplyr::if_else(
        stringr::str_starts(.data$.variable, "prop_"),
        FALSE,
        NA
      )
    )

  aggregated_ts_samples_non_prop <-
    unaggregated_ts_samples |>
    dplyr::filter(.data$resolution == "daily") |>
    dplyr::filter(!stringr::str_starts(.data$.variable, "prop_")) |>
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
      aggregated_numerator = TRUE
    ) |>
    dplyr::select(tidyselect::any_of(required_columns))

  dplyr::bind_rows(
    unaggregated_ts_samples,
    aggregated_ts_samples_non_prop
  )
}

prop_from_timeseries <- function(
  e_denominator_samples,
  e_numerator_samples,
  required_columns
) {
  prop_disease_ed_visits_tbl <-
    dplyr::left_join(
      e_denominator_samples,
      e_numerator_samples,
      by = c("resolution", ".draw", "date", "geo_value", "disease")
    ) |>
    dplyr::mutate(
      prop_disease_ed_visits = .data$observed_ed_visits /
        (.data$observed_ed_visits + .data$other_ed_visits)
    ) |>
    dplyr::rename(.value = "prop_disease_ed_visits") |>
    dplyr::mutate(.variable = "prop_disease_ed_visits") |>
    dplyr::select(tidyselect::any_of(required_columns))

  return(prop_disease_ed_visits_tbl)
}

epiweekly_samples_from_daily <- function(
  daily_samples,
  variables_to_aggregate = "observed_ed_visits",
  required_columns
) {
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

#' Read in and combine training and evaluation
#' data from a model run directory.
#'
#' @param model_run_dir model run directory in which to look
#' for data.
#' @export
read_and_combine_data <- function(model_run_dir) {
  data_cols <- readr::cols(
    date = readr::col_date(),
    geo_value = readr::col_character(),
    disease = readr::col_character(),
    data_type = readr::col_character(),
    .variable = readr::col_character(),
    .value = readr::col_double()
  )

  dat <-
    tidyr::expand_grid(
      epiweekly = c(FALSE, TRUE),
      root = c("combined_training_data", "combined_eval_data")
    ) |>
    dplyr::mutate(
      prefix = ifelse(.data$epiweekly, "epiweekly_", ""),
      aggregated = .data$epiweekly
    ) |>
    tidyr::unite("file_name", "prefix", "root", sep = "") |>
    dplyr::mutate(
      file_path = fs::path(model_run_dir, "data", .data$file_name, ext = "tsv")
    ) |>
    dplyr::mutate(
      data = purrr::map(
        .data$file_path,
        \(x) readr::read_tsv(x, col_types = data_cols)
      )
    ) |>
    dplyr::select("data", "aggregated") |>
    tidyr::unnest("data") |>
    dplyr::mutate(
      resolution = dplyr::if_else(
        .data$aggregated,
        "epiweekly",
        variable_resolution_key[.data$.variable]
      )
    ) |>
    dplyr::select(-"aggregated") |>
    dplyr::distinct()
  # suggest reforms to prep_data to prevent duplicate data being in each table

  non_ww_dat <- dat |>
    dplyr::filter(.variable != "site_level_log_ww_conc") |>
    tidyr::pivot_wider(names_from = ".variable", values_from = ".value") |>
    dplyr::mutate(
      prop_disease_ed_visits = .data$observed_ed_visits /
        (.data$observed_ed_visits + .data$other_ed_visits)
    ) |>
    tidyr::pivot_longer(
      cols = -c(
        "date",
        "geo_value",
        "disease",
        "data_type",
        "resolution"
      ),
      names_to = ".variable",
      values_to = ".value"
    ) |>
    tidyr::drop_na(".value")

  ww_dat <- dat |>
    dplyr::filter(.variable == "site_level_log_ww_conc")

  combined_dat <- dplyr::bind_rows(ww_dat, non_ww_dat)
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
to_tidy_draws_timeseries <- function(
  tidy_forecast,
  observed,
  date_colname = "date",
  sample_id_colname = ".draw",
  value_colname = ".value",
  epiweekly = FALSE
) {
  first_forecast_date <- min(tidy_forecast[[date_colname]])
  resolution <- unique(tidy_forecast$resolution)
  day_count <- ifelse(resolution == "epiweekly", 7, 1)
  n_draws <- max(tidy_forecast[[sample_id_colname]])

  target_variables <- unique(tidy_forecast$.variable)
  transformed_obs <- observed |>
    dplyr::filter(
      .data[[date_colname]] < !!first_forecast_date,
      .data$.variable %in% target_variables
    ) |>
    tidyr::expand_grid(!!sample_id_colname := 1:n_draws) |>
    dplyr::mutate(resolution = !!resolution)

  stopifnot(
    max(as.Date(transformed_obs[[date_colname]])) +
      lubridate::ddays(day_count) ==
      first_forecast_date
  )

  dplyr::bind_rows(
    transformed_obs,
    tidy_forecast
  ) |>
    dplyr::select(!!sample_id_colname, tidyselect::everything())
}


process_pyrenew_model <- function(
  model_run_dir,
  pyrenew_model_name,
  ts_samples,
  required_columns_e,
  n_forecast_days
) {
  model_info <- parse_model_run_dir_path(model_run_dir)

  pyrenew_model_components <- parse_pyrenew_model_name(pyrenew_model_name)

  if (pyrenew_model_components["w"]) {
    required_columns <- c(required_columns_e, "lab_site_index")
  } else {
    required_columns <- required_columns_e
  }

  ## Process PyRenew posterior
  pyrenew_model_dir <- fs::path(
    model_run_dir,
    pyrenew_model_name
  )

  pyrenew_posterior_predictive <-
    forecasttools::read_tabular(
      fs::path(
        pyrenew_model_dir,
        "mcmc_output",
        "tidy_posterior_predictive",
        ext = "parquet"
      )
    ) |>
    dplyr::rename("iteration" = "draw") |>
    dplyr::mutate("date" = as.Date(.data$date)) |>
    dplyr::mutate(dplyr::across(
      c("chain", "iteration"),
      \(x) x + 1L
    )) |>
    dplyr::rename_with(\(x) glue::glue(".{x}"), -"date")

  model_samples_tidy <-
    pyrenew_posterior_predictive |>
    dplyr::mutate(
      geo_value = model_info$location,
      disease = model_info$disease,
      resolution = variable_resolution_key[.data$.variable],
      aggregated_numerator = FALSE,
      aggregated_denominator = NA,
    ) |>
    tidybayes::combine_chains() |>
    dplyr::select(tidyselect::all_of(required_columns))

  # For the E model, do epiweekly and process denominator
  if (pyrenew_model_components["e"]) {
    epiweekly_e_numerator_samples <- epiweekly_samples_from_daily(
      daily_samples = model_samples_tidy,
      variables_to_aggregate = "observed_ed_visits",
      required_columns_e
    )

    e_numerator_samples <- dplyr::bind_rows(
      model_samples_tidy |>
        dplyr::select(tidyselect::all_of(required_columns_e)) |>
        dplyr::filter(.data$.variable == "observed_ed_visits"),
      epiweekly_e_numerator_samples
    ) |>
      dplyr::rename("observed_ed_visits" = ".value") |>
      dplyr::select(-c(".variable", "aggregated_denominator"))

    model_samples_tidy <- dplyr::bind_rows(
      model_samples_tidy,
      epiweekly_e_numerator_samples
    ) |>
      dplyr::distinct()

    ## Process timeseries posterior
    if (!is.null(ts_samples)) {
      e_denominator_samples <- ts_samples |>
        dplyr::filter(.data$.variable == "other_ed_visits") |>
        dplyr::mutate(aggregated_denominator = .data$aggregated_numerator) |>
        dplyr::select(-"aggregated_numerator") |>
        dplyr::rename("other_ed_visits" = ".value") |>
        dplyr::select(-".variable") |>
        dplyr::filter(.data$.draw %in% e_numerator_samples$.draw)

      prop_e_samples <- prop_from_timeseries(
        e_denominator_samples,
        e_numerator_samples,
        required_columns
      )

      model_samples_tidy <- dplyr::bind_rows(
        model_samples_tidy,
        prop_e_samples
      ) |>
        dplyr::distinct()
    }
  }
  return(model_samples_tidy)
}

#' Process loc forecast
#'
#' @param model_run_dir Model run directory
#' @param pyrenew_model_name Name of directory containing pyrenew
#' model outputs
#' @param timeseries_model_name Name of directory containing timeseries
#' model outputs
#' @param n_forecast_days An integer specifying the number of days to forecast.
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
process_loc_forecast <- function(
  model_run_dir,
  n_forecast_days,
  pyrenew_model_name = NA,
  timeseries_model_name = NA,
  ci_widths = c(0.5, 0.8, 0.95),
  save = TRUE
) {
  if (is.na(pyrenew_model_name) && is.na(timeseries_model_name)) {
    stop(
      "Either `pyrenew_model_name` or `timeseries_model_name`",
      "must be provided."
    )
  }

  data_col_types <- readr::cols(
    date = readr::col_date(),
    geo_value = readr::col_character(),
    disease = readr::col_character(),
    data_type = readr::col_character(),
    .variable = readr::col_character(),
    .value = readr::col_double()
  )

  # Used for augmenting denominator forecasts with training period denominator
  daily_training_dat <- readr::read_tsv(
    fs::path(
      model_run_dir,
      "data",
      "combined_training_data",
      ext = "tsv"
    ),
    col_types = data_col_types
  )

  # Used for augmenting denominator forecasts with training period denominator
  epiweekly_training_dat <- readr::read_tsv(
    fs::path(
      model_run_dir,
      "data",
      "epiweekly_combined_training_data",
      ext = "tsv"
    ),
    col_types = data_col_types
  )

  required_columns_e <- c(
    ".chain",
    ".iteration",
    ".draw",
    "date",
    "geo_value",
    "disease",
    ".variable",
    ".value",
    "resolution",
    "aggregated_numerator",
    "aggregated_denominator"
  )

  if (!is.na(timeseries_model_name)) {
    ts_samples <- load_and_aggregate_ts(
      model_run_dir,
      timeseries_model_name,
      daily_training_dat,
      epiweekly_training_dat,
      required_columns = required_columns_e
    )
  }

  if (is.na(pyrenew_model_name)) {
    model_samples_tidy <- ts_samples
  } else {
    model_samples_tidy <- process_pyrenew_model(
      model_run_dir,
      pyrenew_model_name,
      ts_samples,
      required_columns_e,
      n_forecast_days
    )
  }

  ci <- model_samples_tidy |>
    dplyr::select(-tidyselect::any_of(c(".chain", ".iteration", ".draw"))) |>
    dplyr::group_by(dplyr::across(-".value")) |>
    ggdist::median_qi(.width = ci_widths)

  result <- list(
    "samples" = model_samples_tidy,
    "ci" = ci
  )

  if (save) {
    model_name <- dplyr::if_else(
      is.na(pyrenew_model_name),
      timeseries_model_name,
      pyrenew_model_name
    )

    save_dir <- fs::path(model_run_dir, model_name)

    purrr::iwalk(result, \(tab, name) {
      forecasttools::write_tabular(
        tab,
        fs::path(save_dir, name, ext = "parquet")
      )
    })
  }

  return(result)
}
