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
      values_from = "value"
    ) |>
    dplyr::mutate(
      Other = .data$Total - .data$Disease
    ) |>
    with_prop_disease_ed_visits() |>
    dplyr::select(-"Total") |>
    tidyr::pivot_longer(
      c("Disease", "Other", "prop_disease_ed_visits"),
      names_to = "disease",
      values_to = ".value"
    ) |>
    drop_na()

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
    value = readr::col_double()
  )

  prefix <- ifelse(epiweekly, "epiweekly_", "")

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

  combined_dat <- combine_training_and_eval_data(
    train_dat,
    eval_dat,
    disease_name
  )

  combined_dat
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
#' @param epiweekly Is the timeseries epiweekly (as opposed
#' to daily)? Boolean, default `FALSE` (i.e. daily timeseries).
to_tidy_draws_timeseries <- function(tidy_forecast,
                                     observed,
                                     disease_name,
                                     date_colname = "date",
                                     sample_id_colname = ".draw",
                                     value_colname = ".value",
                                     epiweekly = FALSE) {
  first_forecast_date <- min(tidy_forecast[[date_colname]])
  day_count <- ifelse(epiweekly, 7, 1)
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
      lubridate::ddays(day_count) == first_forecast_date
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

parse_pyrenew_model_name <- function(pyrenew_model_name) {
  pyrenew_model_tail <- stringr::str_extract(
    pyrenew_model_name,
    "(?<=_).+$"
  ) |>
    stringr::str_split_1("")
  model_components <- c("h", "e", "w")
  model_components %in% pyrenew_model_tail |>
    purrr::set_names(model_components)
}

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

  first_date_key[variable] + days(step_size_key[variable]) *
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
  disease_name <- parse_model_run_dir_path(model_run_dir)$disease
  pyrenew_model_components <- parse_pyrenew_model_name(pyrenew_model_name)

  ## Process data
  data_for_model_fit <- jsonlite::read_json(
    path(model_run_dir, "data", "data_for_model_fit", ext = "json")
  )

  first_nhsn_date <- data_for_model_fit$nhsn_training_dates[[1]]
  first_nssp_date <- data_for_model_fit$nssp_training_dates[[1]]
  nhsn_step_size <- data_for_model_fit$nhsn_step_size

  daily_combined_dat <- read_and_combine_data(
    model_run_dir, disease_name,
    epiweekly = FALSE
  )

  daily_training_dat <- daily_combined_dat |>
    dplyr::filter(.data$data_type == "train")

  epiweekly_combined_dat <- read_and_combine_data(
    model_run_dir, disease_name,
    epiweekly = TRUE
  )

  epiweekly_training_dat <- epiweekly_combined_dat |>
    dplyr::filter(.data$data_type == "train")

  data_list <- list(
    daily_combined_training_eval_data = daily_combined_dat,
    epiweekly_combined_training_eval_data =
      epiweekly_combined_dat
  )

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
    ungroup() |>
    mutate(date = group_time_index_to_date(
      group_time_index = group_time_index,
      variable = .variable,
      first_nssp_date = first_nssp_date,
      first_nhsn_date = first_nhsn_date,
      nhsn_step_size = nhsn_step_size
    )) |>
    select(-group_time_index)

  if (pyrenew_model_components["e"]) {
    # do epiweekly stuff
    epiweekly_samples_ed <- daily_samples |>
      filter(.variable == "observed_ed_visits") |>
      forecasttools::daily_to_epiweekly(
        value_col = ".value",
        weekly_value_name = ".value",
        id_cols = c(".draw", ".variable"),
        strict = TRUE
      ) |>
      dplyr::mutate(date = forecasttools::epiweek_to_date(
        .data$epiweek,
        .data$epiyear,
        day_of_week = 7
      )) |>
      select(-epiweek, -epiyear, )

    epiweekly_samples <-
      daily_samples |>
      filter(.variable != "observed_ed_visits") |>
      bind_rows(epiweekly_samples_ed) |>
      select(starts_with("."), date, .variable, .value)
  }
  # working here
  ## Process timeseries posterior
  if (!is.null(timeseries_model_name)) {
    timeseries_model_dir <- fs::path(
      model_run_dir,
      timeseries_model_name
    )

    ## augment daily and epiweekly other ed visits forecast
    ## with "sample" format observed data
    daily_other_ed_visits_samples <-
      arrow::read_parquet(
        fs::path(timeseries_model_dir,
          "other_ed_visits_forecast",
          ext = "parquet"
        )
      ) |>
      dplyr::rename(Other = "other_ed_visits") |>
      to_tidy_draws_timeseries(
        daily_training_dat,
        disease_name = "Other",
        epiweekly = FALSE
      )

    ewkly_other_ed_visits_samples <-
      arrow::read_parquet(
        fs::path(timeseries_model_dir,
          "epiweekly_other_ed_visits_forecast",
          ext = "parquet"
        )
      ) |>
      dplyr::rename(Other = "other_ed_visits") |>
      to_tidy_draws_timeseries(
        epiweekly_training_dat,
        disease_name = "Other",
        epiweekly = TRUE
      )

    ewkly_with_ewkly_other_samples <-
      epiweekly_samples_raw |>
      dplyr::select(-"Other") |>
      dplyr::left_join(ewkly_other_ed_visits_samples,
        by = c(".draw", "date")
      ) |>
      with_prop_disease_ed_visits() |>
      pivot_ed_visit_df_longer()
  }

  samples_list <- list(daily_samples = daily_samples)

  ci_list <- purrr::map(
    samples_list |>
      purrr::set_names(~ stringr::str_replace(., "samples", "ci")),
    \(x) {
      dplyr::select(x, "date", ".variable", ".value") |>
        dplyr::group_by(.data$date, .data$.variable) |>
        ggdist::median_qi(.width = ci_widths)
    }
  )

  result <- c(
    data_list,
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
