#' Read in daily forecast draws from a model run directory
#' and output a set of epiweekly quantiles, as a
#' [`tibbble`][tibble::tibble()].
#'
#' @param model_run_dir Path to a directory containing
#' forecast draws to process, whose basename is the forecasted
#' location.
#' @param report_date Report date for which to generate epiweekly
#' quantiles.
#' @param max_lookback_days How many days before the report date
#' to look back when generating epiweekly quantiles (determines how
#' many negative epiweekly forecast horizons (i.e. nowcast/backcast)
#' quantiles will be generated.
#' @param epiweekly_other Use an expressly epiweekly forecast
#' for non-target ED visits instead of a daily forecast aggregated
#' to epiweekly? Boolean, default `FALSE`.
#' @return A [`tibble`][tibble::tibble()] of quantiles.
#' @export
to_epiweekly_quantiles <- function(model_run_dir,
                                   report_date,
                                   max_lookback_days,
                                   epiweekly_other = FALSE) {
  message(glue::glue("Processing {model_run_dir}..."))
  draws_path <- fs::path(model_run_dir,
    "forecast_samples",
    ext = "parquet"
  )

  location <- fs::path_file(model_run_dir)

  draws <- arrow::read_parquet(draws_path) |>
    dplyr::filter(.data$date >= lubridate::ymd(!!report_date) -
      lubridate::days(!!max_lookback_days))

  if (nrow(draws) < 1) {
    return(NULL)
  }

  epiweekly_disease_draws <- draws |>
    dplyr::filter(
      .data$disease == "Disease"
    ) |>
    forecasttools::daily_to_epiweekly(
      date_col = "date",
      value_col = ".value",
      id_cols = ".draw",
      weekly_value_name = "epiweekly_disease",
      strict = TRUE
    )

  if (!epiweekly_other) {
    epiweekly_other_draws <- draws |>
      dplyr::filter(.data$disease == "Other") |>
      forecasttools::daily_to_epiweekly(
        date_col = "date",
        value_col = ".value",
        id_cols = ".draw",
        weekly_value_name = "epiweekly_other",
        strict = TRUE
      )
  } else {
    denom_path <- fs::path(model_run_dir,
      "epiweekly_other_ed_visits_forecast",
      ext = "parquet"
    )

    epiweekly_other_draws <- arrow::read_parquet(denom_path) |>
      dplyr::filter(.data$date >= lubridate::ymd(!!report_date) -
        lubridate::days(!!max_lookback_days)) |>
      dplyr::rename("epiweekly_other" = "other_ed_visits") |>
      dplyr::mutate(
        epiweek = lubridate::epiweek(.data$date),
        epiyear = lubridate::epiyear(.data$date)
      )
  }
  epiweekly_prop_draws <- dplyr::inner_join(
    epiweekly_disease_draws,
    epiweekly_other_draws,
    by = c(
      "epiweek",
      "epiyear",
      ".draw"
    )
  ) |>
    dplyr::mutate(
      "epiweekly_proportion" =
        .data$epiweekly_disease / (.data$epiweekly_disease +
          .data$epiweekly_other)
    )


  epiweekly_quantiles <- epiweekly_prop_draws |>
    forecasttools::trajectories_to_quantiles(
      timepoint_cols = c("epiweek", "epiyear"),
      value_col = "epiweekly_proportion"
    ) |>
    dplyr::mutate(
      "location" = !!location
    )

  message(glue::glue("Done processing {model_run_dir}"))
  return(epiweekly_quantiles)
}

#' Create an epiweekly hubverse-format forecast quantile table
#' from a model batch directory containing forecasts
#' for multiple locations as daily MCMC draws.
#'
#' @param model_batch_dir Model batch directory containing
#' the individual location forecast directories
#' ("model run directories") to process. Name should be in the format
#' `{disease}_r_{reference_date}_f_{first_data_date}_t_{last_data_date}`.
#' @param exclude Locations to exclude, if any, as a list of strings.
#' Default `NULL` (exclude nothing).
#' @param epiweekly_other Use an expressly epiweekly forecast
#' for non-target ED visits instead of a daily forecast aggregated
#' to epiweekly? Boolean, default `FALSE`.
#' @return The complete hubverse-format [`tibble`][tibble::tibble()].
#' @export
to_epiweekly_quantile_table <- function(model_batch_dir,
                                        exclude = NULL,
                                        epiweekly_other = FALSE) {
  model_runs_path <- fs::path(model_batch_dir, "model_runs")

  locations_to_process <- fs::dir_ls(model_runs_path,
    type = "directory"
  )

  if (!is.null(exclude)) {
    locations_to_process <- locations_to_process[
      !(fs::path_file(locations_to_process) %in% exclude)
    ]
  }

  batch_params <- hewr::parse_model_batch_dir_path(
    model_batch_dir
  )
  report_date <- batch_params$report_date
  disease <- batch_params$disease
  disease_abbr <- dplyr::case_when(
    disease == "Influenza" ~ "flu",
    disease == "COVID-19" ~ "covid",
    TRUE ~ disease
  )

  report_epiweek <- lubridate::epiweek(report_date)
  report_epiyear <- lubridate::epiyear(report_date)
  report_epiweek_end <- forecasttools::epiweek_to_date(
    report_epiweek,
    report_epiyear,
    day_of_week = 7
  )

  hubverse_table <- purrr::map(
    locations_to_process,
    \(x) {
      to_epiweekly_quantiles(
        x,
        report_date = report_date,
        max_lookback_days = 8,
        epiweekly_other = epiweekly_other
      )
    }
    ## max_lookback_days = 8 ensures we get
    ## the full -1 horizon but do not waste
    ## time quantilizing draws that will not
    ## be included in the final table.
  ) |>
    dplyr::bind_rows() |>
    forecasttools::get_hubverse_table(
      report_epiweek_end,
      target_name =
        glue::glue("wk inc {disease_abbr} prop ed visits")
    ) |>
    dplyr::arrange(
      .data$target,
      .data$output_type,
      .data$location,
      .data$reference_date,
      .data$horizon,
      .data$output_type_id
    )

  return(hubverse_table)
}
