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
#' @param disease_name Name of the disease quantity for which to
#' produced epiweekly quantiles. Default `"prop_disease_ed_visits"`.
#' @param draws_file_name Name of the parquet file containing
#' forecast draws. Default `"epiweekly_samples"`.
#' @param disease_model_name Name of the model for the target
#' disease. Default `"pyrenew_e"`.
#' @param strict Boolean. If `TRUE`, raise an error if no
#' valid draws are available to aggregate. Otherwise return
#' `NULL` in that case. Default `FALSE`.
#' @return A [`tibble`][tibble::tibble()] of quantiles, or `NULL`
#' if the draws file contains no vale
#' @export
to_epiweekly_quantiles <- function(model_run_dir,
                                   report_date,
                                   max_lookback_days,
                                   disease_name = "prop_disease_ed_visits",
                                   draws_file_name = "epiweekly_samples",
                                   disease_model_name = "pyrenew_e",
                                   strict = FALSE) {
  message(glue::glue("Processing {model_run_dir}..."))
  draws_path <- fs::path(model_run_dir,
    disease_model_name,
    draws_file_name,
    ext = "parquet"
  )

  location <- fs::path_file(model_run_dir)

  draws <- arrow::read_parquet(draws_path) |>
    dplyr::filter(
      .data$date >= lubridate::ymd(!!report_date) -
        lubridate::days(!!max_lookback_days),
      .data$disease == !!disease_name
    )

  if (nrow(draws) < 1) {
    if (strict) {
      stop(glue::glue(
        "to_epiweekly_quantiles() did not find valid draws for ",
        "{disease_name} to convert to quantiles. It is raising an ",
        "error because `strict` was set to `TRUE`. Looked for draws ",
        "in the parquet file {draws_path}, with report date {report_date}",
        "and max max lookback days {max_lookback_days}."
      ))
    }
    return(NULL)
  }

  epiweekly_quantiles <- draws |>
    forecasttools::trajectories_to_quantiles(
      timepoint_cols = c("epiweek", "epiyear"),
      value_col = ".value"
    ) |>
    dplyr::mutate(
      "location" = !!location,
      "source_samples" = !!draws_file_name
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
#' @param epiweekly_other_locations Use an expressly epiweekly forecast
#' for non-target ED visits instead of a daily forecast aggregated
#' to epiweekly for the specified locations. Default `c()` (Use a
#' an aggregated daily other forecast for all.
#' @param strict Boolean. If `TRUE`, raise an error if no
#' valid draws are available to aggregate for any given location.
#' Otherwise return `NULL` for that location but continue with other.
#' locations. Passed as the `strict` argument to [to_epiweekly_quantiles()].
#' Default `FALSE`.
#' @return The complete hubverse-format [`tibble`][tibble::tibble()].
#' @export
to_epiweekly_quantile_table <- function(model_batch_dir,
                                        exclude = NULL,
                                        strict = FALSE,
                                        epiweekly_other_locations = c()) {
  model_runs_path <- fs::path(model_batch_dir, "model_runs")

  model_run_dirs_to_process <- fs::dir_ls(model_runs_path,
    type = "directory"
  ) |>
    purrr::discard(~ fs::path_file(.x) %in% exclude)

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

  get_location_table <- \(model_run_dir) {
    loc <- fs::path_file(model_run_dir)
    use_epiweekly_other <- loc %in% epiweekly_other_locations
    if (use_epiweekly_other) {
      message(glue::glue(
        "Using epiweekly non-target ED visit forecast ",
        "for location {loc}"
      ))
    } else {
      message(glue::glue(
        "Using daily non-target ED visit ",
        "forecast for location {loc}."
      ))
    }
    draws_file <- ifelse(
      use_epiweekly_other,
      "epiweekly_with_epiweekly_other_samples",
      "epiweekly_samples"
    )
    return(to_epiweekly_quantiles(
      model_run_dir,
      report_date = report_date,
      max_lookback_days = 15,
      draws_file_name = draws_file,
      strict = strict
    ))
  }

  quant_table <- purrr::map(
    model_run_dirs_to_process,
    get_location_table
  ) |>
    dplyr::bind_rows()

  loc_sources <- quant_table |>
    dplyr::distinct(.data$location, .data$source_samples)

  hubverse_table <- quant_table |>
    forecasttools::get_hubverse_table(
      report_epiweek_end,
      target_name =
        glue::glue("wk inc {disease_abbr} prop ed visits")
    ) |>
    dplyr::inner_join(loc_sources,
      by = "location"
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
