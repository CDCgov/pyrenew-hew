#' Create an epiweekly hubverse-format forecast quantile table
#' from a model batch directory containing forecasts
#' for multiple locations as daily MCMC draws.
#'
#' @param model_batch_dir Model batch directory containing
#' the individual location forecast directories
#' ("model run directories") to process. Name should be in the format
#' `{disease}_r_{reference_date}_f_{first_data_date}_t_{last_data_date}`.
#' @return The complete hubverse-format [`tibble`][tibble::tibble()].
#' @export
to_epiweekly_quantile_table <- function(model_batch_dir) {
  model_runs_path <- fs::path(model_batch_dir, "model_runs")

  batch_params <- parse_model_batch_dir_path(
    model_batch_dir
  )
  report_date <- batch_params$report_date
  disease <- batch_params$disease
  last_training_date <- batch_params$last_training_date
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

  process_posterior_for_table <- function(file) {
    arrow::read_parquet(file) |>
      dplyr::filter(.data$date > !!last_training_date) |>
      dplyr::rename(location = "geo_value") |>
      dplyr::mutate(
        epiweek = lubridate::epiweek(.data$date),
        epiyear = lubridate::epiyear(.data$date)
      )
  }

  get_location_table <- function(model_run_dir) {
    samples_paths <- fs::dir_ls(model_run_dir,
      recurse = TRUE,
      glob = "*_samples.parquet"
    )
    quantiles_paths <- fs::dir_ls(model_run_dir,
      recurse = TRUE,
      glob = "*_quantiles.parquet"
    )

    quantilized_samples_forecast <- samples_paths |>
      purrr::map(process_posterior_for_table) |>
      purrr::map(\(x) {
        forecasttools::trajectories_to_quantiles(x,
          timepoint_cols = "date",
          value_col = ".value",
          id_cols = c("location", "disease", ".variable", "epiweek", "epiyear")
        )
      })

    quantiles_forecast <- quantiles_paths |>
      purrr::map(process_posterior_for_table) |>
      purrr::map(\(x) dplyr::rename(x, "quantile_value" = .value))

    scorable_datasets <-
      tibble::tibble(
        file_path = c(samples_paths, quantiles_paths),
        forecast_data = c(quantilized_samples_forecast, quantiles_forecast)
      ) |>
      dplyr::mutate(
        forecast_name = .data$file_path |>
          fs::path_file() |>
          fs::path_ext_remove() |>
          stringr::str_remove("_([^_]*)$"),
        forecast_type = .data$file_path |>
          fs::path_file() |>
          fs::path_ext_remove() |>
          stringr::str_extract("(?<=_)([^_]*)$"),
        resolution = .data$file_path |>
          fs::path_file() |>
          fs::path_ext_remove() |>
          stringr::str_extract("^.+?(?=_)"),
        model_name = .data$file_path |>
          fs::path_dir() |>
          fs::path_file()
      ) |>
      tidyr::unite("model", "model_name", "forecast_name", sep = "_") |>
      dplyr::select(-"file_path")

    loc_epiweekly_hubverse_table <-
      scorable_datasets |>
      dplyr::filter(.data$resolution == "epiweekly") |>
      dplyr::mutate(hubverse_data = purrr::map(.data$forecast_data, \(x) {
        x |>
          dplyr::filter(.data$.variable == "prop_disease_ed_visits") |>
          forecasttools::get_hubverse_table(
            report_epiweek_end,
            target_name =
              glue::glue("wk inc {disease_abbr} prop ed visits")
          )
      })) |>
      dplyr::select(-"forecast_data") |>
      tidyr::unnest("hubverse_data")

    return(loc_epiweekly_hubverse_table)
  }



  hubverse_table <- purrr::map(
    fs::dir_ls(model_runs_path), get_location_table
  ) |>
    dplyr::bind_rows()

  return(hubverse_table)
}
