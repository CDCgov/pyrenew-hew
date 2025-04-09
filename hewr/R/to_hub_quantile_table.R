#' Create a hubverse-format forecast quantile table
#' from a model batch directory containing forecasts
#' for multiple locations as daily MCMC draws.
#'
#' @param model_batch_dir Model batch directory containing
#' the individual location forecast directories
#' ("model run directories") to process. Name should be in the format
#' `{disease}_r_{reference_date}_f_{first_data_date}_t_{last_data_date}`.
#' @return The complete hubverse-format [`tibble`][tibble::tibble()].
#' @export
to_hub_quantile_table <- function(model_batch_dir) {
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

  timely_key <- c("daily" = "days", "epiweekly" = "weeks")

  variable_target_key <- c(
    "observed_hospital_admissions" =
      glue::glue("wk inc {disease_abbr} hosp"),
    "observed_ed_visits" = glue::glue("wk inc {disease_abbr} ed visits"),
    "other_ed_visits" = "wk inc other ed visits",
    "prop_disease_ed_visits" =
      glue::glue("wk inc {disease_abbr} prop ed visits")
  )


  get_location_table <- function(model_run_dir) {
    samples_paths <- fs::dir_ls(model_run_dir,
      recurse = TRUE,
      glob = "*/samples.parquet"
    )
    quantiles_paths <- fs::dir_ls(model_run_dir,
      recurse = TRUE,
      glob = "*_quantiles.parquet"
    )

    quantilized_samples_forecast <- samples_paths |>
      purrr::map(arrow::read_parquet) |>
      purrr::map(\(x) {
        forecasttools::trajectories_to_quantiles(x,
          timepoint_cols = "date",
          value_col = ".value",
          id_cols = setdiff(colnames(x), c(
            "date", ".value", ".draw",
            ".chain", ".iteration"
          ))
        )
      }) |>
      tibble::enframe(name = "file_path", value = "data") |>
      dplyr::mutate(model = file_path |>
        path_dir() |>
        path_file()) |>
      dplyr::mutate(model = dplyr::if_else(model == "timeseries_e",
        "baseline_ts", model
      )) |>
      dplyr::select(model, data)

    quantiles_forecast <- quantiles_paths |>
      purrr::map(\(x) {
        arrow::read_parquet(x) |>
          dplyr::rename("quantile_value" = ".value")
      }) |>
      tibble::enframe(name = "file_path", value = "data") |>
      dplyr::mutate(model = "baseline_cdc") |>
      dplyr::select(model, data)
    forecast_data <-
      dplyr::bind_rows(
        quantilized_samples_forecast,
        quantiles_forecast
      ) |>
      tidyr::unnest(data) |>
      dplyr::mutate(.variable = variable_target_key[.variable]) |>
      dplyr::mutate(reference_date = report_date) |>
      dplyr::mutate(horizon_timescale = timely_key[resolution]) |>
      dplyr::mutate(horizon = forecasttools::horizons_from_target_end_dates(
        reference_date = report_date,
        horizon_timescale = horizon_timescale,
        target_end_dates = date
      )) |>
      dplyr::mutate(
        output_type = "quantile",
        output_type_id = round(.data$quantile_level,
          digits = 4
        )
      ) |>
      dplyr::select(
        model_id = model,
        output_type,
        output_type_id,
        value = quantile_value,
        reference_date,
        target = .variable,
        horizon_timescale = horizon,
        target_end_date = date,
        location = geo_value,
        disease,
        aggregated_numerator,
        aggregated_denominator,
        lab_site_index
      )

    return(forecast_data)
  }



  hubverse_table <- purrr::map(
    fs::dir_ls(model_runs_path), get_location_table
  ) |>
    dplyr::bind_rows()

  return(hubverse_table)
}
