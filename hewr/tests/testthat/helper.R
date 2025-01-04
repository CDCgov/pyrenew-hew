create_tidy_forecast_data <- function(directory,
                                      filename,
                                      date_cols,
                                      disease_cols,
                                      n_draw,
                                      with_epiweek = FALSE) {
  data <- tidyr::expand_grid(
    date = date_cols,
    disease = disease_cols,
    .draw = 1:n_draw
  ) |>
    dplyr::mutate(.value = sample(1:100, dplyr::n(), replace = TRUE))
  if (length(disease_cols) == 1) {
    data <- data |>
      dplyr::rename(!!disease_cols := ".value") |>
      dplyr::select(-disease)
  }

  if (with_epiweek) {
    data <- data |>
      dplyr::mutate(
        epiweek = lubridate::epiweek(.data$date),
        epiyear = lubridate::epiyear(.data$date)
      )
  }

  arrow::write_parquet(data, fs::path(directory, filename))
}

create_observation_data <- function(
    date_range, locations) {
  data <- tidyr::expand_grid(
    reference_date = date_range,
    location = locations
  ) |>
    dplyr::mutate(value = sample(1:100, dplyr::n(), replace = TRUE))
  return(data)
}

create_hubverse_table <- function(
    date_range, horizons, locations, output_type, output_type_id) {
  data <- tidyr::expand_grid(
    reference_date = date_range,
    horizon = horizons,
    output_type_id = output_type_id,
    location = locations
  ) |>
    dplyr::group_by(
      reference_date,
      horizon,
      location
    ) |>
    dplyr::arrange(output_type_id,
      .by_group = TRUE
    ) |>
    dplyr::mutate(
      value = sort(
        sample(1:1000, dplyr::n(), replace = TRUE),
        decreasing = FALSE
      ),
      target = "wk inc covid prop ed visits",
      output_type = !!output_type,
      target_end_date = reference_date + 7 * horizon
    ) |>
    dplyr::ungroup()

  return(data)
}
