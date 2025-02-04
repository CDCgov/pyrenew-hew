create_model_results <- function(file,
                                 date_options,
                                 geo_value_options,
                                 disease_options,
                                 variable_options,
                                 n_draw) {
  data <- tidyr::expand_grid(
    .draw = 1:n_draw,
    date = date_options,
    geo_value = geo_value_options,
    disease = disease_options,
    .variable = variable_options
  ) |>
    dplyr::mutate(.value = as.double(rpois(dplyr::n(), lambda = 100)))

  arrow::write_parquet(data, file)
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
