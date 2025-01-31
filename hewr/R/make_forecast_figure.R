#' Make Forecast Figure
#'
#' @param target_variable a variable matching the .variable columns
#' in `combined_dat` and `forecast_ci`
#' @param combined_dat `combined_dat` from the result of
#' [process_state_forecast()]
#' @param forecast_ci `forecast_ci` from the result of
#' [process_state_forecast()]
#' @param data_vintage_date date that the data was collected
#' @param y_transform a character passed as the transform argument to
#' [ggplot2::scale_y_continuous()].
#'
#' @return a ggplot object
#' @export
make_forecast_figure <- function(target_variable,
                                 combined_dat,
                                 forecast_ci,
                                 data_vintage_date,
                                 y_transform = "identity") {
  disease_name <- forecast_ci[["disease"]][1]
  disease_name_pretty <- c(
    "COVID-19" = "COVID-19",
    "Influenza" = "Flu"
  )[disease_name]

  state_abb <- unique(combined_dat$geo_value)[1]
  parsed_variable_name <- parse_variable_name(target_variable)

  y_axis_label <- parsed_variable_name[["full_name"]]
  y_axis_labels <- parsed_variable_name[["y_axis_labels"]]
  core_name <- parsed_variable_name[["core_name"]]

  title_prefix <- ifelse(stringr::str_starts(target_variable, "observed"),
    disease_name_pretty, "Other"
  )
  title <- glue::glue("{title_prefix} {core_name} in {state_abb}")

  last_training_date <- combined_dat |>
    dplyr::filter(.data$data_type == "train") |>
    dplyr::pull(date) |>
    max()

  ggplot2::ggplot(mapping = ggplot2::aes(.data$date, .data$.value)) +
    ggdist::geom_lineribbon(
      data = forecast_ci |> dplyr::filter(.data$.variable == target_variable),
      mapping = ggplot2::aes(ymin = .data$.lower, ymax = .data$.upper),
      color = "#08519c",
      key_glyph = ggplot2::draw_key_rect,
      step = "mid"
    ) +
    ggplot2::scale_fill_brewer(
      name = "Credible Interval Width",
      labels = ~ scales::label_percent()(as.numeric(.))
    ) +
    ggplot2::geom_point(
      mapping = ggplot2::aes(color = .data$data_type), size = 1.5,
      data = combined_dat |>
        dplyr::filter(
          .data$.variable == target_variable,
          .data$date <= max(forecast_ci$date)
        ) |>
        dplyr::mutate(data_type = forcats::fct_rev(.data$data_type)) |>
        dplyr::arrange(dplyr::desc(.data$data_type))
    ) +
    ggplot2::scale_color_manual(
      name = "Data Type",
      values = c("olivedrab1", "deeppink"),
      labels = stringr::str_to_title
    ) +
    ggplot2::geom_vline(xintercept = last_training_date, linetype = "dashed") +
    ggplot2::annotate(
      geom = "text",
      x = last_training_date,
      y = -Inf,
      label = "Fit Period \u2190\n",
      hjust = "right",
      vjust = "bottom"
    ) +
    ggplot2::annotate(
      geom = "text",
      x = last_training_date,
      y = -Inf, label = "\u2192 Forecast Period\n",
      hjust = "left",
      vjust = "bottom",
    ) +
    ggplot2::ggtitle(title,
      subtitle = glue::glue("as of {data_vintage_date}")
    ) +
    ggplot2::scale_y_continuous(y_axis_label,
      labels = y_axis_labels,
      transform = y_transform
    ) +
    ggplot2::scale_x_date("Date") +
    cowplot::theme_minimal_grid() +
    ggplot2::theme(legend.position = "bottom")
}
