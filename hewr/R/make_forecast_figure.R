#' Make Forecast Figure
#'
#' @param target_disease a disease matching the disease columns
#' in `combined_dat` and `forecast_ci`
#' @param combined_dat `combined_dat` from the result of
#' [process_state_forecast()]
#' @param forecast_ci `forecast_ci` from the result of
#' [process_state_forecast()]
#' @param disease_name `"COVID-19"` or `"Influenza"`
#' @param data_vintage_date date that the data was collected
#' @param y_transform a character passed as the transform argument to
#' [ggplot2::scale_y_continuous()].
#'
#' @return a ggplot object
#' @export
make_forecast_figure <- function(target_disease,
                                 combined_dat,
                                 forecast_ci,
                                 disease_name = c("COVID-19", "Influenza"),
                                 data_vintage_date,
                                 y_transform = "identity") {
  disease_name <- rlang::arg_match(disease_name)
  target_variable <- c(
    "Disease" = "observed_ed_visits",
    "Other" = "other_ed_visits",
    "prop_disease_ed_visits" = "prop_disease_ed_visits"
  )[target_disease]

  disease_name_pretty <- c(
    "COVID-19" = "COVID-19",
    "Influenza" = "Flu"
  )[disease_name]
  state_abb <- unique(combined_dat$geo_value)[1]

  y_scale <- if (stringr::str_starts(target_disease, "prop")) {
    ggplot2::scale_y_continuous("Proportion of Emergency Department Visits",
      labels = scales::label_percent(),
      transform = y_transform
    )
  } else {
    ggplot2::scale_y_continuous("Emergency Department Visits",
      labels = scales::label_comma(),
      transform = y_transform
    )
  }


  title <- if (target_disease == "Other") {
    glue::glue("Other ED Visits in {state_abb}")
  } else {
    glue::glue("{disease_name_pretty} ED Visits in {state_abb}")
  }

  last_training_date <- combined_dat |>
    dplyr::filter(data_type == "train") |>
    dplyr::pull(date) |>
    max()

  ggplot2::ggplot(mapping = ggplot2::aes(date, .value)) +
    ggdist::geom_lineribbon(
      data = forecast_ci |> dplyr::filter(.variable == target_variable),
      mapping = ggplot2::aes(ymin = .lower, ymax = .upper),
      color = "#08519c",
      key_glyph = ggplot2::draw_key_rect,
      step = "mid"
    ) +
    ggplot2::scale_fill_brewer(
      name = "Credible Interval Width",
      labels = ~ scales::label_percent()(as.numeric(.))
    ) +
    ggplot2::geom_point(
      mapping = ggplot2::aes(color = data_type), size = 1.5,
      data = combined_dat |>
        dplyr::filter(
          disease == target_disease,
          date <= max(forecast_ci$date)
        ) |>
        dplyr::mutate(data_type = forcats::fct_rev(data_type)) |>
        dplyr::arrange(dplyr::desc(data_type))
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
    y_scale +
    ggplot2::scale_x_date("Date") +
    cowplot::theme_minimal_grid() +
    ggplot2::theme(legend.position = "bottom")
}
