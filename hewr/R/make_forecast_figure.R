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
#' @param display_cutpoints a logical indicating whether to include cutpoints
#' relevant to the `target_variable`.
#' @param highlight_dates a vector of dates to highlight on the plot
#' @param highlight_labels a vector of labels to display at the
#' `highlight_dates`
#'
#' @return a ggplot object
#' @export
make_forecast_figure <- function(target_variable,
                                 combined_dat,
                                 forecast_ci,
                                 data_vintage_date,
                                 y_transform = "identity",
                                 display_cutpoints = TRUE,
                                 highlight_dates = NULL,
                                 highlight_labels = NULL) {
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

  title_prefix <- ifelse(
    stringr::str_starts(target_variable, "other"),
    "Other",
    disease_name_pretty
  )
  title <- glue::glue("{title_prefix} {core_name} in {state_abb}")

  last_training_date <- combined_dat |>
    dplyr::filter(.data$data_type == "train") |>
    dplyr::pull(date) |>
    max()

  lineribbon_dat <- forecast_ci |>
    dplyr::filter(
      .data$.variable == target_variable
    )

  point_dat <- combined_dat |>
    dplyr::filter(
      .data$.variable == target_variable,
      .data$date <= max(forecast_ci$date)
    ) |>
    dplyr::mutate(data_type = forcats::fct_rev(.data$data_type)) |>
    dplyr::arrange(dplyr::desc(.data$data_type))


  if (target_variable == "site_level_log_ww_conc") {
    lineribbon_dat <- lineribbon_dat |>
      dplyr::filter(.data$lab_site_index <= 5)

    point_dat <- point_dat |>
      dplyr::filter(.data$lab_site_index <= 5)

    facet_components <- ggplot2::facet_wrap(~lab_site_index)
  } else {
    facet_components <- list()
  }

  if (display_cutpoints &&
    target_variable == "prop_disease_ed_visits") {
    max_y <- max(lineribbon_dat$.upper, point_dat$.value)

    full_prism_cutpoints <- forecasttools::get_prism_cutpoints(
      state_abb,
      disease_name
    ) |>
      unlist() |>
      utils::head(-1) |>
      utils::tail(-1)


    prism_df <-
      full_prism_cutpoints |>
      tibble::enframe(
        name = "category",
        value = "cutpoint"
      ) |>
      dplyr::mutate("category" = .data$category |>
        stringr::str_remove("^prop_") |>
        stringr::str_replace_all("_", " ") |>
        stringr::str_to_title()) |>
      dplyr::filter(.data$cutpoint <= max_y)

    cutpoint_plot_components <- list(
      ggplot2::geom_hline(
        data = prism_df,
        mapping = ggplot2::aes(
          yintercept = .data$cutpoint,
          color = .data$category
        ),
        linetype = "solid", linewidth = 1
      ),
      forecasttools::scale_color_prism("PRISM Category"),
      ggnewscale::new_scale_color()
    )
  } else {
    cutpoint_plot_components <- list()
  }

  if (!is.null(highlight_dates)) {
    highlight_components <-
      list(
        ggplot2::geom_vline(
          xintercept = highlight_dates,
          linetype = "dashed"
        ),
        ggplot2::annotate(
          geom = "text",
          x = highlight_dates,
          y = -Inf,
          label = stringr::str_c(highlight_labels, "\n"),
          vjust = "bottom"
        )
      )
  } else {
    highlight_components <- list()
  }

  forcast_highlight_components <-
    list(
      ggplot2::geom_vline(
        xintercept = last_training_date,
        linetype = "dashed"
      ),
      ggplot2::annotate(
        geom = "text",
        x = last_training_date,
        y = -Inf,
        label = "Fit Period \u2190\n",
        hjust = "right",
        vjust = "bottom"
      ),
      ggplot2::annotate(
        geom = "text",
        x = last_training_date,
        y = -Inf, label = "\u2192 Forecast Period\n",
        hjust = "left",
        vjust = "bottom",
      )
    )

  ggplot2::ggplot(mapping = ggplot2::aes(.data$date, .data$.value)) +
    cutpoint_plot_components +
    ggdist::geom_lineribbon(
      data = lineribbon_dat,
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
      data = point_dat
    ) +
    ggplot2::scale_color_manual(
      name = "Data Type",
      values = c("olivedrab1", "deeppink"),
      labels = stringr::str_to_title
    ) +
    forcast_highlight_components +
    highlight_components +
    ggplot2::ggtitle(title,
      subtitle = glue::glue("as of {data_vintage_date}")
    ) +
    ggplot2::scale_y_continuous(y_axis_label,
      labels = y_axis_labels,
      transform = y_transform
    ) +
    ggplot2::scale_x_date("Date") +
    facet_components +
    cowplot::theme_minimal_grid() +
    ggplot2::theme(
      legend.position = "bottom",
      legend.direction = "vertical",
      legend.justification = "center"
    )
}
