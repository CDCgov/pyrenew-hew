library(forecasttools)
library(ggplot2)
library(dplyr)
library(argparser)


to_categorized_iqr <- function(hub_table,
                               disease) {
  result <- hub_table |>
    pivot_hubverse_quantiles_wider() |>
    mutate(
      across(c("point", "lower", "upper"),
        ~ forecasttools::categorize_prism(
          .x,
          .data$location,
          !!disease
        ),
        .names = "category_{.col}"
      )
    )
  return(result)
}

plot_category_pointintervals <- function(data, horizon) {
  plot <- data |>
    filter(.data$horizon == !!horizon) |>
    arrange(point) |>
    mutate("location" = factor(.data$location,
      levels = unique(.data$location),
      ordered = TRUE
    )) |>
    ggplot(aes(
      y = location,
      x = point,
      xmin = lower,
      xmax = upper
    )) +
    ggdist::geom_pointinterval() +
    geom_point(
      aes(
        x = lower,
        color = category_lower
      ),
      size = 3,
      show.legend = TRUE
    ) +
    geom_point(
      aes(
        x = upper,
        color = category_upper
      ),
      size = 3,
      show.legend = TRUE
    ) +
    geom_point(
      aes(
        x = point,
        color = category_point
      ),
      size = 5,
      show.legend = TRUE
    ) +
    scale_x_continuous(label = scales::label_percent()) +
    scale_color_prism(drop = FALSE) +
    labs(color = "Activity Level") +
    theme_minimal()

  return(plot)
}


main <- function(hubverse_table_path,
                 disease,
                 output_path,
                 ...) {
  checkmate::check_names(disease,
    subset.of = c("COVID-19", "Influenza")
  )

  dat <- readr::read_tsv(hubverse_table_path) |>
    to_categorized_iqr(disease)

  plots <- list(
    plot_1wk = dat |>
      plot_category_pointintervals(horizon = 0) +
      labs(
        x = "% ED visits",
        y = "Location"
      ) +
      ggtitle(glue::glue("{disease}, 1 week ahead")),
    plot_2wk = dat |>
      plot_category_pointintervals(horizon = 1) +
      labs(
        x = "% ED visits",
        y = "Location"
      ) +
      ggtitle(glue::glue("{disease}, 2 weeks ahead"))
  )


  plots_to_pdf(
    plots,
    output_path
  )
}


p <- arg_parser("Create a pointinterval plot of forecasts") |>
  add_argument(
    "hubverse_table_path",
    help = "Path to a hubverse format forecast table."
  ) |>
  add_argument(
    "disease",
    help = paste0(
      "Name of the disease to plot. ",
      "One of 'COVID-19', 'Influenza'"
    )
  ) |>
  add_argument(
    "output_path",
    help = "Path to save the output plots, as a single PDF"
  )


argv <- parse_args(p)

do.call(main, argv)
