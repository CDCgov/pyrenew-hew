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


main <- function(influenza_table_path,
                 covid_table_path,
                 categories_path,
                 output_path,
                 ...) {
  flu_dat <- readr::read_tsv(influenza_table_path) |>
    to_categorized_iqr("Influenza")

  covid_dat <- readr::read_tsv(covid_table_path) |>
    to_categorized_iqr("COVID-19")

  plots <- list(
    flu_plot_1wk = flu_dat |>
      plot_category_pointintervals(horizon = 0) +
      ggtitle("Influenza, 1 week ahead"),
    flu_plot_2wk = flu_dat |>
      plot_category_pointintervals(horizon = 1) +
      labs(x = "% ED visits") +
      ggtitle("Influenza, 2 weeks ahead"),
    covid_plot_1wk = covid_dat |>
      plot_category_pointintervals(horizon = 0) +
      labs(x = "% ED visits") +
      ggtitle("COVID-19, 1 week ahead"),
    covid_plot_2wk = covid_dat |>
      plot_category_pointintervals(horizon = 1) +
      labs(x = "% ED visits") +
      ggtitle("COVID-19, 2 weeks ahead")
  )


  plots_to_pdf(
    plots,
    output_path
  )
}


p <- arg_parser("Create a pointinterval plot of forecasts") |>
  add_argument(
    "influenza_table_path",
    help = "Path to a hubverse format forecast table for influenza."
  ) |>
  add_argument(
    "covid_table_path",
    help = "Path to a hubverse format forecast table for COVID-19."
  ) |>
  add_argument(
    "output_path",
    help = "Path to save the output plots, as a single PDF"
  )


argv <- parse_args(p)

do.call(main, argv)
