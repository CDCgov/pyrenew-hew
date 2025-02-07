library(forecasttools)
library(ggplot2)
library(dplyr)
library(argparser)
library(fs)
library(hewr)
library(tidyr)
library(purrr)
library(glue)

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
    filter(
      .data$horizon == !!horizon,
      stringr::str_detect(.data$target, "prop")
    ) |>
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
                 output_path,
                 ...) {
  disease <- parse_model_batch_dir_path(path_dir(hubverse_table_path))$disease

  dat <- arrow::read_parquet(hubverse_table_path)

  if (!(".variable" %in% colnames(dat)) ||
    !("prop_disease_ed_visits" %in% dat[[".variable"]])) {
    warning("Input hubverse table must contain a .variable column with
         prop_disease_ed_visits")
  } else {
    dat <- to_categorized_iqr(dat, disease)

    figure_tbl <-
      dat |>
      distinct(model) |>
      expand_grid(horizon = c(0, 1)) |>
      mutate(figure = map2(
        model, horizon,
        \(target_model, horizon) {
          plot_category_pointintervals(
            dat |>
              filter(model == target_model),
            horizon = horizon
          ) +
            labs(
              x = "% ED visits",
              y = "Location"
            ) +
            ggtitle(glue::glue("{disease}, {target_model}"),
              subtitle = glue("{horizon} week ahead")
            )
        }
      ))

    plots_to_pdf(
      figure_tbl$figure,
      output_path
    )
  }
}


p <- arg_parser("Create a pointinterval plot of forecasts") |>
  add_argument(
    "hubverse_table_path",
    help = "Path to a hubverse format forecast table."
  ) |>
  add_argument(
    "output_path",
    help = "Path to save the output plots, as a single PDF"
  )


argv <- parse_args(p)

do.call(main, argv)
