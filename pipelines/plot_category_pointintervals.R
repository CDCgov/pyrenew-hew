library(tidyverse)
library(arrow)
library(forecasttools)
library(fs)
library(hewr)
library(glue)
library(argparser)

plot_category_pointintervals <- function(target_model,
                                         target_horizon,
                                         data_slice,
                                         report_date) {
  target_disease <- unique(data_slice[["disease"]])
  target_end_date <- unique(data_slice[["target_end_date"]])

  figure <- data_slice |>
    arrange(value) |>
    ggplot(aes(y = location)) +
    ggdist::geom_pointinterval(aes(x = value, xmin = .lower, xmax = .upper)) +
    geom_point(aes(x = .lower, color = category_.lower),
      size = 3,
      show.legend = TRUE
    ) +
    geom_point(aes(x = .upper, color = category_.upper),
      size = 3,
      show.legend = TRUE
    ) +
    geom_point(aes(x = value, color = category_value),
      size = 5,
      show.legend = TRUE
    ) +
    scale_x_continuous("Proportion ED Visists with Disease",
      label = scales::label_percent()
    ) +
    scale_y_discrete("Location") +
    scale_color_prism("Activity Level", drop = FALSE) +
    scale_size(guide = "none") +
    ggtitle(glue::glue("{target_disease}, {target_model}"),
      subtitle = glue(
        "Report Date: {report_date}, ",
        "Target: {target_horizon} week ahead ({target_end_date})"
      )
    ) +
    theme_minimal()

  return(figure)
}


main <- function(hubverse_table_path, output_path) {
  model_details <- hubverse_table_path |>
    path_dir() |>
    hewr::parse_model_batch_dir_path()

  report_date <- model_details$report_date

  hub_table <- read_parquet(hubverse_table_path) |>
    filter(
      resolution == "epiweekly",
      str_detect(target, "prop ed visits")
    ) |>
    modify_reference_date(ceiling_mmwr_epiweek,
      horizon_timescale = "weeks"
    ) |>
    filter(horizon %in% 0:1) |>
    forecasttools::hub_quantiles_to_median_qi(.width = 0.95) |>
    rename(value = x) |>
    mutate(across(c(value, .lower, .upper),
      \(x) {
        categorize_prism(diseases = disease, locations = location, values = x)
      },
      .names = "category_{.col}"
    )) |>
    group_by(model_id, horizon) |>
    nest()

  figures <- pmap(
    hub_table,
    \(model_id, horizon, data) {
      plot_category_pointintervals(
        model_id,
        horizon, data, report_date
      )
    }
  )

  plots_to_pdf(figures, output_path)
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
