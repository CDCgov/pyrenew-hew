library(forecasttools)
library(ggplot2)
library(dplyr)

categories <- arrow::read_parquet(path_categories) |>
  transmute(
    disease,
    location = state_abb,
    prop_low = perc_level_low / 100,
    prop_moderate = perc_level_moderate / 100,
    prop_high = perc_level_high / 100,
    prop_very_high = perc_level_very_high / 100
  ) |>
  tidyr::nest(
    bin_breaks = c(
      prop_low,
      prop_moderate,
      prop_high,
      prop_very_high
    ),
    bin_names = c(
      "very low", "low", "moderate", "high"
    )
  )



with_category_cutpoints <- function(df,
                                    disease,
                                    location_column = "location") {
  with_cutpoints <- df |>
    mutate(disease = !!disease) |>
    inner_join(categories, by = c(
      !!location_column == "location",
      "disease"
    ))
  return(with_cutpoints)
}

to_categorized_iqr <- function(hub_table, disease) {
  result <- hub_table |>
    pivot_hubverse_quantiles_wider() |>
    with_category_cutpoints(disease = disease) |>
    mutate(
      category_point = cut(.data$point,
        breaks = .data$bin_breaks,
        labels = .data$bin_names,
        include.lowest = TRUE
      ),
      category_lower = cut(.data$lower,
        breaks = .data$bin_breaks,
        labels = .data$bin_names,
        include.lowest = TRUE
      ),
      category_upper = cut(.data$upper,
        breaks = .data$bin_breaks,
        labels = .data$bin_names,
        include.lowest = TRUE
      )
    )

  return(result)
}

category_pointinterval_plot <- function(data, horizon) {
  plot <- data |>
    filter(.data$horizon == !!horizon) |>
    arrange(point) |>
    mutate("location" = factor(.data$location,
      levels = unique(.data$location),
      ordered = TRUE
    )) |>
    ggplot(aes(
      y = location,
      x = 100 * point,
      xmin = 100 * lower,
      xmax = 100 * upper
    )) +
    ggdist::geom_pointinterval() +
    geom_point(
      aes(
        x = 100 * lower,
        color = category_lower
      ),
      size = 3,
      show.legend = TRUE
    ) +
    geom_point(
      aes(
        x = 100 * upper,
        color = category_upper
      ),
      size = 3,
      show.legend = TRUE
    ) +
    geom_point(
      aes(
        x = 100 * point,
        color = category_point
      ),
      size = 5,
      show.legend = TRUE
    ) +
    scale_color_manual(
      values = list(
        "very low" = "#d3ecea",
        "low" = "#baddab",
        "moderate" = "#faa731",
        "high" = "#f15d54",
        "very high" = "#a03169"
      ),
      breaks = c(
        "very low",
        "low",
        "moderate",
        "high",
        "very high"
      ),
      drop = FALSE
    ) +
    theme_minimal()

  return(plot)
}

flu_dat <- readr::read_tsv(path_flu) |>
  to_categorized_iqr("Influenza")

covid_dat <- readr::read_tsv(path_covid) |>
  to_categorized_iqr("COVID-19")


flu_plot_1wk <- flu_dat |>
  category_pointinterval_plot(horizon = 0) +
  labs(x = "% ED visits") +
  ggtitle("Influenza, 1 week ahead")
flu_plot_2wk <- flu_dat |>
  category_pointinterval_plot(horizon = 1) +
  labs(x = "% ED visits") +
  ggtitle("Influenza, 2 weeks ahead")

covid_plot_1wk <- covid_dat |>
  category_pointinterval_plot(horizon = 0) +
  labs(x = "% ED visits") +
  ggtitle("COVID-19, 1 week ahead")

covid_plot_2wk <- covid_dat |>
  category_pointinterval_plot(horizon = 1) +
  labs(x = "% ED visits") +
  ggtitle("COVID-19, 2 weeks ahead")


plots_to_pdf(
  list(
    flu_plot_1wk,
    flu_plot_2wk,
    covid_plot_1wk,
    covid_plot_2wk
  ),
  "2024-11-20_pointinterval_plots.pdf"
)
