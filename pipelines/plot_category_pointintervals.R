library(forecasttools)
library(ggplot2)
library(dplyr)

categories <- arrow::read_parquet(path_categories) |>
  transmute(
    disease,
    location = state_abb,
    prop_lower_bound = 0,
    prop_low = perc_level_low / 100,
    prop_moderate = perc_level_moderate / 100,
    prop_high = perc_level_high / 100,
    prop_very_high = perc_level_very_high / 100,
    prop_upper_bound = 1,
    very_low_name = "Very Low",
    low_name = "Low",
    moderate_name = "Moderate",
    high_name = "High",
    very_high_name = "Very High"
  ) |>
  tidyr::nest(
    bin_breaks = c(
      prop_lower_bound,
      prop_low,
      prop_moderate,
      prop_high,
      prop_very_high,
      prop_upper_bound
    ),
    bin_names = c(
      very_low_name,
      low_name,
      moderate_name,
      high_name,
      very_high_name
    )
  )



with_category_cutpoints <- function(df,
                                    disease) {
  with_cutpoints <- df |>
    mutate(disease = !!disease) |>
    inner_join(categories, by = c("location", "disease"))
  return(with_cutpoints)
}

categorize_vec <- function(values, break_sets, label_sets) {
  return(purrr::pmap_vec(
    list(
      x = values,
      breaks = break_sets,
      labels = label_sets,
      include.lowest = TRUE,
      order = TRUE,
      right = TRUE
    ),
    cut
  ))
}

to_categorized_iqr <- function(hub_table,
                               disease,
                               .keep = FALSE) {
  result <- hub_table |>
    pivot_hubverse_quantiles_wider() |>
    with_category_cutpoints(disease = disease) |>
    mutate(
      category_point = categorize_vec(
        .data$point,
        .data$bin_breaks,
        .data$bin_names
      ),
      category_lower = categorize_vec(
        .data$lower,
        .data$bin_breaks,
        .data$bin_names
      ),
      category_upper = categorize_vec(
        .data$upper,
        .data$bin_breaks,
        .data$bin_names
      ),
    )

  if (!.keep) {
    result <- result |> select(-c(bin_breaks, bin_names))
  }

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
    scale_color_prism() +
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
  "2024-11-27_pointinterval_plots.pdf"
)
