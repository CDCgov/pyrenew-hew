library(tidyverse)
library(fs)
library(arrow)
library(hewr)
library(forecasttools)
pyrenew_hew_config_dir <- path("private_data/pyrenew-hew-config")
prism_thresholds <- read_parquet(path(pyrenew_hew_config_dir, "prism_thresholds", ext = "parquet")) # nolint
model_run_dir <- path("~/pyrenew-hew/private_data/pyrenew-test-output/influenza_r_2024-11-27_f_2024-08-24_t_2024-11-21/model_runs/CA") # nolint

parsed_model_run_dir <- parse_model_run_dir_path(model_run_dir)


disease_name <- parsed_model_run_dir$disease
data_vintage_date <- parsed_model_run_dir$report_date

target_disease <- "prop_disease_ed_visits"
forecast_ci <- read_parquet(path(model_run_dir, "forecast_ci", ext = "parquet"))
combined_dat <- read_parquet(path(model_run_dir, "combined_training_eval_data",
  ext = "parquet"
))

y_transform <- "identity"


horizon_weeks <- 1
highlight_forecast_date <- epiweek_to_date(
  epiweek = epiweek(data_vintage_date),
  epiyear = epiyear(data_vintage_date),
  day_of_week = 7
) + horizon_weeks * 7

prism_color_scale <- c(
  Minimal = "#D7F2ED", Low = "#B8E5AC", Moderate = "#FEA82F",
  High = "#F05C54", `Very High` = "#A03169", `Data Unavailable` = "#EBEBEB"
)


get_prism_thresholds <- function(disease, state_abb) {
  prism_thresholds |>
    filter(
      disease == !!disease,
      state_abb == !!state_abb
    ) |>
    select(starts_with("perc_")) |>
    pivot_longer(everything()) |>
    mutate(value = value / 100) |>
    mutate(name = name |>
      str_remove("^perc_level_") |>
      str_replace_all("_", " ") |>
      str_to_title()) |>
    deframe()
}


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

state_prism_thresholds <- get_prism_thresholds(disease_name, state_abb)

ggplot2::ggplot(mapping = ggplot2::aes(date, .value)) +
  ggdist::geom_lineribbon(
    data = forecast_ci |> dplyr::filter(disease == target_disease),
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
    y = Inf,
    label = "\nFit Period \u2190",
    hjust = "right",
    vjust = "top"
  ) +
  ggplot2::annotate(
    geom = "text",
    x = last_training_date,
    y = Inf, label = "\n\u2192 Forecast Period",
    hjust = "left",
    vjust = "top",
  ) +
  ggplot2::ggtitle(title,
    subtitle = glue::glue("as of {data_vintage_date}")
  ) +
  y_scale +
  ggplot2::scale_x_date("Date") +
  cowplot::theme_minimal_grid() +
  ggplot2::theme(legend.position = "bottom") +
  geom_hline(
    yintercept = state_prism_thresholds,
    color = prism_color_scale[names(state_prism_thresholds)]
  ) +
  annotate(
    geom = "text",
    x = min(combined_dat$date),
    y = state_prism_thresholds,
    label = names(state_prism_thresholds),
    hjust = "left",
    vjust = "bottom",
    color = prism_color_scale[names(state_prism_thresholds)]
  ) +
  geom_vline(xintercept = highlight_forecast_date) +
  annotate(
    geom = "text",
    x = highlight_forecast_date,
    y = Inf,
    label = "\n\nTarget Forecast Horizon",
    vjust = "top"
  )
