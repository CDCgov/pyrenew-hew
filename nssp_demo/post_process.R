library(tidyverse)
library(tidybayes)
library(fs)
library(cowplot)
library(glue)
library(scales)
library(here)
library(argparser)
library(arrow)

theme_set(theme_minimal_grid())

disease_name_formatter <- c("covid-19" = "COVID-19", "influenza" = "Flu")
disease_name_nssp_map <- c(
  "covid-19" = "COVID-19/Omicron",
  "influenza" = "Influenza"
)

# Create a parser
p <- arg_parser("Generate forecast figures") %>%
  add_argument(p, "--model_dir",
    help = "Directory containing the model data",
    required = TRUE
  ) %>%
  add_argument(p, "--filter_bad_chains",
    help = "Filter out bad chains from the samples",
    flag = TRUE
  ) %>%
  add_argument(p, "--good_chain_tol",
    help = "Tolerance level for determining good chains",
    default = 2L
  )

argv <- parse_args(p)
model_dir <- path(argv$model_dir)
filter_bad_chains <- argv$filter_bad_chains
good_chain_tol <- argv$good_chain_tol

base_dir <- path_dir(model_dir)

disease_name_raw <- base_dir %>%
  path_file() %>%
  str_extract("^.+(?=_r_)")

disease_name_nssp <- unname(disease_name_nssp_map[disease_name_raw])
disease_name_pretty <- unname(disease_name_formatter[disease_name_raw])

# To be replaced with reading tidy data from forecasttools
read_pyrenew_samples <- function(inference_data_path,
                                 filter_bad_chains = TRUE,
                                 good_chain_tol = 2) {
  arviz_split <- function(x) {
    x %>%
      select(-distribution) %>%
      split(f = as.factor(x$distribution))
  }

  pyrenew_samples <-
    read_csv(inference_data_path) %>%
    rename_with(\(varname) str_remove_all(varname, "\\(|\\)|\\'|(, \\d+)")) |>
    rename(
      .chain = chain,
      .iteration = draw
    ) |>
    mutate(across(c(.chain, .iteration), \(x) as.integer(x + 1))) |>
    mutate(
      .draw = tidybayes:::draw_from_chain_and_iteration_(.chain, .iteration),
      .after = .iteration
    ) |>
    pivot_longer(-starts_with("."),
      names_sep = ", ",
      names_to = c("distribution", "name")
    ) |>
    arviz_split() |>
    map(\(x) pivot_wider(x, names_from = name) |> tidy_draws())

  if (filter_bad_chains) {
    good_chains <-
      pyrenew_samples$log_likelihood %>%
      pivot_longer(-starts_with(".")) %>%
      group_by(.iteration, .chain) %>%
      summarize(value = sum(value)) %>%
      group_by(.chain) %>%
      summarize(value = mean(value)) %>%
      filter(value >= max(value) - 2) %>%
      pull(.chain)
  } else {
    good_chains <- unique(pyrenew_samples$log_likelihood$.chain)
  }

  good_pyrenew_samples <- map(
    pyrenew_samples,
    \(x) filter(x, .chain %in% good_chains)
  )
  good_pyrenew_samples
}

make_one_forecast_fig <- function(target_disease,
                                  dat,
                                  last_training_date,
                                  last_data_date,
                                  posterior_predictive_ci,
                                  state_abb) {
  y_scale <- if (str_starts(target_disease, "prop")) {
    scale_y_continuous("Proportion of Emergency Department Admissions",
      labels = percent
    )
  } else {
    scale_y_continuous("Emergency Department Admissions", labels = comma)
  }

  title <- if (target_disease == "Total") {
    glue("Total ED Admissions in {state_abb}")
  } else {
    glue("{disease_name_pretty} ED Admissions in {state_abb}")
  }

  ggplot(mapping = aes(date, .value)) +
    geom_lineribbon(
      data = posterior_predictive_ci %>% filter(disease == target_disease),
      mapping = aes(ymin = .lower, ymax = .upper),
      color = "#08519c", key_glyph = draw_key_rect, step = "mid"
    ) +
    geom_point(
      mapping = aes(shape = data_type),
      data = dat %>% filter(disease == target_disease)
    ) +
    geom_vline(xintercept = last_training_date, linetype = "dashed") +
    annotate(
      geom = "text",
      x = last_training_date,
      y = -Inf,
      label = "Fit Period ←\n",
      hjust = "right",
      vjust = "bottom"
    ) +
    annotate(
      geom = "text",
      x = last_training_date,
      y = -Inf, label = "→ Forecast Period\n",
      hjust = "left",
      vjust = "bottom",
    ) +
    ggtitle(title, subtitle = glue("as of {last_data_date}")) +
    y_scale +
    scale_x_date("Date") +
    scale_shape_discrete("Data Type", labels = str_to_title) +
    scale_fill_brewer(
      name = "Credible Interval Width",
      labels = ~ percent(as.numeric(.))
    ) +
    theme(legend.position = "bottom")
}


make_forecast_figs <- function(model_dir,
                               filter_bad_chains = TRUE,
                               good_chain_tol = 2) {
  state_abb <- model_dir %>%
    path_split() %>%
    pluck(1) %>%
    tail(1)

  data_path <- path(model_dir, "data", ext = "csv")
  inference_data_path <- path(model_dir, "inference_data",
    ext = "csv"
  )
  total_ed_admissions_path <- path(model_dir, "other_ed_admissions_forecast",
    ext = "parquet"
  )

  dat <-
    read_csv(data_path) %>%
    mutate(disease = if_else(disease == disease_name_nssp,
      "Disease", # assign a common name for use in plotting functions
      disease
    )) %>%
    pivot_wider(names_from = disease, values_from = ED_admissions) %>%
    mutate(prop_disease_ed_admissions = Disease / (Disease + Total)) %>%
    mutate(time = dense_rank(date)) %>%
    pivot_longer(c(Total, Disease, prop_disease_ed_admissions),
      names_to = "disease",
      values_to = ".value"
    )

  last_training_date <- dat %>%
    filter(data_type == "train") %>%
    pull(date) %>%
    max()

  last_data_date <- dat %>%
    pull(date) %>%
    max()

  pyrenew_samples <- read_pyrenew_samples(inference_data_path,
    filter_bad_chains = filter_bad_chains,
    good_chain_tol = good_chain_tol
  )

  total_ed_admission_forecast <-
    read_parquet(total_ed_admissions_path) %>%
    rename(Total = total_ED_admissions)


  total_ed_admission_samples <-
    bind_rows(
      dat %>%
        filter(
          disease == "Total",
          date <= last_training_date
        ) %>%
        select(date, Total = .value) %>%
        expand_grid(.draw = 1:max(total_ed_admission_forecast$.draw)),
      total_ed_admission_forecast
    )

  posterior_predictive_samples <-
    pyrenew_samples$posterior_predictive %>%
    gather_draws(observed_hospital_admissions[time]) %>%
    pivot_wider(names_from = .variable, values_from = .value) %>%
    rename(Disease = observed_hospital_admissions) %>%
    ungroup() %>%
    mutate(date = min(dat$date) + time) %>%
    left_join(total_ed_admission_samples) %>%
    mutate(prop_disease_ed_admissions = Disease / Total) %>%
    pivot_longer(c(Total, Disease, prop_disease_ed_admissions),
      names_to = "disease",
      values_to = ".value"
    )

  posterior_predictive_ci <-
    posterior_predictive_samples %>%
    select(date, disease, .value) %>%
    group_by(date, disease) %>%
    median_qi(.width = c(0.5, 0.8, 0.95))


  all_forecast_plots <- map(
    set_names(unique(dat$disease)),
    ~ make_one_forecast_fig(
      .x,
      dat,
      last_training_date,
      last_data_date,
      posterior_predictive_ci,
      state_abb
    )
  )

  all_forecast_plots
}

forecast_figs <- make_forecast_figs(
  model_dir,
  filter_bad_chains,
  good_chain_tol
)

iwalk(forecast_figs, ~ save_plot(
  filename = path(model_dir, glue("{.y}_forecast_plot"), ext = "pdf"),
  plot = .x,
  device = cairo_pdf, base_height = 6
))


# File will end here once command line version is working
# Temp code to run for all states while command line version doesn't work
# Command line version is dependent on https://github.com/rstudio/renv/pull/2018
base_dir <- path(
  "nssp_demo",
  "private_data",
  "influenza_r_2024-10-21_f_2024-07-16_t_2024-10-13"
)

# Save all figures for each state
walk(dir_ls(base_dir, type = "dir"), function(model_dir) {
  print(model_dir)
  forecast_figs <- make_forecast_figs(model_dir,
    filter_bad_chains = TRUE,
    good_chain_tol = 2
  )

  iwalk(forecast_figs, ~ save_plot(
    filename = path(model_dir, glue("{.y}_forecast_plot"), ext = "pdf"),
    plot = .x,
    device = cairo_pdf, base_height = 6
  ))
})


# Combine figures across states
tibble(
  full_path = dir_ls(base_dir,
    type = "file",
    glob = "*_forecast_plot.pdf",
    recurse = TRUE
  ),
  plot_type = path_file(full_path)
) %>%
  group_by(plot_type) %>%
  summarize(all_fig_paths = str_c(full_path, collapse = " ")) %>%
  mutate(combined_plot_path = path(
    base_dir,
    glue("{path_file(base_dir)}_{plot_type}")
  )) %>%
  select(-plot_type) %>%
  as.list() %>%
  pwalk(~ system2("pdfunite", args = glue("{.x} {.y}")))
