script_packages <- c(
  "dplyr",
  "stringr",
  "purrr",
  "ggplot2",
  "tidybayes",
  "fs",
  "cowplot",
  "glue",
  "scales",
  "argparser",
  "arrow",
  "tidyr",
  "readr",
  "here",
  "forcats"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


# To be replaced with reading tidy data from forecasttools
read_pyrenew_samples <- function(inference_data_path,
                                 filter_bad_chains = TRUE,
                                 good_chain_tol = 2) {
  arviz_split <- function(x) {
    x |>
      select(-distribution) |>
      split(f = as.factor(x$distribution))
  }

  pyrenew_samples <-
    read_csv(inference_data_path,
      show_col_types = FALSE
    ) |>
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
      pyrenew_samples$log_likelihood |>
      pivot_longer(-starts_with(".")) |>
      group_by(.iteration, .chain) |>
      summarize(value = sum(value)) |>
      group_by(.chain) |>
      summarize(value = mean(value)) |>
      filter(value >= max(value) - 2) |>
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
                                  combined_dat,
                                  last_training_date,
                                  data_vintage_date,
                                  posterior_predictive_ci,
                                  state_abb,
                                  y_transform = "identity") {
  y_scale <- if (str_starts(target_disease, "prop")) {
    scale_y_continuous("Proportion of Emergency Department Visits",
      labels = percent,
      transform = y_transform
    )
  } else {
    scale_y_continuous("Emergency Department Visits",
      labels = comma,
      transform = y_transform
    )
  }

  title <- if (target_disease == "Other") {
    glue("Other ED Visits in {state_abb}")
  } else {
    glue("{disease_name_pretty} ED Visits in {state_abb}")
  }

  ggplot(mapping = aes(date, .value)) +
    geom_lineribbon(
      data = posterior_predictive_ci |> filter(disease == target_disease),
      mapping = aes(ymin = .lower, ymax = .upper),
      color = "#08519c",
      key_glyph = draw_key_rect,
      step = "mid"
    ) +
    scale_fill_brewer(
      name = "Credible Interval Width",
      labels = ~ percent(as.numeric(.))
    ) +
    geom_point(
      mapping = aes(color = data_type), size = 1.5,
      data = combined_dat |>
        filter(
          disease == target_disease,
          date <= max(posterior_predictive_ci$date)
        ) |>
        mutate(data_type = fct_rev(data_type)) |>
        arrange(desc(data_type))
    ) +
    scale_color_manual(
      name = "Data Type",
      values = c("olivedrab1", "deeppink"),
      labels = str_to_title
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
    ggtitle(title, subtitle = glue("as of {data_vintage_date}")) +
    y_scale +
    scale_x_date("Date") +
    theme(legend.position = "bottom")
}


postprocess_state_forecast <- function(model_run_dir,
                                       filter_bad_chains = TRUE,
                                       good_chain_tol = 2) {
  state_abb <- model_run_dir |>
    path_split() |>
    pluck(1) |>
    tail(1)

  train_data_path <- path(model_run_dir, "data", ext = "csv")
  eval_data_path <- path(model_run_dir, "eval_data", ext = "tsv")
  inference_data_path <- path(model_run_dir, "inference_data",
    ext = "csv"
  )
  other_ed_visits_path <- path(
    model_run_dir,
    "other_ed_visits_forecast",
    ext = "parquet"
  )

  train_dat <- read_csv(train_data_path, show_col_types = FALSE)

  data_vintage_date <- max(train_dat$date) + 1
  # this should be stored as metadata somewhere else, instead of being
  # computed like this

  eval_dat <- read_tsv(eval_data_path, show_col_types = FALSE) |>
    mutate(data_type = "eval")

  combined_dat <-
    bind_rows(
      train_dat |>
        filter(data_type == "train"),
      eval_dat
    ) |>
    mutate(
      disease = if_else(
        disease == disease_name_nssp,
        "Disease", # assign a common name for
        # use in plotting functions
        disease
      )
    ) |>
    pivot_wider(names_from = disease, values_from = ed_visits) |>
    mutate(
      Other = Total - Disease,
      prop_disease_ed_visits = Disease / Total
    ) |>
    select(-Total) |>
    mutate(time = dense_rank(date)) |>
    pivot_longer(c(Disease, Other, prop_disease_ed_visits),
      names_to = "disease",
      values_to = ".value"
    )


  last_training_date <- combined_dat |>
    filter(data_type == "train") |>
    pull(date) |>
    max()

  pyrenew_samples <- read_pyrenew_samples(inference_data_path,
    filter_bad_chains = filter_bad_chains,
    good_chain_tol = good_chain_tol
  )

  other_ed_visits_forecast <-
    read_parquet(other_ed_visits_path) |>
    rename(Other = other_ed_visits)

  other_ed_visits_samples <-
    bind_rows(
      combined_dat |>
        filter(
          data_type == "train",
          disease == "Other",
          date <= last_training_date
        ) |>
        select(date, Other = .value) |>
        expand_grid(.draw = 1:max(other_ed_visits_forecast$.draw)),
      other_ed_visits_forecast
    )

  posterior_predictive_samples <-
    pyrenew_samples$posterior_predictive |>
    gather_draws(observed_hospital_admissions[time]) |>
    pivot_wider(names_from = .variable, values_from = .value) |>
    rename(Disease = observed_hospital_admissions) |>
    ungroup() |>
    mutate(date = min(combined_dat$date) + time) |>
    left_join(other_ed_visits_samples,
      by = c(".draw", "date")
    ) |>
    mutate(prop_disease_ed_visits = Disease / (Disease + Other)) |>
    pivot_longer(c(Other, Disease, prop_disease_ed_visits),
      names_to = "disease",
      values_to = ".value"
    )

  arrow::write_parquet(
    posterior_predictive_samples,
    path(model_run_dir, "forecast_samples",
      ext = "parquet"
    )
  )

  posterior_predictive_ci <-
    posterior_predictive_samples |>
    select(date, disease, .value) |>
    group_by(date, disease) |>
    median_qi(.width = c(0.5, 0.8, 0.95))


  arrow::write_parquet(
    posterior_predictive_ci,
    path(model_run_dir, "forecast_ci",
      ext = "parquet"
    )
  )


  all_forecast_plots <- map(
    set_names(unique(combined_dat$disease)),
    ~ make_one_forecast_fig(
      .x,
      combined_dat,
      last_training_date,
      data_vintage_date,
      posterior_predictive_ci,
      state_abb,
    )
  )

  all_forecast_plots_log <- map(
    set_names(unique(combined_dat$disease)),
    ~ make_one_forecast_fig(
      .x,
      combined_dat,
      last_training_date,
      data_vintage_date,
      posterior_predictive_ci,
      state_abb,
      y_transform = "log10"
    )
  )

  iwalk(all_forecast_plots, ~ save_plot(
    filename = path(model_run_dir, glue("{.y}_forecast_plot"), ext = "pdf"),
    plot = .x,
    device = cairo_pdf, base_height = 6
  ))
  iwalk(all_forecast_plots_log, ~ save_plot(
    filename = path(model_run_dir, glue("{.y}_forecast_plot_log"), ext = "pdf"),
    plot = .x,
    device = cairo_pdf, base_height = 6
  ))
}


theme_set(theme_minimal_grid())

disease_name_formatter <- c("covid-19" = "COVID-19", "influenza" = "Flu")
disease_name_nssp_map <- c(
  "covid-19" = "COVID-19",
  "influenza" = "Influenza"
)

# Create a parser
p <- arg_parser("Generate forecast figures") |>
  add_argument(
    "--model-run-dir",
    help = "Directory containing the model data and output.",
  ) |>
  add_argument(
    "--no-filter-bad-chains",
    help = paste0(
      "By default, postprocess_state_forecast.R filters ",
      "any bad chains from the samples. Set this flag ",
      "to retain them"
    ),
    flag = TRUE
  ) |>
  add_argument(
    "--good-chain-tol",
    help = "Tolerance level for determining good chains.",
    default = 2L
  )

argv <- parse_args(p)
model_run_dir <- path(argv$model_run_dir)
filter_bad_chains <- !argv$no_filter_bad_chains
good_chain_tol <- argv$good_chain_tol

base_dir <- path_dir(model_run_dir)

disease_name_raw <- base_dir |>
  path_file() |>
  str_extract("^.+(?=_r_)")

disease_name_nssp <- unname(disease_name_nssp_map[disease_name_raw])
disease_name_pretty <- unname(disease_name_formatter[disease_name_raw])


postprocess_state_forecast(
  model_run_dir,
  filter_bad_chains,
  good_chain_tol
)
