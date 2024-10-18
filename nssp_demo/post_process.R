library(tidyverse)
library(tidybayes)
library(fs)
library(cowplot)
library(glue)
library(scales)
library(here)
library(argparser)

# Create a parser
p <- arg_parser("Generate forecast figures") %>%
  add_argument(p, "--model_dir",
    help = "Directory containing the model data",
    required = TRUE
  )

argv <- parse_args(p)
model_dir <- path(argv$model_dir)

base_dir <- path_dir(model_dir)

theme_set(theme_minimal_grid())

disease_name_formatter <- c("covid-19" = "COVID-19", "influenza" = "Flu")

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

make_forecast_fig <- function(model_dir,
                              filter_bad_chains = TRUE,
                              good_chain_tol = 2) {
  disease_name_raw <- base_dir %>%
    path_file() %>%
    str_extract("^.+(?=_r_)")

  state_abb <- model_dir %>%
    path_split() %>%
    pluck(1) %>%
    tail(1)

  data_path <- path(model_dir, "data", ext = "csv")
  inference_data_path <- path(model_dir, "inference_data",
    ext = "csv"
  )

  dat <- read_csv(data_path) %>%
    arrange(date) %>%
    mutate(time = row_number() - 1) %>%
    rename(.value = ED_admissions)

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

  hosp_ci <-
    pyrenew_samples$posterior_predictive %>%
    gather_draws(observed_hospital_admissions[time]) %>%
    median_qi(.width = c(0.5, 0.8, 0.95)) %>%
    mutate(date = min(dat$date) + time)



  forecast_plot <-
    ggplot(mapping = aes(date, .value)) +
    geom_lineribbon(
      data = hosp_ci,
      mapping = aes(ymin = .lower, ymax = .upper),
      color = "#08519c", key_glyph = draw_key_rect, step = "mid"
    ) +
    geom_point(mapping = aes(shape = data_type), data = dat) +
    scale_y_continuous("Emergency Department Admissions") +
    scale_x_date("Date") +
    scale_fill_brewer(
      name = "Credible Interval Width",
      labels = ~ percent(as.numeric(.))
    ) +
    scale_shape_discrete("Data Type", labels = str_to_title) +
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
    ggtitle(
      glue(
        "{disease_name_formatter[disease_name_raw]} ",
        "NSSP-based forecast for {state_abb}"
      ),
      subtitle = glue("as of {last_data_date}")
    ) +
    theme(legend.position = "bottom")

  forecast_plot
}

forecast_fig <- make_forecast_fig(model_dir)

save_plot(
  filename = path(model_dir, "forecast_plot", ext = "pdf"),
  plot = forecast_fig,
  device = cairo_pdf, base_height = 6
)


# Temp code while command line version doesn't work
base_dir <- path(
  "nssp_demo",
  "private_data",
  "influenza_r_2024-10-10_f_2024-04-12_l_2024-10-09_t_2024-10-05"
)

walk(dir_ls(base_dir), function(model_dir) {
  forecast_fig <- make_forecast_fig(model_dir)

  save_plot(
    filename = path(model_dir, "forecast_plot", ext = "pdf"),
    plot = forecast_fig,
    device = cairo_pdf, base_height = 6
  )
})

path(dir_ls(base_dir, type = "directory"), "forecast_plot", ext = "pdf") %>%
  str_c(collapse = " ") %>%
  str_c(
    path(base_dir,
      glue("{path_file(base_dir)}_all_forecasts"),
      ext = "pdf"
    ),
    sep = " "
  ) %>%
  system2("pdfunite", args = .)
