#!/usr/bin/env Rscript

# Simple plotting script for EpiAutoGP forecasts
# Uses already-processed samples.parquet and ci.parquet files

script_packages <- c(
  "argparser",
  "arrow",
  "cowplot",
  "dplyr",
  "fs",
  "glue",
  "hewr",
  "purrr",
  "lubridate"
)

purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


plot_epiautogp_forecast <- function(
  model_run_dir,
  epiautogp_model_name
) {
  model_dir <- fs::path(model_run_dir, epiautogp_model_name)
  figure_dir <- fs::path(model_dir, "figures")
  fs::dir_create(figure_dir)

  # Read already-processed forecast outputs
  samples <- arrow::read_parquet(fs::path(model_dir, "samples.parquet"))
  ci <- arrow::read_parquet(fs::path(model_dir, "ci.parquet"))

  # Read observed data
  data <- hewr::read_and_combine_data(model_dir)

  # Parse the report date from the model_run_dir path
  parsed_dir <- hewr::parse_model_run_dir_path(model_run_dir)
  report_date <- parsed_dir$report_date

  # Get unique combinations to plot
  plot_specs <- ci |>
    distinct(
      geo_value,
      disease,
      .variable,
      resolution,
      aggregated_numerator,
      aggregated_denominator
    )

  # Create and save each plot
  for (i in seq_len(nrow(plot_specs))) {
    spec <- plot_specs[i, ]

    # Create the figure
    fig <- hewr::make_forecast_figure(
      dat = data,
      geo_value = spec$geo_value,
      disease = spec$disease,
      .variable = spec$.variable,
      resolution = spec$resolution,
      aggregated_numerator = spec$aggregated_numerator,
      aggregated_denominator = spec$aggregated_denominator,
      y_transform = "identity",
      ci = ci,
      data_vintage_date = report_date
    )

    # Create filename
    agg_num_suffix <- if (isTRUE(spec$aggregated_numerator)) {
      "_agg_num"
    } else {
      ""
    }
    agg_denom_suffix <- if (isTRUE(spec$aggregated_denominator)) {
      "_agg_denom"
    } else {
      ""
    }

    filename <- glue::glue(
      "{epiautogp_model_name}_{spec$.variable}_{spec$resolution}",
      "{agg_num_suffix}{agg_denom_suffix}.pdf"
    )

    figure_path <- fs::path(figure_dir, filename)

    # Save the plot
    cowplot::save_plot(
      filename = figure_path,
      plot = fig,
      device = cairo_pdf,
      base_height = 6
    )

    message(glue::glue("Saved plot: {filename}"))
  }

  message(glue::glue("All plots saved to: {figure_dir}"))
}


# Command-line interface
p <- arg_parser("Generate EpiAutoGP forecast figures") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model run (e.g., .../model_runs/CA)"
  ) |>
  add_argument(
    "--epiautogp-model-name",
    help = "Name of EpiAutoGP model directory (e.g., 'epiautogp_nhsn')",
    default = "epiautogp_nhsn"
  )

argv <- parse_args(p)

plot_epiautogp_forecast(
  model_run_dir = fs::path(argv$model_run_dir),
  epiautogp_model_name = argv$epiautogp_model_name
)
