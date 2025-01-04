script_packages <- c(
  "argparser", "cowplot", "dplyr", "fs", "glue", "hewr", "purrr",
  "tidyr"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


save_forecast_figures <- function(model_run_dir,
                                  pyrenew_model_name,
                                  timeseries_model_name) {
  parsed_model_run_dir <- parse_model_run_dir_path(model_run_dir)
  processed_forecast <- process_state_forecast(
    model_run_dir,
    pyrenew_model_name,
    timeseries_model_name
  )
  diseases <- unique(
    processed_forecast$daily_combined_training_eval_data$disease
  )

  y_transforms <- c("identity" = "", "log10" = "_log")

  timescales <- c(
    "daily",
    "epiweekly",
    "epiweekly_with_epiweekly_other"
  )

  figure_save_tbl <-
    expand_grid(
      target_disease = diseases,
      y_transform = names(y_transforms),
      timescale = timescales
    ) |>
    filter(!(.data$target_disease == "Disease" &
      .data$timescale == "epiweekly_with_epiweekly_other")) |>
    mutate(
      transform_name = y_transforms[y_transform],
      dat_timescale = ifelse(timescale == "daily",
        "daily",
        "epiweekly"
      )
    ) |>
    mutate(
      figure_path = path(
        model_run_dir,
        glue(
          "{target_disease}_",
          "forecast_plot{transform_name}_",
          "{timescale}"
        ),
        ext = "pdf"
      ),
      dat_to_use = glue("{dat_timescale}_combined_training_eval_data"),
      ci_to_use = glue("{timescale}_ci")
    ) |>
    mutate(figure = pmap(
      list(
        target_disease,
        y_transform,
        dat_to_use,
        ci_to_use
      ),
      \(target_disease,
        y_transform,
        dat_to_use,
        ci_to_use) {
        make_forecast_figure(
          target_disease = target_disease,
          combined_dat = processed_forecast[[dat_to_use]],
          forecast_ci = processed_forecast[[ci_to_use]],
          disease_name = parsed_model_run_dir$disease,
          data_vintage_date = parsed_model_run_dir$report_date,
          y_transform = y_transform
        )
      }
    ))


  walk2(
    figure_save_tbl$figure, figure_save_tbl$figure_path,
    \(figure, figure_path) {
      save_plot(
        filename = figure_path,
        plot = figure,
        device = cairo_pdf, base_height = 6
      )
    }
  )
}

p <- arg_parser("Generate forecast figures") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output.",
  ) |>
  add_argument(
    "--pyrenew-model-name",
    help = "Name of directory containing pyrenew model outputs"
  ) |>
  add_argument(
    "--timeseries-model-name",
    help = "Name of directory containing timeseries model outputs"
  )

argv <- parse_args(p)

model_run_dir <- path(argv$model_run_dir)
pyrenew_model_name <- argv$pyrenew_model_name
timeseries_model_name <- argv$timeseries_model_name

save_forecast_figures(model_run_dir, pyrenew_model_name, timeseries_model_name)
