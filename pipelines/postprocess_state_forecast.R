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


save_forecast_figures <- function(model_run_dir) {
  parsed_model_run_dir <- parse_model_run_dir_path(model_run_dir)
  processed_forecast <- process_state_forecast(model_run_dir)

  figure_save_tbl <-
    expand_grid(
      target_disease = unique(processed_forecast$combined_dat$disease),
      y_transform = c("identity", "log10")
    ) |>
    mutate(path_suffix = c("identity" = "", "log10" = "_log")[y_transform]) |>
    mutate(figure_path = path(model_run_dir,
      glue("{target_disease}_forecast_plot{path_suffix}"),
      ext = "pdf"
    )) |>
    mutate(figure = map2(
      target_disease, y_transform,
      \(target_disease, y_transform) {
        make_forecast_figure(
          target_disease = target_disease,
          combined_dat = processed_forecast$combined_dat,
          forecast_ci = processed_forecast$forecast_ci,
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
  )

argv <- parse_args(p)

model_run_dir <- path(argv$model_run_dir)
save_forecast_figures(model_run_dir)
