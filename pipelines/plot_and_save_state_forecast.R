script_packages <- c(
  "argparser", "cowplot", "dplyr", "fs", "glue", "hewr", "purrr",
  "tidyr", "stringr", "lubridate"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


save_forecast_figures <- function(model_run_dir,
                                  pyrenew_model_name,
                                  timeseries_model_name = NULL) {
  parsed_model_run_dir <- parse_model_run_dir_path(model_run_dir)
  pyrenew_model_components <- parse_pyrenew_model_name(pyrenew_model_name)
  processed_forecast <- process_state_forecast(
    model_run_dir,
    pyrenew_model_name,
    timeseries_model_name,
    save = TRUE
  )
  processed_forecast$daily_data <- read_and_combine_data(model_run_dir,
    epiweekly = FALSE
  )


  variables <- unique(processed_forecast$daily_samples[[".variable"]])

  y_transforms <- c("identity" = "", "log10" = "_log")

  timescales <- "daily"
  if (pyrenew_model_components[["e"]]) {
    timescales <- c(timescales, "epiweekly", "epiweekly_with_epiweekly_other")
    processed_forecast$epiweekly_data <- read_and_combine_data(model_run_dir,
      epiweekly = TRUE
    )
  }
  # This isn't quite right. Gives misleading file names to h figures
  # They are labelled "daily" but are actually epiweekly
  # No prefix at all would also be fine
  # This section is a mess. It produces redundant plots.
  figure_save_tbl <-
    expand_grid(
      target_variable = variables,
      y_transform = names(y_transforms),
      timescale = timescales
    ) |>
    filter(
      !(target_variable == "site_level_log_ww_conc" & y_transform == "log10")
    ) |>
    filter(!(.data$target_variable == "observed_ed_visits" &
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
        pyrenew_model_name,
        glue(
          "{target_variable}_",
          "{timescale}",
          "{transform_name}"
        ),
        ext = "pdf"
      ),
      dat_to_use = glue("{dat_timescale}_data"),
      ci_to_use = glue("{timescale}_ci")
    ) |>
    mutate(figure = pmap(
      list(
        target_variable,
        y_transform,
        dat_to_use,
        ci_to_use
      ),
      \(target_variable,
        y_transform,
        dat_to_use,
        ci_to_use) {
        make_forecast_figure(
          target_variable = target_variable,
          combined_dat = processed_forecast[[dat_to_use]],
          forecast_ci = processed_forecast[[ci_to_use]],
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
