script_packages <- c(
  "argparser",
  "cowplot",
  "dplyr",
  "fs",
  "glue",
  "hewr",
  "purrr",
  "tidyr",
  "stringr",
  "lubridate"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


save_forecast_figures <- function(
  model_run_dir,
  n_forecast_days,
  pyrenew_model_name = NA,
  timeseries_model_name = NA
) {
  if (is.na(pyrenew_model_name) && is.na(timeseries_model_name)) {
    stop(
      "Either `pyrenew_model_name` or `timeseries_model_name`",
      "must be provided."
    )
  }

  create_file_name <- function(
    model_name,
    .variable,
    resolution,
    aggregated_numerator,
    aggregated_denominator,
    y_transform
  ) {
    glue(
      "{model_name}_",
      "{.variable}_{resolution}",
      "{dplyr::if_else(vctrs::vec_equal(",
      "aggregated_numerator,TRUE, na_equal = TRUE),'_agg_num', '')}",
      "{dplyr::if_else(vctrs::vec_equal(",
      "aggregated_denominator, TRUE, na_equal = TRUE), '_agg_denom', '')}",
      "{y_transforms[y_transform]}"
    ) |>
      str_replace_all("_+", "_")
  }

  model_name <- dplyr::if_else(
    is.na(pyrenew_model_name),
    timeseries_model_name,
    pyrenew_model_name
  )

  figure_dir <- fs::path(model_run_dir, model_name, "figures")
  dir_create(figure_dir)

  parsed_model_run_dir <- parse_model_run_dir_path(model_run_dir)
  processed_forecast <- process_loc_forecast(
    model_run_dir,
    n_forecast_days,
    pyrenew_model_name,
    timeseries_model_name,
    save = TRUE
  )

  y_transforms <- c("identity" = "", "log10" = "_log")

  processed_forecast$data <- read_and_combine_data(model_run_dir)

  distinct_fig_type_tbl <-
    processed_forecast$ci |>
    distinct(
      geo_value,
      disease,
      .variable,
      resolution,
      aggregated_numerator,
      aggregated_denominator
    ) |>
    expand_grid(y_transform = names(y_transforms)) |>
    filter(!(.variable == "site_level_log_ww_conc" & y_transform == "log10"))

  figure_save_tbl <-
    distinct_fig_type_tbl |>
    mutate(
      figure = pmap(
        list(
          geo_value,
          disease,
          .variable,
          resolution,
          aggregated_numerator,
          aggregated_denominator,
          y_transform
        ),
        \(
          geo_value,
          disease,
          .variable,
          resolution,
          aggregated_numerator,
          aggregated_denominator,
          y_transform
        ) {
          make_forecast_figure(
            processed_forecast$data,
            geo_value,
            disease,
            .variable,
            resolution,
            aggregated_numerator,
            aggregated_denominator,
            y_transform,
            processed_forecast$ci,
            parsed_model_run_dir$report_date
          )
        }
      )
    ) |>
    mutate(
      file_name = create_file_name(
        model_name,
        .data$.variable,
        .data$resolution,
        .data$aggregated_numerator,
        .data$aggregated_denominator,
        .data$y_transform
      )
    ) |>
    mutate(figure_path = path(figure_dir, file_name, ext = "pdf"))

  walk2(
    figure_save_tbl$figure,
    figure_save_tbl$figure_path,
    \(figure, figure_path) {
      save_plot(
        filename = figure_path,
        plot = figure,
        device = cairo_pdf,
        base_height = 6
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
    help = "Name of directory containing timeseries model outputs",
  ) |>
  add_argument(
    "--n-forecast-days",
    help = "Number of days to forecast"
  )

argv <- parse_args(p)

model_run_dir <- path(argv$model_run_dir)
n_forecast_days <- as.numeric(argv$n_forecast_days)
pyrenew_model_name <- argv$pyrenew_model_name
timeseries_model_name <- argv$timeseries_model_name

save_forecast_figures(
  model_run_dir,
  n_forecast_days,
  pyrenew_model_name,
  timeseries_model_name
)
