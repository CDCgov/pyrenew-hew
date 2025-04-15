script_packages <- c(
  "argparser",
  "arrow",
  "dplyr",
  "forecasttools",
  "scoringutils",
  "tidyr",
  "readr",
  "hewr",
  "fs",
  "stringr",
  "purrr"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})


#' Score Forecasts
#'
#' This function scores forecast data using the `scoringutils` package. It takes
#' in scorable data, that is data which has a joined eval data and forecast
#' data, and scores it.
#'
#' NB: this function assumes that _log-scale_ scoring is the default. If you
#' want to vary this behavior, you can splat additional arguments to
#' `scoringutils::transform_forecasts` such as the identity transformation e.g.
#' `fun = identity` with `label = "identity"`.
#'
#' If more than one model is present in the data, in the column `model_col` the
#' function will add relative skill metrics to the output.
#'
#' @param samples_scorable A data frame to be scored using
#' `scoringutils::as_forecast_sample`
#' @param quantiles_scorable A data frame to be scored using
#' `scoringutils::as_forecast_quantile`
#' @param ... Additional arguments passed to
#' `scoringutils::transform_forecasts`.
#'
#' @return A data frame with scored forecasts and relative skill metrics.
#' @export
score_single_run <- function(samples_scorable,
                             quantiles_scorable,
                             ...) {
  quants <- unique(quantiles_scorable$quantile_level)

  forecast_sample_df <- as_forecast_sample(samples_scorable)
  forecast_quantile_df <- bind_rows(
    as_forecast_quantile(quantiles_scorable),
    as_forecast_quantile(forecast_sample_df,
      probs = quants
    )
  ) |>
    as_forecast_quantile()


  sample_scores <- forecast_sample_df |>
    scoringutils::transform_forecasts(...) |>
    scoringutils::score()

  interval_coverage_95 <- purrr::partial(scoringutils::interval_coverage,
    interval_range = 95
  )

  quantile_metrics <- c(get_metrics(forecast_quantile_df),
    interval_coverage_95 = interval_coverage_95
  )

  quantile_scores <- forecast_quantile_df |>
    scoringutils::transform_forecasts(...) |>
    scoringutils::score(metrics = quantile_metrics)

  # Add relative skill if more than one model is present
  if (n_distinct(forecast_sample_df[["model"]]) > 1) {
    sample_scores <- scoringutils::add_relative_skill(sample_scores)
  }
  if (n_distinct(forecast_quantile_df[["model"]]) > 1) {
    quantile_scores <- scoringutils::add_relative_skill(quantile_scores)
  }

  return(list(
    sample_scores = sample_scores,
    quantile_scores = quantile_scores
  ))
}

read_and_score_location <- function(model_run_dir,
                                    strict = TRUE) {
  first_forecast_date <- parse_model_run_dir_path(
    model_run_dir
  )$last_training_date + lubridate::days(1)

  samples_paths <- dir_ls(model_run_dir,
    recurse = TRUE,
    glob = "*_samples.parquet"
  )
  quantiles_paths <- dir_ls(model_run_dir,
    recurse = TRUE,
    glob = "*_quantiles.parquet"
  )

  scorable_datasets <-
    tibble(file_path = c(samples_paths, quantiles_paths)) |>
    mutate(
      forecast_name = file_path |>
        path_file() |>
        path_ext_remove() |>
        str_remove("_([^_]*)$"),
      forecast_type = file_path |>
        path_file() |>
        path_ext_remove() |>
        str_extract("(?<=_)([^_]*)$"),
      resolution = file_path |>
        path_file() |>
        path_ext_remove() |>
        str_extract("^.+?(?=_)"),
      model_name = file_path |>
        path_dir() |>
        path_file()
    ) |>
    unite("model", model_name, forecast_name, sep = "_") |>
    mutate(forecast_data = map(file_path, \(x) {
      read_parquet(x) |>
        rename(predicted = .value) |>
        filter(date > !!first_forecast_date)
    })) |>
    select(-file_path)

  eval_data <-
    tibble(
      file_path = c(
        path(model_run_dir, "data", "combined_eval_data", ext = "tsv"),
        path(model_run_dir, "data", "epiweekly_combined_eval_data", ext = "tsv")
      ),
      resolution = c("daily", "epiweekly"),
      eval_data = map(file_path, \(x) {
        read_tsv(x,
          col_types = cols(
            date = col_date(),
            geo_value = col_character(),
            disease = col_character(),
            data_type = col_character(),
            .variable = col_character(),
            .value = col_double()
          )
        ) |>
          rename(observed = .value) |>
          filter(date > !!first_forecast_date)
      })
    ) |>
    select(-file_path) |>
    unnest(eval_data)

  if (nrow(eval_data) == 0 && strict) {
    stop(paste0(
      "Nothing to score. ",
      "If you want to permit this, ",
      "set `strict` to `FALSE`."
    ))
  }


  # Removing lab_site_index column
  # NWSS data is not added to eval data yet

  samples_scorable <-
    scorable_datasets |>
    filter(forecast_type == "samples") |>
    unnest(forecast_data) |>
    inner_join(eval_data,
      by = join_by(
        resolution, date, geo_value, disease, .variable
      )
    ) |>
    rename(sample_id = .draw) |>
    select(-c(.chain, .iteration, forecast_type, starts_with("lab_site_index")))

  quantiles_scorable <-
    scorable_datasets |>
    filter(forecast_type == "quantiles") |>
    unnest(forecast_data) |>
    inner_join(eval_data,
      by = join_by(resolution, date, geo_value, disease, .variable)
    ) |>
    select(-c(forecast_type, lab_site_index))

  scored <- score_single_run(
    quantiles_scorable = quantiles_scorable,
    samples_scorable = samples_scorable,
    offset = 1
  )


  write_rds(scored, path(model_run_dir, "scored", ext = "rds"))
}

# Create a parser
p <- arg_parser("Score a single location forecast") |>
  add_argument(
    "model_run_dir",
    help = "Directory containing the model data and output."
  )

argv <- parse_args(p)

read_and_score_location(argv$model_run_dir)
