suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(scoringutils))
suppressPackageStartupMessages(library(arrow))
suppressPackageStartupMessages(library(argparser))


#' Score Forecasts
#'
#' This function scores forecast data using the `scoringutils` package. It takes
#' in scorable data, that is data which has a joined truth data and forecast
#' data, and scores this.
#'
#' This function aims at scoring _sampled_ forecasts. Care must be taken to
#' select the appropriate columns for the observed and predicted values, as well
#' as the forecast unit. The expected `sample_id` column is `.draw` due to
#' expecting input from a tidybayes format.
#'
#' NB: this function assumes that _log-scale_ scoring is the default. If you
#' want to vary this behaviour, you can splat additional arguments to
#' `scoringutils::transform_forecasts` such as the identity transformation e.g.
#' `fun = identity` with `label = "identity"`.
#'
#' If more than one model is present in the data, in the column `model_col` the
#' function will add relative skill metrics to the output.
#'
#' @param scorable_data A data frame containing the data to be scored.
#' @param forecast_unit A character string specifying the forecast unit.
#' @param observed A character string specifying the column name for observed
#' values.
#' @param predicted A string vector specifying the column name for predicted
#' values.
#' @param sample_id A string specifying the column name for sample
#' IDs. Default is ".draw".
#' @param model_col A character string specifying the column name for models.
#' @param ... Additional arguments passed to
#' `scoringutils::transform_forecasts`.
#'
#' @return A data frame with scored forecasts and relative skill metrics.
#' @export
score_single_run <- function(
    scorable_data, quantile_only_data, forecast_unit, observed, predicted,
    sample_id = ".draw", model_col = "model", ...) {
  forecast_sample_df <- scorable_data |>
    scoringutils::as_forecast_sample(
      forecast_unit = forecast_unit,
      observed = observed,
      predicted = predicted,
      sample_id = sample_id
    )

  quantile_only_df <- quantile_only_data |>
    scoringutils::as_forecast_quantile(
      forecast_unit = forecast_unit,
      observed = observed,
      predicted = predicted
    )
  quants <- unique(quantile_only_df$quantile_level)
  quantiles_from_samples_df <- forecast_sample_df |>
    scoringutils::as_forecast_quantile(probs = quants)

  forecast_quantile_df <-
    dplyr::bind_rows(
      quantile_only_df,
      quantiles_from_samples_df
    ) |>
    scoringutils::as_forecast_quantile()

  sample_scores <- forecast_sample_df |>
    scoringutils::transform_forecasts(...) |>
    scoringutils::score()

  quantile_scores <- forecast_quantile_df |>
    scoringutils::transform_forecasts(...) |>
    scoringutils::score()
  # Add relative skill if more than one model is present
  if (n_distinct(scorable_data[[model_col]]) != 1) {
    sample_scores <- scoringutils::add_relative_skill(sample_scores)
    quantile_scores <- scoringutils::add_relative_skill(quantile_scores)
  }
  return(list(
    sample_scores = sample_scores,
    quantile_scores = quantile_scores
  ))
}


prep_truth_data <- function(truth_data_path) {
  dat <- readr::read_tsv(truth_data_path,
    show_col_types = FALSE
  ) |>
    filter(data_type == "test") |>
    rename(true_value = ed_visits) |>
    mutate(disease = case_when(
      disease %in% c("Influenza", "COVID-19") ~ "Disease",
      disease == "Total" ~ "Total",
      TRUE ~ NA
    )) |>
    filter(!is.na(disease)) |>
    tidyr::pivot_wider(
      names_from = "disease",
      values_from = "true_value"
    ) |>
    mutate(prop_disease_ed_visits = Disease / Total) |>
    tidyr::pivot_longer(
      c(Disease, Total, prop_disease_ed_visits),
      names_to = "disease",
      values_to = "true_value"
    )

  return(dat)
}

read_and_score_location <- function(model_run_dir,
                                    eval_data_filename = "eval_data",
                                    eval_data_file_ext = "tsv") {
  message(glue::glue("Scoring {model_run_dir}..."))
  forecast_path <- fs::path(
    model_run_dir,
    "forecast_samples.parquet"
  )
  ts_baseline_path <- fs::path(
    model_run_dir,
    "baseline_ts_prop_ed_visits_forecast.parquet"
  )
  cdc_baseline_path <- fs::path(
    model_run_dir,
    "baseline_cdc_prop_ed_visits_forecast.parquet"
  )

  truth_path <- fs::path(model_run_dir,
    eval_data_filename,
    ext = eval_data_file_ext
  )

  actual_data <- prep_truth_data(truth_path)

  pyrenew <- arrow::read_parquet(forecast_path) |>
    mutate(model = "pyrenew-hew") |>
    select(date, .draw, disease, model, .value)

  ts_baseline <- arrow::read_parquet(ts_baseline_path) |>
    mutate(
      model = "ts_baseline",
      disease = "prop_disease_ed_visits"
    ) |>
    select(date,
      .draw,
      disease,
      model,
      .value = prop_disease_ed_visits
    )

  cdc_baseline <- arrow::read_parquet(cdc_baseline_path) |>
    mutate(
      model = "cdc_baseline",
      disease = "prop_disease_ed_visits"
    ) |>
    select(date,
      disease,
      quantile_level,
      .value = baseline_ed_visit_prop_forecast,
      model
    )


  quantile_forecasts_to_score <- inner_join(
    cdc_baseline,
    actual_data,
    by = c("disease", "date")
  ) |>
    filter(disease == "prop_disease_ed_visits")

  sample_forecasts_to_score <- bind_rows(
    pyrenew,
    ts_baseline
  ) |>
    inner_join(actual_data,
      by = c("disease", "date")
    ) |>
    filter(disease == "prop_disease_ed_visits")

  max_visits <- actual_data |>
    filter(disease == "Total") |>
    pull(true_value) |>
    max()

  scored <- score_single_run(
    sample_forecasts_to_score,
    quantile_forecasts_to_score,
    forecast_unit = c("date", "model"),
    observed = "true_value",
    sample_id = ".draw",
    predicted = ".value",
    offset = 1 / max_visits
  )

  saveRDS(scored, fs::path(model_run_dir, "score_table", ext = "rds"))
}

# Create a parser
p <- arg_parser("Score a single location forecast") |>
  add_argument(
    "model-run-dir",
    help = "Directory containing the model data and output."
  )

argv <- parse_args(p)

read_and_score_location(argv$model_run_dir)
