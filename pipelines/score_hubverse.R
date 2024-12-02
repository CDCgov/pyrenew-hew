library(ggplot2)
library(hewr)

score_hubverse <- function(path_observed,
                           path_forecast,
                           observed_column,
                           horizons = c(0, 1),
                           log_shift_offset = 1) {
  obs <- readr::read_tsv(path_observed,
    show_col_types = FALSE
  )
  forecast <- readr::read_tsv(path_forecast,
    show_col_types = FALSE
  )

  interval_coverage_95 <- purrr::partial(
    scoringutils::interval_coverage,
    interval_range = 95
  )

  to_score <- dplyr::inner_join(
    forecast |> dplyr::filter(horizon %in% !!horizons),
    obs |> dplyr::select(
      location,
      target_end_date = reference_date,
      observed = {{ observed_column }}
    ),
    by = c("location", "target_end_date")
  ) |>
    scoringutils::as_forecast_quantile(
      predicted = "value",
      observed = "observed",
      quantile_level = "output_type_id"
    ) |>
    scoringutils::transform_forecasts(
      fun = scoringutils::log_shift,
      offset = log_shift_offset
    )

  scored <- to_score |>
    scoringutils::score(
      metrics = c(
        scoringutils::get_metrics(to_score),
        list(interval_coverage_95 = interval_coverage_95)
      )
    ) |>
    dplyr::filter(scale == "log")

  return(scored)
}


last_target_date <- lubridate::ymd("2024-11-30")

truth_path <- "~/epiweekly.tsv"
all_paths_flu <- c(
  "~/2024-11-04-influenza-hubverse-table.tsv",
  "~/2024-11-13-influenza-hubverse-table.tsv",
  "~/2024-11-20-influenza-hubverse-table.tsv",
  "~/2024-11-27-influenza-hubverse-table.tsv"
)
all_paths_covid <- c(
  "~/2024-11-04-covid-hubverse-table.tsv",
  "~/2024-11-13-covid-19-hubverse-table.tsv",
  "~/2024-11-20-covid-19-hubverse-table.tsv",
  "~/2024-11-27-covid-19-hubverse-table.tsv"
)

flu_scores <- purrr::map(
  all_paths_flu,
  \(x) score_hubverse(truth_path, x, prop_influenza)
) |>
  dplyr::bind_rows() |>
  dplyr::filter(target_end_date <= !!last_target_date)
covid_scores <- purrr::map(
  all_paths_covid,
  \(x) score_hubverse(truth_path, x, prop_covid)
) |>
  dplyr::bind_rows() |>
  dplyr::filter(target_end_date <= !!last_target_date)

full_scores <- dplyr::bind_rows(
  flu_scores,
  covid_scores
)


summary_by_epiweek <- full_scores |>
  scoringutils::summarise_scores(by = c("horizon", "reference_date", "target"))

summary_overall <- full_scores |>
  scoringutils::summarise_scores(by = c("horizon", "target"))


summary_by_loc <- full_scores |>
  dplyr::filter(horizon %in% c(0, 1)) |>
  scoringutils::summarise_scores(by = c("horizon", "location", "target"))


coverage_figs <- purrr::map(
  c(0.5, 0.95),
  \(x) {
    forecasttools::plot_coverage_by_date(
      full_scores, x
    ) +
      ggplot2::theme_minimal()
  }
)


forecasttools::plots_to_pdf(
  coverage_figs,
  "2024-11-29-coverage.pdf",
  width = 11,
  height = 8.5
)

readr::write_tsv(full_summary, "2024-11-30-scoring-summary.tsv")
