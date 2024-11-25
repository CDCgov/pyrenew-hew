library(ggplot2)
library(hewr)

score_hubverse <- function(path_observed,
                           path_forecast,
                           observed_column,
                           horizons = c(0, 1, 2)) {
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
    )

  scored <- to_score |>
    scoringutils::score(
      metrics = c(
        scoringutils::get_metrics(to_score),
        list(interval_coverage_95 = interval_coverage_95)
      )
    )

  return(scored)
}


truth_path <- "~/epiweekly.tsv"
all_paths_flu <- c(
  "~/2024-11-04-influenza-hubverse-table.tsv",
  "~/2024-11-13-influenza-hubverse-table.tsv"
)
all_paths_covid <- c(
  "~/2024-11-04-covid-hubverse-table.tsv",
  "~/2024-11-13-covid-19-hubverse-table.tsv"
)

flu_scores <- purrr::map(
  all_paths_flu,
  \(x) score_hubverse(truth_path, x, prop_influenza)
) |>
  dplyr::bind_rows() |>
  dplyr::filter(target_end_date < lubridate::today() + 2)
covid_scores <- purrr::map(
  all_paths_covid,
  \(x) score_hubverse(truth_path, x, prop_covid)
) |>
  dplyr::bind_rows() |>
  dplyr::filter(target_end_date < lubridate::today() + 2)

full_scores <- dplyr::bind_rows(
  flu_scores,
  covid_scores
)


flu_summary <- flu_scores |>
  scoringutils::summarise_scores(by = c("horizon", "reference_date", "target"))

covid_summary <- covid_scores |>
  scoringutils::summarise_scores(by = c("horizon", "reference_date", "target"))


full_summary <- dplyr::bind_rows(flu_summary, covid_summary)

fig <- plot_coverage_by_date(full_scores, 0.95)
