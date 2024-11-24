library(scoringutils)
library(dplyr)
library(ggplot2)

score_path <- "~/Influenza_epiweekly_score_table.rds"

scores <- readr::read_rds(score_path)[["quantile_scores"]] |>
  mutate(horizon = floor(as.numeric(date - report_date) / 7))

for_coverage_plots <- scores |>
  summarise_scores(by = c("model", "date", "horizon")) |>
  mutate(horizon_name = glue::glue("{horizon + 1} week ahead")) |>
  filter(model == "pyrenew-hew")


coverage_plot <- function(data, coverage_level) {
  coverage_column <-
    glue::glue("interval_coverage_{100 * coverage_level}")
  return(
    ggplot(
      data = data,
      mapping = aes(
        x = date,
        y = .data[[coverage_column]]
      )
    ) +
      geom_line(linewidth = 2) +
      geom_point(shape = 21, size = 3, fill = "darkgreen") +
      geom_hline(
        yintercept = coverage_level,
        linewidth = 1.5,
        linetype = "dashed"
      ) +
      facet_wrap(~horizon_name) +
      scale_y_continuous(label = scales::label_percent()) +
      scale_x_date() +
      coord_cartesian(ylim = c(0, 1)) +
      theme_minimal()
  )
}

coverage_plot_50 <- for_coverage_plots |>
  coverage_plot(0.50)
