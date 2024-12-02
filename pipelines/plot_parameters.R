library(dplyr)
library(ggplot2)
library(tidybayes)
library(ggdist)
library(tibble)

pathogen <- "influenza"
job_path <- ""
state <- "US"

fit <- arrow::read_parquet(fs::path(job_path,
  "model_runs",
  state,
  "mcmc_tidy",
  "pyrenew_posterior",
  ext = "parquet"
)) |>
  tibble()


inf_feedback <- fit |>
  spread_draws(inf_feedback_raw, p_ed_visit_mean) |>
  ggplot(aes(
    x = inf_feedback_raw,
    y = p_ed_visit_mean,
    group = .draw
  )) +
  geom_point() +
  scale_x_continuous(transform = "log10") +
  scale_y_continuous(transform = "logit") +
  theme_minimal()
