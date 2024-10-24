library(tidyverse)
library(fs)
library(fable)
library(jsonlite)
library(glue)
library(argparser)
library(arrow)

p <- arg_parser("Forecast total ED admissions") %>%
  add_argument(p, "--model_dir",
    help = "Directory containing the model data",
    required = TRUE
  ) %>%
  add_argument("--n_forecast_days",
    help = "Number of days to forecast",
    default = 28L
  ) %>%
  add_argument("--n_samples",
    help = "Number of samples to generate",
    default = 2000L
  )

argv <- parse_args(p)
model_dir <- path(argv$model_dir)
n_forecast_days <- argv$n_forecast_days
n_samples <- arv$n_samples

fit_and_forecast <- function(denom_data,
                             n_forecast_days = 28,
                             n_samples = 2000) {
  forecast_horizon <- glue("{n_forecast_days} days")

  fit <-
    denom_data %>%
    filter(data_type == "train") %>%
    model(
      comb_model = combination_ensemble(
        ETS(log(ED_admissions) ~ trend(method = c("N", "M", "A"))),
        ARIMA(log(ED_admissions))
      ),
      arima = ARIMA(log(ED_admissions)),
      ets = ETS(log(ED_admissions) ~ trend(method = c("N", "M", "A")))
    )

  forecast_samples <- fit |>
    generate(h = forecast_horizon, times = n_samples) |>
    as_tibble() %>%
    mutate(ED_admissions = .sim, .draw = as.integer(.rep)) |>
    filter(.model == "comb_model") %>%
    select(date, .draw, ED_admissions)

  forecast_samples
}

main <- function(model_dir, n_forecast_days = 28, n_samples = 2000) {
  # to do: do this with json data that has dates
  data_path <- path(model_dir, "data", ext = "csv")

  denom_data <- read_csv(data_path) %>%
    filter(disease == "Total") %>%
    select(-disease) %>%
    as_tsibble(index = date)

  forecast_samples <- fit_and_forecast(denom_data, n_forecast_days, n_samples)

  save_path <- path(model_dir, "total_ed_admissions_forecast", ext = "parquet")
  write_parquet(forecast_samples, save_path)
}


main(model_dir, n_forecast_days)
# File will end here once command line version is working
# Temp code to run for all states while command line version doesn't work
# Command line version is dependent on https://github.com/rstudio/renv/pull/2018

base_dir <- path(
  "nssp_demo/private_data/influenza_r_2024-10-21_f_2024-07-16_t_2024-10-13"
)

dir_ls(base_dir, type = "dir") %>%
  map(.f = function(model_dir) {
    print(path_file(model_dir))
    main(model_dir, n_forecast_days = 28)
  })
