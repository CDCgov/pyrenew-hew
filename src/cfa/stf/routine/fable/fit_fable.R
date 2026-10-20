script_packages <- c(
  "dplyr",
  "tidyr",
  "tibble",
  "readr",
  "stringr",
  "fs",
  "fable",
  "glue",
  "argparser",
  "rlang",
  "stfroutineforecasting",
  "forecasttools",
  "urca"
)

## load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

# Draw jointly from the Gaussian state-space forecast. Fable's ARIMA generate()
# conditions on lagged observations/residuals directly, which propagates tail
# NAs into every draw. The filtered state retains uncertainty across those gaps.
generate_missing_arima <- function(
  fit,
  horizon,
  n_samples,
  target_col,
  offset
) {
  predicted <- forecast(fit, h = horizon)
  arima <- fit$comb_model[[1]]$fit$model
  state <- arima$model
  n <- nrow(predicted)
  covariance <- matrix(0, n, n)
  P <- state$P
  for (i in seq_len(n)) {
    P <- state$T %*% P %*% t(state$T) + state$V
    covariance[i, i] <- drop(crossprod(state$Z, P %*% state$Z)) + state$h
    cross <- P
    if (i < n) {
      for (j in seq.int(i + 1L, n)) {
        cross <- cross %*% t(state$T)
        covariance[i, j] <- covariance[j, i] <- drop(crossprod(
          state$Z,
          cross %*% state$Z
        ))
      }
    }
  }
  covariance <- covariance * arima$sigma2
  # The response distribution is exp(N(mu, sigma^2)) - offset, so its median
  # recovers the Gaussian location without confusing it with the response mean.
  location <- log(
    unlist(stats::quantile(predicted[[target_col]], 0.5)) + offset
  )
  draws <- mvtnorm::rmvnorm(n_samples, mean = location, sigma = covariance)
  tibble::tibble(
    date = rep(predicted$date, each = n_samples),
    .rep = rep(seq_len(n_samples), n),
    .sim = as.vector(exp(draws) - offset)
  )
}


#' Fit and Forecast Time Series Data
#'
#' This function fits a combination ensemble model to the training data and
#' generates forecast samples for a specified number of days.
#'
#' @param data A data frame containing only the training time series data.
#' @param n_forecast_days An integer specifying the number of days to forecast.
#' Default is 28.
#' @param n_samples An integer specifying the number of forecast samples to
#' generate. Default is 2000.
#' @param target_col A string specifying the name of the target column in the
#' data. Default is "ed_visits".
#' @param output_col A string specifying the name of the output column for the
#' forecasted values. Default is "other_ed_visits".
#
#' @return A tibble containing the forecast samples with columns for date,
#' draw number, and forecasted values.
fit_and_forecast_ensemble <- function(
  data,
  n_forecast_days = 28,
  n_samples = 2000,
  target_col = "ed_visits",
  output_col = "other_ed_visits"
) {
  forecast_horizon <- glue::glue("{n_forecast_days} days")
  target_sym <- rlang::sym(target_col)

  offset <- 1

  series <- data |> as_tsibble(index = date) |> tsibble::fill_gaps()
  has_missing <- anyNA(series[[target_col]])
  if (all(is.na(series[[target_col]]))) {
    stop("At least one finite training observation is required")
  }
  # Seasonal ETS cannot estimate its initial state with internal missing values.
  # ARIMA integrates missing observations through its state-space likelihood.
  # Do not fill gaps with point estimates and treat them as observed counts.
  fit <- if (has_missing) {
    warning(
      "Missing training observations: using ARIMA without the ETS component"
    )
    series |>
      model(
        comb_model = ARIMA(log(!!target_sym + !!offset), approximation = FALSE)
      )
  } else {
    series |>
      model(
        comb_model = combination_ensemble(
          ETS(
            log(!!target_sym + !!offset) ~
              trend(
                method = c("N", "M", "A")
              )
          ),
          ARIMA(log(!!target_sym + !!offset))
        )
      )
  }

  generated <- if (has_missing) {
    generate_missing_arima(fit, forecast_horizon, n_samples, target_col, offset)
  } else {
    generate(fit, h = forecast_horizon, times = n_samples)
  }
  forecast_samples <- generated |>
    as_tibble() |>
    mutate(
      !!output_col := pmax(.data$.sim, 0), # clip values
      .draw = as.integer(.data$.rep)
    ) |>
    select("date", ".draw", all_of(output_col))

  if (
    !all(is.finite(forecast_samples[[output_col]])) ||
      any(forecast_samples[[output_col]] < 0)
  ) {
    stop(glue::glue("Nonfinite or negative count forecast for {output_col}"))
  }

  attr(forecast_samples, "fitted") <- augment(fit)
  attr(forecast_samples, "components") <- if (has_missing) {
    "ARIMA"
  } else {
    "ETS + ARIMA"
  }
  forecast_samples
}

main <- function(
  model_dir,
  n_forecast_days = 28,
  n_samples = 2000
) {
  training_data <- stfroutineforecasting::load_training_data(
    model_dir,
    "combined_data"
  )

  target_and_other_data <- training_data$data
  geo_value <- training_data$geo_value
  disease <- training_data$disease
  resolution <- training_data$resolution

  ## Fit and forecast other (non-target-disease) ED visits using a combination
  ## ensemble model
  fable_e_other <- fit_and_forecast_ensemble(
    target_and_other_data,
    n_forecast_days,
    n_samples,
    target_col = "other_ed_visits",
    output_col = "other_ed_visits"
  )

  readr::write_tsv(
    attr(fable_e_other, "fitted"),
    fs::path(model_dir, "fitted.tsv")
  )
  readr::write_lines(
    attr(fable_e_other, "components"),
    fs::path(model_dir, "components.txt")
  )

  fable_e_other_forecast <-
    fable_e_other |>
    stfroutineforecasting::format_timeseries_output(
      geo_value = geo_value,
      disease = disease,
      resolution = resolution,
      output_type_id = ".draw"
    )

  # Save the forecast
  write_tabular(
    fable_e_other_forecast,
    path(
      model_dir,
      "samples",
      ext = "parquet"
    )
  )
}


run_cli <- function() {
  p <- arg_parser(
    "Forecast other (non-target-disease) ED visits for a given location."
  ) |>
    add_argument(
      "--model-dir",
      help = "Directory containing the model data and output."
    ) |>
    add_argument(
      "--n-forecast-days",
      help = "Number of days to forecast.",
      default = 28L
    ) |>
    add_argument(
      "--n-samples",
      help = "Number of samples to generate.",
      default = 2000L
    )

  argv <- parse_args(p)
  model_dir <- path(argv$model_dir)
  n_forecast_days <- argv$n_forecast_days
  n_samples <- argv$n_samples

  main(
    model_dir,
    n_forecast_days = n_forecast_days,
    n_samples = n_samples
  )
}

if (sys.nframe() == 0L) {
  run_cli()
}
