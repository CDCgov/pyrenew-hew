library(fable)
library(fabletools)
library(tsibble)
library(dplyr)

#' Forecast Counts
#'
#' This function forecasts counts from a given dataset using an ensemble of
#' Exponential Smoothing State Space Model (ETS) and seasonal AutoRegressive Integrated
#' Moving Average (SARIMA) models. It returns both predictive samples and a
#' forecast object  with prediction intervals. In the presence of missing values,
#' the function defaults to an SARIMA model.
#'
#' The expected use-case of this function is short term forecasting of count data
#' for the numberator of a rate calculation. For example, the numerator for
#' calculating the proportion of emergency department visits that are due to
#' a target pathogen.
#'
#' @param edvisitdata A data frame containing the count data.
#' @param count_col A string specifying the column name of the count data.
#' @param date_col A string specifying the column name of the date data.
#' @param h A string specifying the forecast horizon (default is "3 weeks").
#'
#' @return A list containing:
#' \item{predictive_samples}{A tsibble of predictive samples generated from the model.}
#' \item{fc}{A fable object containing the forecasted values with prediction intervals.}
forecast_counts <- function(edvisitdata, count_col, date_col, times = 2000, h = "3 weeks") {
  # Convert the data frame to a tsibble
  count_tsibble <- edvisitdata |>
    as_tsibble(index = .data[[date_col]]) |>
    mutate(target = .data[[count_col]])
  # Fit a model using fable with a combination ensemble of ETS and ARIMA
  count_sym <- rlang::sym(count_col)
  date_sym <- rlang::sym(date_col)
  fit <- if (anyNA(edvisitdata[[count_col]])) {
    message("The count column contains missing values. Defaulting to (S)ARIMA model.")
    count_tsibble |>
      model(
        arima = ARIMA(log(!!count_sym)),
      )
  } else {
    count_tsibble |>
      model(
        comb_model = combination_ensemble(
          ETS(log(!!count_sym) ~ trend(method = c("N", "M", "A"))),
          ARIMA(log(!!count_sym))
        ),
        arima = ARIMA(log(!!count_sym)),
        ets = ETS(log(!!count_sym) ~ trend(method = c("N", "M", "A")))
      )
  }
  # Produce forecasts
  predictive_samples <- fit |>
    generate(h = h, times = times) |>
    mutate(value = .sim, .draw = as.numeric(.rep)) |>
    select(!!date_sym, .draw, value)
  # Calculate the forecasted values with prediction intervals
  fc <- fit |>
    forecast(h = h)
  # Return the results
  return(list(predictive_samples = predictive_samples, fc = fc))
}
