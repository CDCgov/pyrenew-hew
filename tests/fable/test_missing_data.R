source("src/cfa/stf/routine/fable/fit_fable.R")

testthat::test_that("joint draws match the analytic Kalman forecast across gaps", {
  set.seed(1343)
  x <- tibble::tibble(
    date = as.Date("2026-01-01") + 0:83,
    y = exp(
      4 + as.numeric(stats::arima.sim(list(ar = 0.8), n = 84, sd = 0.1))
    ) -
      1
  )
  x$y[c(1:4, 25:33, 80:84)] <- NA_real_
  fit <- x |>
    tsibble::as_tsibble(index = date) |>
    model(comb_model = ARIMA(log(y + 1) ~ 1 + pdq(1, 0, 0) + PDQ(0, 0, 0)))
  predicted <- forecast(fit, h = 7)
  location <- log(unlist(stats::quantile(predicted$y, 0.5)) + 1)
  sd <- log(unlist(stats::quantile(predicted$y, stats::pnorm(1))) + 1) -
    location
  draws <- generate_missing_arima(fit, "7 days", 40000L, "y", 1)
  samples <- matrix(log(draws$.sim + 1), ncol = 7)
  testthat::expect_true(all(is.finite(samples)))
  testthat::expect_equal(colMeans(samples), location, tolerance = 0.005)
  testthat::expect_equal(apply(samples, 2, stats::sd), sd, tolerance = 0.005)
  phi <- unname(fit$comb_model[[1]]$fit$model$coef["ar1"])
  expected_covariance <- sd[1]^2 * phi
  testthat::expect_equal(
    stats::cov(samples)[1, 2],
    expected_covariance,
    tolerance = 0.002
  )
})

testthat::test_that("missing observations select ARIMA and yield finite forecasts", {
  set.seed(1343)
  x <- tibble::tibble(
    date = as.Date("2026-01-01") + 0:83,
    ed_visits = 100 + stats::rpois(84, 10)
  )
  x$ed_visits[c(1:4, 40:45, 80:84)] <- NA_real_
  testthat::expect_warning(
    draws <- fit_and_forecast_ensemble(x, n_samples = 100),
    "using ARIMA without the ETS component"
  )
  testthat::expect_identical(attr(draws, "components"), "ARIMA")
  testthat::expect_true(all(is.finite(draws$other_ed_visits)))
  testthat::expect_equal(max(draws$date), max(x$date) + 28)
})

testthat::test_that("joint draws also handle ordinary and seasonal differencing", {
  set.seed(1343)
  x <- tibble::tibble(
    date = as.Date("2026-01-01") + 0:83,
    y = exp(4 + cumsum(stats::rnorm(84, sd = 0.03))) - 1
  )
  x$y[c(1:4, 25:33, 80:84)] <- NA_real_
  series <- tsibble::as_tsibble(x, index = date)
  specifications <- list(
    ARIMA(log(y + 1) ~ 0 + pdq(0, 1, 1) + PDQ(0, 0, 0)),
    ARIMA(log(y + 1) ~ 0 + pdq(0, 0, 1) + PDQ(0, 1, 1))
  )
  purrr::walk(specifications, function(specification) {
    fit <- model(series, comb_model = specification)
    predicted <- forecast(fit, h = 7)
    location <- log(unlist(stats::quantile(predicted$y, 0.5)) + 1)
    sd <- log(unlist(stats::quantile(predicted$y, stats::pnorm(1))) + 1) -
      location
    draws <- generate_missing_arima(fit, "7 days", 40000L, "y", 1)
    samples <- matrix(log(draws$.sim + 1), ncol = 7)
    testthat::expect_equal(colMeans(samples), location, tolerance = 0.005)
    testthat::expect_equal(apply(samples, 2, stats::sd), sd, tolerance = 0.005)
  })
})
