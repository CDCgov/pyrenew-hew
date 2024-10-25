library(arrow)
library(dplyr)
library(lubridate)

#' Generate Example Prediction Data
#'
#' This function generates example prediction data for a specified number of
#' days, areas, chains and repetitions per chain. The data is log-normally
#' distributed with a meanlog of 1.0 and a sdlog of 0.25. The data is returned
#' as a tibble in tidybayes format and can be optionally saved to a specified
#' path with a default path of "scortingutilhelpers/assets" and a default file
#' format of Arrow/Parquet.
#'
#' @param save_path A character string specifying the path where the data
#' should be saved if `save_data` is TRUE.
#' @param ndays An integer specifying the number of days for which to generate
#' data. Default is 21.
#' @param reps An integer specifying the number of repetitions for each
#' day/location. Default is 100.
#' @param nchains An integer specifying the number of chains for which to
#' generate data. NB: this is not a MCMC method but meant to test expected
#' output. Default is 4.
#' @param nareas An integer specifying the number of areas for which to generate
#' data. Areas are "A", "B", etc. Default is 3.
#' @param save_data A logical value indicating whether to save the generated data
#' to `save_path`. Default is FALSE.
#' @param ... Additional arguments passed to `arrow::write_dataset` if
#' `save_data` is TRUE.
#'
#' @return A tibble containing the generated example prediction data with
#' columns:
#' \describe{
#'  \item{area}{Area identifier for each observation.}
#'   \item{date}{Date for each observation.}
#'   \item{prediction}{Generated value for each observation.}
#'   \item{.chain}{Chain identifier for the data generation process.}
#'   \item{.iteration}{Iteration number for each observation.}
#'   \item{.draw}{Row number for each observation.}
#'   \item{model}{Model identifier for the data generation process.}
#' }
#' @export
example_prediction <- function(
    save_path = "scoringutilhelpers/assets/example_predictions", ndays = 21,
    reps = 100, nchains = 4, nareas = 3, save_data = FALSE, ...) {
  # Generate a sequence of dates for 3 weeks
  dates <- seq.Date(
    from = lubridate::ymd("2024-10-24"), by = "day",
    length.out = ndays
  )
  dates_1wkahead <- seq.Date(
    from = lubridate::ymd("2024-10-31"), by = "day",
    length.out = ndays
  )
  dates_2wkahead <- seq.Date(
    from = lubridate::ymd("2024-11-7"), by = "day",
    length.out = ndays
  )
  areas <- LETTERS[1:nareas]
  # Create log-normally distributed data for each date and area
  # sending to tidydata
  exampledata <- lapply(
    areas,
    function(area) {
      lapply(
        1:nchains,
        function(i) {
          data1wk <- tibble(
            area = area,
            reference_date = rep(dates, each = reps),
            target_end_date = rep(dates_1wkahead, each = reps),
            prediction = rlnorm(reps * ndays, meanlog = log(1.0), sdlog = 0.25),
            .chain = i,
            .iteration = 1:(reps * ndays),
          )
          data2wk <- tibble(
            area = area,
            reference_date = rep(dates, each = reps),
            target_end_date = rep(dates_2wkahead, each = reps),
            prediction = rlnorm(reps * ndays, meanlog = log(1.0), sdlog = 0.25),
            .chain = i,
            .iteration = 1:(reps * ndays),
          )
          data <- bind_rows(data1wk, data2wk)
        }
      ) |>
        bind_rows()
    }
  ) |>
    bind_rows() |>
    mutate(
      .draw = row_number(), model = "example"
    )
  if (save_data) {
    arrow::write_dataset(exampledata, save_path, ...)
  }
  return(exampledata)
}
