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
#' @param sourcepath A character string specifying the path where the data
#' should be saved if `savedata` is TRUE.
#' @param ndays An integer specifying the number of days for which to generate
#' data. Default is 21.
#' @param reps An integer specifying the number of repetitions for each
#' day/location. Default is 100.
#' @param nchains An integer specifying the number of chains for which to
#' generate data. NB: this is not a MCMC method but meant to test expected
#' output. Default is 4.
#' @param nareas An integer specifying the number of areas for which to generate
#' data. Areas are "A", "B", etc. Default is 3.
#' @param savedata A logical value indicating whether to save the generated data
#' to `sourcepath`. Default is FALSE.
#' @param ... Additional arguments passed to `arrow::write_dataset` if
#' `savedata` is TRUE.
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
#' }
#' @export
exampleprediction <- function(
  sourcepath = "scoringutilhelpers/assets/examplepredictions", ndays = 21,
  reps = 100, nchains = 4, nareas = 3, savedata = FALSE, ...) {
  # Generate a sequence of dates for 3 weeks
  dates <- seq.Date(from = lubridate::ymd("2024-10-24"), by = "day",
    length.out = ndays)
  areas <- LETTERS[1:nareas]
  # Create log-normally distributed data for each date and area
  # sending to tidydata
  exampledata <- lapply(areas,
    function(area) {
    lapply(1:nchains,
        function(i) {
        data <- tibble(
            area = area,
            date = rep(dates, each = reps),
            prediction = rlnorm(reps * ndays, meanlog = log(1.0), sdlog = 0.25),
            .chain = i,
            .iteration = 1:(reps * ndays),
            )
        }) |>
        bind_rows()
    }) |>
    bind_rows() |>
    mutate(
      .draw = row_number()
    )
    if (savedata) {
        arrow::write_dataset(exampledata, sourcepath, ...)
    }
    return(exampledata)
}
