library(dplyr)
library(readr)

#' Generate Example Truth Data
#'
#' This function generates example truth data for a specified number of days and
#' areas. The data is log-normally distributed and can optionally be saved to a
#' specified path.
#'
#' @param savepath A character string specifying the path where the data
#' should be saved. Default is "scortingutilhelpers/assets".
#' @param ndays An integer specifying the number of days for which to generate
#' data. Default is 21.
#' @param nareas An integer specifying the number of areas for which to generate
#' data. Default is 3.
#' @param savedata A logical value indicating whether to save the generated
#' data. Default is FALSE.
#' @param ... Additional arguments passed to `arrow::write_dataset` if
#' `savedata` is TRUE.
#'
#' @return A tibble containing the generated example truth data with columns for
#' area, date, and truthdata.
#' @export
exampletruthdata <- function(
  savepath = "scortingutilhelpers/assets/exampletruthdata",
  ndays = 21, nareas = 3, savedata = FALSE, ...) {
  # Generate a sequence of dates and a sequence of areas
  dates <- seq.Date(from = lubridate::ymd("2024-10-24"), by = "day",
    length.out = ndays)
  areas <- LETTERS[1:nareas]
  # Create log-normally distributed truth data
  exampledata <- lapply(areas,
    function(area) {
    data <- tibble(
            area = area,
            date = dates,
            truthdata = rlnorm(ndays, meanlog = log(1.0), sdlog = 0.25),
            )
    }) |>
    bind_rows()
    if (savedata) {
        readr::write_tsv(exampledata, savepath, ...)
    }
    return(exampledata)
}
