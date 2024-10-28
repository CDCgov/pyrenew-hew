library(dplyr)
library(readr)
library(tidyr)

#' Generate Example Truth Data
#'
#' This function generates example truth data for a specified number of days and
#' areas. The data is log-normally distributed and can optionally be saved to a
#' specified path.
#'
#' @param save_path A character string specifying the path where the data
#' should be saved. Default is "scortingutilhelpers/assets".
#' @param n_days An integer specifying the number of days for which to generate
#' data. Default is 21.
#' @param n_areas An integer specifying the number of areas for which to
#' generate data. Default is 3.
#' @param save_data A logical value indicating whether to save the generated
#' data. Default is FALSE.
#' @param ... Additional arguments passed to `readr::write_tsv` if
#' `save_data` is TRUE.
#'
#' @return A tibble containing the generated example truth data with columns for
#' area, date, and truthdata.
#' @export
example_truthdata <- function(
    save_path = "scoringutilhelpers/assets",
    n_days = 21, n_areas = 3, save_data = FALSE, ...) {
  # Generate a sequence of dates and a sequence of areas
  dates <- seq.Date(
    from = lubridate::ymd("2024-10-24"), by = "day",
    length.out = n_days
  )
  areas <- LETTERS[1:n_areas]
  # Create log-normally distributed truth data
  exampledata <- tidyr::expand_grid(
    area = areas,
    date = list(dates)
  ) |>
    tidyr::unnest(date) |>
    mutate(truthdata = rlnorm(n(), meanlog = log(1.0), sdlog = 0.25))



  if (save_data) {
    readr::write_tsv(
      exampledata,
      file.path(save_path, "example_truthdata.tsv"), ...
    )
  }
  return(exampledata)
}
