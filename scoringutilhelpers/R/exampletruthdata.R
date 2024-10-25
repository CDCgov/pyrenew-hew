library(dplyr)
library(readr)

exampletruthdata <- function(sourcepath = "scortingutilhelpers/assets",
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
        arrow::write_dataset(exampledata, sourcepath, ...)
    }
    return(exampledata)
}
