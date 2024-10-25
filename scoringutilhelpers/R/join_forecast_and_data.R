library(arrow)
library(readr)
library(dplyr)

#' Join Forecast and Actual Data
#'
#' This function reads forecast data from a Parquet file and actual data from a
#' TSV file, then joins them using a specified key.
#'
#' @param forecast_source A character vector specifying the path to the Parquet
#' file containing forecast data.
#' @param data_path A character string specifying the path to the TSV file
#' containing actual data.
#' @param join_key A character vector specifying the key(s) to join the forecast
#' and actual data on. Default is NULL.
#' @param ... Additional arguments passed to `arrow::open_dataset`.
#'
#' @return A data frame resulting from the left join of the forecast and actual
#' data.
#' @export
join_forecast_and_data <- function(forecast_source, data_path, join_key = NULL,
  ...) {
  predictions <- arrow::open_dataset(forecast_source, ...)
  actual_data <- readr::read_tsv(data_path)
  joined_data <- dplyr::left_join(predictions, actual_data, by = join_key)
  return(joined_data)
}
