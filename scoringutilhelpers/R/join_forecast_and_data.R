library(arrow)
library(readr)
library(dplyr)

#' Join Forecast and Actual Data
#'
#' This function reads forecast data from a Parquet file and actual data from a
#' TSV file, then joins them using a specified key.
#'
#' @param forecast_sources A character vector specifying the path to the Parquet file containing forecast data.
#' @param data_path A character string specifying the path to the TSV file containing actual data.
#' @param join_key A character vector specifying the key(s) to join the forecast and actual data on.
#' @param ... Additional arguments passed to `arrow::read_parquet`.
#'
#' @return A data frame resulting from the left join of the forecast and actual data.
#' @importFrom arrow read_parquet
#' @importFrom readr read_tsv
#' @importFrom dplyr left_join
#' @export
join_forecast_and_data <- function(forecast_sources, data_path, join_key, ...) {
  predictions <- arrow::read_parquet(forecast_sources, ...)
  actual_data <- readr::read_tsv(data_path)
  joined_data <- dplyr::left_join(predictions, actual_data, by = join_key)
  return(joined_data)
}