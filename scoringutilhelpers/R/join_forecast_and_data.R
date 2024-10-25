# Load necessary libraries
library(arrow)
library(readr)
library(dplyr)

join_forecast_and_data <- function(forecast_sources, data_path, join_key, ...) {
  predictions <- arrow::read_parquet(forecast_sources, ...)
  actual_data <- readr::read_tsv(data_path)
  joined_data <- dplyr::left_join(predictions, actual_data, by = join_key)
  return(joined_data)
}