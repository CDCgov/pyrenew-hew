library(epidatr)
library(dplyr)

#' Generate Example Data
#'
#' This function generates example data for a specified target
#' respiratory pathogen e.g., "covid", "influenza", or "rsv",
#' and saves it to a CSV file in the specified output directory.
#'
#' We assume that the total number of ED visits per week for the whole
#' US that are not respiratory virus related are ~ 2.6 million,
#' based on https://www.cdc.gov/nchs/dhcs/ed-visits/index.htm . We can
#' then estimate the number of respiratory virus related ED visits
#' per week using the percentage of ED visits that are respiratory virus
#' related. We can then estimate the number of ED visits related to
#' the target respiratory virus and all other ED visits that are
#' either respiratory virus related for a non-target respiratory
#' virus or not respiratory virus related.
#'
#' @param target A character string specifying the target for which to
#' generate data. Valid options are "covid", "influenza", or "rsv".
#' Default is "covid".
#' @param est_non_resp A numeric value representing the estimated number of
#' non-respiratory ED visits per week. Default is 2.6e6.
#' @param output_dir A character string specifying the directory where the
#' output CSV file will be saved. Default is "countpredr/local_assets".
#' @param savedata A logical value indicating whether to save the generated data
#' to a CSV file. Default is TRUE.
#'
#' @return An example dataset containing the estimated number of ED visits
#' related to the target respiratory virus and all other ED visits.
#'
#' @details The function performs the following steps:
#' \itemize{
#'   \item Creates the output directory if it does not exist.
#'   \item Validates the target respiratory virus is one of the available
#' NSSP options.
#'   \item Retrieves and processes data from the "nssp" source using the
#' `epidatr::pub_covidcast` function.
#'   \item Calculates various estimates and combines them into a
#' `fabletools::tsibble`.
#'   \item Saves the resulting data to a CSV file in the specified output
#' directory.
#' }
#' @export
generate_example_data <- function(
    target = "covid",
    est_non_resp = 2.6e6,
    output_dir = "countpredr/local_assets",
    savedata = TRUE) {
  # Create the output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # check if the target is valid
  signal <- if (target == "covid") {
    "pct_ed_visits_covid"
  } else if (target == "influenza") {
    "pct_ed_visits_influenza"
  } else if (target == "rsv") {
    "pct_ed_visits_rsv"
  } else {
    stop(
      "Invalid target: ", target,
      ", please choose from 'covid', 'influenza', or 'rsv'"
    )
  }

  all_resp_nssp <- pub_covidcast(
    source = "nssp",
    signals = "pct_ed_visits_combined",
    geo_type = "nation",
    time_type = "week",
  ) |>
    rename(all_pct = value, date = time_value) |>
    select(date, all_pct)

  target_resp_nssp <- pub_covidcast(
    source = "nssp",
    signals = signal,
    geo_type = "nation",
    time_type = "week",
  ) |>
    rename(target_pct = value, date = time_value) |>
    select(date, target_pct)

  exampledata <- left_join(all_resp_nssp, target_resp_nssp,
    by = join_by(date)
  ) |>
    mutate(resp_est = all_pct * est_non_resp / (100 - all_pct)) |>
    mutate(target_resp_est = resp_est * target_pct / all_pct) |>
    mutate(non_target_resp_est = resp_est - target_resp_est) |>
    mutate(other_ed_visits = non_target_resp_est + est_non_resp) |>
    select(date, target_resp_est, other_ed_visits)

  # Save the data to a CSV file
  if (savedata) {
    output_file <- file.path(output_dir, paste0("exampledata_", target, ".csv"))
    write.csv(exampledata, output_file, row.names = FALSE)

    message("Example data saved to ", output_file)
  }

  return(exampledata)
}
