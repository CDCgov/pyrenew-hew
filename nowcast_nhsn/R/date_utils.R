#' Validate that all dates are Saturdays
#'
#' @description
#' Internal helper function to validate that all provided dates fall on
#' Saturday.
#' @noRd
validate_all_saturdays <- function(dates) {
  dates <- as.Date(dates)
  weekdays_check <- weekdays(dates)

  if (!all(weekdays_check == "Saturday")) {
    bad_dates <- dates[weekdays_check != "Saturday"]
    rlang::abort(sprintf(
      "All dates must be Saturdays. Got: %s",
      paste(format(bad_dates, "%Y-%m-%d (%A)"), collapse = ", ")
    ))
  }

  invisible(NULL)
}


#' Convert Saturday dates to `epidatr::epirange`
#'
#' @description
#' Converts a sequence of Saturday dates (forecast hub convention) to an
#' `epidatr::epirange` object (Sunday-ending weeks in YYYYWW format).
#' This allows using the same date inputs for both GitHub fetcher (Saturdays)
#' and epidatr fetcher (epiweeks).
#'
#' @param dates A vector of dates. Must all be Saturdays.
#'
#' @return An `epidatr::epirange` object covering the epiweeks corresponding
#'   to the input dates.
#'
#' @details
#' The function:
#' 1. Validates that all dates are Saturdays
#' 2. Adds one day to convert to Sunday (epiweeks end on Sunday)
#' 3. Converts to MMWR epiweek format (YYYYWW)
#' 4. Returns an `epidatr::epirange` spanning from the earliest to latest week
#'
#' **Why the conversion?**
#' - Forecast hub uses Saturday as the week-ending date
#' - MMWR/CDC epiweeks end on Sunday
#' - epidatr API requires epiweek format (YYYYWW)
#' @export
saturdays_to_epirange <- function(dates) {
  dates <- as.Date(dates)

  # Validate all dates are Saturdays
  validate_all_saturdays(dates)

  # Convert Saturdays to Sundays (epiweeks end on Sunday)
  sundays <- dates + 1

  # Convert to MMWR epiweek format (YYYYWW)
  epiweeks <- vapply(
    sundays,
    function(d) {
      mmwr <- MMWRweek::MMWRweek(d)
      mmwr$MMWRyear * 100 + mmwr$MMWRweek
    },
    numeric(1)
  )

  # Create epirange from min to max
  epidatr::epirange(min(epiweeks), max(epiweeks))
}
