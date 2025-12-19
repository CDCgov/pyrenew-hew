#' Validate NHSN signal and geo_type parameters
#'
#' @description
#' Internal helper function to validate `signal` and `geo_type` parameters for
#' target nowcast NHSN data fetching functions. Note that the preliminary
#' signals are the targets for nowcasting in this project due to Wednesday
#' releases.
#'
#' @param signal Character string to validate against target NHSN signals
#' @param geo_type Character string to validate against valid geographic types
#'
#' @return NULL (called for side effects - throws error if invalid)
#' @keywords internal
#' @noRd
validate_nhsn_params <- function(signal, geo_type) {
  valid_signals <- c(
    "confirmed_admissions_covid_ew_prelim",
    "confirmed_admissions_flu_ew_prelim",
    "confirmed_admissions_rsv_ew_prelim"
  )

  if (!signal %in% valid_signals) {
    rlang::abort(
      paste0(
        "Not a target signal. Must be one of: ",
        paste(valid_signals, collapse = ", ")
      )
    )
  }

  valid_geo_types <- c("state", "hhs", "nation")
  if (!geo_type %in% valid_geo_types) {
    rlang::abort(
      paste0(
        "Invalid geo_type. Must be one of: ",
        paste(valid_geo_types, collapse = ", ")
      )
    )
  }

  invisible(NULL)
}


#' Fetch NHSN hospital admissions data
#'
#' @description
#' Fetches target NHSN (National Healthcare Safety Network) hospital respiratory
#' admissions data from the `epidatr` R client for the Delphi Epidata API. This
#' is just a closure around `epidatr::pub_covidcast()` with validation for the
#' target signals and parameters used in this nowcasting and checking that the
#' output is non-empty.
#' NB:
#' - For `time_values` and `issues`, you can pass an `epidatr::timeset` object
#' however this source only supports epiweeks (not dates).
#' - Output week dates are converted from Sundays (Epidata) to Saturdays to
#' align with reporting dates from forecast hub data.
#'
#' @param signal Character string specifying the NHSN signal to fetch.
#'   Default is `"confirmed_admissions_covid_ew_prelim"`.
#' @param geo_type Character string specifying geographic resolution.
#'   Options: `"state"`, `"hhs"`, `"nation"`. Default is `"state"`.
#' @param geo_values Character vector of geography codes or `"*"` for all.
#'   Default is `"*"`.
#' @param time_values A timeset object (e.g., `epirange(202401, 202404)`)
#'   or `"*"` for all available dates. Default is `"*"`.
#' @param issues A timeset object for specific issue dates, or `"*"` for most
#'   recent. Default is `"*"`.
#' @return A tibble with columns including:
#'   - `geo_value`: Geographic identifier
#'   - `time_value`: Reference date (week-ending Sunday from Epidata)
#'   - `value`: Number of confirmed admissions
#'   - `issue`: Issue date (week-ending Sunday of the epiweek when data was
#' published)
#'   - `lag`: Days between reference date and issue date
#'   - `signal`: Signal name
#'   - Additional metadata columns from `epidatr::pub_covidcast()`
#' @export
fetch_nhsn_data <- function(
  signal = "confirmed_admissions_covid_ew_prelim",
  geo_type = "state",
  geo_values = "*",
  time_values = "*",
  issues = "*"
) {
  # Input validation
  validate_nhsn_params(signal, geo_type)

  # Fetch data using epidatr
  nhsn_data <- epidatr::pub_covidcast(
    source = "nhsn",
    signals = signal,
    geo_type = geo_type,
    time_type = "week",
    time_values = time_values,
    geo_values = geo_values,
    issues = issues,
  )

  # Output validation
  if (nrow(nhsn_data) == 0) {
    rlang::warn("No data returned for the specified parameters")
  }
  nhsn_data
}


#' Fetch NHSN data formatted for reporting triangle construction
#'
#' @description
#' Wrapper around [fetch_nhsn_data()] that reshapes and converts the output
#' to a long format suitable for [baselinenowcast::as_reporting_triangle()].
#' Converts week-ending dates from Sunday (Epidata format) to Saturday
#' (forecast hub format).
#'
#' @param signal Character string specifying the NHSN signal to fetch.
#'   Default is `"confirmed_admissions_covid_ew_prelim"`.
#'
#' @param geo_type Character string specifying geographic resolution.
#'   Options: `"state"`, `"hhs"`, `"nation"`. Default is `"state"`.
#'
#' @param geo_values Character vector of geography codes,
#'   or `"*"` for all geographies. Default is `"*"`.
#'
#' @param reference_dates A timeset object (e.g., `epirange(202401, 202404)`)
#'   or `"*"` for all available dates. Default is `"*"`.
#'
#' @param report_dates A timeset object for specific report dates, or `"*"`
#'   for most all. Default is `"*"`.
#'
#' @return A data frame in long format suitable for
#' [baselinenowcast::as_reporting_triangle()]
#'   with columns:
#'   - `reference_date`: Date of the event (week-ending Saturday)
#'   - `report_date`: Date when data was reported. This is the
#' week-ending Saturday _after_ the epiweek when data was published.
#'   - `count`: Number of confirmed admissions
#'   - `location`: Geographic identifier
#'   - `signal`: Signal name
#' @export
fetch_reporting_data_epidatr <- function(
  signal = "confirmed_admissions_covid_ew_prelim",
  geo_type = "state",
  geo_values = "*",
  reference_dates = "*",
  report_dates = "*"
) {
  results <- fetch_nhsn_data(
    signal = signal,
    geo_type = geo_type,
    geo_values = geo_values,
    time_values = reference_dates,
    issues = report_dates
  )
  # Format for baselinenowcast
  results <- results |>
    dplyr::select(
      reference_date = time_value,
      report_date = issue,
      count = value,
      location = geo_value,
      signal
    ) |>
    dplyr::mutate(
      # Convert dates for weekly reporting from Sunday (epidatr) to Saturday
      # (as reported on GitHub)
      # reference_date: shift back 1 day
      # report_date: shift forward 6 days
      reference_date = reference_date - 1,
      report_date = report_date + 6
    ) |>
    dplyr::arrange(reference_date, report_date)

  results
}
