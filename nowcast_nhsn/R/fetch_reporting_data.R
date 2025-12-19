#' Convert dates to epirange format
#'
#' @param x Date vector or epirange object
#' @return epirange object
#' @noRd
as_epirange <- function(x) {
  if (inherits(x, "Date")) {
    validate_all_saturdays(x)
    x <- saturdays_to_epirange(x)
  }
  x
}

#' Create an epidata data source object
#'
#' @param signal Character, epidata signal name (e.g., "confirmed_admissions_covid_ew_prelim")
#' @param geo_types Character vector, geographic types to query (e.g., "state", "nation")
#' @return A source object of class "epidata_source"
#' @export
epidata_source <- function(
  signal,
  geo_types = c("state", "nation")
) {
  structure(
    list(
      signal = signal,
      geo_types = geo_types
    ),
    class = c("epidata_source", "reporting_source")
  )
}

#' Create a GitHub data source object
#'
#' @param signal Character, disease signal ("covid", "flu", or "rsv")
#' @param repo Character, GitHub repository (e.g., "CDCgov/covid19-forecast-hub"). Defaults based on signal.
#' @param file_path Character, path to target data file within repo. Defaults based on signal.
#' @param github_token Character, GitHub personal access token for authentication
#' @return A source object of class "github_source"
#' @export
github_source <- function(
  signal = "covid",
  repo = NULL,
  file_path = NULL,
  github_token = NULL
) {
  # Validate signal
  signal <- match.arg(signal, c("covid", "flu", "rsv"))

  # Set defaults based on signal if not provided
  if (is.null(repo)) {
    repo <- switch(
      signal,
      covid = "CDCgov/covid19-forecast-hub",
      flu = "cdcepi/Flusight-forecast-data",
      rsv = "CDCgov/rsv-forecast-hub"
    )
  }

  if (is.null(file_path)) {
    file_path <- switch(
      signal,
      covid = "target-data/covid-hospital-admissions.csv",
      flu = "target-data/target-hospital-admissions.csv",
      rsv = "target-data/time-series.parquet"
    )
  }

  structure(
    list(
      signal = signal,
      repo = repo,
      file_path = file_path,
      github_token = github_token
    ),
    class = c("github_source", "reporting_source")
  )
}

#' Fetch reporting data from a source object
#'
#' Generic function to fetch reporting triangle data using S3 method dispatch.
#'
#' @param source A source object created by [epidata_source()] or [github_source()]
#' @param reference_dates Date vector or epirange of reference dates
#' @param report_dates Date vector or epirange of report dates
#' @param locations Character vector of locations
#' @param ... Additional arguments passed to methods
#' @return data.frame with columns: reference_date, report_date, location, count, signal
#' @export
fetch_reporting_data <- function(source, reference_dates, report_dates, locations, ...) {
  UseMethod("fetch_reporting_data")
}

#' Fetch reporting data from epidata
#'
#' @param source An epidata_source object
#' @param reference_dates Date vector or epirange of reference dates
#' @param report_dates Date vector or epirange of report dates
#' @param locations Character vector of locations ("*" for all)
#' @param ... Additional arguments (unused)
#' @return data.frame with reporting triangle data
#' @export
fetch_reporting_data.epidata_source <- function(
  source,
  reference_dates,
  report_dates,
  locations = "*",
  ...
) {
  # Convert to epirange if Dates are provided, otherwise use as-is
  report_dates <- as_epirange(report_dates)
  reference_dates <- as_epirange(reference_dates)

  purrr::map_dfr(source$geo_types, function(geo_type) {
    fetch_reporting_data_epidatr(
      signal = source$signal,
      geo_type = geo_type,
      geo_values = locations,
      reference_dates = reference_dates,
      report_dates = report_dates
    )
  })
}

#' Fetch reporting data from GitHub
#'
#' @param source A github_source object
#' @param reference_dates Date vector of reference dates (Saturdays)
#' @param report_dates Date vector of report dates (Saturdays)
#' @param locations Character vector of locations (NULL for all)
#' @param ... Additional arguments (unused)
#' @return data.frame with reporting triangle data
#' @export
fetch_reporting_data.github_source <- function(
  source,
  reference_dates,
  report_dates,
  locations = NULL,
  ...
) {
  # Validate report_dates are Saturdays
  validate_all_saturdays(report_dates)
  validate_all_saturdays(reference_dates)
  # Dispatch based on signal
  fetch_github_by_signal(
    signal = source$signal,
    reference_dates = reference_dates,
    report_dates = report_dates,
    locations = locations,
    repo = source$repo,
    file_path = source$file_path,
    github_token = source$github_token
  )
}

#' Internal S3 generic for signal-based GitHub dispatch
#'
#' @param signal Character, disease signal
#' @param ... Additional arguments passed to methods
#' @return data.frame with reporting triangle data
#' @noRd
fetch_github_by_signal <- function(signal, ...) {
  UseMethod("fetch_github_by_signal", structure(list(), class = signal))
}

#' Fetch COVID data from GitHub
#' @noRd
fetch_github_by_signal.covid <- function(
  signal,
  reference_dates,
  report_dates,
  locations,
  repo,
  file_path,
  github_token,
  ...
) {
  fetch_reporting_data_gh_csv(
    reference_dates = reference_dates,
    report_dates = report_dates,
    locations = locations,
    repo = repo,
    file_path = file_path,
    github_token = github_token
  )
}

#' Default method for unimplemented signals
#' @noRd
fetch_github_by_signal.default <- function(signal, ...) {
  rlang::abort(sprintf(
    "GitHub fetching not yet implemented for signal '%s'. Only 'covid' is currently supported.",
    signal
  ))
}

#' Default method for unsupported source types
#' @export
fetch_reporting_data.default <- function(source, ...) {
  rlang::abort(sprintf(
    "Don't know how to fetch data from source of class '%s'",
    paste(class(source), collapse = "', '")
  ))
}
