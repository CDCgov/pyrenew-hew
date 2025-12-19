#' Fetch commit history for a file
#' @noRd
fetch_file_commits <- function(repo, file_path, token) {
  url <- sprintf("https://api.github.com/repos/%s/commits", repo)

  headers <- c(
    "Accept" = "application/vnd.github.v3+json",
    "User-Agent" = "nowcastNHSN-R-package"
  )

  if (nchar(token) > 0) {
    headers <- c(headers, "Authorization" = paste("token", token))
  }

  # Fetch commits (paginated)
  all_commits <- list()
  page <- 1
  per_page <- 100

  repeat {
    response <- httr::GET(
      url,
      query = list(path = file_path, per_page = per_page, page = page),
      httr::add_headers(.headers = headers)
    )

    if (httr::status_code(response) != 200) {
      rlang::abort(sprintf(
        "GitHub API request failed with status %d: %s",
        httr::status_code(response),
        httr::content(response, "text", encoding = "UTF-8")
      ))
    }

    commits <- httr::content(response)
    if (length(commits) == 0) {
      break
    }

    all_commits <- c(all_commits, commits)

    # Check if there are more pages
    if (length(commits) < per_page) {
      break
    }
    page <- page + 1
  }

  # Parse commit data
  tibble::tibble(
    sha = purrr::map_chr(all_commits, "sha"),
    date = purrr::map_chr(all_commits, c("commit", "committer", "date")) |>
      lubridate::as_datetime() |>
      as.Date(),
    message = purrr::map_chr(all_commits, c("commit", "message"))
  )
}


#' Find commit SHA for a given date
#' NB: finds most recent commit on or before target_date but not more than a
#' week before.
#' @noRd
find_commit_for_date <- function(commits, target_date) {
  week_ago <- target_date - 7
  eligible <- commits |>
    dplyr::filter(date <= target_date, date >= week_ago) |>
    dplyr::arrange(dplyr::desc(date))

  if (nrow(eligible) == 0) {
    return(NULL)
  }

  eligible$sha[1]
}


#' Fetch file content at a specific commit
#' @noRd
fetch_file_at_commit <- function(repo, file_path, sha, token) {
  url <- sprintf(
    "https://raw.githubusercontent.com/%s/%s/%s",
    repo,
    sha,
    file_path
  )

  headers <- c("User-Agent" = "nowcastNHSN-R-package")
  if (nchar(token) > 0) {
    headers <- c(headers, "Authorization" = paste("token", token))
  }

  response <- httr::GET(url, httr::add_headers(.headers = headers))

  if (httr::status_code(response) != 200) {
    rlang::abort(sprintf(
      "Failed to fetch file at commit %s: status %d",
      sha,
      httr::status_code(response)
    ))
  }

  # Parse CSV
  content <- httr::content(response, "text", encoding = "UTF-8")
  readr::read_csv(content, show_col_types = FALSE)
}


#' Fetch reporting data from GitHub commit history
#'
#' This function fetches historical versions of the COVID-19 hospital admissions
#' target data from the covid19-forecast-hub GitHub repository. Each version
#' represents a "snapshot" of the data as it existed at a particular report date,
#' allowing you to track how reference date values were revised over time.
#'
#' @param locations Character vector of geographic locations (e.g., state abbreviations
#'   like "CA", "NY", or "US" for national). If NULL (default), all locations are returned.
#' @param reference_dates Date vector of reference dates to include in the triangle.
#'   These are the dates for which hospital admissions are being reported.
#' @param report_dates Date vector of report dates (Saturdays). These represent when
#'   the forecaster sees the data (the Saturday following a Wednesday data update).
#'   For each date, the function will find the nearest GitHub commit on or before that date.
#'   Must be Saturdays.
#' @param repo Character, GitHub repository in format "owner/repo". Default is
#'   "CDCgov/covid19-forecast-hub".
#' @param file_path Character, path to the target data file within the repository.
#'   Default is "target-data/covid-hospital-admissions.csv".
#' @param github_token Optional character, GitHub personal access token for API
#'   authentication. If NULL, uses GITHUB_PAT or GITHUB_TOKEN environment variable.
#'   Anonymous requests are limited to 60/hour.
#'
#' @return A data.frame with columns:
#'   \describe{
#'     \item{reference_date}{Date, the date of hospital admissions}
#'     \item{report_date}{Date, the date this version of the data was available}
#'     \item{location}{Character, geographic location code}
#'     \item{count}{Numeric, hospital admissions count}
#'   }
#'
#' @details
#' The function uses the GitHub API to:
#' 1. List commits that modified the target data file
#' 2. For each search_date, find the most recent commit on or before that date
#' 3. Fetch the file content at that commit SHA
#' 4. Parse the CSV and extract rows matching locations and reference_dates
#' 5. Combine all versions into a reporting triangle format
#'
#' **Date conventions:**
#' - report_dates: Saturdays (when forecaster sees data after Wednesday update)
#' - The function searches for GitHub commits on or before each report_date
#'
#' The resulting data.frame can be passed to
#' \code{\link[baselinenowcast]{as_reporting_triangle}} for nowcasting.
#'
#' @export
#' @examples
#' \dontrun{
#' # Fetch data using report_dates (Saturdays)
#' data <- fetch_reporting_data_github(
#'   locations = c("CA", "NY"),
#'   reference_dates = seq(as.Date("2024-11-22"), as.Date("2024-12-13"), by = "week"),
#'   report_dates = seq(as.Date("2024-11-22"), as.Date("2024-12-13"), by = "week")
#' )
#'
#' # Create reporting triangle
#' library(baselinenowcast)
#' triangle <- as_reporting_triangle(data, delays_unit = "weeks")
#' }
fetch_reporting_data_gh_csv <- function(
  reference_dates,
  report_dates,
  locations = NULL,
  repo = "CDCgov/covid19-forecast-hub",
  file_path = "target-data/covid-hospital-admissions.csv",
  github_token = NULL
) {
  # Validate report_dates are Saturdays
  report_dates <- as.Date(report_dates)
  validate_all_saturdays(report_dates)

  # Get GitHub token
  token <- github_token %||%
    Sys.getenv("GITHUB_PAT", Sys.getenv("GITHUB_TOKEN", ""))

  # Convert inputs to proper types
  reference_dates <- as.Date(reference_dates)
  if (!is.null(locations)) {
    locations <- tolower(as.character(locations))
  }

  # Get commit history for the file
  commits <- fetch_file_commits(repo, file_path, token)
  signal = "confirmed_admissions_covid_ew_prelim"

  # For each report date, find the nearest commit and fetch data
  results <- purrr::map_dfr(
    report_dates,
    function(report_date) {
      # Find most recent commit on or before this report date
      # but not more than a week before
      commit_sha <- find_commit_for_date(commits, report_date)

      if (is.null(commit_sha)) {
        rlang::warn(sprintf("No commits found on or before %s", report_date))
        return(NULL)
      }

      # Fetch file content at this commit
      csv_data <- fetch_file_at_commit(repo, file_path, commit_sha, token)

      # Filter to requested locations and reference_dates
      filtered_data <- csv_data |>
        dplyr::filter(
          date >= min(reference_dates),
          date <= max(reference_dates)
        )

      if (!is.null(locations)) {
        filtered_data <- filtered_data |>
          dplyr::filter(tolower(state) %in% locations)
      }

      filtered_data |>
        dplyr::mutate(report_date = report_date, signal = signal) |>
        dplyr::select(
          reference_date = date,
          report_date,
          location = state,
          count = value,
          signal
        )
    },
    .progress = TRUE
  )

  results
}
