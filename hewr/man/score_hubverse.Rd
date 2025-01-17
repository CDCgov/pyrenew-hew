% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/score_hubverse.R
\name{score_hubverse}
\alias{score_hubverse}
\title{Read in and score hubverse-format output from \code{pyrenew-hew}.}
\usage{
score_hubverse(
  forecast,
  observed,
  horizons = c(0, 1),
  transform = scoringutils::log_shift,
  append_transformed = FALSE,
  offset = 1,
  observed_value_column = "value",
  observed_location_column = "location",
  observed_date_column = "reference_date",
  quantile_tol = 10,
  ...
)
}
\arguments{
\item{forecast}{forecasts, as a hubverse-format
\code{\link[tibble:tibble]{tibble}} produced by
\code{\link[=to_epiweekly_quantile_table]{to_epiweekly_quantile_table()}}, with columns
\code{target_end_date}, \code{value}, and \code{horizon}}

\item{observed}{observations, as a \code{\link[tibble:tibble]{tibble}}.}

\item{horizons}{Horizons to score. Default \code{c(0, 1)}}

\item{transform}{transformation passed as the
\code{fun} argument to \code{\link[scoringutils:transform_forecasts]{scoringutils::transform_forecasts()}}.
Default \code{\link[scoringutils:log_shift]{scoringutils::log_shift()}}.}

\item{append_transformed}{When calling
\code{\link[scoringutils:transform_forecasts]{scoringutils::transform_forecasts()}}, append
the transformed scale forecasts to the base scale forecasts
or keep only the transformed scale forecasts? Passed as the
\code{append} argument to \code{\link[scoringutils:transform_forecasts]{scoringutils::transform_forecasts()}}.
Boolean, default \code{FALSE} (keep only transformed scale).}

\item{offset}{Offset for the transform passed to
\code{\link[scoringutils:transform_forecasts]{scoringutils::transform_forecasts()}}.}

\item{observed_value_column}{Name of the column containing
observed values in the \code{observed} table, as a string.
Default \code{"value"}}

\item{observed_location_column}{Name of the column containing
location values in the \code{observed} table, as a string.
Default \code{"location"}}

\item{observed_date_column}{Name of the column containing
date values in the \code{observed} table, as a string.
Default \code{"reference_date"}}

\item{quantile_tol}{Round quantile level values to this many
decimal places, to avoid problems with floating point number
comparisons. Passed as the \code{digits} argument to \code{\link[base:Round]{base::round()}}.
Default 10.}

\item{...}{Other keyword arguments passed to
\code{\link[scoringutils:transform_forecasts]{scoringutils::transform_forecasts()}}.}
}
\value{
A table of scores, as the output of
\code{\link[scoringutils:score]{scoringutils::score()}}, filtered to include only the
transformed_scale.
}
\description{
Expects forecast output created by \code{\link[=to_epiweekly_quantile_table]{to_epiweekly_quantile_table()}}
and an observed data table with location, date, and value columns.
The column names in the observed data table can be configured;
defaults are \code{"location"}, \code{"reference_date"}, and
\code{"value"}, respectively, to reflecting the format used
in pyrenew-hew pipeline scripts such as
\code{pipelines/create_observed_data_table.py}.
}
