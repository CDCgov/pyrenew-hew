% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/to_hub_quantile_table.R
\name{to_hub_quantile_table}
\alias{to_hub_quantile_table}
\alias{model_fit_dir_to_hub_q_tbl}
\alias{model_loc_dir_to_hub_q_tbl}
\alias{model_runs_dir_to_hub_q_tbl}
\title{Create a hubverse-format forecast quantile table}
\usage{
model_fit_dir_to_hub_q_tbl(model_fit_dir)

model_loc_dir_to_hub_q_tbl(model_loc_dir)

model_runs_dir_to_hub_q_tbl(model_runs_dir)
}
\arguments{
\item{model_fit_dir}{Model fit directory containing samples and/or quantiles}

\item{model_loc_dir}{Model location directory containing model fits}

\item{model_runs_dir}{Model runs directory containing model locations}
}
\value{
A hubverse quantile table
}
\description{
Create a hubverse-format forecast quantile table
}
