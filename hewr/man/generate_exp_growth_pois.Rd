% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/generate_exp_growth_process.R
\name{generate_exp_growth_pois}
\alias{generate_exp_growth_pois}
\title{Generate Exponential Growth Process with Poisson Noise}
\usage{
generate_exp_growth_pois(rt, initial)
}
\arguments{
\item{rt}{A numeric vector of exponential growth rates.}

\item{initial}{A numeric value representing the initial value of the process.}
}
\value{
An integer vector of Poisson samples generated from the exponential
growth process.
}
\description{
This function generates a sequence of samples from an exponential growth
process through Poisson sampling:

\if{html}{\out{<div class="sourceCode math">}}\preformatted{\\begin\{aligned\}
\\( \\lambda_t &= I_0 \\exp(\\sum_\{t=1\}^\{t\} r_t) \\) \\\\
I_t &\\sim \\text\{Poisson\}(\\lambda_t).
}\if{html}{\out{</div>}}
}
\examples{
rt <- c(0.1, 0.2, 0.15)
initial <- 10
generate_exp_growth_pois(rt, initial)

}
