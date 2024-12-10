script_packages <- c(
  "forecasttools",
  "dplyr",
  "readr",
  "lubridate",
  "argparser"
)

# load in packages without messages
purrr::walk(script_packages, \(pkg) {
  suppressPackageStartupMessages(
    library(pkg, character.only = TRUE)
  )
})

disease_nhsn_key <- c(
  "COVID-19" = "totalconfc19newadm",
  "Influenza" = "totalconfflunewadm"
)

p <- arg_parser(
  "Pull NHSN data"
) |>
  add_argument(
    "--start-date",
    type = "character",
    default = NULL,
    help = "Start date in YYYY-MM-DD format"
  ) |>
  add_argument(
    "--end-date",
    type = "character",
    default = NULL,
    help = "End date in YYYY-MM-DD format"
  ) |>
  add_argument(
    "--disease",
    type = "character",
    help = "Disease name"
  ) |>
  add_argument(
    "--output-file",
    type = "character",
    help = "Path to output file"
  )


argv <- parse_args(p)
start_date <- argv$start_date
end_date <- argv$end_date
disease <- argv$disease
output_file <- argv$output_file

if (is.na(output_file)) {
  output_file <- stdout()
}

columns <- disease_nhsn_key[disease]

dat <- pull_nhsn(
  start_date = start_date,
  end_date = end_date,
  columns = columns
) |>
  mutate(weekendingdate = as_date(weekendingdate)) |>
  rename(nhsn_admissions = !!unname(columns)) |>
  mutate(nhsn_admissions = as.numeric(nhsn_admissions)) |>
  write_tsv(output_file)
