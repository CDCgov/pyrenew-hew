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
  add_argument("--jurisdictions",
               type = "character",
               help = "space-separated list of jurisdictions to keep",
               nargs = Inf
               ) |> 
  add_argument(
    "--output-file",
    type = "character",
    help = "Path to output file"
  )

null_if_na <- function(x) {
  if (all(is.na(x))) {
    NULL
  } else {
    x
  }
}

argv <- parse_args(p)
start_date <- null_if_na(argv$start_date)
end_date <- null_if_na(argv$end_date)
disease <- null_if_na(argv$disease)
jurisdictions <- null_if_na(argv$jurisdictions)
output_file <- argv$output_file

if (is.na(output_file)) {
  output_file <- stdout()
}

columns <- disease_nhsn_key[disease]

dat <- pull_nhsn(
  start_date = start_date,
  end_date = end_date,
  columns = columns,
  jurisdictions = jurisdictions,
) |>
  mutate(weekendingdate = as_date(weekendingdate)) |>
  rename(hospital_admissions = !!unname(columns)) |>
  mutate(hospital_admissions = as.numeric(hospital_admissions))

write_tsv(dat, output_file)
