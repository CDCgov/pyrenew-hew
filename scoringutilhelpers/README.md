# Scoring helpers for `pyrenew-hew`


## Scoring helpers

The goal of this note is to demonstrate the model scoring helpers in
`scoringutilhelpers`.

First, we load the code:

``` r
library(scoringutils)
library(ggplot2)
source("R/example_prediction.R")
source("R/example_truth_data.R")
source("R/join_forecast_and_data.R")
source("R/score_forecasts.R")
```

## Example data

Now, we can generate some example data.

The `example_prediction` function generates log-normal “forecasts”
across a user selected number of areas and dates with 1 and 2 week
lookaheads. In order to make the example data match the likely use-case
we also generate `.chain` and `.iteration` fields to match a
tidybayes-type data structure. The example data is serialised locally in
`/assets`.

The `example_truth_data` function generates “truth_data” in a similar way,
and also serialises to `/assets`.

``` r
# Example predictions
examplepreds <- example_prediction(save_data = TRUE)
# Example truth data
exampledata <- example_truth_data(save_data = TRUE)
```

Note that the `examplepreds` is serialised in `parquet` format and
`exampledata` as `tsv` format.

## Joining forecasts and the truth data

Preparation for scoring the forecasts is done by:

1.  Ingesting forecasts from the `forecast_source` directory and
    truth_data from `truth_data_file`. The forecasts are treated as an
    Arrow dataset and loaded with `arrow::open_dataset`. The truth_data
    is assumed to be a `tsv` file.
2.  The joining operation is determined by `join_key`. In this case the
    forecast is *to* a date in `target_end_date` scored with data
    recorded at that `date`.

``` r
forecast_source <- "scoringutilhelpers/assets/example_predictions"
truth_data_file <- "scoringutilhelpers/assets/example_truth_data.tsv"
scorable_data <- join_forecast_and_data(forecast_source, truth_data_file,
  join_key = join_by(area, target_end_date == date)
) |>
  collect()
scorable_data |> print(n = 10)
```

    # A tibble: 50,400 × 9
       .chain .iteration .draw area  reference_date target_end_date prediction model
        <int>      <int> <int> <chr> <date>         <date>               <dbl> <chr>
     1      1          1     1 A     2024-10-24     2024-10-31           0.808 exam…
     2      1          1     1 A     2024-10-25     2024-11-01           1.13  exam…
     3      1          1     1 A     2024-10-26     2024-11-02           0.957 exam…
     4      1          1     1 A     2024-10-27     2024-11-03           0.735 exam…
     5      1          1     1 A     2024-10-28     2024-11-04           0.876 exam…
     6      1          1     1 A     2024-10-29     2024-11-05           0.841 exam…
     7      1          1     1 A     2024-10-30     2024-11-06           0.946 exam…
     8      1          1     1 A     2024-10-31     2024-11-07           0.968 exam…
     9      1          1     1 A     2024-11-01     2024-11-08           1.02  exam…
    10      1          1     1 A     2024-11-02     2024-11-09           1.18  exam…
    # ℹ 50,390 more rows
    # ℹ 1 more variable: truth_data <dbl>

## Scoring prepared forecasts

Scoring the forecast is done by:

1.  Setting the [forecast
    unit](https://epiforecasts.io/scoringutils/dev/index.html#the-forecast-unit).
    In this case, we want to forecast on the area, reference date (date
    of forecast submission), the target date of the data and model
    (although only one “model” is in this example).
2.  Transforming the data and forecasts. The forecasts and data in this
    example are non-negative, and the default is to score on the
    log-transformed data. `score_forecasts` function splats arguments to
    `scoringutils::transform_forecasts` to modify this.
3.  Scoring the transformed forecasts and data.
4.  If more than one model is provided `score_forecasts` does pairwise
    comparisons.

``` r
forecast_unit <- c("area", "reference_date", "target_end_date", "model")
observed <- "truth_data"
predicted <- "prediction"

scored_forecasts <- score_forecasts(scorable_data,
  forecast_unit = forecast_unit,
  observed = observed,
  predicted = predicted,
)
scored_forecasts
```

           area reference_date target_end_date   model   scale   bias        dss
         <char>         <Date>          <Date>  <char>  <char>  <num>      <num>
      1:      A     2024-10-24      2024-10-31 example natural  0.140 -2.6763044
      2:      A     2024-10-25      2024-11-01 example natural -0.935  2.4547245
      3:      A     2024-10-26      2024-11-02 example natural  0.425 -2.1458903
      4:      A     2024-10-27      2024-11-03 example natural  0.900 -0.7754065
      5:      A     2024-10-28      2024-11-04 example natural -0.345 -2.4865248
     ---
    122:      C     2024-10-26      2024-11-09 example     log  0.465 -2.4427319
    123:      C     2024-10-27      2024-11-10 example     log -0.400 -2.4484693
    124:      C     2024-10-28      2024-11-11 example     log  0.790 -1.2586868
    125:      C     2024-10-29      2024-11-12 example     log -0.520 -2.3397407
    126:      C     2024-10-30      2024-11-13 example     log -0.600 -2.0584358
               crps overprediction underprediction dispersion   log_score       mad
              <num>          <num>           <num>      <num>       <num>     <num>
      1: 0.05575651    0.002872917      0.00000000 0.05288359 -0.54396045 0.2239094
      2: 0.44779650    0.000000000      0.39047100 0.05732550  2.00155135 0.2336494
      3: 0.08848366    0.027502432      0.00000000 0.06098122 -0.34948006 0.2551530
      4: 0.22418138    0.165827644      0.00000000 0.05835374  0.46835458 0.2431496
      5: 0.08151398    0.000000000      0.01977016 0.06174382 -0.23570654 0.2577069
     ---
    122: 0.09389622    0.036703625      0.00000000 0.05719260 -0.25670024 0.2550321
    123: 0.09200240    0.000000000      0.03310884 0.05889356 -0.23960228 0.2605521
    124: 0.18640590    0.129375108      0.00000000 0.05703079  0.11453058 0.2421037
    125: 0.09689695    0.000000000      0.04225864 0.05463830 -0.25469114 0.2299873
    126: 0.12561510    0.000000000      0.06801631 0.05759880 -0.08084834 0.2333985
          ae_median     se_mean
              <num>       <num>
      1: 0.03415607 0.003687819
      2: 0.60012752 0.333929320
      3: 0.12810090 0.028707289
      4: 0.33397977 0.128301402
      5: 0.11466343 0.008236991
     ---
    122: 0.14678095 0.022094766
    123: 0.16057194 0.021453558
    124: 0.30535951 0.090633935
    125: 0.16179927 0.025360140
    126: 0.21541919 0.044438693

## Summary scores

`scored_forecasts` is the level of forecast we (likely) want to operate
at; that is that is likely to be better to score multiple scenarios and
*then* summarise rather than summarising multiple scenarios.

However, we should note that we can further summarise
`scored_forecasts`.

``` r
summ <- summarise_scores(scored_forecasts, by = c("model", "area"))
summ
```

         model   area         bias       dss      crps overprediction
        <char> <char>        <num>     <num>     <num>          <num>
    1: example      A -0.001190476 -1.753875 0.1357817     0.02379426
    2: example      B -0.082619048 -1.391587 0.1702695     0.04832772
    3: example      C -0.135952381 -2.265463 0.1050017     0.01772348
       underprediction dispersion    log_score       mad ae_median    se_mean
                 <num>      <num>        <num>     <num>     <num>      <num>
    1:      0.05403425 0.05795320  0.002301762 0.2458696 0.1774663 0.06048685
    2:      0.06372649 0.05821529  0.204155195 0.2475007 0.2432254 0.09036541
    3:      0.02844568 0.05883253 -0.180353247 0.2517263 0.1591800 0.02939690
