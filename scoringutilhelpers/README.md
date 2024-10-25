# Scoring helpers for `pyrenew-hew`


## Scoring helpers

The goal of this note is to demonstrate the model scoring helpers in
`scoringutilhelpers`.

First, we load the code:

``` r
library(scoringutils)
library(ggplot2)
source("R/exampleprediction.R")
source("R/exampletruthdata.R")
source("R/join_forecast_and_data.R")
source("R/score_forecasts.R")
```

## Example data

Now, we can generate some example data.

The `exampleprediction` function generates log-normal “forecasts” across
a user selected number of areas and dates with 1 and 2 week lookaheads.
In order to make the example data match the likely use-case we also
generate `.chain` and `.iteration` fields to match a tidybayes-type data
structure. The example data is serialised locally in `/assets`.

The `exampletruthdata` function generates “truthdata” in a similar way,
and also serialises to `/assets`.

``` r
#Example predictions
examplepreds <- exampleprediction(savedata = TRUE)
#Example truth data
exampledata <- exampletruthdata(savedata = TRUE)
```

Note that the `examplepreds` is serialised in `parquet` format and
`exampledata` as `tsv` format.

## Joining forecasts and the truth data

Preparation for scoring the forecasts is done by:

1.  Ingesting forecasts from the `forecast_source` directory and
    truthdata from `truthdata_file`. The forecasts are treated as an
    Arrow dataset and loaded with `arrow::open_dataset`. The truthdata
    is assumed to be a `tsv` file.
2.  The joining operation is determined by `join_key`. In this case the
    forecast is *to* a date in `target_end_date` scored with data
    recorded at that `date`.

``` r
forecast_source <- "scoringutilhelpers/assets/examplepredictions"
truthdata_file <- "scoringutilhelpers/assets/exampletruthdata.tsv"
scorable_data <- join_forecast_and_data(forecast_source, truthdata_file,
        join_key = join_by(area, target_end_date == date)) |>
        collect()
scorable_data |> print(n = 10)        
```

    # A tibble: 50,400 × 9
       area  reference_date target_end_date prediction .chain .iteration .draw model
       <chr> <date>         <date>               <dbl>  <int>      <int> <int> <chr>
     1 A     2024-10-24     2024-10-31           0.794      1          1     1 exam…
     2 A     2024-10-24     2024-10-31           0.722      1          2     2 exam…
     3 A     2024-10-24     2024-10-31           0.728      1          3     3 exam…
     4 A     2024-10-24     2024-10-31           1.09       1          4     4 exam…
     5 A     2024-10-24     2024-10-31           0.774      1          5     5 exam…
     6 A     2024-10-24     2024-10-31           0.835      1          6     6 exam…
     7 A     2024-10-24     2024-10-31           0.938      1          7     7 exam…
     8 A     2024-10-24     2024-10-31           0.848      1          8     8 exam…
     9 A     2024-10-24     2024-10-31           1.05       1          9     9 exam…
    10 A     2024-10-24     2024-10-31           0.745      1         10    10 exam…
    # ℹ 50,390 more rows
    # ℹ 1 more variable: truthdata <dbl>

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
observed <- "truthdata"
predicted <- "prediction"

scored_forecasts <- score_forecasts(scorable_data,
        forecast_unit = forecast_unit,
        observed = observed,
        predicted = predicted,
    )
scored_forecasts
```

           area reference_date target_end_date   model   scale   bias       dss
         <char>         <Date>          <Date>  <char>  <char>  <num>     <num>
      1:      A     2024-10-24      2024-10-31 example natural  0.690 -1.899113
      2:      A     2024-10-25      2024-11-01 example natural  0.355 -2.432046
      3:      A     2024-10-26      2024-11-02 example natural  0.590 -2.034498
      4:      A     2024-10-27      2024-11-03 example natural  0.165 -2.708927
      5:      A     2024-10-28      2024-11-04 example natural -0.135 -2.664006
     ---                                                                       
    122:      C     2024-10-26      2024-11-09 example     log -0.290 -2.687959
    123:      C     2024-10-27      2024-11-10 example     log -0.335 -2.519673
    124:      C     2024-10-28      2024-11-11 example     log  0.265 -2.744870
    125:      C     2024-10-29      2024-11-12 example     log  0.635 -1.953640
    126:      C     2024-10-30      2024-11-13 example     log  0.510 -2.245857
               crps overprediction underprediction dispersion   log_score       mad
              <num>          <num>           <num>      <num>       <num>     <num>
      1: 0.12957669     0.07436022     0.000000000 0.05521648 -0.23164035 0.2347068
      2: 0.07441519     0.01779923     0.000000000 0.05661596 -0.46635109 0.2301188
      3: 0.11412323     0.05737499     0.000000000 0.05674824 -0.24048324 0.2443253
      4: 0.06377707     0.00566926     0.000000000 0.05810781 -0.36939558 0.2503712
      5: 0.06492466     0.00000000     0.003168703 0.06175595 -0.33169361 0.2615177
     ---                                                                           
    122: 0.06858898     0.00000000     0.012036178 0.05655281 -0.43157759 0.2448634
    123: 0.08295334     0.00000000     0.021735876 0.06121747 -0.22906599 0.2689030
    124: 0.06834722     0.01025714     0.000000000 0.05809008 -0.33604102 0.2500228
    125: 0.13285992     0.07128174     0.000000000 0.06157818 -0.07006896 0.2600511
    126: 0.11281805     0.05407785     0.000000000 0.05874020 -0.11339420 0.2612102
          ae_median      se_mean
              <num>        <num>
      1: 0.21587968 5.501567e-02
      2: 0.09690666 1.775464e-02
      3: 0.18998518 4.504192e-02
      4: 0.06482301 4.683591e-03
      5: 0.04729166 4.089016e-05
     ---                        
    122: 0.08151715 8.190394e-03
    123: 0.13094724 1.345912e-02
    124: 0.08239299 4.713940e-03
    125: 0.22294504 5.035666e-02
    126: 0.19103536 3.422620e-02

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

         model   area        bias       dss      crps overprediction
        <char> <char>       <num>     <num>     <num>          <num>
    1: example      A  0.17785714 -2.186645 0.1092928     0.03390710
    2: example      B -0.12428571 -1.588940 0.1641718     0.04487413
    3: example      C  0.04547619 -2.004585 0.1218931     0.03797486
       underprediction dispersion  log_score       mad ae_median    se_mean
                 <num>      <num>      <num>     <num>     <num>      <num>
    1:      0.01632374 0.05906194 -0.1770303 0.2485820 0.1540629 0.03605570
    2:      0.06011481 0.05918290  0.1367940 0.2506604 0.2493126 0.07565414
    3:      0.02442670 0.05949156 -0.0909736 0.2549414 0.1698421 0.04635933
