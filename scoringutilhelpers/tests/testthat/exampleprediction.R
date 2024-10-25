test_that("exampleprediction generates data with correct dimensions", {
    ndays <- 21
    reps <- 100
    nchains <- 4
    nareas <- 3
    nlookaheads <- 2
    result <- exampleprediction(ndays = ndays, reps = reps, nchains = nchains,
    nareas = nareas)
    expect_equal(nrow(result), ndays * reps * nchains * nareas * nlookaheads)
    expect_equal(ncol(result), 8)
})

test_that("exampleprediction generates data with correct columns", {
    result <- exampleprediction()
    expect_true(all(c("area", "reference_date", "target_end_date", "prediction",
     ".chain", ".iteration", ".draw") %in% colnames(result)))
})

test_that("exampleprediction generates data with correct date range", {
    ndays <- 24
    start_date <- ymd("2024-10-24")
    end_date <- start_date + days(ndays - 1)
    result <- exampleprediction(ndays = ndays)
    expect_true(
    all(result$reference_date >= start_date & result$reference_date <= end_date
    ))
})

test_that("exampleprediction generates data with correct areas", {
    nareas <- 2
    result <- exampleprediction(nareas = nareas)
    expect_true(all(result$area %in% LETTERS[1:nareas]))
})

test_that("exampleprediction generates data with correct number of chains", {
    nchains <- 2
    result <- exampleprediction(nchains = nchains)
    expect_equal(length(unique(result$.chain)), nchains)
})

test_that("exampleprediction saves data when savedata is TRUE", {
    temp_dir <- tempdir()
    result <- exampleprediction(savepath = temp_dir, savedata = TRUE)
    expect_true(file.exists(file.path(temp_dir, "part-0.parquet")))
})

