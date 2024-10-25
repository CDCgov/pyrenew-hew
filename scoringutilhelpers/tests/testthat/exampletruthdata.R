test_that("exampletruthdata generates correct number of rows and columns", {
    ndays <- 10
    nareas <- 2
    result <- exampletruthdata(ndays = ndays, nareas = nareas)
    expect_equal(nrow(result), ndays * nareas)
    expect_equal(ncol(result), 3)
})

test_that("exampletruthdata generates correct column names", {
    result <- exampletruthdata()
    expect_true(all(c("area", "date", "truthdata") %in% colnames(result)))
})

test_that("exampletruthdata generates correct date range", {
    ndays <- 17
    start_date <- ymd("2024-10-24")
    result <- exampletruthdata(ndays = ndays)
    expected_dates <- seq.Date(from = start_date, by = "day", 
        length.out = ndays)
    expect_equal(unique(result$date), expected_dates)
})

test_that("exampletruthdata generates correct number of unique areas", {
    nareas <- 5
    result <- exampletruthdata(nareas = nareas)
    expect_equal(length(unique(result$area)), nareas)
})

test_that("exampletruthdata saves data when savedata is TRUE", {
    savepath <- tempdir()
    filename <- file.path(savepath, "exampletruthdata.tsv")
    result <- exampletruthdata(savepath = savepath, savedata = TRUE)
    expect_true(file.exists(filename))
    saved_data <- readr::read_tsv(filename)
    expect_equal(nrow(saved_data), nrow(result))
    expect_equal(ncol(saved_data), ncol(result))
    expect_equal(colnames(saved_data), colnames(result))
    unlink(savepath)  # Clean up the temporary file
})
