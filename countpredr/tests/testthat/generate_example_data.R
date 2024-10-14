test_that("`generate_example_data` throws error if invalid target is used", {
  generate_example_data(target = "mpox") |> expect_error()
})

test_that("generate_example_data saves the CSV file with correct data", {
    temp_dir <- tempdir()
    output_dir <- file.path(temp_dir, "test_output")
    output_file <- file.path(output_dir, "exampledata_covid.csv")
    
    generate_example_data(output_dir = output_dir)
    
    expect_true(file.exists(output_file))
    
    exampledata <- read.csv(output_file)
    expect_true("date" %in% names(exampledata))
    expect_true("target_resp_est" %in% names(exampledata))
    expect_true("other_ed_visits" %in% names(exampledata))
})

test_that("generate_example_data doesnt the CSV file when savedata = FALSE", {
    temp_dir <- tempdir()
    output_dir <- file.path(temp_dir, "test_output")
    output_file <- file.path(output_dir, "exampledata_rsv.csv")
    
    generate_example_data(target = "rsv", output_dir = output_dir, savedata = FALSE)
    
    expect_false(file.exists(output_file))
})

test_that("generate_example_data has non-negative values and above test non resp est", {
    temp_dir <- tempdir()
    output_dir <- file.path(temp_dir, "test_output")
    output_file <- file.path(output_dir, "exampledata_covid.csv")
    
    test_est_non_resp <- 2e6
    generate_example_data(est_non_resp = test_est_non_resp, output_dir = output_dir)
    
    exampledata <- read.csv(output_file)
    
    # Check that the estimates are within expected ranges
    expect_true(all(exampledata$target_resp_est >= 0))
    expect_true(all(exampledata$other_ed_visits >= test_est_non_resp))
})