#!/usr/bin/env Rscript

# Plot EpiAutoGP Forecast Results
#
# This script creates plots of the EpiAutoGP forecast output using the
# forecasttools::plot_hubverse_file_quantiles function.
#
# Usage: Rscript plot_forecast.R <forecast_csv_file> <output_directory>

library(forecasttools)
library(ggplot2)
library(readr)
library(dplyr)
library(lubridate)
library(jsonlite)

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  cat(
    "Usage: Rscript plot_forecast.R
        <forecast_csv_file> <output_directory>\n"
  )
  quit(status = 1)
}

forecast_csv_file <- args[1]
output_directory <- args[2]

# Path to the original vintaged data and JSON input
vintaged_data_file <- "vintaged_us_nhsn_data.csv"
json_input_file <- "epiautogp_input_2025-08-16.json"

# Check if forecast file exists
if (!file.exists(forecast_csv_file)) {
  cat("❌ Error: Forecast CSV file not found:", forecast_csv_file, "\n")
  quit(status = 1)
}

# Check if vintaged data file exists
if (!file.exists(vintaged_data_file)) {
  cat("❌ Error: Vintaged data file not found:", vintaged_data_file, "\n")
  quit(status = 1)
}

# Check if JSON input file exists
if (!file.exists(json_input_file)) {
  cat("❌ Error: JSON input file not found:", json_input_file, "\n")
  quit(status = 1)
}

# Create output directory if it doesn't exist
if (!dir.exists(output_directory)) {
  dir.create(output_directory, recursive = TRUE)
  cat("📁 Created output directory:", output_directory, "\n")
}

# Read and process the vintaged data
vintaged_data <- read_csv(vintaged_data_file, show_col_types = FALSE)

# Read JSON input to get nowcast information
json_data <- fromJSON(json_input_file, simplifyVector = FALSE)
forecast_date <- as.Date(json_data$forecast_date)
nowcast_date <- as.Date(json_data$nowcast_dates[[1]])
nowcast_samples <- sapply(json_data$nowcast_reports, function(x) x[[1]])

# Calculate nowcast statistics
nowcast_median <- median(nowcast_samples)
nowcast_q25 <- quantile(nowcast_samples, 0.25)
nowcast_q75 <- quantile(nowcast_samples, 0.75)
cat(
  "Nowcast uncertainty: median =",
  round(nowcast_median, 1),
  ", IQR = [",
  round(nowcast_q25, 1),
  ",",
  round(nowcast_q75, 1),
  "]\n"
)

# Find the most recent report_date
most_recent_report_date <- max(vintaged_data$report_date)


# Prepare observed data for latest report date
# (black line - extends through forecast)
observed_data_latest <- vintaged_data %>%
  filter(report_date == most_recent_report_date) %>%
  select(date = reference_date, location = geo_value, value = confirm) %>%
  mutate(
    date = as.Date(date),
    location = toupper(location), # Ensure location is uppercase (US)
    data_type = "Latest data"
  ) %>%
  arrange(date)

# Prepare observed data as it was on forecast date 2025-08-16 (red line)
observed_data_forecast_date <- vintaged_data %>%
  filter(report_date == forecast_date) %>%
  select(date = reference_date, location = geo_value, value = confirm) %>%
  mutate(
    date = as.Date(date),
    location = toupper(location),
    data_type = "Data on forecast date"
  ) %>%
  arrange(date)

# Read forecast data to determine forecast start date and extend latest data
forecast_data <- read_csv(forecast_csv_file, show_col_types = FALSE)
forecast_start_date <- min(forecast_data$target_end_date)
forecast_end_date <- max(forecast_data$target_end_date)

# Use 8 weeks (2 months) of lookback for plotting
lookback_start_date <- forecast_start_date - weeks(8)

# Filter both datasets for plotting
observed_data_latest_filtered <- observed_data_latest %>%
  filter(date >= lookback_start_date)

observed_data_ff <- observed_data_forecast_date %>%
  filter(date >= lookback_start_date)

# Extend latest data through forecast period by projecting the trend
# Use the last observed value and extend it as a flat line through
# forecast period
max_observed_date <- max(observed_data_latest$date)
last_observed_value <-
  observed_data_latest$value[observed_data_latest$date == max_observed_date]

if (forecast_end_date > max_observed_date) {
  # Create weekly extension points through the forecast period
  extended_dates <-
    seq(max_observed_date + weeks(1), forecast_end_date, by = "week")
  extended_data <- data.frame(
    date = extended_dates,
    location = "US",
    value = last_observed_value, # Use last observed value
    data_type = "Latest data (extended)"
  )
  observed_data_latest_filtered <-
    bind_rows(observed_data_latest_filtered, extended_data)
}


# Combine datasets for forecasttools (using only latest data as base)
temp_observed_file <- tempfile(fileext = ".csv")
write_csv(
  observed_data_latest_filtered %>%
    select(date, location, value),
  temp_observed_file
)


tryCatch(
  {
    # Create base forecast plot using forecasttools
    plots <- plot_hubverse_file_quantiles(
      forecast_file_path = forecast_csv_file,
      locations = NULL, # Plot all locs (should be just "US" in our case)
      observed_data_path = temp_observed_file, # Include latest observed data
      start_date = as.character(lookback_start_date),
      end_date = as.character(forecast_end_date),
      location_input_format = "abbr",
      location_output_format = "abbr",
      y_transform = "identity",
      linewidth = 2,
      pointsize = 4,
      forecast_linecolor = "darkblue",
      forecast_pointcolor = "darkblue",
      obs_linecolor = "black",
      obs_pointcolor = "black",
      autotitle = TRUE
    )

    # Enhance the first plot with additional data layers
    if (length(plots) > 0) {
      first_location <- names(plots)[1]
      base_plot <- plots[[first_location]]

      cat("🎨 Adding custom data layers to plot...\n")

      # Enhanced plot with additional layers and proper legend
      enhanced_plot <- base_plot +
        # Add extended latest data line (black line through forecast period)
        geom_line(
          data = observed_data_latest_filtered,
          aes(x = date, y = value, color = "Latest data"),
          linewidth = 2,
          alpha = 0.9,
          inherit.aes = FALSE
        ) +
        geom_point(
          data = observed_data_latest_filtered %>%
            filter(!is.na(value)),
          aes(x = date, y = value, color = "Latest data"),
          size = 2,
          alpha = 0.9,
          inherit.aes = FALSE
        ) +

        # Add forecast date data as red line/points
        geom_line(
          data = observed_data_ff,
          aes(x = date, y = value, color = "Vintage data (forecast date)"),
          linewidth = 1.5,
          alpha = 0.8,
          inherit.aes = FALSE
        ) +
        geom_point(
          data = observed_data_ff,
          aes(x = date, y = value, color = "Vintage data (forecast date)"),
          size = 2.5,
          alpha = 0.8,
          inherit.aes = FALSE
        ) +

        # Add nowcast uncertainty as error bar
        geom_errorbar(
          data = data.frame(
            x = nowcast_date,
            y = nowcast_median,
            ymin = nowcast_q25,
            ymax = nowcast_q75
          ),
          aes(x = x, y = y, ymin = ymin, ymax = ymax),
          color = "#FF8C00",
          width = 2,
          linewidth = 1.5,
          alpha = 0.9,
          inherit.aes = FALSE
        ) +

        # Add nowcast median point
        geom_point(
          data = data.frame(x = nowcast_date, y = nowcast_median),
          aes(x = x, y = y, color = "Nowcast uncertainty"),
          size = 4,
          alpha = 0.9,
          inherit.aes = FALSE
        ) +

        # Add vertical line at forecast date
        geom_vline(
          xintercept = as.numeric(forecast_date),
          linetype = "dashed",
          color = "gray50",
          alpha = 0.7
        ) +

        # Add manual color scale for legend
        scale_color_manual(
          name = "Data Sources",
          values = c(
            "Latest data" = "#D73027",
            "Vintage data (forecast date)" = "#228B22",
            "Nowcast uncertainty" = "#FF8C00" # Dark orange for nowcast
          ),
          breaks = c(
            "Latest data",
            "Vintage data (forecast date)",
            "Nowcast uncertainty"
          )
        ) +

        # Update labels and legend
        labs(
          title = paste(
            "EpiAutoGP Forecast - COVID-19 Hospital Admissions (US)"
          ), # nolint
          subtitle = paste(
            "Forecast generated on:",
            as.character(forecast_date),
            "| Nowcast date:",
            as.character(nowcast_date)
          ),
          x = "Date",
          y = "Hospital Admissions",
          caption = paste(
            "Dashed line marks forecast date |",
            "Red: Latest data extended through forecast period |", # nolint
            "Teal bands show forecast uncertainty"
          )
        ) +

        # Improve theme
        theme_minimal() +
        theme(
          plot.title = element_text(size = 14, face = "bold"),
          plot.subtitle = element_text(size = 11, color = "gray30"),
          plot.caption = element_text(size = 10, color = "gray50", hjust = 0), # nolint
          axis.title = element_text(size = 12),
          axis.text = element_text(size = 10),
          legend.position = "bottom",
          legend.title = element_text(size = 11, face = "bold"),
          legend.text = element_text(size = 10),
          panel.grid.minor = element_blank()
        )

      # Save the forecast plot
      plot_filename <-
        file.path(
          output_directory,
          paste0("forecast_plot_", first_location, ".png")
        )

      ggsave(
        filename = plot_filename,
        plot = enhanced_plot,
        width = 14,
        height = 10,
        dpi = 300,
        bg = "white"
      )

      cat("📊 Forecast plot saved successfully\n")
    } else {
      cat("⚠️  No plots were generated\n")
    }

    # Clean up temporary file
    if (file.exists(temp_observed_file)) {
      file.remove(temp_observed_file)
    }
  },
  error = function(e) {
    cat("❌ Error during plotting:\n")
    cat("   ", as.character(e), "\n")
    cat("\n💡 Troubleshooting tips:\n")
    cat("   - Check that the CSV file is properly formatted\n")
    cat("   - Ensure forecasttools package is installed and loaded\n")
    cat("   - Verify the forecast data contains expected columns\n")
    quit(status = 1)
  }
)
