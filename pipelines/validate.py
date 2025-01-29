import polars as pl


def validate_ww_conc_data(ww_data: pl.DataFrame, conc_col_name: str, lod_col_name: str):
    """
    Validate wastewater concentration data.
    """
    if ww_data.is_empty():
        raise ValueError("Input DataFrame 'ww_data' is empty.")

    ww_conc = ww_data[conc_col_name]
    if ww_conc.is_null().any():
        raise ValueError(
            f"{conc_col_name} has missing values. "
            "Observations below the limit of detection "
            "must indicate a numeric value less than "
            "the limit of detection."
        )

    if not isinstance(ww_conc, pl.Series):
        raise TypeError("Wastewater concentration is expected to be a 1D Series.")

    if ww_conc.dtype not in [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.Float32,
        pl.Float64,
    ]:
        raise TypeError("Expected numeric values for wastewater concentration.")

    if ww_data["date", "site", "lab"].is_duplicated().any():
        raise ValueError(
            "Duplicate observations found for the same site, lab, and date."
        )

    ww_lod = ww_data[lod_col_name]
    if ww_lod.is_null().any():
        raise ValueError("There are missing values in the limit of detection data.")

    if not isinstance(ww_lod, pl.Series):
        raise TypeError("Limit of detection data is expected to be a 1D Series.")

    ww_obs_dates = ww_data["date"]
    if ww_obs_dates.is_null().any():
        raise ValueError("Date column has missing values.")

    if ww_obs_dates.dtype != pl.Date:
        raise TypeError("Date column has to be of Date type.")

    site_labels = ww_data["site"]
    if site_labels.is_null().any():
        raise TypeError("Site labels are missing.")

    if site_labels.dtype not in [pl.Int32, pl.Int64] and site_labels.dtype != pl.Utf8:
        raise TypeError("Site labels not of integer/string type.")

    lab_labels = ww_data["lab"]
    if lab_labels.is_null().any():
        raise TypeError("Lab labels are missing.")

    if lab_labels.dtype not in [pl.Int32, pl.Int64] and lab_labels.dtype != pl.Utf8:
        raise TypeError("Lab labels are not of integer/string type.")

    site_pops = ww_data["site_pop"]
    if site_pops.is_null().any():
        raise ValueError("Site populations are missing.")
    if site_pops.dtype not in [pl.Int32, pl.Int64] or (site_pops < 0).any():
        raise ValueError("Site populations are not integers, or have negative values.")

    records_per_site_per_pop = (
        ww_data[["site", "site_pop"]].unique().group_by("site").len()
    )
    if (records_per_site_per_pop["len"] != 1).any():
        raise ValueError("The data contains sites with varying population sizes.")

    return None


def validate_count_data(
    count_data: pl.DataFrame, count_col_name: str, pop_size_col_name: str
):
    if count_data.is_empty():
        raise ValueError("Input data is empty")

    counts = count_data[count_col_name]
    if counts.is_null().any():
        raise ValueError(f"{count_col_name} has missing values. ")

    if not isinstance(counts, pl.Series):
        raise TypeError(f"{count_col_name} is expected to be a 1D Series.")

    if counts.dtype not in [pl.Int32, pl.Int64]:
        raise ValueError(f"{count_col_name} should be integers")

    if (counts < 0).any():
        raise ValueError(f"{count_col_name} should not contain negative values")

    # Assuming the model expects daily data, check for daily data consistency
    dates = count_data["date"]
    if len(set(dates)) != len(dates):
        raise ValueError("Count dataset does not appear to be daily.")

    if dates.dtype != pl.Date:
        raise TypeError("Date column should be of date type")

    pop_sizes = count_data[pop_size_col_name]
    if pop_sizes.is_null().any():
        raise ValueError(f"{pop_size_col_name} has missing values. ")

    if not isinstance(pop_sizes, pl.Series):
        raise TypeError(f"{count_col_name} is expected to be a 1D Series.")

    if pop_sizes.dtype not in [pl.Int32, pl.Int64]:
        raise ValueError(f"{pop_size_col_name} should be an integer vector")

    if (pop_sizes < 0).any():
        raise ValueError(f"{pop_size_col_name} should not contain negative values")

    if len(set(pop_sizes)) > 1:
        raise ValueError(
            "Multiple/time-varying count catchment area populations are not currently supported."
        )

    # Ensure that there are no repeated dates (i.e., no duplicate observations for the same day)
    date_counts = count_data["date"].value_counts()
    if (date_counts["count"] > 1).any():
        raise ValueError(
            "Check that data is from a single location, and ensure that there are not multiple count data streams on the same date."
        )

    return None
