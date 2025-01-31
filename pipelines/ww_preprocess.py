import polars as pl


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


def preprocess_count_data(
    count_data: pl.DataFrame,
    count_col_name: str = "daily_hosp_admits",
    pop_size_col_name: str = "state_pop",
) -> pl.DataFrame:
    """
    Pre-processes hospital admissions data, converting column names to match the
    expected format for further analysis.

    Parameters:
    ----------
    count_data : pl.DataFrame
        A Polars DataFrame containing the following columns: date,
        a count column and  a population size column

    count_col_name : str | optional
        The name of the column containing the epidemiological indiator.
        Default is `daily_hosp_admits`.

    pop_size_col_name : str | optional
        The name of the column containing the population size for each record.
        Default is `state_pop`.

    Returns:
    -------
    pl.DataFrame
        A Polars DataFrame with the columns renamed to:
        - "count" for the admissions count column
        - "total_pop" for the population size column
        - "date" for the date column.
    """

    assert (
        count_col_name in count_data.columns
    ), f"Column '{count_col_name}' is missing in the input data."
    assert (
        pop_size_col_name in count_data.columns
    ), f"Column '{pop_size_col_name}' is missing in the input data."

    validate_count_data(
        count_data, count_col_name=count_col_name, pop_size_col_name=pop_size_col_name
    )
    count_data_preprocessed = count_data.rename(
        {count_col_name: "count", pop_size_col_name: "total_pop"}
    )
    return count_data_preprocessed
