"""
Process EpiAutoGP forecast outputs to match the format expected by plotting functions.

This module provides functionality equivalent to the R function process_loc_forecast()
but specifically tailored for EpiAutoGP model outputs.
"""

from pathlib import Path

import polars as pl


def combine_forecast_with_observed(
    forecast: pl.DataFrame,
    observed: pl.DataFrame,
) -> pl.DataFrame:
    """
    Combine forecast samples with observed data.

    For observed time points, creates duplicate rows for each draw with the
    observed value, matching the structure of forecast samples.

    Parameters
    ----------
    forecast : pl.DataFrame
        Forecast samples with columns: date, .draw, .variable, .value, resolution
    observed : pl.DataFrame
        Observed data with columns: date, geo_value, disease, .variable, .value, resolution

    Returns
    -------
    pl.DataFrame
        Combined samples with observed data repeated for each draw
    """
    # Get forecast metadata
    first_forecast_date = forecast.select(pl.col("date").min()).item()
    n_draws = forecast.select(pl.col(".draw").max()).item()
    target_variables = forecast.select(".variable").unique().to_series().to_list()

    # Ensure we're working with proper date types
    if isinstance(first_forecast_date, str):
        import datetime as dt

        first_forecast_date = dt.datetime.strptime(
            first_forecast_date, "%Y-%m-%d"
        ).date()

    # Filter observed data to before forecast period and target variables
    observed_filtered = observed.filter(
        (pl.col("date") < pl.lit(first_forecast_date))
        & (pl.col(".variable").is_in(target_variables))
    )

    # Create draws for observed data (duplicate each observation n_draws times)
    draw_numbers = pl.DataFrame({".draw": list(range(1, n_draws + 1))})
    observed_with_draws = observed_filtered.join(draw_numbers, how="cross")

    # Ensure schemas match before concatenation
    # Add any columns that are in forecast but not in observed (with null values)
    for col in forecast.columns:
        if col not in observed_with_draws.columns:
            # Get the dtype from forecast for this column
            dtype = forecast.schema[col]
            observed_with_draws = observed_with_draws.with_columns(
                pl.lit(None, dtype=dtype).alias(col)
            )

    # Cast observed columns to match forecast types
    for col in observed_with_draws.columns:
        if col in forecast.columns:
            forecast_dtype = forecast.schema[col]
            if observed_with_draws.schema[col] != forecast_dtype:
                observed_with_draws = observed_with_draws.with_columns(
                    pl.col(col).cast(forecast_dtype)
                )

    # Select columns in the same order as forecast
    observed_with_draws = observed_with_draws.select(forecast.columns)

    # Combine observed and forecast
    combined = pl.concat([observed_with_draws, forecast], how="vertical")

    return combined


def calculate_credible_intervals(
    samples: pl.DataFrame,
    ci_widths: list[float] = [0.5, 0.8, 0.95],
) -> pl.DataFrame:
    """
    Calculate median and credible intervals from posterior samples.

    Parameters
    ----------
    samples : pl.DataFrame
        Samples with .draw, .value, and grouping columns
    ci_widths : list[float]
        Widths of credible intervals to compute (e.g., 0.5 = 50% interval)

    Returns
    -------
    pl.DataFrame
        Credible intervals with columns for median, lower, upper bounds at each width
    """
    # Group by everything except .draw and .value
    group_cols = [c for c in samples.columns if c not in [".draw", ".value"]]

    # Calculate quantiles for each CI width
    quantile_exprs = []
    for width in ci_widths:
        lower_q = (1 - width) / 2
        upper_q = 1 - lower_q

        quantile_exprs.extend(
            [
                pl.col(".value").quantile(lower_q).alias(f".lower_{width}"),
                pl.col(".value").quantile(upper_q).alias(f".upper_{width}"),
            ]
        )

    ci = samples.group_by(group_cols).agg(
        [
            pl.col(".value").median().alias(".value"),
            *quantile_exprs,
            pl.len().alias(".n_samples"),
        ]
    )

    # Add .width column for compatibility with R output
    # Create separate rows for each width
    ci_list = []
    for width in ci_widths:
        ci_width = ci.select(
            [
                *group_cols,
                pl.col(".value"),
                pl.col(f".lower_{width}").alias(".lower"),
                pl.col(f".upper_{width}").alias(".upper"),
                pl.lit(width).alias(".width"),
                pl.col(".n_samples"),
            ]
        )
        ci_list.append(ci_width)

    return pl.concat(ci_list, how="vertical")


def process_epiautogp_forecast(
    model_run_dir: Path,
    model_name: str,
    target: str,
    n_forecast_days: int,
    ci_widths: list[float] = [0.5, 0.8, 0.95],
    save: bool = True,
) -> dict[str, pl.DataFrame]:
    """
    Process EpiAutoGP forecast outputs to create samples and credible intervals.

    This function mimics the behavior of the R process_loc_forecast() function
    but is specifically designed for EpiAutoGP outputs.

    Parameters
    ----------
    model_run_dir : Path
        Directory containing model run outputs
    model_name : str
        Name of the EpiAutoGP model directory
    target : str
        Target type ("nhsn" or "nssp") - used to determine input filename
    n_forecast_days : int
        Number of days in the forecast period (currently unused but kept for API compatibility)
    ci_widths : list[float], default=[0.5, 0.8, 0.95]
        Widths of credible intervals to compute
    save : bool, default=True
        Whether to save the processed outputs as parquet files

    Returns
    -------
    dict[str, pl.DataFrame]
        Dictionary with keys:
        - "samples": Combined observed and forecast samples
        - "ci": Credible intervals
    """
    model_dir = Path(model_run_dir) / model_name
    data_dir = model_dir / "data"

    # Map target to file suffix (matching Julia's DEFAULT_TARGET_LETTER)
    target_suffix = {
        "nhsn": "h",  # hospital admissions
        "nssp": "e",  # emergency department visits
    }
    suffix = target_suffix.get(target, "e")  # default to "e" if unknown

    suffix = target_suffix.get(target, "e")  # default to "e" if unknown

    # Read training data (observed values)
    epiweekly_training = pl.read_csv(
        data_dir / "epiweekly_combined_training_data.tsv",
        separator="\t",
        try_parse_dates=True,
    )

    # Ensure date column is proper date type
    if epiweekly_training.schema["date"] != pl.Date:
        epiweekly_training = epiweekly_training.with_columns(
            pl.col("date").str.to_date()
        )

    # Read EpiAutoGP forecast samples
    # EpiAutoGP outputs epiweekly_epiautogp_samples_{suffix}.parquet
    # where suffix is 'h' for nhsn or 'e' for nssp
    forecast_samples = pl.read_parquet(
        model_dir / f"epiweekly_epiautogp_samples_{suffix}.parquet"
    )

    # Ensure forecast dates are also proper date type
    if forecast_samples.schema["date"] != pl.Date:
        forecast_samples = forecast_samples.with_columns(pl.col("date").str.to_date())

    # Prepare observed data for joining
    # Only select columns that exist in the training data
    obs_cols = ["date", ".variable", ".value"]
    for col in ["geo_value", "disease", "resolution"]:
        if col in epiweekly_training.columns:
            obs_cols.append(col)

    observed_for_joining = epiweekly_training.select(obs_cols)

    # Combine forecast with observed data
    model_samples_tidy = combine_forecast_with_observed(
        forecast=forecast_samples,
        observed=observed_for_joining,
    )

    # Add aggregation metadata columns for compatibility with plotting functions
    model_samples_tidy = model_samples_tidy.with_columns(
        [
            pl.lit(False).alias("aggregated_numerator"),
            pl.when(pl.col(".variable").str.starts_with("prop_"))
            .then(pl.lit(False))
            .otherwise(pl.lit(None))
            .alias("aggregated_denominator"),
            # Add placeholder columns for PyRenew compatibility
            pl.lit(None).cast(pl.Int32).alias(".chain"),
            pl.lit(None).cast(pl.Int32).alias(".iteration"),
        ]
    )

    # Calculate credible intervals
    ci = calculate_credible_intervals(model_samples_tidy, ci_widths=ci_widths)

    result = {
        "samples": model_samples_tidy,
        "ci": ci,
    }

    if save:
        save_dir = Path(model_run_dir) / model_name
        for name, df in result.items():
            df.write_parquet(save_dir / f"{name}.parquet")

    return result
