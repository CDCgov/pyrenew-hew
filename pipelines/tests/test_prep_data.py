from pipelines import prep_data


def test_get_loc_pop_df():
    """
    Confirm get_loc_pop_df()
    returns a polars data frame
    with the expected number of rows
    and expected column names
    """
    df = prep_data.get_loc_pop_df()
    assert df.height == 58  # 50 locs, 7 other jursidictions, US national
    assert set(df.columns) == set(["name", "abb", "population"])
