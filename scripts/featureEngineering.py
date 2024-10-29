import pandas as pd


def extract_year(df, column, new_column_name):
    df[new_column_name] = df[column].dt.year
    return df


def difference(df, column1, column2):
    newcol = f"diff_btw_{column1}_and_{column2}"
    df[newcol] = df[column1] - df[column2]
    return df


def adjust_prices(kingcoSales, kingcoIndex, base_year=2024):
    """
    Adjust sale prices, improvement values, and land values for inflation 
    using the HPI from the kingcoIndex DataFrame.

    Parameters:
    - kingcoSales (DataFrame): DataFrame containing sales data with 'sale_price', 'join_year', and 'sale_year'.
    - kingcoIndex (DataFrame): DataFrame containing HPI data with 'year' and 'HPI'.
    - base_year (int): The year to use as the base for inflation adjustment (default is 2024).

    Returns:
    - DataFrame: Updated DataFrame with adjusted sale prices, adjusted improvement values, 
                  and adjusted land values, without inflation factors and HPI.
    """

    # Merge the two DataFrames based on the sale_year
    merged_sale_df = pd.merge(
        kingcoSales,
        kingcoIndex[['year', 'HPI']],
        left_on='sale_year',
        right_on='year',
        how='left'
    )

    # Debug: Check columns in merged_sale_df
    print("Columns after sale year merge:", merged_sale_df.columns)

    # Calculate the inflation adjustment factor using sale_year
    base_year_hpi_sale = merged_sale_df.loc[merged_sale_df['year']
                                            == base_year, 'HPI']
    if base_year_hpi_sale.empty:
        raise ValueError(f"No HPI data found for base year {
                         base_year} in sale year data.")

    merged_sale_df['inflation_factor_sale'] = base_year_hpi_sale.iloc[0] / \
        merged_sale_df['HPI']

    # Adjust the sale_price for inflation
    merged_sale_df['adjusted_sale_price'] = merged_sale_df['sale_price'] * \
        merged_sale_df['inflation_factor_sale']

    # Now merge with kingcoIndex again for join_year calculations
    merged_join_df = pd.merge(
        merged_sale_df,
        kingcoIndex[['year', 'HPI']],
        left_on='join_year',
        right_on='year',
        how='left'
    )

    # Debug: Check columns in merged_join_df
    print("Columns after join year merge:", merged_join_df.columns)

    # Calculate the inflation adjustment factor using join_year
    base_year_hpi_join = merged_join_df.loc[merged_join_df['year']
                                            == base_year, 'HPI']
    if base_year_hpi_join.empty:
        raise ValueError(f"No HPI data found for base year {
                         base_year} in join year data.")

    merged_join_df['inflation_factor_join'] = base_year_hpi_join.iloc[0] / \
        merged_join_df['HPI']

    # Calculate adjusted_imp_value and adjusted_land_value using the join_year inflation factor
    merged_join_df['adjusted_imp_value'] = merged_join_df['imp_val'] * \
        merged_join_df['inflation_factor_join']
    merged_join_df['adjusted_land_value'] = merged_join_df['land_val'] * \
        merged_join_df['inflation_factor_join']

    # Select only the relevant columns and drop unnecessary ones
    final_df = merged_join_df[['adjusted_sale_price',
                               'adjusted_imp_value', 'adjusted_land_value']]

    return final_df

# Example usage:
# adjusted_df = adjust_prices(king_county_sales_df, king_county_index_df, base_year=2023)
