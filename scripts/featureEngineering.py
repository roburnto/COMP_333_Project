import pandas as pd


def extract_year(df, column, new_column_name):
    df[new_column_name] = df[column].dt.year
    return df


def difference(df, column1, column2):
    newcol = f"diff_btw_{column1}_and_{column2}"
    df[newcol] = df[column1] - df[column2]
    return df


def create_hpi_mapping(df_hpi):
    """
    Create a mapping of HPI for the years relevant to the sales and joining.

    Parameters:
        df_hpi (DataFrame): DataFrame containing 'HPI' and 'year'.

    Returns:
        dict: A dictionary mapping year to HPI.
    """
    return df_hpi.set_index('year')['HPI'].to_dict()


def adjust_prices(row, hpi_mapping):
    """
    Adjust sale_price, imp_val, and land_val based on HPI values.

    Parameters:
        row (Series): A row from the sales DataFrame.
        hpi_mapping (dict): A dictionary mapping year to HPI.

    Returns:
        Series: Adjusted values for sale_price, imp_val, and land_val.
    """
    # Extract the HPI for sale_year and join_year
    sale_year = row['sale_year']
    join_year = row['join_year']

    hpi_sale = hpi_mapping.get(sale_year, 1)  # Default to 1 if year not found
    hpi_join = hpi_mapping.get(join_year, 1)  # Default to 1 if year not found
    hpi_2023 = hpi_mapping.get(2023, 1)  # Default to 1 if year not found

    # Adjust sale_price, imp_val, and land_val
    adjusted_sale_price = row['sale_price'] * (hpi_2023 / hpi_sale)
    adjusted_imp_val = row['imp_val'] * (hpi_2023 / hpi_join)
    adjusted_land_val = row['land_val'] * (hpi_2023 / hpi_join)

    return pd.Series({
        'adjusted_sale_price': adjusted_sale_price,
        'adjusted_imp_val': adjusted_imp_val,
        'adjusted_land_val': adjusted_land_val
    })


def process_sales_data(df_hpi, df_sales):
    """
    Process sales data by adjusting prices based on HPI.

    Parameters:
        df_hpi (DataFrame): DataFrame containing HPI data.
        df_sales (DataFrame): DataFrame containing sales data.

    Returns:
        DataFrame: The original sales DataFrame with adjusted price columns added.
    """
    # Create HPI mapping
    hpi_mapping = create_hpi_mapping(df_hpi)

    # Apply the adjustment function to the sales DataFrame
    adjusted_values = df_sales.apply(
        adjust_prices, axis=1, hpi_mapping=hpi_mapping)

    # Combine the adjusted values back into the original sales DataFrame
    df_sales = pd.concat([df_sales, adjusted_values], axis=1)

    return df_sales


def target_encode(df, target_col, cat_cols):
    for col in cat_cols:
        # Calculate mean of the target variable for each category
        mean_target = df.groupby(col)[target_col].mean()
        # Map the mean values back to the original DataFrame
        df[col + '_encoded'] = df[col].map(mean_target)
    return df
