import pandas as pd


def extract_year(df, column, new_column_name):
    df[new_column_name] = df[column].dt.year
    return df


def difference(df, column1, column2):
    newcol = f"diff_btw_{column1}_and_{column2}"
    df[newcol] = df[column1] - df[column2]
    return df


def adjust_for_inflation(df1, df2, year1, attribute):
    # Create a subset of df1 with year1 and attribute
    subset_df1 = df1[[year1, attribute]].copy()

    # Merge the subset with df2 to get HPI
    merged_df = pd.merge(subset_df1, df2[['year', 'HPI']],
                         left_on=year1, right_on='year', how='left')

    # Calculate the inflation factor based on the base year HPI
    base_year_hpi = merged_df.loc[merged_df['year'] == 2023, 'HPI'].iloc[0]
    merged_df['inflation_factor'] = base_year_hpi / merged_df['HPI']

    # Calculate the adjusted attribute
    adjusted_attribute = f"adjusted_{attribute}"
    merged_df[adjusted_attribute] = merged_df[attribute] * \
        merged_df['inflation_factor']

    # Merge the adjusted attribute back to the original df1
    df1 = df1.merge(merged_df[[year1, adjusted_attribute]],
                    on=year1,
                    how='left')

    return df1
