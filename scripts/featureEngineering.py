import pandas as pd


def extract_year(df, column, new_column_name):
    df[new_column_name] = df[column].dt.year
    return df


def difference(df, column1, column2):
    newcol = f"diff_btw_{column1}_and_{column2}"
    df[newcol] = df[column1] - df[column2]
    return df


def adjust_for_inflation(df1, df2, year1, attribute):

    merged_df = pd.merge(df1, df2[['year', 'HPI']],
                         left_on=year1, right_on='year', how='left')

    base_year_hpi = merged_df.loc[merged_df['year'] == 2023, 'HPI'].iloc[0]
    merged_df['inflation_factor'] = base_year_hpi / merged_df['HPI']
    adjusted_attribute = f"adjusted_{attribute}"
    merged_df[adjusted_attribute] = merged_df[attribute] * \
        merged_df['inflation_factor']
    merged_df = merged_df.drop(['year', 'inflation_factor', 'HPI'], axis=1)
    return merged_df
