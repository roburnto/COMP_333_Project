import pandas as pd


def load_data(file_path):
    return pd.read_csv(file_path)


def view_data(df):
    df.info()
    df.describe()
    return


def missing_data(df):
    df.isnull().sum()
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()
               ).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1,
                             keys=['Total', 'Percent'])
    return missing_data


def clean_data(df):
    df.drop_duplicates()
    return df


def drop_columns(df, columns):
    return df.drop(columns, axis=1)


def parse_data_types(df, column_types):
    """
    Parse the data types of specified columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column_types (dict): A dictionary where keys are column names and values are the desired data types.

    Returns:
        pd.DataFrame: The DataFrame with updated data types.
    """

    for column, dtype in column_types.items():
        if column in df.columns:
            if dtype == 'datetime':
                # Convert to datetime
                df[column] = pd.to_datetime(
                    df[column], errors='coerce')  # Coerce errors to NaT
            elif dtype in ['float', 'int']:
                # Convert to numeric (float or int)
                df[column] = pd.to_numeric(
                    df[column], errors='coerce')  # Coerce errors to NaN
                if dtype == 'int':
                    df[column] = df[column].fillna(0).astype(
                        int)  # Fill NaN with 0 for integers
            elif dtype == 'category':
                # Convert to categorical
                df[column] = df[column].astype('category')
            else:
                print(f"Unsupported data type: {dtype} for column: {column}")

    return df
