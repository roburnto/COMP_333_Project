import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def display_correlation_heatmap(data, title='Correlation Matrix'):
    """
    Computes and displays a heatmap of the correlation matrix for the provided DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame to compute the correlation matrix from.
    - title (str): Title for the heatmap.
    """
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True,
                fmt=".2f", cmap='coolwarm', square=True)
    plt.title(title)
    plt.show()


def analyze_missing_values(data):
    """
    Analyzes the missing values in the DataFrame.

    Parameters:
    - data (pd.DataFrame): DataFrame to analyze.

    Returns:
    - None: Prints the percentage of missing values for each column.
    """
    missing_values = data.isnull().sum()
    total = data.shape[0]
    percent_missing = (missing_values / total) * 100
    missing_info = pd.DataFrame(
        {'Missing Values': missing_values, 'Percentage': percent_missing})
    print(missing_info[missing_info['Missing Values'] > 0])


def perform_tree_regression(X, y, drop_missing=True):
    """
    Performs a Decision Tree Regression on the dataset, either dropping missing values
    or imputing them with the mean.

    Parameters:
    - X (pd.DataFrame): Feature DataFrame.
    - y (pd.Series): Target variable.
    - drop_missing (bool): Whether to drop missing values or impute.

    Returns:
    - model (DecisionTreeRegressor): The fitted regression model.
    - mse (float): The mean squared error of the model.
    """
    # Handle missing values
    if drop_missing:
        # Drop rows with missing values in X and align y accordingly
        X = X.dropna()
        y = y.loc[X.index]  # Select only rows in y that match the new X index
    else:
        X = X.fillna(X.mean())  # Impute with mean

    # Reset indices to avoid misalignment during train-test split
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and fit the model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and compute the mean squared error

    return model
