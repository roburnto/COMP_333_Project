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


def perform_tree_regression(data, target_column, drop_missing=True):
    """
    Performs a Decision Tree Regression on the dataset, either dropping missing values
    or imputing them with the mean.

    Parameters:
    - data (pd.DataFrame): DataFrame to analyze.
    - target_column (str): The target column to predict.
    - drop_missing (bool): Whether to drop missing values or impute.

    Returns:
    - model (DecisionTreeRegressor): The fitted regression model.
    - mse (float): The mean squared error of the model.
    """
    # Prepare the features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Handle missing values
    if drop_missing:
        X = X.dropna()
        y = y[X.index]  # Ensure y matches the X after dropping
    else:
        X = X.fillna(X.mean())  # Impute with mean

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Create and fit the model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and compute the mean squared error
    predictions = model.predict(X_test)

    return model
