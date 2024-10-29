import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer


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


def plot_regression_results(y_true, y_pred):
    """
    Plots the true vs. predicted values for the regression model.

    Parameters:
    - y_true (pd.Series or np.array): True target values.
    - y_pred (np.array): Predicted target values by the model.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [
             y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title("True vs. Predicted Values")
    plt.show()


def perform_tree_regression(X, y):
    """
    Performs a Decision Tree Regression on the dataset, assuming missing values
    have already been handled.

    Parameters:
    - X (pd.DataFrame): Feature DataFrame, without missing values.
    - y (pd.Series): Target variable, without missing values.

    Returns:
    - model (DecisionTreeRegressor): The fitted regression model.
    """

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and fit the model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predict and compute metrics
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    medae = median_absolute_error(y_test, predictions)

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Median Absolute Error (MedAE): {medae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    plot_regression_results(y_test, predictions)

    return model


def knn_impute_column(data, target_column, n_neighbors=5, attributes=None):
    """
    Performs KNN imputation for a specified column based on selected attributes.

    Parameters:
    - data (pd.DataFrame): DataFrame containing the data.
    - target_column (str): The column with missing values to impute.
    - n_neighbors (int): Number of neighbors to use for KNN.
    - attributes (list of str): List of attribute columns to use for finding neighbors.

    Returns:
    - pd.DataFrame: DataFrame with the target column imputed.
    """
    # If no specific attributes are provided, use all columns except the target
    if attributes is None:
        attributes = data.columns.drop(target_column)

    # Create a copy of the DataFrame with only the necessary columns for imputation
    impute_data = data[attributes].copy()
    impute_data[target_column] = data[target_column]

    # Initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Perform KNN imputation
    imputed_array = imputer.fit_transform(impute_data)

    # Update the target column in the original data with the imputed values
    data_imputed = data.copy()
    # Last column is the target
    data_imputed[target_column] = imputed_array[:, -1]

    return data_imputed
