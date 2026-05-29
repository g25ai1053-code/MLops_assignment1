"""
misc.py
--------
Generic, reusable helper functions for the ML Ops Assignment 1 workflow.

The functions here are intentionally model-agnostic so that the SAME code can
train a DecisionTreeRegressor (train.py) or a KernelRidge model (train2.py),
or any other scikit-learn regressor, simply by passing in a different model
object.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def load_data():
    """
    Load the Boston Housing dataset manually.

    The dataset was removed from scikit-learn due to ethical concerns, so we
    rebuild it directly from the original source as recommended in the
    scikit-learn 1.0 documentation.

    Returns
    -------
    pandas.DataFrame
        DataFrame with 13 feature columns and the target column 'MEDV'.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)

    # The raw file stores each record across two lines, so we stitch them back
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
    ]

    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target  # MEDV is the target variable (median home value)
    return df


def preprocess_data(df, target_column="MEDV", test_size=0.2, random_state=42):
    """
    Split the data into train/test sets and standard-scale the features.

    Scaling is fit ONLY on the training data and then applied to the test data
    to avoid leakage. This is generic for any tabular regression dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset including the target column.
    target_column : str
        Name of the target column.
    test_size : float
        Fraction of data held out for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy.ndarray
    """
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def train_model(model, X_train, y_train):
    """
    Fit any scikit-learn regressor on the training data.

    Parameters
    ----------
    model : estimator
        Any scikit-learn regression model implementing .fit().
    X_train, y_train : array-like
        Training features and targets.

    Returns
    -------
    model : the fitted estimator
    """
    model.fit(X_train, y_train)
    return model


def test_model(model, X_test, y_test):
    """
    Evaluate a fitted model on the test set and return the MSE.

    Parameters
    ----------
    model : fitted estimator
    X_test, y_test : array-like

    Returns
    -------
    float
        Mean squared error on the test set.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse


def average_mse_cv(model, X, y, cv=5):
    """
    Compute the average MSE using k-fold cross-validation.

    Returns a positive MSE (scikit-learn returns negative MSE for scoring).

    Parameters
    ----------
    model : estimator
    X, y : array-like
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    float
        Average MSE across the folds.
    """
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv)
    return -np.mean(scores)
