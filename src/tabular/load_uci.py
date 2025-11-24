# src/tabular/load_uci.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


def load_housing(test_size=0.2, seed=42):
    """
    California Housing dataset (safe alternative to Boston Housing)
    """
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()

    X = data.data
    y = data.target
    feature_names = data.feature_names

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names



def load_energy(test_size=0.2, seed=42):
    """
    Energy Efficiency dataset (UCI)
    Target: Heating Load (Y1)
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    df = pd.read_excel(url)

    # Features in columns 0..7, target Y1 in column 8
    X = df.iloc[:, 0:8].values
    y = df.iloc[:, 8].values
    feature_names = df.columns[:8].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, feature_names
