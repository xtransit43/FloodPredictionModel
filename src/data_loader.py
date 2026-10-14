import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath='data/flood.csv'):
    """Loads dataset and separates features and target with fallback paths."""
    if not os.path.exists(filepath) and os.path.exists('flood.csv'):
        filepath = 'flood.csv'
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return df, X, y
def get_train_test_splits(X, y):
    """Returns train, validation, and test splits including y_test."""
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.1764, random_state=42
    )
    return X_train, X_val, X_test, y_train, y_val, X_train_full, y_train_full, y_test
