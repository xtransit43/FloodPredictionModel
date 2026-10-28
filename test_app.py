import os
import pytest
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from src.data_loader import load_data, get_train_test_splits

def test_data_loading():
    """Verify that dataset loads correctly and contains expected columns."""
    df, X, y = load_data('data/flood.csv')
    assert df is not None, "Dataset failed to load."
    assert X.shape[0] > 0, "Feature set is empty."
    assert len(y) == len(X), "Features and target length mismatch."
    assert 'FloodProbability' in df.columns or y.name is not None

def test_train_test_splits():
    """Verify train, validation, and test split shapes."""
    df, X, y = load_data('data/flood.csv')
    X_train, X_val, X_test, y_train, y_val, X_train_full, y_train_full, y_test = get_train_test_splits(X, y)
    
    assert len(X_train_full) > len(X_test), "Test split should be smaller than full training set."
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions must match across splits."

@pytest.mark.skipif(not os.path.exists('models/model_rf.pkl'), reason="Models not trained yet.")
def test_model_predictions():
    """Verify that trained models successfully generate predictions."""
    df, X, y = load_data('data/flood.csv')
    _, _, X_test, _, _, _, _, _ = get_train_test_splits(X, y)
    
    # Take a small sample for fast testing
    sample_X = X_test.head(5)
    
    # Test Random Forest
    rf_model = joblib.load('models/model_rf.pkl')
    preds = rf_model.predict(sample_X)
    assert len(preds) == len(sample_X), "Prediction count mismatch for Random Forest."
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be probabilities between 0 and 1."
