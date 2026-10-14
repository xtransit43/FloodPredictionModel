import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from src.data_loader import load_data, get_train_test_splits

class KMeansFeatureRegressor:
    def __init__(self, n_clusters=15, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.regressor = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        train_distances = self.kmeans.transform(X_scaled)
        self.regressor.fit(train_distances, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        test_distances = self.kmeans.transform(X_scaled)
        return self.regressor.predict(test_distances)

def train_and_save_models():
    os.makedirs('models', exist_ok=True)
    
    print("Loading data...")
    df, X, y = load_data('data/flood.csv')
    X_train, X_val, X_test, y_train, y_val, X_train_full, y_train_full = get_train_test_splits(X, y)
    
    # 1. K-Means Feature Regressor
    print("Training K-Means Feature Regressor...")
    kmeans_model = KMeansFeatureRegressor(n_clusters=15, random_state=42)
    kmeans_model.fit(X_train_full, y_train_full)
    joblib.dump(kmeans_model, 'models/model_kmeans.pkl')
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_full, y_train_full)
    joblib.dump(rf_model, 'models/model_rf.pkl')
    
    # 3. XGBoost / Gradient Boosting
    print("Training XGBoost / Gradient Boosting...")
    xgb_model = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train_full, y_train_full)
    joblib.dump(xgb_model, 'models/model_xgb.pkl')
    
    # 4. Deep Neural Network
    print("Training Deep Neural Network...")
    nn_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    nn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
    nn_model.save('models/model_nn.keras')
    
    # Save feature names metadata
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    print("All 4 models successfully trained and serialized inside 'models/'!")

if __name__ == "__main__":
    train_and_save_models()
