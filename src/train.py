import os
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from src.data_loader import load_data, get_train_test_splits

def train_and_save_models():
    os.makedirs('models', exist_ok=True)
    
    print("Loading data...")
    df, X, y = load_data('data/flood.csv')
    X_train, X_val, X_test, y_train, y_val, X_train_full, y_train_full = get_train_test_splits(X, y)
    
    # 1. Linear Regression
    print("Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train_full, y_train_full)
    joblib.dump(lr_model, 'models/model_lr.pkl')
    
    # 2. Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_full, y_train_full)
    joblib.dump(rf_model, 'models/model_rf.pkl')
    
    # 3. Neural Network
    print("Training Neural Network...")
    nn_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='linear')
    ])
    nn_model.compile(optimizer='adam', loss='mean_squared_error')
    nn_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
    nn_model.save('models/model_nn.keras')
    
    # Save feature columns metadata
    joblib.dump(list(X.columns), 'models/feature_names.pkl')
    print("All models successfully trained and serialized inside 'models/'!")

if __name__ == "__main__":
    train_and_save_models()
