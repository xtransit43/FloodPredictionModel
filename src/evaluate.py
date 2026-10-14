import os
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from data_loader import load_data, get_train_test_splits

class KMeansFeatureRegressor:
    """K-Means clustering distance-based feature regressor."""
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

def evaluate_models():
    print("Loading test data and models...")
    _, X, y = load_data('data/flood.csv')
    _, _, X_test, _, y_test, _, _ = get_train_test_splits(X, y)
    
    # Load trained model artifacts
    kmeans_model = joblib.load('models/model_kmeans.pkl')
    rf_model = joblib.load('models/model_rf.pkl')
    xgb_model = joblib.load('models/model_xgb.pkl')
    nn_model = tf.keras.models.load_model('models/model_nn.keras')
    
    # Fix Keras 3 steps_per_execution bug if present
    if getattr(nn_model, 'steps_per_execution', None) is None:
        nn_model.steps_per_execution = 1
    
    preds = {
        'K-Means Cluster Model': kmeans_model.predict(X_test),
        'Random Forest': rf_model.predict(X_test),
        'XGBoost': xgb_model.predict(X_test),
        'Neural Network': nn_model.predict(X_test.to_numpy(), verbose=0).flatten()
    }
    
    results = []
    print("\n--- Comprehensive 4-Model Evaluation Benchmark ---")
    for name, y_pred in preds.items():
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            'Model': name, 
            'MSE': mse, 
            'RMSE': rmse, 
            'MAE': mae, 
            'R2 Score': r2
        })
        print(f"{name} -> MSE: {mse:.6f} | RMSE: {rmse:.6f} | MAE: {mae:.6f} | R²: {r2:.4f}")
        
    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    
    # Generate and save evaluation scatter plots
    os.makedirs('models', exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    for i, (name, y_pred) in enumerate(preds.items()):
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color='teal', ax=axes[i])
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_title(f'{name}', fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Actual Probability', fontsize=9)
        if i == 0:
            axes[i].set_ylabel('Predicted Probability', fontsize=9)
            
    plt.suptitle('Multi-Model Evaluation: Actual vs. Predicted Probabilities', fontsize=15, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig('models/evaluation_plot.png')
    plt.close(fig)
    print("\nEvaluation plot saved successfully to models/evaluation_plot.png!")

if __name__ == "__main__":
    evaluate_models()
