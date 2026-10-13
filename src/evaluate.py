import joblib
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from src.data_loader import load_data, get_train_test_splits

def evaluate_models():
    print("Loading test data and models...")
    _, X, y = load_data('data/flood.csv')
    _, _, X_test, _, y_test, _, _ = get_train_test_splits(X, y)
    
    # Load trained models
    lr_model = joblib.load('models/model_lr.pkl')
    rf_model = joblib.load('models/model_rf.pkl')
    nn_model = tf.keras.models.load_model('models/model_nn.keras')
    
    # Generate predictions
    preds = {
        'Linear Regression': lr_model.predict(X_test),
        'Random Forest': rf_model.predict(X_test),
        'Neural Network': nn_model.predict(X_test, verbose=0).flatten()
    }
    
    # Evaluate metrics
    results = []
    for name, y_pred in preds.items():
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'MSE': mse, 'R2 Score': r2})
        print(f"{name} -> MSE: {mse:.6f} | R²: {r2:.4f}")
        
    # Optional: Plot Actual vs Predicted evaluation chart
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, (name, y_pred) in enumerate(preds.items()):
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.3, color='teal', ax=axes[i])
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Actual Probability', fontsize=10)
        
    plt.suptitle('Model Evaluation: Actual vs. Predicted', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('models/evaluation_plot.png')
    print("Evaluation plot saved successfully to models/evaluation_plot.png!")

if __name__ == "__main__":
    evaluate_models()
