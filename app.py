import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

st.set_page_config(
    page_title="Flood Risk Multi-Model Dashboard",
    page_icon="🌊",
    layout="wide"
)

@st.cache_resource
def load_assets():
    try:
        kmeans = joblib.load('models/model_kmeans.pkl')
        rf = joblib.load('models/model_rf.pkl')
        xgb = joblib.load('models/model_xgb.pkl')
        nn = tf.keras.models.load_model('models/model_nn.keras')
        
        if getattr(nn, 'steps_per_execution', None) is None:
            nn.steps_per_execution = 1
            
        features = joblib.load('models/feature_names.pkl')
        df, X, y = load_data('data/flood.csv')
        _, _, X_test, _, y_test, _, _, y_test = get_train_test_splits(X, y)
        return kmeans, rf, xgb, nn, features, df, X_test, y_test
    except Exception as e:
        return None, None, None, None, None, None, None, None

kmeans_model, rf_model, xgb_model, nn_model, feature_names, df, X_test, y_test = load_assets()

if kmeans_model is None or feature_names is None or df is None:
    st.error("⚠️ Model artifacts or dataset not found! Please run `python src/train.py` first to generate the models.")
    st.stop()

os.makedirs('models', exist_ok=True)

st.title("🌊 Flood Risk Multi-Model Benchmark Dashboard")
st.markdown("""
Modularized enterprise-grade ML dashboard comparing **K-Means Clustering**, **Random Forest**, **XGBoost / Gradient Boosting**, and **Deep Neural Network**. Use the sidebar to tweak environmental factors and configure model parameters.
""")

# Sidebar Controls
st.sidebar.header("🎛️ Environmental Factors")
input_data = {}
for col in feature_names:
    input_data[col] = st.sidebar.slider(
        label=str(col).replace('_', ' '), 
        min_value=int(df[col].min()), 
        max_value=int(df[col].max()), 
        value=int(df[col].median())
    )

input_df = pd.DataFrame([input_data])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ K-Means Hyperparameters")
n_clusters_param = st.sidebar.slider("Number of Clusters (k)", min_value=5, max_value=30, value=15, step=1)

# Live Predictions
pred_km = float(kmeans_model.predict(input_df)[0])
pred_rf = float(rf_model.predict(input_df)[0])
pred_xgb = float(xgb_model.predict(input_df)[0])
pred_nn = float(nn_model.predict(input_df.to_numpy(), verbose=0).flatten()[0])

st.subheader("🎯 Live Model Prediction Comparison")
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="🔵 K-Means Cluster", value=f"{pred_km:.2%}")
c2.metric(label="🌲 Random Forest", value=f"{pred_rf:.2%}")
c3.metric(label="🚀 XGBoost / GB", value=f"{pred_xgb:.2%}")
c4.metric(label="🧠 Neural Network", value=f"{pred_nn:.2%}")

st.markdown("---")

# Visualizations
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Architecture Prediction Spread")
    pred_df = pd.DataFrame({
        'Model': ['K-Means', 'Random Forest', 'XGBoost', 'Neural Network'],
        'Predicted Flood Probability': [pred_km, pred_rf, pred_xgb, pred_nn]
    })
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=pred_df, x='Model', y='Predicted Flood Probability', hue='Model', palette='crest', ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability Score")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    chart_path_1 = 'models/chart_spread.png'
    fig.savefig(chart_path_1, bbox_inches='tight', dpi=150)
    plt.close(fig)
    st.image(chart_path_1, use_container_width=True)

with col_right:
    st.subheader("📈 K-Means Elbow Method Analysis")
    X_sample = df.iloc[:, :-1].sample(min(2000, len(df)), random_state=42)
    inertias = []
    k_range = range(2, 16)
    for k in k_range:
        km_temp = KMeans(n_clusters=k, random_state=42, n_init=5)
        km_temp.fit(X_sample)
        inertias.append(km_temp.inertia_)
    
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(list(k_range), inertias, marker='o', color='teal', lw=2)
    ax2.axvline(x=n_clusters_param, color='red', linestyle='--', label=f'Selected k={n_clusters_param}')
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Inertia")
    ax2.legend()
    
    chart_path_2 = 'models/chart_elbow.png'
    fig2.savefig(chart_path_2, bbox_inches='tight', dpi=150)
    plt.close(fig2)
    st.image(chart_path_2, use_container_width=True)

st.markdown("---")

# Evaluation Benchmark Statistics Section
st.subheader("📐 Global Test Set Evaluation Statistics")
st.markdown("Performance benchmarks computed across the holdout test set (`n=7,500` samples):")

@st.cache_data
def compute_metrics():
    preds_test = {
        'K-Means Cluster': kmeans_model.predict(X_test),
        'Random Forest': rf_model.predict(X_test),
        'XGBoost / GB': xgb_model.predict(X_test),
        'Neural Network': nn_model.predict(X_test.to_numpy(), verbose=0).flatten()
    }
    
    metrics_list = []
    for name, y_pred in preds_test.items():
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics_list.append({
            'Model Architecture': name,
            'MSE': f"{mse:.5f}",
            'RMSE': f"{rmse:.5f}",
            'MAE': f"{mae:.5f}",
            'R² Score': f"{r2:.4f}"
        })
    return pd.DataFrame(metrics_list)

metrics_df = compute_metrics()
st.dataframe(metrics_df, use_container_width=True, hide_index=True)
