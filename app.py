import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

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
    kmeans = joblib.load('models/model_kmeans.pkl')
    rf = joblib.load('models/model_rf.pkl')
    xgb = joblib.load('models/model_xgb.pkl')
    nn = tf.keras.models.load_model('models/model_nn.keras')
    features = joblib.load('models/feature_names.pkl')
    df = pd.read_csv('data/flood.csv')
    return kmeans, rf, xgb, nn, features, df

kmeans_model, rf_model, xgb_model, nn_model, feature_names, df = load_assets()

st.title("🌊 Flood Risk Multi-Model Benchmark Dashboard")
st.markdown("""
Modularized enterprise-grade ML dashboard comparing **K-Means Clustering**, **Random Forest**, **XGBoost**, and **Deep Neural Network**. Use the sidebar to tweak environmental factors and configure the K-Means elbow parameter.
""")

st.sidebar.header("🎛️ Environmental Factors")
input_data = {}
for col in feature_names:
    input_data[col] = st.sidebar.slider(
        label=col.replace('_', ' '), 
        min_value=int(df[col].min()), 
        max_value=int(df[col].max()), 
        value=int(df[col].median())
    )

input_df = pd.DataFrame([input_data])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ K-Means Hyperparameters")
n_clusters_param = st.sidebar.slider("Number of Clusters (k)", min_value=5, max_value=30, value=15, step=1)

pred_km = kmeans_model.predict(input_df)[0]
pred_rf = rf_model.predict(input_df)[0]
pred_xgb = xgb_model.predict(input_df)[0]
pred_nn = nn_model.predict(input_df, verbose=0).flatten()[0]

st.subheader("🎯 Live Model Prediction Comparison")
c1, c2, c3, c4 = st.columns(4)
c1.metric(label="🔵 K-Means Cluster", value=f"{pred_km:.2%}")
c2.metric(label="🌲 Random Forest", value=f"{pred_rf:.2%}")
c3.metric(label="🚀 XGBoost", value=f"{pred_xgb:.2%}")
c4.metric(label="🧠 Neural Network", value=f"{pred_nn:.2%}")

st.markdown("---")
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Architecture Prediction Spread")
    pred_df = pd.DataFrame({
        'Model': ['K-Means', 'Random Forest', 'XGBoost', 'Neural Network'],
        'Predicted Flood Probability': [pred_km, pred_rf, pred_xgb, pred_nn]
    })
    
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=pred_df, x='Model', y='Predicted Flood Probability', palette='crest', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability Score")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    st.pyplot(fig)

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
    st.pyplot(fig2)
