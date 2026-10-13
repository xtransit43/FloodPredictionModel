import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Flood Risk Multi-Model Dashboard",
    page_icon="🌊",
    layout="wide"
)

# Load Saved Assets
@st.cache_resource
def load_assets():
    lr = joblib.load('models/model_lr.pkl')
    rf = joblib.load('models/model_rf.pkl')
    nn = tf.keras.models.load_model('models/model_nn.keras')
    features = joblib.load('models/feature_names.pkl')
    df = pd.read_csv('data/flood.csv')
    return lr, rf, nn, features, df

lr_model, rf_model, nn_model, feature_names, df = load_assets()

# App Header
st.title("🌊 Flood Risk Multi-Model Benchmark Dashboard")
st.markdown("""
Modularized enterprise-grade ML dashboard comparing **Linear Regression**, **Random Forest**, and a **Deep Neural Network**. Use the sidebar to tweak environmental factors and visualize real-time architecture predictions.
""")

# Sidebar Inputs
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

# Predictions
pred_lr = lr_model.predict(input_df)[0]
pred_rf = rf_model.predict(input_df)[0]
pred_nn = nn_model.predict(input_df, verbose=0).flatten()[0]

# Metrics Display
st.subheader("🎯 Live Model Prediction Comparison")
c1, c2, c3 = st.columns(3)
c1.metric(label="📈 Linear Regression", value=f"{pred_lr:.2%}")
c2.metric(label="🌲 Random Forest", value=f"{pred_rf:.2%}")
c3.metric(label="🧠 Neural Network", value=f"{pred_nn:.2%}")

# Bar Plot Spread
st.markdown("---")
st.subheader("📊 Architecture Prediction Spread")
pred_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Neural Network'],
    'Predicted Flood Probability': [pred_lr, pred_rf, pred_nn]
})

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=pred_df, x='Model', y='Predicted Flood Probability', palette='crest', ax=ax)
ax.set_ylim(0, 1)
ax.set_ylabel("Probability Score")
ax.set_xlabel("")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, fontweight='bold')
st.pyplot(fig)
