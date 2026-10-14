# Flood Risk Multi-Model Benchmark Dashboard

## Project Overview
**FloodPredictionModel** is an interactive multi-model machine learning dashboard built with Streamlit that benchmarks flood risk predictions across four distinct architectures: **K-Means Feature Regressor**, **Random Forest**, **XGBoost / Gradient Boosting**, and a **Deep Neural Network**. It utilizes modular training pipelines, interactive environmental parameter controls, and detailed statistical evaluations comparing Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and $R^2$ scores across models.

---
## Features
- **Interactive Streamlit Dashboard**: Provides a modular web interface equipped with sidebar sliders to dynamically tweak environmental factors, configure K-Means clusters, and visualize real-time flood probability predictions.
- **Rigorous Statistical Model Comparison**: Evaluates and contrasts model performance using robust statistical metrics including **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and the **Coefficient of Determination ($R^2$ Score)**.
- **Automated Data Processing & Training Pipeline**: Implements modular scripts for data loading, cross-validation splits, serialized model saving, and offline evaluation.
- **Visual Analytics & Spread Visualization**: Features probability spread bar plots, K-Means elbow method analysis, actual vs. predicted evaluation scatter plots, and live metric cards.

---

## Tech Stack
- **Framework & UI**: Streamlit
- **Machine Learning & Deep Learning**: Scikit-Learn, TensorFlow / Keras 3, XGBoost / Scikit-Learn Gradient Boosting
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib

---

## Project Structure
- `app.py`: Main interactive Streamlit multi-model dashboard.
- `src/train.py`: Training script for generating and serializing model artifacts inside `models/`.
- `src/evaluate.py`: Model evaluation script calculating benchmark metrics and outputting comparison plots.
- `src/data_loader.py`: Utility functions for loading and splitting data.
- `models/`: Directory housing serialized models (`model_kmeans.pkl`, `model_rf.pkl`, `model_xgb.pkl`, `model_nn.keras`), feature metadata, and evaluation outputs.

---

## Getting Started & Installation

### Prerequisites
Make sure you have [Python](https://www.python.org/) installed on your machine.

### Installation Steps
1. Clone the repository and navigate to the project directory:
   ```bash
   cd FloodPredictionModel
2. Install project dependencies:
   ``` bash
   pip install -r requirements.txt
   ```
3. Run the training pipeline to generate model assets:
   ``` bash
   python train.py
   ```
4. Launch the Streamlit Dashboard
   ``` bash
   streamlit run app.py
   ```
