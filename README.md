# Flood Risk Multi-Model Benchmark Dashboard

## Project Overview
**Flood Risk Multi-Model Benchmark Dashboard** is an enterprise-grade machine learning and interactive Streamlit web application designed to benchmark and compare environmental flood risk predictions across multiple modeling architectures (Linear Regression, Random Forest, and a Deep Neural Network).

---

## Features
* **Interactive Streamlit Dashboard**: Provides a modular web interface equipped with sidebar sliders to dynamically tweak environmental factors and visualize real-time flood probability predictions.
* **Rigorous Statistical Model Comparison**: Evaluates and contrasts model performance using robust statistical metrics including **Mean Squared Error (MSE)** to quantify residual error variance and the **Coefficient of Determination ($R^2$ Score)** to measure goodness-of-fit and variance explanation across architectures.
* **Automated Data Processing & Training Pipeline**: Implements modular scripts for data loading, cross-validation tuning (`GridSearchCV`, `KFold`), serialized model saving, and offline evaluation.
* **Visual Analytics & Spread Visualization**: Features probability spread bar plots, feature correlation heatmaps, scatterplot grids, and actual vs. predicted evaluation charts.

---

## Tech Stack
* **Framework & UI**: Streamlit
* **Machine Learning & Deep Learning**: Scikit-Learn, TensorFlow / Keras
* **Data Manipulation & Analysis**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Model Serialization**: Joblib

---

## Project Structure
* `app.py`: Main interactive Streamlit multi-model dashboard.
* `train.py`: Training script for generating and serializing model artifacts.
* `evaluate.py`: Model evaluation script calculating MSE and $R^2$ scores and outputting plots.
* `neural_network.py`: Exploratory data analysis, hyperparameter tuning, and training pipeline.
* `src/data_loader.py`: Utility functions for loading and splitting data.
* `models/`: Directory housing serialized models (`model_lr.pkl`, `model_rf.pkl`, `model_nn.keras`), feature metadata, and evaluation outputs.

---

## Getting Started & Installation

### Prerequisites
Make sure you have [Python](https://www.python.org/) installed on your machine.

### Installation Steps
1. Clone the repository and navigate to the project directory:
   ```bash
   cd FloodPredictionModel
   ```
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