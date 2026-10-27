# Flood Risk Multi-Model Benchmark Dashboard

## Project Overview
**FloodPredictionModel** is an enterprise-grade, production-ready machine learning application that pairs a high-performance Streamlit flood risk benchmarking dashboard with a secure, cloud-native infrastructure. Designed to transition machine learning workflows seamlessly from local research to robust production deployment, the application benchmarks flood risk predictions across four distinct architectures—**K-Means Feature Regressor**, **Random Forest**, **XGBoost / Gradient Boosting**, and a **Deep Neural Network**—while running on a high-availability **Google Kubernetes Engine (GKE)** cluster backed by an automated **GitHub Actions CI/CD pipeline** and **Workload Identity Federation (OIDC)**.

---
## Production Architecture & DevOps (GCP & GKE)
Migrated from Streamlit Community Cloud to a professional production-grade cloud environment on **Google Cloud Platform (GCP)**:
- **Containerization**: Packaged with a custom optimized `Dockerfile` for stateless container execution.
- **Orchestration**: Deployed on **Google Kubernetes Engine (GKE)** (`flood-cluster`) using high-availability Kubernetes deployments and services (`k8s-deployment.yaml`).
- **CI/CD Pipeline**: Automated continuous integration and deployment via **GitHub Actions** (`deploy.yaml`) utilizing **Google Cloud Build** for image compilation and Google Artifact Registry for image hosting.
- **Secure Authentication**: Implements **Workload Identity Federation (OIDC)** to securely authenticate GitHub Actions with GCP without relying on long-lived static JSON keys.
- **Infrastructure as Code (IaC)**: Provisioned and managed using **Terraform**.

---
## Features
- **Cloud-Native Scalability**: Deployed on GKE with zero-downtime rolling updates and automated container builds.
- **Interactive Streamlit Dashboard**: Provides a modular web interface equipped with sidebar sliders to dynamically tweak environmental factors, configure K-Means clusters, and visualize real-time flood probability predictions.
- **Rigorous Statistical Model Comparison**: Evaluates and contrasts model performance using robust statistical metrics including **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and the **Coefficient of Determination ($R^2$ Score)**.
- **Automated Data Processing & Training Pipeline**: Implements modular scripts for data loading, cross-validation splits, serialized model saving, and offline evaluation.
- **Visual Analytics & Spread Visualization**: Features probability spread bar plots, K-Means elbow method analysis, actual vs. predicted evaluation scatter plots, and live metric cards.

---

## Tech Stack
- **Cloud & DevOps**: Google Cloud Platform (GCP), Google Kubernetes Engine (GKE), Google Artifact Registry, Google Cloud Build, Terraform, GitHub Actions, Workload Identity Federation (OIDC)
- **Containerization**: Docker, Kubernetes (`kubectl`)
- **Framework & UI**: Streamlit
- **Machine Learning & Deep Learning**: Scikit-Learn, TensorFlow / Keras 3, XGBoost / Scikit-Learn Gradient Boosting
- **Data Manipulation & Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Model Serialization**: Joblib

---

## Project Structure
- `app.py`: Main interactive Streamlit multi-model dashboard.
- `Dockerfile`: Container configuration for production deployment.
- `k8s-deployment.yaml`: Kubernetes manifests for GKE orchestration.
- `.github/workflows/deploy.yaml`: Automated GitHub Actions CI/CD workflow.
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
