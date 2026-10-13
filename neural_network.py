import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Set Professional Seaborn Theme
sns.set_theme(style="whitegrid", palette="muted")

# 2. Load and Inspect Data
df = pd.read_csv('flood.csv')
print("Dataset Shape:", df.shape)

# 3. Separate Features and Target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 4. Exploratory Data Analysis: Correlation Heatmap
plt.figure(figsize=(12, 10))
corr_matrix = df.corr(method='pearson')
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.show()

# 5. Exploratory Data Analysis: Scatterplot Grid (Each Feature vs. FloodProbability)
print("Generating Scatterplots for Each Independent Variable vs. Flood Probability...")
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 14))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    sns.scatterplot(
        data=df, 
        x=col, 
        y='FloodProbability', 
        ax=axes[i], 
        alpha=0.2, 
        color='royalblue'
    )
    axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=9)
    axes[i].set_ylabel('Flood Probability', fontsize=9)

plt.suptitle('Independent Variables vs. Flood Probability', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# 6. Train / Test Split for Final Evaluation
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Further split training into train and validation for the Neural Network
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1764, random_state=42  # ~15% validation split
)

# ==========================================
# MODEL 1: LINEAR REGRESSION (Baseline)
# ==========================================
print("\n--- Training Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_full, y_train_full)
lr_pred = lr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print(f"Linear Regression -> MSE: {lr_mse:.6f} | R² Score: {lr_r2:.4f}")

# ==========================================
# MODEL 2: RANDOM FOREST + GRIDSEARCHCV + KFOLD
# ==========================================
print("\n--- Tuning Random Forest with GridSearchCV & K-Fold ---")
# Setup 5-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define hyperparameter grid to search over
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [5, 10]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=kf,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_full, y_train_full)
print(f"Best Hyperparameters found: {grid_search.best_params_}")

# Use best estimator from grid search
rf_model = grid_search.best_estimator_
rf_pred = rf_model.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print(f"Optimized Random Forest -> MSE: {rf_mse:.6f} | R² Score: {rf_r2:.4f}")

# ==========================================
# MODEL 3: DEEP LEARNING (Neural Network)
# ==========================================
print("\n--- Training Neural Network ---")
nn_model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='linear')
])

nn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mean_squared_error',
    metrics=['mae']
)

history = nn_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32,
    verbose=1
)

nn_pred = nn_model.predict(X_test).flatten()
nn_mse = mean_squared_error(y_test, nn_pred)
nn_r2 = r2_score(y_test, nn_pred)
print(f"Neural Network          -> MSE: {nn_mse:.6f} | R² Score: {nn_r2:.4f}")

# ==========================================
# MODEL COMPARISON & VISUALIZATION
# ==========================================
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Optimized Random Forest', 'Neural Network'],
    'MSE': [lr_mse, rf_mse, nn_mse],
    'R2 Score': [lr_r2, rf_r2, nn_r2]
})

print("\n--- Final Performance Benchmark ---")
print(results_df.to_string(index=False))

# Bar Plot comparing R² Scores
plt.figure(figsize=(10, 5))
sns.barplot(data=results_df, x='Model', y='R2 Score', palette='viridis')
plt.title('Model Performance Comparison with K-Fold & GridSearch ($R^2$ Score)', fontsize=14, fontweight='bold')
plt.ylim(0, 1.05)
plt.ylabel('$R^2$ Score (Higher is Better)', fontsize=12)
plt.xlabel('Model Architecture', fontsize=12)
plt.tight_layout()
plt.show()

# Side-by-Side Actual vs Predicted Scatter Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

models_dict = {
    'Linear Regression': lr_pred,
    'Optimized Random Forest': rf_pred,
    'Neural Network': nn_pred
}

for i, (name, preds) in enumerate(models_dict.items()):
    sns.scatterplot(x=y_test, y=preds, alpha=0.3, color='teal', ax=axes[i])
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[i].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('Actual Probability', fontsize=10)
    if i == 0:
        axes[i].set_ylabel('Predicted Probability', fontsize=10)
    else:
        axes[i].set_ylabel('')

plt.suptitle('Actual vs. Predicted Flood Probabilities Across Tuned Models', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.show()
