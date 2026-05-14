# CO₂ Emissions Prediction Using Machine Learning

This project builds and compares multiple machine learning models to predict vehicle CO₂ emissions based on engine characteristics, fuel consumption, and vehicle attributes. It includes hyperparameter tuning, model evaluation, explainability using SHAP, and statistical comparison of models.

## Dataset

The dataset used is the **CO₂ Emissions for Vehicles (Canada)** dataset, which contains:

- Engine size
- Cylinders
- Fuel type
- Vehicle class
- Fuel consumption (city, highway, combined)
- CO₂ emissions (target variable)

## 🎯 Objective

To predict **CO₂ emissions (g/km)** using vehicle features and:

- Compare multiple ML models
- Optimize hyperparameters using Optuna
- Evaluate model performance using R², MAE, RMSE
- Interpret models using SHAP
- Perform statistical comparison of models

## Workflow

### 1. Data Preprocessing
- Removed duplicates
- Renamed columns for readability
- Feature engineering:
  - Engine per cylinder
  - Engine × cylinders
  - Polynomial features

### 2. Exploratory Data Analysis (EDA)
- Distribution analysis
- Correlation heatmaps
- PCA visualization
- Fuel consumption vs emissions trends

### 3. Machine Learning Models
The following models were trained and compared:

- Decision Tree
- Random Forest (Optuna-tuned)
- Gradient Boosting
- Lasso Regression
- K-Nearest Neighbors (Optuna-tuned)
- Support Vector Regression (Optuna-tuned)
- XGBoost (Optuna-tuned)
- LightGBM (Optuna-tuned)
- CatBoost (Optuna-tuned)

## Hyperparameter Optimization

- Tool: **Optuna**
- Method: Tree-structured Parzen Estimator (TPE)
- Metric: 5-fold cross-validated R² score

## Model Evaluation Metrics

Each model is evaluated using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- 5-Fold Cross Validation

## Model Explainability

Model interpretability is performed using:

- **SHAP (SHapley Additive Explanations)**
  - Feature importance ranking
  - Summary plots
  - Dependence plots
  - Waterfall plots

## Statistical Analysis

To compare models statistically:

- **Friedman Test** → checks if models differ significantly
- **Nemenyi Post-hoc Test** → identifies best-performing models
- Critical Difference diagram visualizes ranking differences

## Best Performing Model

```markdown
## Feature Engineering Comparison

A toggle-based system was implemented:

- `USE_ENGINEERED_FEATURES = False` → without engineered features
- `USE_ENGINEERED_FEATURES = True` → with engineered features

Each run is executed separately and saved, ensuring a fair comparison between feature sets.

The best model is selected based on test R² score.

Example:
