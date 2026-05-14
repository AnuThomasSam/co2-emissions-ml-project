# %%
import numpy as np 
import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 
from sklearn.linear_model import Lasso 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import shap
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
def run_optuna(objective, n_trials=100):
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)
    return study
import scipy.stats as stats
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import time

kf = KFold(n_splits=5, shuffle=True, random_state=42)

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

import warnings
warnings.filterwarnings("ignore")

# %%
co2_vehicles= pd.read_csv('data/CO2 Emissions_Canada.csv')
df=co2_vehicles.copy()
df.head(5)

# %%
df.info()

# %%
df.shape

# %%
df.columns

# %%
df.isnull().values.any()

# %%
df.isnull().sum()

# %%
def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(df)

# %%
# Duplicated data

df_duplicated=df[df.duplicated() == True]
df_duplicated

# %%
sum(df_duplicated.groupby(by =['Make','Model','Vehicle Class','Engine Size(L)','Cylinders','Transmission','Fuel Type'])['Fuel Consumption City (L/100 km)'].nunique())

# %%
df = df.drop_duplicates().reset_index(drop=True)

# %%
df.describe().T

# %%
df.describe(include= 'object').T

# %% [markdown]
# Rename the Columns

# %%
df.rename(columns={ 'Make': 'make',
                    'Model': 'model',
                    'Vehicle Class': 'vehicle_class',
                    'Engine Size(L)': 'engine_size',
                    'Cylinders': 'cylinders',
                    'Transmission': 'transmission',
                    'Fuel Type': 'fuel_type',
                    'Fuel Consumption City (L/100 km)': 'fuel_cons_city',
                    'Fuel Consumption Hwy (L/100 km)': 'fuel_cons_hwy',
                    'Fuel Consumption Comb (L/100 km)': 'fuel_cons_comb',
                    'Fuel Consumption Comb (mpg)': 'fuel_cons_comb_mpg',
                    'CO2 Emissions(g/km)': 'co2'
                    }, inplace=True)

# %%
# Human‑readable label map

label_map = {
    'engine_size': 'Engine Displacement',
    'cylinders': 'Cylinder Count',
    'engine_per_cylinder': 'Engine Displacement per Cylinder',
    'engine_x_cylinders': 'Engine × Cylinders',
    'engine_size_sq': 'Engine Displacement²',
    'cylinders_sq': 'Cylinder Count²',
    'vehicle_class_COMPACT': 'Compact Car',
    'vehicle_class_FULL-SIZE': 'Full-Size Car',
    'vehicle_class_MID-SIZE': 'Mid-Size Car',
    'vehicle_class_SUBCOMPACT': 'Subcompact Car',
    'vehicle_class_TWO-SEATER': 'Two-Seater',
    'vehicle_class_STATION WAGON - SMALL': 'Station Wagon (Small)',
    'vehicle_class_STATION WAGON - MID-SIZE': 'Station Wagon (Mid-Size)',
    'vehicle_class_SUV - SMALL': 'SUV (Small)',
    'vehicle_class_SUV - STANDARD': 'SUV (Standard)',
    'vehicle_class_PICKUP TRUCK - SMALL': 'Pickup Truck (Small)',
    'vehicle_class_PICKUP TRUCK - STANDARD': 'Pickup Truck (Standard)',
    'vehicle_class_MINIVAN': 'Minivan',
    'vehicle_class_VAN - PASSENGER': 'Van (Passenger)',
    'vehicle_class_VAN - CARGO': 'Van (Cargo)',
    'transmission_A4': 'Auto (4‑speed)',
    'transmission_A5': 'Auto (5‑speed)',
    'transmission_A6': 'Auto (6‑speed)',
    'transmission_A7': 'Auto (7‑speed)',
    'transmission_A8': 'Auto (8‑speed)',
    'transmission_A9': 'Auto (9‑speed)',
    'transmission_AV': 'CVT',
    'transmission_AV6': 'CVT (6‑speed)',
    'transmission_M5': 'Manual (5‑speed)',
    'transmission_M6': 'Manual (6‑speed)',
    'transmission_AM6': 'Automated Manual (6‑speed)',
    'transmission_AV10': 'CVT (10‑speed)',
    'transmission_AV8': 'CVT (8‑speed)',
    'fuel_type_X': 'Regular Gasoline',
    'fuel_type_Z': 'Premium Gasoline',
    'fuel_type_D': 'Diesel',
    'fuel_type_E': 'Ethanol (E85)',
    'fuel_type_N': 'Natural Gas',
    'make_FORD': 'Ford',
    'make_TOYOTA': 'Toyota',
    'make_HONDA': 'Honda',
    'make_CHEVROLET': 'Chevrolet',
    'make_BMW': 'BMW',
    'make_MERCEDES-BENZ': 'Mercedes-Benz',
    'make_VOLKSWAGEN': 'Volkswagen',
    'make_HYUNDAI': 'Hyundai',
    'make_KIA': 'Kia',
    'make_NISSAN': 'Nissan',
    'make_MASERATI': 'Maserati',
    'make_GENESIS': 'Genesis',
    'make_LAND ROVER': 'Land Rover',
    'make_MITSUBISHI': 'Mitsubishi'
}


def pretty(col):
    return label_map.get(col, col)

# %%
# Feature Engineering

USE_ENGINEERED_FEATURES = True

if USE_ENGINEERED_FEATURES:
    df['engine_per_cylinder'] = df['engine_size'] / df['cylinders']
    df['engine_x_cylinders'] = df['engine_size'] * df['cylinders']
    df['engine_size_sq'] = df['engine_size'] ** 2
    df['cylinders_sq'] = df['cylinders'] ** 2

    numeric_features = [
        'engine_size', 'cylinders',
        'engine_per_cylinder', 'engine_x_cylinders',
        'engine_size_sq', 'cylinders_sq'
    ]
else:
    numeric_features = ['engine_size', 'cylinders']

# %%
df.columns 

# %% [markdown]
# Data Visualisation

# %% [markdown]
# Distribution of Categorical Features

# %%
def plot_bar_graphs(df, columns):
    for column in columns:
        plt.figure(figsize=(15, 5))
        ax = sns.countplot(x=column, data=df, order=df[column].value_counts().index)
        ax.bar_label(ax.containers[0],rotation=45)
        plt.xlabel(column, fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plt.title(f'Bar Graph of {column}', fontsize=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.show()
        
eda_cat_features = ['make','vehicle_class', 'engine_size', 'cylinders', 'transmission', 'fuel_type']

plot_bar_graphs(df, eda_cat_features)

# %%
# Model Feature

df.model.unique()

# %%
fig = plt.figure(figsize = (10,6))
ax = fig.add_axes([0,0,1,1])
counts = df.model.value_counts().sort_values(ascending=False).head(20)
counts.plot(kind = "bar")
plt.title('Top 20 Model')   
plt.xlabel('model') 
plt.ylabel('Number of Cars')
plt.xticks(rotation = 45)
ax.bar_label(ax.containers[0], labels=counts.values, fontsize=12);

# %% [markdown]
# Target Variable vs Categorical Features

# %%
def plot_bar_with_co2(df, columns):
    for column in columns:
        plt.figure(figsize=(15, 5))
        grouped_data = df.groupby(column)['co2'].mean().round(1).reset_index()
        grouped_data_sorted = grouped_data.sort_values(by='co2', ascending=False)
        ax = sns.barplot(x=column, y='co2', data=grouped_data_sorted, order=grouped_data_sorted[column])
        ax.bar_label(ax.containers[0],rotation=90)
        plt.xlabel(column, fontsize=18)
        plt.ylabel('Mean Co2 Emission', fontsize=15)
        plt.title(f'Mean Co2 Emission by {column}', fontsize=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.show()
        
plot_bar_with_co2(df, eda_cat_features)

# %% [markdown]
# Correlation Matrix

# %%
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('figures/correlation_heatmap.pdf', bbox_inches='tight')
plt.show()

# %% [markdown]
# PCA – DIMENSIONALITY REDUCTION

# %%
numeric_df = df.select_dtypes(include=['int64', 'float64']) 
# Scale numeric data 
scaler_pca = StandardScaler() 
scaled_data = scaler_pca.fit_transform(numeric_df) 
# Apply PCA to 2 components 
pca = PCA(n_components=2) 
pca_result = pca.fit_transform(scaled_data) 
# Add PCA components to dataframe 
df_pca = df.copy() 
df_pca['PCA1'] = pca_result[:, 0] 
df_pca['PCA2'] = pca_result[:, 1] 

plt.figure(figsize=(10,6)) 
sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='fuel_type', palette='tab10') 
plt.title('PCA Visualization (Colored by Fuel Type)') 
plt.savefig('figures/pca_scatter.pdf', bbox_inches='tight')
plt.show() 

print("PCA Explained Variance Ratio:", pca.explained_variance_ratio_)

# %% [markdown]
# Distribution of Numerical Features

# %%
numerical_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(15, 10))

num_vars = len(numerical_df.columns)

for i, var in enumerate(numerical_df.columns, 1):
    plt.subplot((num_vars // 3) + 1, 3, i)
    sns.histplot(data=df, x=var, kde=True)
    plt.title(f'Distribution of {var}')
    
plt.tight_layout()
plt.savefig('figures/numerical_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Consumption of Highway and City

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="fuel_cons_hwy", kde=True, label = "Fuel Consumption in Highway",color = "blue")
sns.histplot(data=df, x="fuel_cons_city", kde=True, label = "Fuel Consumption in City")
plt.xlabel('Consumption (L/100 km)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title(f'Histogram of Highway and City', fontsize=10)
plt.legend()
plt.show()

# %% [markdown]
# Target Variable vs Numerical Features

# %%
# Target vs Fuel Consumption Combined (city+hwy)
# Hue: Vehicle Class

plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='fuel_cons_comb',y='co2',hue='vehicle_class')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', title='Vehicle Class')
plt.tight_layout()

plt.title('Fuel Consumption Comb vs CO2 Emissions by Vehicle Class')
plt.xlabel('Fuel Consumption Comb (L/100 km)')
plt.ylabel('CO2 Emissions (g/km)')
plt.savefig('figures/fuel_consumption_vs_co2.pdf', bbox_inches='tight')
plt.show()

# %%
# Target vs Fuel Consumption in Miles Per Gallon (mpg)
# Hue: Fuel Type 

plt.figure(figsize=(8,5))
sns.scatterplot(data=df,x='fuel_cons_comb_mpg',y='co2',hue='fuel_type')

plt.title('Fuel Consumption mpg(efficiency) vs CO2 Emissions by Fuel_Type')
plt.xlabel('Fuel Consumption in Miles Per Gallon (L/100 km)')
plt.ylabel('CO2 Emissions (g/km)')

# %%
# Target vs Fuel Consumption in Miles Per Gallon
# Hue: Vehicle Class

plt.figure(figsize=(10,6))
sns.scatterplot(data=df, x='fuel_cons_comb_mpg',y='co2',hue='vehicle_class')
plt.legend(bbox_to_anchor=(1, 1), loc='upper left', title='Vehicle Class')
plt.tight_layout()

plt.title('Fuel Consumption mpg(efficiency) vs CO2 Emissions by Vehicle_Class')
plt.xlabel('Fuel Consumption in Miles Per Gallon (L/100 km)')
plt.ylabel('CO2 Emissions (g/km)')

# %%
# Target vs Engine Size

plt.figure(figsize=(8, 6))
sns.scatterplot(x='engine_size', y='co2', data=df, hue='fuel_type')
plt.title('Engine Size vs CO2 Emissions')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.savefig('figures/engine_size_vs_co2.pdf', bbox_inches='tight')
plt.show()

# %%
# Target vs Cylinders Number

plt.figure(figsize=(8,5))
sns.scatterplot(data=df,x='cylinders',y='co2')

plt.title('Cylinders vs CO2 Emissions')
plt.xlabel('Number of cylinders')
plt.ylabel('CO2 Emissions (g/km)')
plt.show()

# %% [markdown]
# Correlations of Numerical Features

# %%
correlation_matrix = df.corr(numeric_only=True)

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

# %%
# Pairplot for the dataframe

sns.pairplot(df,
             kind="reg",
             diag_kind="kde", 
             plot_kws={"line_kws": {"color": "red"}}
            )

# %% [markdown]
# Skewness

# %%
# Calculate skewness for numeric features

# A skewness value greater than 1 indicates positive skewness,
# a skewness value less than -1 indicates negative skewness,
# and a skewness value close to zero indicates a relatively symmetric distribution.

num_cols= df.select_dtypes('number').columns

skew_limit = 0.75               # define a limit above which we will log transform
skew_vals = df[num_cols].skew()


# Showing the skewed columns
skew_cols = (skew_vals
             .sort_values(ascending=False)
             .to_frame()
             .rename(columns={0:'Skew'})
             .query('abs(Skew) > {}'.format(skew_limit)))
skew_cols

# %% [markdown]
# MACHINE LEARNING MODELS

# %%
categorical_features = ['make', 'vehicle_class', 'transmission', 'fuel_type']

X = df[numeric_features + categorical_features]
y = df['co2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# %%
# Fit preprocessing on training data
preprocess.fit(X_train)

# Count numeric features
num_feature_count = len(numeric_features)

# Count encoded categorical features
encoded_cat_features = preprocess.named_transformers_['cat'] \
    .get_feature_names_out(categorical_features)

cat_feature_count = len(encoded_cat_features)

# Total features after preprocessing
total_features = num_feature_count + cat_feature_count

print("Numeric Features:", num_feature_count)
print("Encoded Categorical Features:", cat_feature_count)
print("Total Features After One-Hot Encoding:", total_features)

# %% [markdown]
# Evaluation Function

# %%
def evaluate_model(model_name, model, X_train, X_test, y_train, y_test, cat_features=None):
    # Define KFold ONCE for all models
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Training time
    train_start = time.time()

    # Special handling for CatBoost
    if model_name == "CatBoost":
        model.fit(X_train, y_train, cat_features=cat_features)
    else:
        model.fit(X_train, y_train)

    training_time = time.time() - train_start

    # Prediction time
    pred_start = time.time()

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    prediction_time = time.time() - pred_start

    # Train metrics
    r2_train = r2_score(y_train, y_train_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    rmse_train = mse_train ** 0.5
    mae_train = mean_absolute_error(y_train, y_train_pred)

    # Test metrics
    r2_test = r2_score(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = mse_test ** 0.5
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # 5-fold CV on TRAINING DATA ONLY (no leakage)
    # Custom CV for CatBoost
    if model_name == "CatBoost":
        cv_scores = []

        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            temp_model = model.__class__(**model.get_params())
            temp_model.fit(X_tr, y_tr, cat_features=cat_features)

            preds = temp_model.predict(X_val)
            cv_scores.append(r2_score(y_val, preds))

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

    else:
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()

    print(f"\n{model_name}")
    print("-" * len(model_name))

    print("\nTRAIN METRICS")
    print(f"R²:   {r2_train:.4f}")
    print(f"MSE:  {mse_train:.4f}")
    print(f"RMSE: {rmse_train:.4f}")
    print(f"MAE:  {mae_train:.4f}")

    print("\nTEST METRICS")
    print(f"R²:   {r2_test:.4f}")
    print(f"MSE:  {mse_test:.4f}")
    print(f"RMSE: {rmse_test:.4f}")
    print(f"MAE:  {mae_test:.4f}")

    print("\n5-FOLD CROSS VALIDATION (Train Only)")
    print(f"CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

    print("\nCOMPUTATIONAL COST")
    print(f"Training Time:  {training_time:.4f} seconds")
    print(f"Prediction Time:{prediction_time:.4f} seconds")

    return {
        "Model": model_name,
        "Train_R2": r2_train,
        "Test_R2": r2_test,
        "Train_MSE": mse_train,
        "Test_MSE": mse_test,
        "Train_RMSE": rmse_train,
        "Test_RMSE": rmse_test,
        "Train_MAE": mae_train,
        "Test_MAE": mae_test,
        "CV_R2_Mean": cv_mean,
        "CV_R2_Std": cv_std,
        "Training_Time_Seconds": training_time,
        "Prediction_Time_Seconds": prediction_time
    }


# %% [markdown]
# Model Implementations

# %%
# Decision Tree
dt_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', DecisionTreeRegressor(max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42))])

dt_pipeline.fit(X_train, y_train)
dt_results = evaluate_model("Decision Tree", dt_pipeline, X_train, X_test, y_train, y_test)

# %%
# Random Forest
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
    }

    model = RandomForestRegressor(**params, random_state=42)

    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
    return scores.mean()

# Run Optuna
rf_study = run_optuna(objective_rf)
print("Best Params:", rf_study.best_params)

# Build tuned model
best_rf_model = RandomForestRegressor(
    **rf_study.best_params,
    random_state=42
)

best_rf_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', best_rf_model)
])

# Fit tuned model
best_rf_pipeline.fit(X_train, y_train)

# Evaluate tuned model
rf_results = evaluate_model(
    "Random Forest (Tuned)",
    best_rf_pipeline,
    X_train,
    X_test,
    y_train,
    y_test
)

# %%
# Gradient Boosting
gbr_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', GradientBoostingRegressor(n_estimators=500,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.8,
                random_state=42))])

gbr_pipeline.fit(X_train, y_train)
gbr_results = evaluate_model("Gradient Boosting", gbr_pipeline, X_train, X_test, y_train, y_test)

# %%
#Lasso Regression
lasso_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', Lasso(alpha=0.01, max_iter=10000))
])

lasso_pipeline.fit(X_train, y_train)
lasso_results = evaluate_model("Lasso Regression", lasso_pipeline, X_train, X_test, y_train, y_test)

# %%
# KNN Optuna Tuning

def objective_knn(trial):
    model = KNeighborsRegressor(
        n_neighbors=trial.suggest_int('n_neighbors', 3, 20),
        weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
        p=trial.suggest_int('p', 1, 2)
    )

    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
    return scores.mean()


# Run Optuna
knn_study = run_optuna(objective_knn)
print("Best Params:", knn_study.best_params)

# Build Tuned KNN Model

best_knn_model = KNeighborsRegressor(
    **knn_study.best_params
)

best_knn_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', best_knn_model)
])

# Fit tuned model
best_knn_pipeline.fit(X_train, y_train)

# Evaluate Tuned KNN Model

knn_results = evaluate_model(
    "KNN Regression (Tuned)",
    best_knn_pipeline,
    X_train,
    X_test,
    y_train,
    y_test
)


# %%
# SVR Optuna Tuning

def objective_svr(trial):
    model = SVR(
        kernel='rbf',
        C=trial.suggest_float('C', 1, 200, log=True),
        epsilon=trial.suggest_float('epsilon', 0.01, 1.0),
        gamma=trial.suggest_categorical('gamma', ['scale', 'auto'])
    )

    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=kf, scoring='r2')
    return scores.mean()


# Run Optuna
svr_study = run_optuna(objective_svr)
print("Best Params:", svr_study.best_params)

# Build Tuned SVR Model

best_svr_model = SVR(
    **svr_study.best_params
)

best_svr_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', best_svr_model)
])

# Fit tuned model
best_svr_pipeline.fit(X_train, y_train)

# Evaluate Tuned SVR Model

svr_results = evaluate_model(
    "Support Vector Regression (Tuned)",
    best_svr_pipeline,
    X_train,
    X_test,
    y_train,
    y_test
)


# %%
# XGBoost Optuna Tuning

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5)
    }

    model = XGBRegressor(**params, random_state=42)

    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])

    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
    return scores.mean()


# Run Optuna
xgb_study = run_optuna(objective_xgb)
print("Best Params:", xgb_study.best_params)

# Build Tuned XGBoost Model

best_xgb_model = XGBRegressor(
    **xgb_study.best_params,
    random_state=42
)

best_xgb_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', best_xgb_model)
])

# Fit tuned model
best_xgb_pipeline.fit(X_train, y_train)

# Evaluate Tuned XGBoost Model

xgb_results = evaluate_model(
    "XGBoost (Tuned)",
    best_xgb_pipeline,
    X_train,
    X_test,
    y_train,
    y_test
)


# %%
# LightGBM: Data Preparation

# Copy data
X_train_lgb = X_train.copy()
X_test_lgb = X_test.copy()

# Convert categorical features to category dtype
for col in categorical_features:
    X_train_lgb[col] = X_train_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

# Optuna Objective Function

def objective_lgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", -1, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }

    model = LGBMRegressor(**params, random_state=42, verbose=-1)

    pipe = Pipeline([
        ('prep', preprocess),
        ('model', model)
    ])

    scores = cross_val_score(
        pipe,
        X_train_lgb,
        y_train,
        cv=kf,
        scoring="r2"
    )

    return scores.mean()

# Run Optuna Tuning

study_lgb = run_optuna(objective_lgb)

print("Best Params:", study_lgb.best_params)
print("Best CV R2:", study_lgb.best_value)

# Train Final LightGBM Model

best_lgb_model = LGBMRegressor(
    **study_lgb.best_params,
    random_state=42,
    verbose=-1
)

best_lgb_pipeline = Pipeline([
    ('prep', preprocess),
    ('model', best_lgb_model)
])

best_lgb_pipeline.fit(
    X_train_lgb,
    y_train
)

# Store Model for SHAP

shap_models = {}  
shap_models["LightGBM (Tuned)"] = best_lgb_pipeline

# Evaluate Model

lgb_results = evaluate_model(
    "LightGBM (Tuned)",
    best_lgb_pipeline,
    X_train_lgb,
    X_test_lgb,
    y_train,
    y_test
)

# %%
# Prepare CatBoost data

catboost_cat_features = ['make', 'vehicle_class', 'transmission', 'fuel_type']

X_cat = df[numeric_features + catboost_cat_features]

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X_cat, y, test_size=0.2, random_state=42
)

# Get categorical indices
cat_feature_indices = [X_cat.columns.get_loc(col) for col in catboost_cat_features]

# Optuna Hyperparameter Tuning

def objective_cat(trial):
    params = {
        'iterations': trial.suggest_int('iterations', 200, 800),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
    }

    model = CatBoostRegressor(
        **params,
        cat_features=catboost_cat_features,
        verbose=0,
        random_state=42
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kf.split(X_train_cat):
        X_tr = X_train_cat.iloc[train_idx]
        X_val = X_train_cat.iloc[val_idx]
        y_tr = y_train_cat.iloc[train_idx]
        y_val = y_train_cat.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        scores.append(r2_score(y_val, preds))

    return np.mean(scores)


# Run Optuna
study_cat = run_optuna(objective_cat)

print("Best Params:", study_cat.best_params)
print("Best CV R2:", study_cat.best_value)

# Train Tuned CatBoost Model

cat_model = CatBoostRegressor(
    **study_cat.best_params,
    cat_features=catboost_cat_features,
    verbose=0,
    random_state=42
)

cat_model.fit(X_train_cat, y_train_cat)

# Evaluate Tuned Model

cat_results = evaluate_model(
    "CatBoost",
    cat_model,
    X_train_cat,
    X_test_cat,
    y_train_cat,
    y_test_cat
)

# %%
summary_df = pd.DataFrame([
    dt_results,
    rf_results,
    gbr_results,
    lasso_results,
    knn_results,
    svr_results,
    xgb_results,
    lgb_results,
    cat_results
])

# Drop CV columns
# summary_df_clean = summary_df.drop(columns=['CV_R2_Mean', 'CV_R2_Std'])

# Sort by Test_R2 (best model first)
summary_df_clean = summary_df.sort_values(by='Test_R2', ascending=False)

# Round for cleaner presentation
summary_df_clean = summary_df_clean.round(4)

# Reset index
summary_df_clean.reset_index(drop=True, inplace=True)

# summary_df_clean.to_csv("results/results_before.csv", index=False)
summary_df_clean.to_csv("results/results_after.csv", index=False)

summary_df_clean

# %%
# Computational cost comparison

cost_comparison = summary_df_clean[
    [
        "Model",
        "Test_R2",
        "Test_MAE",
        "Training_Time_Seconds",
        "Prediction_Time_Seconds"
    ]
]

print(cost_comparison)

# %%
before = pd.read_csv("results/results_before.csv")
after = pd.read_csv("results/results_after.csv")

comparison = before.merge(after, on="Model", suffixes=("_Before", "_After"))
comparison["Improvement"] = comparison["Test_R2_After"] - comparison["Test_R2_Before"]

print(comparison.sort_values(by='Improvement', ascending=False))

# %% [markdown]
# SHAP IMPLEMENTATION

# %%
#Get transformed data + feature names

def get_transformed_data(pipeline, X):
    preprocess = pipeline.named_steps['prep']
    
    X_transformed = preprocess.transform(X)

    # Convert sparse → dense
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Force numeric dtype
    X_transformed = np.asarray(X_transformed, dtype=np.float64)

    # Get feature names
    num_features = numeric_features
    cat_features_ohe = preprocess.named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    feature_names = list(num_features) + list(cat_features_ohe)

    return X_transformed, feature_names

# SHAP importance

def get_shap_importance(shap_values, feature_names):
    return pd.DataFrame({
        "Feature": feature_names,
        "SHAP Importance": np.abs(shap_values).mean(axis=0)
    }).sort_values(by="SHAP Importance", ascending=False)


# Tree model importance

def get_model_importance(model, feature_names):
    return pd.DataFrame({
        "Feature": feature_names,
        "Model Importance": model.feature_importances_
    }).sort_values(by="Model Importance", ascending=False)

# %%
# SHAP for Tree-Based Models

tree_models = {
    "Decision Tree": dt_pipeline,
    "Random Forest (Tuned)": best_rf_pipeline,
    "Gradient Boosting": gbr_pipeline,
    "XGBoost (Tuned)": best_xgb_pipeline,
    "LightGBM (Tuned)": best_lgb_pipeline
}

shap_results = {}

for name, pipeline in tree_models.items():
    print(f"\nProcessing SHAP for {name}...")

    X_test_transformed, feature_names = get_transformed_data(pipeline, X_test)
    model = pipeline.named_steps.get('model', None)

    if model is None:
        raise ValueError(f"{name}: 'model' step not found in pipeline")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    shap_results[name] = {
        "shap_values": shap_values,
        "features": feature_names,
        "X": X_test_transformed,
        "model": model
    }

# %%
# SHAP for KNN & SVR (KernelExplainer)

kernel_models = {
    "KNN Regression (Tuned)": best_knn_pipeline,
    "Support Vector Regression (Tuned)": best_svr_pipeline
}

for name, pipeline in kernel_models.items():
    print(f"\nProcessing SHAP for {name} (KernelExplainer)...")

    # Use RAW data
    background = shap.sample(X_train, 100)

    explainer = shap.KernelExplainer(
        lambda x: pipeline.predict(pd.DataFrame(x, columns=X_train.columns)),
        background
    )

    shap_values = explainer.shap_values(X_test.iloc[:50])

    shap_results[name] = {
        "shap_values": shap_values,
        "features": X_train.columns.tolist(),
        "X": X_test.iloc[:50],
        "model": pipeline.named_steps['model']
    }

# %%
# SHAP for CatBoost

print("\nProcessing SHAP for CatBoost...")

explainer = shap.TreeExplainer(cat_model)
shap_values = explainer.shap_values(X_test_cat)

shap_results["CatBoost"] = {
    "shap_values": shap_values,
    "features": X_test_cat.columns.tolist(),
    "X": X_test_cat,
    "model": cat_model
}

# %% [markdown]
# Best-performing model

# %%
best_model_name = summary_df_clean.iloc[0]['Model']
print("Best Model:", best_model_name)

best_shap = shap_results[best_model_name]

shap_values = best_shap["shap_values"]
X_data = best_shap["X"]
feature_names = best_shap["features"]
model = best_shap.get("model", None)

if model is None:
    raise ValueError("No model stored in SHAP results")

if not hasattr(model, "feature_importances_") and not hasattr(model, "get_booster"):
    print("Skipping non-tree model for waterfall")
else:
    explainer = shap.TreeExplainer(model)
    shap_values_single = explainer(X_data)
    shap_values_single.feature_names = feature_names
    shap.plots.waterfall(shap_values_single[0])

# %%
# Human-readable feature names
feature_labels = [pretty(col) for col in feature_names]

# Summary Plot (Beeswarm)

plt.figure(figsize=(12, 7))
shap.summary_plot(
    shap_values,
    X_test_transformed, 
    feature_names=feature_labels,
    show=False
)
plt.tight_layout()
plt.savefig("figures/shap_summary_plot.pdf", bbox_inches="tight", dpi=300)
plt.show()

# Bar Plot

shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_labels, plot_type="bar")
plt.savefig("figures/shap_bar_plot.pdf", bbox_inches="tight")
plt.show()

# Dependence Plots (Top 3 Features)

X_plot = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

mean_abs = np.abs(shap_values).mean(axis=0)

# Grouped SHAP Importance
grouped_importance = {}

for feature, importance in zip(feature_names, mean_abs):

    # Group one-hot encoded features

    if feature.startswith("make_"):
        group = "Make"

    elif feature.startswith("vehicle_class_"):
        group = "Vehicle Class"

    elif feature.startswith("transmission_"):
        group = "Transmission"

    elif feature.startswith("fuel_type_"):
        group = "Fuel Type"

    # Numerical features stay separate

    else:
        group = pretty(feature)

    # Sum SHAP values within groups

    grouped_importance[group] = (
        grouped_importance.get(group, 0) + importance
    )

# Create grouped SHAP dataframe

grouped_shap_df = pd.DataFrame({
    "Feature Group": grouped_importance.keys(),
    "SHAP Importance": grouped_importance.values()
})

grouped_shap_df = grouped_shap_df.sort_values(
    by="SHAP Importance",
    ascending=False
)
print(grouped_shap_df)

# Grouped SHAP Bar Plot

plt.figure(figsize=(10, 6))

sns.barplot(
    data=grouped_shap_df,
    x="SHAP Importance",
    y="Feature Group",
    palette="viridis"
)

plt.title("Grouped SHAP Feature Importance")
plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Feature Group")

plt.tight_layout()
plt.savefig("figures/grouped_shap_importance.pdf", bbox_inches="tight", dpi=300)
plt.show()

top_idx = np.argsort(mean_abs)[-3:][::-1]

top_features = [feature_names[i] for i in top_idx]
top_feature_labels = [feature_labels[i] for i in top_idx]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (feature, label) in enumerate(zip(top_features, top_feature_labels)):

    shap.dependence_plot(
        feature,
        shap_values,
        X_plot,             
        ax=axes[i],
        show=False
    )
    axes[i].set_title(f"({chr(97+i)}) {label}")
    axes[i].set_xlabel(label)
    axes[i].set_ylabel("SHAP value for " + label)

plt.tight_layout()
plt.savefig("figures/shap_dependence_plots.pdf", bbox_inches="tight", dpi=300)
plt.show()

# Waterfall Plot

i = 0  # first instance

explainer = shap.TreeExplainer(model)
shap_values_single = explainer(X_test_transformed)
shap_values_single.feature_names = feature_labels

plt.figure(figsize=(10, 6))

shap.plots.waterfall(
    shap_values_single[i],
    max_display=10,
    show=False
)
plt.tight_layout()
plt.savefig("figures/shap_waterfall_plot.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
# Feature Importance Comparison Table

importance_rows = []

for model_name, result in shap_results.items():
    shap_vals = result["shap_values"]
    features = result["features"]
    model = result["model"]

    # SHAP importance (works for ALL models)
    shap_importance = np.abs(shap_vals).mean(axis=0)

    # Try to get model-based importance (tree models only)
    if hasattr(model, "feature_importances_"):
        model_importance = model.feature_importances_
    elif model_name == "CatBoost":
        model_importance = model.get_feature_importance()
    else:
        model_importance = None  # KNN, SVR, Lasso

    # Store row
    importance_rows.append({
        "Model": model_name,
        "Top SHAP Feature": features[np.argmax(shap_importance)],
        "Top SHAP Importance": shap_importance.max(),
        "Has Model Importance": model_importance is not None,
        "Top Model Feature": features[np.argmax(model_importance)] if model_importance is not None else None,
        "Top Model Importance": model_importance.max() if model_importance is not None else None
    })

importance_df = pd.DataFrame(importance_rows)
importance_df

# %% [markdown]
# Residual analysis section

# %%
# Best Model (XGBoost (Tuned))

best_model = best_xgb_pipeline

y_pred = best_model.predict(X_test)

residuals = y_test - y_pred

# Predicted vs Actual (with y = x line)

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred, alpha=0.5, color='blue')

# identity line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

plt.xlabel("Actual CO2 Emissions")
plt.ylabel("Predicted CO2 Emissions")
plt.title("XGBoost: Predicted vs Actual CO2 Emissions")

plt.grid(True)
plt.tight_layout()
plt.savefig("figures/xgboost_predicted_vs_actual.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
# Residual histogram + normal curve

plt.figure(figsize=(8,6))

sns.histplot(residuals, kde=True, bins=40, color='purple', stat='density')

# Fit normal distribution
mu, std = stats.norm.fit(residuals)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)

plt.plot(x, p, 'r', linewidth=2)

plt.title("Residual Distribution (XGBoost)")
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Density")

plt.tight_layout()
plt.savefig("figures/xgboost_residuals.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
# Residuals vs Predicted (heteroscedasticity check)

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residuals, alpha=0.5)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted CO2")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values (Heteroscedasticity Check)")

plt.grid(True)
plt.tight_layout()
plt.savefig("figures/xgboost_residuals_vs_predicted.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%
# Find which categories have biggest errors

test_results = X_test.copy()
test_results['actual'] = y_test.values
test_results['predicted'] = y_pred
test_results['residual'] = residuals
test_results['abs_error'] = np.abs(residuals)

# %%
# Biggest error by Vehicle Class

vehicle_errors = test_results.groupby('vehicle_class')['abs_error'].mean().sort_values(ascending=False)

print(vehicle_errors.head(10))

# %%
# Biggest error by Fuel Type

fuel_errors = test_results.groupby('fuel_type')['abs_error'].mean().sort_values(ascending=False)

print(fuel_errors)

# %%
plt.figure(figsize=(10,5))
sns.barplot(x=vehicle_errors.values, y=vehicle_errors.index)
plt.title("Mean Absolute Error by Vehicle Class")
plt.xlabel("MAE")
plt.ylabel("Vehicle Class")
plt.show()

# %%
# CV Results

# Models (exclude CatBoost here)
models = {
    "Decision Tree": dt_pipeline,
    "Random Forest": best_rf_pipeline,
    "Gradient Boosting": gbr_pipeline,
    "Lasso": lasso_pipeline,
    "KNN": best_knn_pipeline,
    "SVR": best_svr_pipeline,
    "XGBoost": best_xgb_pipeline,
    "LightGBM": best_lgb_pipeline
}

cv_results = pd.DataFrame()

# Standard sklearn models

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    cv_results[name] = scores

# CatBoost (manual CV)

cat_scores = []

for train_idx, val_idx in kf.split(X_train_cat):
    X_tr = X_train_cat.iloc[train_idx]
    X_val = X_train_cat.iloc[val_idx]
    y_tr = y_train_cat.iloc[train_idx]
    y_val = y_train_cat.iloc[val_idx]

    temp_model = CatBoostRegressor(
        **study_cat.best_params,
        cat_features=catboost_cat_features,
        verbose=0,
        random_state=42
    )

    temp_model.fit(X_tr, y_tr)
    preds = temp_model.predict(X_val)

    cat_scores.append(r2_score(y_val, preds))

cv_results["CatBoost"] = cat_scores

print(cv_results)

# %%
# Run Friedman Test

stat, p = friedmanchisquare(*[cv_results[col] for col in cv_results.columns])

print("Friedman Test Statistic:", stat)
print("p-value:", p)

# %%
# Nemenyi Post-hoc Test

nemenyi = sp.posthoc_nemenyi_friedman(cv_results)

print(nemenyi)

# %%
ranks = cv_results.rank(axis=1, ascending=False)
avg_ranks = ranks.mean()

print(avg_ranks)

# Plot Critical Difference Diagram

plt.figure(figsize=(10, 5))

sp.critical_difference_diagram(avg_ranks, nemenyi)

plt.title("Nemenyi Critical Difference Diagram")
plt.tight_layout()
plt.savefig("figures/nemenyi_critical_difference.pdf", bbox_inches="tight", dpi=300)
plt.show()

# %%



