#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def load_data(data_dir):
    """Loads training and testing datasets from a given directory."""
    data_files = [f for f in os.listdir(data_dir) if f.endswith('csv')]
    data_train_name = [f for f in data_files if 'TRAIN' in f]
    data_test_name = [f for f in data_files if 'TEST' in f]

    if not data_train_name or not data_test_name:
        raise FileNotFoundError("TRAIN or TEST dataset not found in directory!")

    data_train = pd.read_csv(os.path.join(data_dir, data_train_name[0]))
    data_test = pd.read_csv(os.path.join(data_dir, data_test_name[0]))

    return data_train, data_test


def preprocess_data(data_train, data_test, drop_cols=None):
    """Preprocess the datasets: removes unwanted columns and handles NaNs."""
    if drop_cols is None:
        drop_cols = ['highavse', 'lowavse', 'truedcr', 'lq', 'id', 'tdrift50', 'tdrift10']

    data_train_filtered = data_train.drop(columns=drop_cols).dropna()
    data_test_filtered = data_test.drop(columns=drop_cols).dropna()

    X_train = data_train_filtered.drop(columns=['energylabel'])
    y_train = data_train_filtered['energylabel']

    X_test = data_test_filtered.drop(columns=['energylabel'])
    y_test = data_test_filtered['energylabel']

    return X_train, X_test, y_train, y_test


def standardize_data(X_train, X_test):
    """Standardizes the feature data."""
    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)
    X_test_standardized = scaler.transform(X_test)
    return X_train_standardized, X_test_standardized, scaler


def train_svm_model(X_train, y_train):
    """Trains an SVM Regressor with hyperparameter tuning."""
    param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.5], 
        'loss': ['epsilon_insensitive'],
        'dual': [True]  
    }

    svr = LinearSVR(max_iter=1000)

    # Perform GridSearchCV with 3-fold Crossvalidation (to minimize runtime)
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,  
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_svr_model = grid_search.best_estimator_

    return best_svr_model, best_params


def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(abs(y_pred - y_test))

    return mse, mae, r2, y_test, y_pred


def plot_results(y_test, y_pred, best_params, save_dir="plots/"):
    """Generates and saves plots for evaluation."""
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    # Scatter plot of predicted vs actual values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree reference line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values (SVM)")
    plt.savefig(os.path.join(save_dir, 'svm_predicted_vs_actual.png'), dpi=300)
    plt.close()

    # Distribution Plot: Predicted vs Actual Test Values
    plt.figure(figsize=(8, 6))
    sns.kdeplot(y_test, label="Actual Test Values", color="blue", fill=True)
    sns.kdeplot(y_pred, label="Predicted Values", color="red", linestyle="--", fill=True)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Distribution of Predicted vs Actual Test Values")
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'svm_predicted_vs_test_distribution.png'), dpi=300)
    plt.close()


def main():
    """Main function to run the SVM Model."""
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "Data/")

    print("Loading data...")
    data_train, data_test = load_data(data_dir)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data_train, data_test)

    print("Standardizing data...")
    X_train_standardized, X_test_standardized, scaler = standardize_data(X_train, X_test)

    print("Training SVM model...")
    svm_model, best_params = train_svm_model(X_train_standardized, y_train)

    print(f"Best Hyperparameters: {best_params}")

    print("Evaluating model...")
    mse, mae, r2, y_test, y_pred = evaluate_model(svm_model, X_test_standardized, y_test)

    print("Generating plots...")
    plot_results(y_test, y_pred, best_params)

    print('------------------------------------------------------------------------------------------')
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    print('------------------------------------------------------------------------------------------')
    print("Although this SVM model has a relatively higher MSE, it serves as a reference for comparison.")
    print("Run `DeepLearning_NN.py` to acquire the most accurate result on the NPML dataset instead.")
    print("Plots have been saved for reference.")
    print('------------------------------------------------------------------------------------------')


if __name__ == "__main__":
    main()
