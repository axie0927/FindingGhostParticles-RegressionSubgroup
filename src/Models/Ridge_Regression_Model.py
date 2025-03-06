#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
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


def train_ridge_model(X_train, y_train):
    """Trains a Ridge Regression model using GridSearchCV."""
    alpha_range = np.logspace(-5, 2.5, 100)
    param_grid = {"alpha": alpha_range}

    ridge = Ridge()
    grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_alpha = grid_search.best_params_['alpha']
    ridge_reg = Ridge(alpha=best_alpha)
    ridge_reg.fit(X_train, y_train)

    return ridge_reg, best_alpha


def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set."""
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = np.mean(abs(y_pred - y_test))

    return mse, mae, r2, y_test, y_pred


def plot_results(y_test, y_pred, save_dir="plots/"):
    """Generates and saves plots for evaluation."""
    os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

    # Scatter plot of predicted vs actual values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree reference line
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values")
    plt.savefig(os.path.join(save_dir, 'ridge_predicted_vs_Actual.png'), dpi=300)
    plt.close()


def plot_alpha_vs_mse(X_train, X_test, y_train, y_test, save_dir="plots/"):
    """Plots training and testing MSE for different alpha values in Ridge Regression."""
    os.makedirs(save_dir, exist_ok=True)

    alpha_range = np.logspace(-5, 2.5, 100)
    train_errors, test_errors = [], []

    for alpha in alpha_range:
        ridge_reg = Ridge(alpha=alpha)
        ridge_reg.fit(X_train, y_train)

        train_pred = ridge_reg.predict(X_train)
        test_pred = ridge_reg.predict(X_test)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        train_errors.append(train_mse)
        test_errors.append(test_mse)

    plt.figure(figsize=(8, 6))
    plt.semilogx(alpha_range, train_errors, label='Training Error', color='blue')
    plt.semilogx(alpha_range, test_errors, label='Testing Error', color='red')
    plt.xlabel('Regularization Parameter (alpha)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training and Testing Error vs Alpha')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'ridge_alpha_vs_mse.png'), dpi=300)
    plt.close()


def main():
    """Main function to run the Ridge Regression Model."""
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "Data/")

    print("Loading data...")
    data_train, data_test = load_data(data_dir)

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(data_train, data_test)

    print("Standardizing data...")
    X_train_standardized, X_test_standardized, scaler = standardize_data(X_train, X_test)

    print("Training Ridge model...")
    ridge_model, best_alpha = train_ridge_model(X_train_standardized, y_train)

    print(f"Best Regularization Parameter (Alpha): {best_alpha:.6f}")

    print("Evaluating model...")
    mse, mae, r2, y_test, y_pred = evaluate_model(ridge_model, X_test_standardized, y_test)

    print("Generating plots...")
    plot_results(y_test, y_pred)
    plot_alpha_vs_mse(X_train_standardized, X_test_standardized, y_train, y_test)

    print('------------------------------------------------------------------------------------------')
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
    print('------------------------------------------------------------------------------------------')
    print("The Ridge Regression model performs poorly on the test set like the Baseline Model and will not be used on our final NPML dataset.")
    print("Run `DeepLearning_NN.py` to acquire the most accurate result on the NPML dataset instead.")
    print("Two plots are generated just for reference.")
    print('------------------------------------------------------------------------------------------')


if __name__ == "__main__":
    main()
