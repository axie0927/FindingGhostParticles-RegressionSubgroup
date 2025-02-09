#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


# In[2]:


# Assign processed data to data_files
cwd = os.getcwd()
data_dir = os.path.join(cwd, "Data/")
data_files = [f for f in os.listdir(str(data_dir)) if f.endswith('csv')]

data_train_name = [f for f in data_files if 'TRAIN' in f]
data_test_name = [f for f in data_files if 'TEST' in f]

data_train = pd.read_csv(os.path.join(data_dir,data_train_name[0]))
data_test = pd.read_csv(os.path.join(data_dir,data_test_name[0]))


# In[3]:


# Drop columns needed for classification group
boolean_col = ['highavse','lowavse','truedcr','lq']
data_train_filtered = data_train.drop(columns=boolean_col+['id'])
data_test_filtered = data_test.drop(columns=boolean_col+['id'])

# Find and Drop rows with missing values
data_train_filtered = data_train_filtered.dropna()
data_test_filtered = data_test_filtered.dropna()

# Drop irrelevant features and feature with perfect multicollinearity 
data_train_filtered = data_train_filtered.drop(columns=['tdrift50','tdrift10'])
data_test_filtered = data_test_filtered.drop(columns=['tdrift50','tdrift10'])


# In[4]:


# Train Test split
X_train = data_train_filtered.drop(columns=['energylabel'])
X_test = data_test_filtered.drop(columns=['energylabel'])
y_train = data_train_filtered['energylabel']
y_test = data_test_filtered['energylabel']

# Standardizing our columns
scaler = StandardScaler()
X_train_standardized = scaler.fit_transform(X_train)
X_test_standardized = scaler.transform(X_test)

param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.5], 
    'loss': ['epsilon_insensitive']
}

# Train SVR with default parameters
svr = LinearSVR(random_state=42, max_iter=10000)

# Perform GridSearchCV with 3-fold Crossvalidation (using 3-fold to minimize runtime)
grid_search = GridSearchCV(
    estimator=svr,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=3,  
)

# Fit the model
grid_search.fit(X_train_standardized, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_svr_model = grid_search.best_estimator_

print(f"Best Hyperparameters: {best_params}")

# Predictions
y_pred = best_svr_model.predict(X_test_standardized)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2 Score: {r2}")


# We see here that an SVM model may not be the best model to use since it has a higher MSE than our base model, however we might also want to look into whether MSE is our best measure of model performance.

# In[5]:


print(np.mean(abs(y_pred - y_test)))

