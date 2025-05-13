import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Generating synthetic dataset
N = 10000  # Number of samples
np.random.seed(42)

X = np.random.rand(N, 1) * 100000  # Random GDP values
y = (0.5 * X + np.random.randn(N, 1) * 1000).ravel()  # Life satisfaction with some noise

# Single test point and multiple test points
X_new = [[37655.2]]
X_batch = np.random.rand(1000, 1) * 100000  # 1000 new test samples for testing predict scalability

# Linear Regression
lr_model = LinearRegression()

# Measure fit time
start_time = time.time()
lr_model.fit(X, y)
lr_fit_time = time.time() - start_time

# Measure predict time for 1 point
start_time = time.time()
lr_model.predict(X_new)
lr_predict_single_time = time.time() - start_time

# Measure predict time for multiple points
start_time = time.time()
lr_model.predict(X_batch)
lr_predict_batch_time = time.time() - start_time

print(f"Linear Regression fit time: {lr_fit_time:.6f} seconds")
print(f"Linear Regression predict time (single point): {lr_predict_single_time:.6f} seconds")
print(f"Linear Regression predict time (batch of 1000): {lr_predict_batch_time:.6f} seconds")

# K-Nearest Neighbors
knn_model = KNeighborsRegressor(n_neighbors=5)

# Measure fit time
start_time = time.time()
knn_model.fit(X, y)
knn_fit_time = time.time() - start_time

# Measure predict time for 1 point
start_time = time.time()
knn_model.predict(X_new)
knn_predict_single_time = time.time() - start_time

# Measure predict time for multiple points
start_time = time.time()
knn_model.predict(X_batch)
knn_predict_batch_time = time.time() - start_time

print(f"KNN fit time: {knn_fit_time:.6f} seconds")
print(f"KNN predict time (single point): {knn_predict_single_time:.6f} seconds")
print(f"KNN predict time (batch of 1000): {knn_predict_batch_time:.6f} seconds")