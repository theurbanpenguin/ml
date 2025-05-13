import pandas as pd
import time
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# Load dataset
DATA_FILE = "lifesat.csv"
lifesat = pd.read_csv(DATA_FILE)

# Features and targets
X = lifesat[["GDP per capita (USD)"]].values  # Features
y = lifesat[["Life satisfaction"]].values  # Targets

# Linear Regression
lr_model = LinearRegression()

# Measure time for fitting Linear Regression
start_time = time.time()
lr_model.fit(X, y)
lr_fit_time = time.time() - start_time
print(f"Linear Regression fit time: {lr_fit_time:.6f} seconds")

# Measure time for predicting using Linear Regression
X_new = [[37655.2]]
start_time = time.time()
lr_prediction = lr_model.predict(X_new)
lr_predict_time = time.time() - start_time
print(f"Linear Regression predict time: {lr_predict_time:.6f} seconds")
print(f"Linear Regression prediction: {lr_prediction}")

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsRegressor(n_neighbors=3)

# Measure time for fitting KNN
start_time = time.time()
knn_model.fit(X, y)
knn_fit_time = time.time() - start_time
print(f"KNN fit time: {knn_fit_time:.6f} seconds")

# Measure time for predicting using KNN
start_time = time.time()
knn_prediction = knn_model.predict(X_new)
knn_predict_time = time.time() - start_time
print(f"KNN predict time: {knn_predict_time:.6f} seconds")
print(f"KNN prediction: {knn_prediction}")