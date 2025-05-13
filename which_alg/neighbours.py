import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
DATA_FILE = "lifesat.csv"
lifesat = pd.read_csv(DATA_FILE)

X = lifesat[["GDP per capita (USD)"]].values # features
y = lifesat[["Life satisfaction"]].values # targets

model = KNeighborsRegressor(n_neighbors=3)
model.fit(X,y) # In KNN, this doesn't "train" the model; it simply stores the training data for use during prediction

X_new = [[571]]
print(model.predict(X_new))