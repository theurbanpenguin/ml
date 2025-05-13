import pandas as pd
from sklearn.linear_model import LinearRegression
DATA_FILE = "lifesat.csv"
lifesat = pd.read_csv(DATA_FILE)

X = lifesat[["GDP per capita (USD)"]].values # features
y = lifesat[["Life satisfaction"]].values # targets

model = LinearRegression()  # In this simple example, we are not splitting into training and testing sets for demonstration purposes
model.fit(X,y)

X_new = [[571]]
print(model.predict(X_new))
