import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read the dataset
data = pd.read_csv("house_price.csv")

print("Dataset")
print(data)

# Input and Output
X = data[["AREA"]]
y = data["PRICE"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Predict price for a new house
area = [[1600]]
predicted_price = model.predict(area)

print("\nPredicted Price for 1600 sq.ft house:")
print(predicted_price[0])

# Plot graph
plt.scatter(X, y, color="blue")
plt.plot(X, model.predict(X), color="red")

plt.title("House Price Prediction")
plt.xlabel("Area (sq.ft)")
plt.ylabel("Price")

plt.show()