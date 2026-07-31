import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Read CSV file
data = pd.read_csv("student_marks.csv")

# Display dataset
print("Dataset:")
print(data)

# Input and Output
X = data[["STUDY_HOURS"]]
y = data["MARKS"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

print("\nActual Marks:")
print(y_test.values)

print("\nPredicted Marks:")
print(predictions)

# Predict marks for a new student
hours = [[7.5]]
predicted_marks = model.predict(hours)

print("\nPredicted Marks for 7.5 study hours:")
print(predicted_marks[0])
import matplotlib.pyplot as plt

# Scatter plot of original data
plt.scatter(X, y, color="blue", label="Actual Data")

# Regression line
plt.plot(X, model.predict(X), color="red", label="Regression Line")

plt.title("Student Marks Prediction")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.show()