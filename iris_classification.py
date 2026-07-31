import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
X = iris.data
y = iris.target
print(X[:5])
print(y[:5])
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
sample = [[5.2, 3.4, 1.5, 0.2]]

prediction = model.predict(sample)

print(prediction)
print(iris.target_names[prediction[0]])
plt.scatter(X[:,0], X[:,2], c=y, s=60)

plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
plt.title("Iris Flower Classification")
plt.grid(True)



