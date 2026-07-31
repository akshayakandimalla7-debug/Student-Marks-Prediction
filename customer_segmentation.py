import pandas as pd

data = pd.read_csv("Mall_Customers.csv")

print(data.head())
X = data[['Annual_Income', 'Spending_Score']]
from sklearn.cluster import KMeans

model = KMeans(n_clusters=5, random_state=42)

model.fit(X)
data['Cluster'] = model.labels_

print(data)
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

plt.scatter(
    X['Annual_Income'],
    X['Spending_Score'],
    c=data['Cluster'],
    s=100
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")

plt.grid(True)

plt.show()