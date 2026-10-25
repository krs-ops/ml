import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

data = pd.read_csv(r"C:\Users\ADMIN\Documents\DATASET(ML)\Mall_Customers.csv")

X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

X = StandardScaler().fit_transform(X)

model = DBSCAN(eps=0.3, min_samples=5)

labels = model.fit_predict(X)

data["Cluster"] = labels

print(data.head())

print("\nCluster Labels:")
print(set(labels))

noise = list(labels).count(-1)
clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("\nNumber of clusters:", clusters)
print("Number of noise points:", noise)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("DBSCAN Clustering - Mall Customers")
plt.show()
