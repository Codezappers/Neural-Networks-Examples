from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

candies = np.array([[1],[2],[3],[8],[9],[10]])

kmeans = KMeans (n_clusters=2)

kmeans.fit(candies)

labels = kmeans.labels_

plt.scatter(candies, np.zeros_like(candies), c=labels, cmap='viridis')
plt.title('Candies')
plt.xlabel('Candies')
plt.show()