import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Define the features
X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]])

# Define the target 
# 0 represents wear shorts, 1 represents grab a raincoat, 2 represents wear a jacket
y = [0, 1, 2, 1]  # Adjusted to match the class names

# Create a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Plot the decision tree
fig, ax = plt.subplots(figsize=(10, 8))
tree.plot_tree(clf, filled=True, feature_names=['Sunny', 'Rainy', 'Cloudy'], class_names=['Shorts', 'Raincoat', 'Jacket'])
plt.show()