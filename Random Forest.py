import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

# Define the features
X = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0]])

# Define the target labels
# 0 represents wear shorts, 1 represents grab a raincoat, 2 represents wear a jacket
y = [0, 1, 2, 1]  # Adjusted to match the class names

# Create a random forest classifier
rf_clf = RandomForestClassifier(n_estimators=10, random_state=42)
rf_clf.fit(X, y)

# Visualize one of the decision trees in the random forest
tree_to_visualize = 0
fig, ax = plt.subplots(figsize=(10, 8))
tree.plot_tree(rf_clf.estimators_[tree_to_visualize], feature_names=['sunny', 'rainy', 'cloudy'], class_names=['wear shorts', 'grab a raincoat', 'wear a jacket'], filled=True, rounded=True)
plt.show()