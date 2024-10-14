import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


np.random.seed(42)
X = np.random.rand(100, 2)
y = (X[:,0] + X[:,1] > 1).astype(int) # Calculating the sum of the two columns and checking if it is greater than 1

print(f'X: {X}')
print(f'y: {y}')   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'y_pred: {y_pred}')

accuracy_score = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy_score}')
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix: {conf_matrix}')

plt.figure(figsize=(7,6))
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, s=50, cmap= 'viridis', edgecolors='k')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression')
plt.grid()
plt.show()