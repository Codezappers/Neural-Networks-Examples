import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(0)

# Generate random data for X and y
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

print(f'X value: {X}')
print(f'y value: {y}')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train.ravel())  # Use ravel() to ensure y_train is 1D

# Predict the target values for the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')

# Plot the test data and the model's predictions
plt.scatter(X_test, y_test, color='red', label='Actual Data')
plt.plot(X_test, y_pred, color='blue', label='Predicted Data')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
