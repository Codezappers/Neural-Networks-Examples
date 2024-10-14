import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
np.random.seed(0)

# Generate random data for X and y
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = model.predict(X_test)

# Calculate the Mean Absolute Error
mse = mean_absolute_error(y_test, y_pred)

# Plot the test data and the model's predictions
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Logistic Regression')
plt.show()

# Print the Mean Absolute Error, model coefficients, and intercept
print(f'Mean Absolute Error: {mse:.2f}')
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')