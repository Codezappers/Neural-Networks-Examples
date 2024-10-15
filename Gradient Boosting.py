from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  # Corrected import
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {'X': [1, 2, 3, 4, 5], 'y': [2, 4, 5, 4, 5]}
df = pd.DataFrame(data)

# Split the data into features and target
X = df[['X']]
y = df['y']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.scatter(X, y, color='blue', label='Real data')
plt.plot(X_test, y_pred, color='red', label='Predicted data')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()