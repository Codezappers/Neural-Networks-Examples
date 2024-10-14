from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Correctly import pandas
data = {'X': [1, 2, 3, 4, 5], 'y': [2, 4, 5, 4, 5]}
df = pd.DataFrame(data)

X = df[['X']]  # Reshape X to be a 2-dimensional array
y = df['y']

# Split the data
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