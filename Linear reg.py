import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Assuming bedrooms and prices are already defined
bedrooms = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Example data
prices = np.array([150, 200, 250, 300, 350])  # Example data

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(bedrooms, prices)
predicted_prices = model.predict(bedrooms)

# Predict the price for a house with 6 bedrooms
new_bedrooms = np.array([6]).reshape(-1, 1)  # Reshape to 2D array
predicted_price = model.predict(new_bedrooms)

# Plot the data
plt.scatter(bedrooms, prices, color='blue', label='Real data', alpha=0.5)
plt.plot(bedrooms, predicted_prices, color='red', label='Predicted data')
plt.scatter(new_bedrooms, predicted_price, color='green', label='Predicted price for 6 bedrooms')

plt.xlabel('Number of Bedrooms')
plt.ylabel('Price')
plt.legend()
plt.show()