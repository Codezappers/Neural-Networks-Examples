import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Realistic data for the bedrooms and house prices
bedrooms = np.array([1, 2, 3, 4, 5])
prices = np.array([100000, 200000, 300000, 400000, 500000])

# Reshape the data
bedrooms = bedrooms.reshape(-1, 1)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(bedrooms, prices)
predicted_prices = model.predict(bedrooms)

# Plot the data
plt.scatter(bedrooms, prices, color = 'blue', label = 'Real data', alpha = 0.5)
plt.plot(bedrooms, predicted_prices, color = 'red', label = 'Predicted data')
plt.scatter(new_bedrooms, predicted_prices, color = 'green', label = 'Predicted price for 6 bedrooms')

plt.title('House prices')
plt.xlabel('Number of bedrooms')
plt.ylabel('Price')
plt.legend()
plt.show()