import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def mean_squared_error(y_true, y_pred):
    N = len(y_true)
    mse = np.sum((y_true - y_pred)**2)/N
    return mse

np.random.seed(42)
X = np.random.rand(100,2)
y = 4 +3 * X + np.random.randn(100,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'y_pred: {y_pred}')

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
