import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Hyperparameters
input_size = 2  # Adjust this based on your input data
hidden_size = 5
output_size = 1
num_epochs = 1000
learning_rate = 0.01

# Create the model
model = SimpleNN(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Example data
X_tensor = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]], dtype=torch.float32)
y_tensor = torch.tensor([[0], [0], [1], [1], [1]], dtype=torch.float32)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Test the model on new data
test_data = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
predicted_probs = model(test_data)
print(f'Predicted probabilities: {predicted_probs}')
print("\nPredictions:")
print(predicted_probs.detach().numpy())
# Convert probabilities to binary labels (0 or 1)
predicted_labels = (predicted_probs > 0.5).float()
print("\nPredicted labels:")
print(predicted_labels.detach().numpy())