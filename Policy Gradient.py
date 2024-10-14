import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple policy network
class SimplePolicy(nn.Module):
    def __init__(self):
        super(SimplePolicy, self).__init__()
        self.fc = nn.Linear(1, 2)
        
    def forward(self, state):
        return torch.softmax(self.fc(state), dim=1)
    
policy = SimplePolicy()
optimizer = optim.Adam(policy.parameters(), lr=0.01)

# Training Loop
num_episodes = 1000
for episode in range(num_episodes):
    state = torch.FloatTensor([[np.random.rand()]])
    action_prob = policy(state)  # Corrected from policy_net to policy
    
    # Sample an action based on the probabilities
    action = np.random.choice([0, 1], p=action_prob.detach().numpy().flatten())
    
    reward = 1
    
    loss = -torch.log(action_prob[0, action]) * reward
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Test the policy
    test_state = torch.FloatTensor([[0.5]])
    test_action_prob = policy(test_state)
    
    action_probability = test_action_prob.detach().numpy()
    
    chosen_action = np.argmax(action_probability)
    
    print(f"Episode: {episode}, Chosen action: {chosen_action}, Action probabilities: {action_probability}")