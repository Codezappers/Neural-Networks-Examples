import numpy as np

# Define the environment
# S: Start, G: Goal, -: Empty space

env = np.array([
    ['S', '-', '-'],
    ['-', '-', '-'],
    ['-', '-', 'G']
])

# Map each state to a unique integer
state_mapping = {
    (0,0): 0,
    (0,1): 1,
    (0,2): 2,
    (1,0): 3,
    (1,1): 4,
    (1,2): 5,
    (2,0): 6,
    (2,1): 7,
    (2,2): 8,
}

num_states = len(state_mapping)
num_actions = 4  # Up, Down, Left, Right
q_table = np.zeros((num_states, num_actions))

# Define parameters
learning_rate = 0.8
discount_factor = 0.95
exploration_rate = 0.2
num_episodes = 1000

# Q-learning algorithm
for episode in range(num_episodes):
    state = state_mapping[(0, 0)]
    done = False
    
    while not done:
        if np.random.rand() < exploration_rate:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[state, :])
            
        # Get current position
        current_row, current_col = [key for key, value in state_mapping.items() if value == state][0]
        
        # Determine new position based on action
        if action == 0:  # Up
            new_row, new_col = current_row - 1, current_col
        elif action == 1:  # Down
            new_row, new_col = current_row + 1, current_col
        elif action == 2:  # Left
            new_row, new_col = current_row, current_col - 1
        elif action == 3:  # Right
            new_row, new_col = current_row, current_col + 1

        # Ensure new position is within bounds
        new_row = max(0, min(new_row, env.shape[0] - 1))
        new_col = max(0, min(new_col, env.shape[1] - 1))

        # Get new state
        new_state = state_mapping[(new_row, new_col)]
        
        # Determine reward
        reward = 1 if env[new_row, new_col] == 'G' else 0
        
        # Update Q-table
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state, :]))
        
        # Transition to new state
        state = new_state
        
        # Check if goal is reached
        if env[new_row, new_col] == 'G':
            done = True
            
print(q_table)