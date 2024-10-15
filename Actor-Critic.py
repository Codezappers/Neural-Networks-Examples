import numpy as np

# Environment 
goal_position = (2,2)
current_position = np.random.randint(0,3,size=(2))

# Actor
def taken_action():
    return np.random.choice(['up','down','left','right'])

# Critic
def provide_feedback(current_position, goal_position):
    distance_before = np.linalg.norm(current_position - goal_position)
    new_position = current_position + np.random.randint(-1,2,size=(2))
    distance_after = np.linalg.norm(new_position - goal_position)
    
    if distance_after < distance_before:
        return "+ 1 (Closer to goal)"
    else:
        return "- 1 (Farther from goal)"
    
num_episodes = 100
for episode in range(num_episodes):
    action = taken_action()  
    
    feedback = provide_feedback(current_position, goal_position)
    
    reward = 1
    
    if feedback == "- 1 (Closer to goal)":
        print(f"Episode: {episode + 1} | Action: {action} | Feedback: {feedback} | Reward: {reward}".format(episode, action, feedback, reward))
    else:
        print(f"Episode: {episode + 1} | Action: {action} | Feedback: {feedback} | Reward: {reward}) - Adjusting action")
        
    if action == 'up':
        current_position[0] += 1
    elif action == 'down':
        current_position[0] -= 1
    elif action == 'left':
        current_position[1] -= 1
    elif action == 'right':
        current_position[1] += 1