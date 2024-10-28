import gym
import numpy as np
import random
import sys, time

# Set up the Taxi-v3 environment.
env = gym.make("Taxi-v3")

# Print observation and action space details
print("Observation space (number of states):", env.observation_space.n)
print("Action space (number of actions):", env.action_space.n)

# Initialize Q-table
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Model parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor that determines the importance of future rewards
epsilon = 0.1  # Exploration rate

# Number of training episodes. I choose it higher so that both SARSA and Q learning can be able to learn it.
episodes = 10000

def choose_action(state, q_table, epsilon):
    """
    Choose an action based on epsilon-greedy policy.
    
    Parameters:
        - state: The current state of the agent.
        - q_table: The Q-table containing Q-values for each state-action pair.
        - epsilon: The exploration rate (probability of choosing a random action).
    
    Returns:
        - action: The selected action (either exploratory or exploitative).
    """

    if random.uniform(0, 1) < epsilon:
        # choose a random action
        action = random.choice(range(len(q_table[state])))
    else:
        # choose the action with the highest Q-value for the current state
        action = np.argmax(q_table[state])
    
    return action

def SARSA_training(env, q_table, epsilon, gamma, alpha, episodes):
    # SARSA training loop
    for episode in range(episodes):
        # Reset the environment and get the initial state
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info
        done = False # done means whether current episode is finished or not

        # Choose initial action using epsilon-greedy policy
        action = choose_action(state, q_table, epsilon)

        # Loop until this episode is done
        while not done:
            # Perform the action and get the next state, reward, and done flag
            step_info = env.step(action)
            '''
                step_info: (107, -1, False, False, {'prob': 1.0, 'action_mask': array([1, 1, 1, 0, 0, 0], dtype=int8)})
                107-> state number, -1 -> reward, False-> episode is not finished.
            '''
            next_state = step_info[0]
            reward = step_info[1]
            done = step_info[2]

            # Choose the next action using epsilon-greedy policy.
            next_action = choose_action(next_state, q_table, epsilon)

            # Update Q-value[state, action] using the Bellman equation for SARSA.
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, next_action] - q_table[state, action]
            )

            # Move to the next state and action
            state = next_state
            action = next_action

    print("SARSA training completed.")

def Qlearning_training(env, q_table, epsilon, gamma, alpha, episodes):
    # Q-learning training loop
    for episode in range(episodes):
        # Reset the environment and get the initial state
        state_info = env.reset()
        state = state_info[0] if isinstance(state_info, tuple) else state_info  # Handle tuple for Gym's reset output
        done = False # done means whether current episode is finished or not
        
        # Loop until the episode is done
        while not done:
            # Choose an action using epsilon-greedy policy.
            action = choose_action(state, q_table, epsilon)
   
            # Perform the action and get the next state, reward, and done flag
            step_info = env.step(action)
            next_state = step_info[0] if isinstance(step_info, tuple) else step_info
            reward = step_info[1]
            done = step_info[2]
            
            # Update Q-table using the Bellman equation for Q-learning
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            
            # Update the current state only.
            state = next_state

    print("Q-learning based Training completed.")

# Based on argument, either SARSA or Q-learning based training will be happened.
if int(sys.argv[1]) == 1 :
    SARSA_training(env, q_table, epsilon, gamma, alpha, episodes)
else :
    Qlearning_training(env, q_table, epsilon, gamma, alpha, episodes)

print("Simulating the taxi after training:")

# Testing and evaluation
test_episodes = 100
success_count = 0

for _ in range(test_episodes):
    # Reset the environment for each test
    state_info = env.reset()
    state = state_info[0] if isinstance(state_info, tuple) else state_info
    done = False
    steps = 0
    
    # Loop until the episode is done or a step limit is reached
    while not done and steps < 100:
        action = np.argmax(q_table[state])  # Choose the best action
        
        # Perform the action
        step_info = env.step(action)
        next_state = step_info[0] if isinstance(step_info, tuple) else step_info
        done = step_info[2]
        
        # Update the state
        state = next_state
        steps += 1
    
    # If the episode completes successfully within 100 steps, increment the success count
    if done and steps < 100:
        success_count += 1

print(f"The agent succeeded in {success_count} of {test_episodes} test episodes.")

# Visualization with human render mode
env = gym.make("Taxi-v3", render_mode="human")

state_info = env.reset()
state = state_info[0] if isinstance(state_info, tuple) else state_info
done = False
steps = 0

# Render the environment after training to see the taxi's behavior
while not done and steps < 100:
    action = np.argmax(q_table[state])  # Choose the best action
    
    step_info = env.step(action)
    print(step_info)
    next_state = step_info[0] if isinstance(step_info, tuple) else step_info
    done = step_info[2]
    
    env.render()  # Visualize the environment after each action
    time.sleep(0.5)  # Pause to allow observation
    
    state = next_state
    steps += 1
    print(steps)

env.close()
