import numpy as np
import gym
import random
import time

env = gym.make("Taxi-v2")
env.render()

# get #Actions and #States
action_size = env.action_space.n
state_size = env.observation_space.n

# hyperparameters
total_episodes = 10000        # Total episodes
max_steps = 99                # Max steps per episode

learning_rate = 0.7           # Learning rate
gamma = 0.5                   # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

# initialise Q-table
q_table = np.zeros((state_size, action_size))

# training
for episode in range(total_episodes):
    print("Executing episode %d" % episode)
    state = env.reset()
    step = 0
    finished = False

    while step < max_steps and not finished:
        r = random.uniform(0,1)
        if r <= epsilon:
            # exploration - apply random applicable action
            action = env.action_space.sample()
        else:
            # exploitation - apply best action according to q_table
            action = np.argmax(q_table[state,:])

        # observe new state and reward after applying the action
        new_state, reward, finished, info = env.step(action)
        
        # update respective q_table entry according to Bellman Equation
        q_table[state, action] = q_table[state, action] + learning_rate * (reward + gamma *
                                    np.max(q_table[new_state, :]) - q_table[state, action])

        state = new_state
        step += 1
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

# testing
env.reset()
state = env.reset()
finished = False
while not finished:
    env.render()
    # choose best action according to q_table
    action = np.argmax(q_table[state, :])
    # apply action
    new_state, reward, finished, info = env.step(action)
    state = new_state
    time.sleep(1)

env.close()
