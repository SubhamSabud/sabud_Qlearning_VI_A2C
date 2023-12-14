# Import numpy library
import numpy as np

# Define the states
A = 0
B = 1

# Define the actions
MOVE = 0
STAY = 1

# Define the rewards
R = np.array([[0, 1], # R(s=A, a=MOVE), R(s=A, a=STAY)
              [0, 1]]) # R(s=B, a=MOVE), R(s=B, a=STAY)

# Define the state transition matrix
P = np.array([[[0, 1], [1, 0]], # P(s'=B|s=A, a=MOVE), P(s'=A|s=A, a=STAY)
              [[1, 0], [0, 1]]]) # P(s'=A|s=B, a=MOVE), P(s'=B|s=B, a=STAY)

# Define the discount factor, the learning rate and the exploration rate
gamma = 0.8
alpha = 0.5
epsilon = 0.5

# Initialize the Q-table
Q = np.zeros((2, 2))

# Set the initial state
s = A

# Run Q-learning for 200 steps
for i in range(200):
  # Choose the action according to an epsilon-greedy policy
  # With probability 1 - epsilon, choose the best action
  # With probability epsilon, choose a random action
  if np.random.random() < 1 - epsilon:
    a = np.argmax(Q[s])
  else:
    a = np.random.choice([MOVE, STAY])

  # Observe the next state and the reward
  s_next = np.random.choice([A, B], p=P[s, a])
  r = R[s, a]

  # Update the Q-value using the Q-learning formula
  Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (r + gamma * np.max(Q[s_next]))

  # Set the next state as the current state
  s = s_next

# Print the final Q-table
print(Q)
