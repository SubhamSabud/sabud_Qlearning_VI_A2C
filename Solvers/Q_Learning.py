from collections import defaultdict
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from Solvers.Abstract_Solver import AbstractSolver
from lib import plotting

class QLearning(AbstractSolver):

    def __init__(self, env, options):
        assert (str(env.action_space).startswith('Discrete') or
                str(env.action_space).startswith('Tuple(Discrete')), str(
            self) + " cannot handle non-discrete action spaces"
        super().__init__(env, options)
        # The final action-value function.
        # A nested dictionary that maps state -> (action -> action-value).
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def train_episode(self):
        """
        Run a single episode of the Q-Learning algorithm: Off-policy TD control.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Use:
            self.env: OpenAI environment.
            self.options.steps: steps per episode
            self.epsilon_greedy_action(state): returns an epsilon greedy action
            np.argmax(self.Q[next_state]): action with highest q value
            self.options.gamma: Gamma discount factor.
            self.Q[state][action]: q value for ('state', 'action')
            self.options.alpha: TD learning rate.
            next_state, reward, done, _ = self.step(action): advance one step in the environment
        """

        # Reset the environment
        state = self.env.reset()

        for _ in range(self.options.steps):
            # Choose action using epsilon-greedy policy
            action = self.epsilon_greedy_action(state)

            # Take a step in the environment
            next_state, reward, done, _ = self.step(action)

            # Update Q value using the Q-learning update rule
            self.Q[state][action] += self.options.alpha * (
                    reward + self.options.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

            # Update the current state
            state = next_state

            # Check for episode termination
            if done:
                break

    def __str__(self):
        return "Q-Learning"

    def plot(self, stats):
        plotting.plot_episode_stats(stats)

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on Q values.

        Returns:
            A function that takes an observation as input and returns a greedy
            action.
        """

        def policy_fn(state):
            best_action = np.argmax(self.Q[state])
            return best_action

        return policy_fn

    def epsilon_greedy_action(self, state):
        """
        Return an epsilon-greedy action based on the current Q-values and epsilon.

        Use:
            self.env.action_space.n: size of the action space
            np.argmax(self.Q[state]): action with highest q value

        Returns:
            An epsilon-greedy action.
        """
        if np.random.rand() < self.options.epsilon:
            # Exploration: choose a random action
            return np.random.choice(self.env.action_space.n)
        else:
            # Exploitation: choose the action with the highest Q value
            return np.argmax(self.Q[state])

