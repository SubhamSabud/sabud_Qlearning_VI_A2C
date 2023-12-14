import numpy as np
import heapq
from Solvers.Abstract_Solver import AbstractSolver, Statistics
import os

class ValueIteration(AbstractSolver):

    def __init__(self, env, options):
        assert str(env.observation_space).startswith('Discrete'), str(self) + " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env, options)
        self.V = np.zeros(env.nS)
        self.options.tol = getattr(options, 'tol', 1e-4)  # Set tol to a default value if not provided

    def train_episode(self):
        """
        Update the estimated value of each state using the value iteration algorithm.
        """
        delta = 0
        for state in range(self.env.nS):
            v = self.V[state]
            # Compute the new value for the state by doing a one-step lookahead
            new_value = max([sum([probability * (reward + self.options.gamma * self.V[next_state])
                                             for probability, next_state, reward, _ in self.env.P[state][action]])
                                              for action in range(self.env.nA)])
            # Update the value of the state
            self.V[state] = new_value
            delta = max(delta, abs(v - new_value))
        
        # Check for convergence
        if delta < self.options.tol:
            self.converged = True

        # Update statistics
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on state values.
        """
        def policy_fn(state):
            """
            Returns a greedy action based on the current state values.
            """
            return np.argmax([sum([probability * (reward + self.options.gamma * self.V[next_state])
                                   for probability, next_state, reward, _ in self.env.P[state][action]])
                              for action in range(self.env.nA)])

        return policy_fn


class PriorityQueue:
    """
    Implements a priority queue data structure.
    """
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
