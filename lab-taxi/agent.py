import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.epsilon = 1
        self.episode = 1
        self.alpha = 0.01
        self.gamma = 1

        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        prop = self.find_prop(state)
        action = np.random.choice(np.arange(self.nA), p=prop)
        return action

    def find_prop(self, state):
        prop = [self.epsilon / self.nA] * self.nA
        prop[np.argmax(self.Q[state])] += 1 - self.epsilon
        return prop

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_state_props = self.find_prop(next_state)
        next_reward = np.sum(np.multiply(next_state_props, self.Q[next_state]))
        # next_action = self.select_action(next_state)
        self.Q[state][action] = self.Q[state][action] + self.alpha * \
                                (reward + self.gamma * next_reward - self.Q[state][action])
        if done:
            self.update_epsilon()

    def update_epsilon(self):
        self.epsilon = self.epsilon/self.episode
        if self.epsilon < 0.001:
            self.epsilon = 0.001
        self.episode +=1
