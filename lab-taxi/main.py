from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v3')
# print(env.action_space)
# print(env.observation_space)
agent = Agent()
avg_rewards, best_avg_reward = interact(env, agent)