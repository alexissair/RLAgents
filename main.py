from ddpg import DDPG
import numpy as np
from dqn import DQN
from models import *
import gym
import matplotlib.pyplot as plt
from normalization_env import NormalizedEnv

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    model = CriticModel(size_in=env.observation_space.shape[0], hidden_units=256, size_out=env.action_space.n)
    agent = DQN(env, actions_size=env.action_space.n, states_size=env.observation_space.shape[0], batch_size=64,
                learning_rate=3e-4, epsilon_decay=0.9999, epsilon_min=0.01, gamma=0.9999, memory_size=5000,
                models_update=10, rho=0.99, model=model)

    r_train = agent.train(5000, verbose=True)
    r_test = agent.test(500)
    print(r_test)
    print(np.mean(r_train))
    print(np.mean(r_test))
