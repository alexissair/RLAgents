import numpy as np
import torch
from collections import deque
import random
import torch.nn as nn
from base_agent import BaseAgent


class DQN(BaseAgent):
    def __init__(self,
                 env,
                 model,
                 states_size,
                 actions_size,
                 models_update,
                 rho,
                 gamma,
                 batch_size,
                 learning_rate,
                 memory_size,
                 epsilon_min,
                 epsilon_decay):
        super(DQN, self).__init__(env=env, memory_size=memory_size)
        # Environment parameters
        self.env = env
        self.states_size = states_size
        self.actions_size = actions_size

        # Model parameters
        self.model = model
        self.target_model = self.model

        self.models_update = models_update
        self.rho = rho
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)

        # Parameters handling exploration
        self.epsilon = 1.
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def act(self, state, mode):
        state = torch.tensor(state, dtype=torch.float)
        if mode == 'train' and np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions_size)
            self.epsilon *= self.epsilon_decay
        else:
            action = torch.argmax(self.model(state)).item()
        return action

    def build_batch(self):
        batch = np.asarray(random.sample(self.memory, self.batch_size))

        # Extracting tuples
        states_tensor = torch.zeros((self.batch_size, self.states_size))
        Q_targets = torch.zeros((self.batch_size, self.actions_size))

        # Compute Q targets
        with torch.no_grad():
            for i in range(self.batch_size):
                states_tensor[i] = torch.tensor(batch[i, 0], dtype=torch.float)
                action_i = torch.tensor(batch[i, 1], dtype=torch.float)
                next_state_i = torch.tensor(batch[i, 2], dtype=torch.float)

                # Defining Q_targets for non selected action
                Q_targets[i] = self.model(states_tensor[i])

                # Dealing with selected action
                action_i = int(action_i.item())
                Q_targets[i, action_i] = torch.tensor(batch[i, 3], dtype=torch.float).item()
                if not batch[i, 4]:
                    Q_targets[i, action_i] += self.gamma * self.target_model(next_state_i).max()

        return states_tensor, Q_targets

    def train(self, n_episodes, verbose):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        loss = nn.MSELoss()
        rewards = list()

        for _ in range(n_episodes):
            reward_ep = self.run_episode(mode='train')
            rewards.append(reward_ep)

            if _ % 50 == 0 & verbose:
                print('Episode n°{}'.format(_))

            if _ % self.models_update == 0 and (len(self.memory) > self.batch_size):
                # Updating the critic
                states, Q_targets = self.build_batch()
                Q_outputs = self.model(states)
                print(Q_outputs)

                loss_Q = loss(Q_outputs, Q_targets)

                optimizer.zero_grad()
                loss_Q.backward()
                optimizer.step()

        return rewards
