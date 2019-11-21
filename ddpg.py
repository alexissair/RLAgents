import numpy as np
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from collections import deque
from models import ActorModel, CriticModel
from base_agent import BaseAgent


class DDPG(BaseAgent):
    def __init__(self,
                 env,
                 actor_model,
                 critic_model,
                 states_size,
                 actions_size,
                 amin,
                 amax,
                 models_update,
                 rho,
                 gamma,
                 batch_size,
                 critic_learning_rate,
                 actor_learning_rate,
                 memory_size,
                 noise_std,
                 noise_decay,
                 noise_min):

        # Inheriting from Base agent
        super(DDPG, self).__init__(env=env, memory_size=memory_size)

        # Environment parameters
        self.states_size = states_size
        self.actions_size = actions_size
        self.amin = amin
        self.amax = amax

        # Training parameters

        # Models parameters
        self.actor_model = actor_model
        self.actor_target_model = self.actor_model
        self.critic_model = critic_model
        self.critic_target_model = self.critic_model
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        # Updates model
        self.models_update = models_update
        self.rho = rho  # rho used for soft updates

        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = deque(maxlen=memory_size)

        # Noise handling exploration
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.noise_min = noise_min
        return

    def act(self, state, mode):
        state = torch.tensor(state, dtype=torch.float)
        action = self.actor_model(state) + (mode == 'train') * np.random.normal(loc=0, scale=self.noise_std)
        # Updating the current noise for selecting actions
        self.noise_std *= self.noise_decay
        self.noise_std = max(self.noise_min, self.noise_std)
        torch.clamp(action, min=self.amin, max=self.amax)
        return np.array([action.item()])

    def run_episode(self, mode):
        state = self.env.reset()
        done = False
        total_reward_ep = 0
        while not done:
            action = self.act(state, mode=mode)
            next_state, reward, done, _ = self.env.step(action)
            total_reward_ep += reward
            self.memory.append(np.array([state, action, next_state, reward, done]))
            state = next_state
        return total_reward_ep

    def build_critic_target(self):
        batch = np.asarray(random.sample(self.memory, self.batch_size))

        # Extracting tuples
        states_tensor = torch.zeros((self.batch_size, self.states_size))
        actions_tensor = torch.zeros((self.batch_size, self.actions_size))
        Q_targets = torch.zeros(self.batch_size)

        # Compute Q targets
        with torch.no_grad():
            for i in range(self.batch_size):
                states_tensor[i] = torch.tensor(batch[i, 0], dtype=torch.float)
                actions_tensor[i] = torch.tensor(batch[i, 1], dtype=torch.float)
                next_state_i = torch.tensor(batch[i, 2], dtype=torch.float)
                Q_targets[i] = torch.tensor(batch[i, 3], dtype=torch.float)
                if not batch[i, 4]:
                    next_action_i = self.actor_target_model(next_state_i)
                    next_Q = self.critic_target_model(torch.cat([next_state_i, next_action_i]))
                    Q_targets[i] += self.gamma * next_Q.item()

        return states_tensor, actions_tensor, Q_targets

    def train(self, n_episodes, verbose):
        rewards = list()
        critic_optimizer = Adam(params=self.critic_model.parameters(), lr=self.critic_learning_rate)
        loss_critic = nn.MSELoss()

        actor_optimizer = Adam(params=self.actor_model.parameters(), lr=self.actor_learning_rate)

        for _ in range(n_episodes):
            if _ % 50 == 0 & verbose:
                print('Episode n°{}'.format(_))
            reward_ep = self.run_episode(mode='train')
            rewards.append(reward_ep)

            if _ % self.models_update == 0 and (len(self.memory) > self.batch_size):
                # Updating the critic
                states, actions, Q_targets = self.build_critic_target()
                Q_inputs = torch.cat([states, actions], dim=1)
                Q_outputs = self.critic_model(Q_inputs).view(-1)
                loss = loss_critic(Q_outputs, Q_targets)

                critic_optimizer.zero_grad()
                loss.backward()
                critic_optimizer.step()

                # Updating the actor
                chosen_actions = self.actor_model(states)
                loss_actor = - self.critic_model(torch.cat([states, chosen_actions], dim=1)).mean()
                actor_optimizer.zero_grad()
                loss_actor.backward()
                actor_optimizer.step()

                # Soft updating target models
                self.soft_update_models(self.actor_model, self.actor_target_model, self.rho)
                self.soft_update_models(self.critic_model, self.critic_target_model, self.rho)

        return rewards
