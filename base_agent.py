from collections import deque
import numpy as np


class BaseAgent:
    def __init__(self,
                 env,
                 memory_size,
                 gamma,
                 batch_size,
                 ):
        self.env = env
        self.memory = deque(maxlen=memory_size)
        return

    def act(self, state, mode):
        return NotImplementedError

    def train(self, n_episodes, verbose):
        return NotImplementedError

    @staticmethod
    def soft_update_models(model, target_model, rho):
        # update target networks
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(param.data * (1.0 - rho) + target_param.data * rho)
        return

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

    def test(self, n_episodes):
        rewards = list()
        for _ in range(n_episodes):
            reward_ep = self.run_episode(mode='test')
            rewards.append(reward_ep)
        return rewards
