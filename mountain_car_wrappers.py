import gym

import numpy as np


class RewardEnergy(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_energy = 0
        self.prev_x = -0.5

    def reward(self, reward):
        x, v = self.unwrapped.state
        v = v * 1e2
        
        # Reward with energy
        current_energy = v**2 / 2 + 25/3 * np.sin(3 * x)
        total_rwd = reward + current_energy - self.prev_energy
        self.prev_energy = current_energy
        
        # Reward with x
        # total_rwd = 100 * (x - self.prev_x)
        # self.prev_x = x
        # total_rwd = x

        return total_rwd
