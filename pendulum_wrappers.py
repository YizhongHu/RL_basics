import gym

import numpy as np


class RewardEnergy(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_energy = -10

    def reward(self, reward):
        theta, theta_dt = self.unwrapped.state

        # Reward with energy
        current_energy = 10 * np.sin(theta) + theta_dt ** 2 / 2
        total_rwd = current_energy - self.prev_energy + reward + theta**2 + 0.1 * theta_dt**2
        self.prev_energy = current_energy

        # Reward with x
        # total_rwd = 100 * (x - self.prev_x)
        # self.prev_x = x
        # total_rwd = x

        return total_rwd
