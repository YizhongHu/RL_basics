import tensorflow as tf

import gym

from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tensorforce.execution import Runner
from tensorforce.core.parameters import Decaying

environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)

network = [
    dict(type='dense', size=32, activation='relu'),
    dict(type='dense', size=32, activation='relu')
]

epsilon = dict(
    type='exponential', unit='episodes', num_steps=1000,
    initial_value=1.0, decay_rate=0.95, dtype=tf.float32
)

agent = Agent.create(
    agent='dqn', environment=environment, memory=10000, batch_size=32, network=network,
    update_frequency=1, start_updating=300, learning_rate=1e-3, exploration=epsilon
)

runner = Runner(
    agent=agent, environment=environment, max_episode_timesteps=500
)

runner.run(num_episodes=500)
runner.run(num_episodes=100, evaluation=True)
runner.close()

with gym.make('CartPole-v1') as env:
    done = False
    obs = env.reset()
    total_reward = 0
    while not done:
        env.render()
        action = agent.act(obs)
        obs, rwd, done, _ = env.step(action)
        total_reward += rwd
        
    print(f'Total Reward: {total_reward}')
