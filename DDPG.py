import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import pickle
import os

from nn import create_mlp
from DQN import Experience, ExperienceReplay


class DeepDeterministicPolicyGradient(tf.keras.Model):

    def __init__(self, actor_dims, critic_dims, action_bounds,
                 exploration, exploration_decay, gamma,
                 memory, start_updating,
                 batch_size, actor_lr, critic_lr) -> None:
        super().__init__()

        assert critic_dims[-1] == 1

        self.mu = create_mlp(actor_dims, final_activation='linear')
        self.q = create_mlp(critic_dims, final_activation='linear')

        self.action_bounds = np.array(action_bounds)

        self.sigma_init = exploration
        self.sigma = exploration
        self.sigma_decay = exploration_decay
        self.gamma = gamma

        self.memory = memory
        self.replay = ExperienceReplay(memory)
        self.start_updating = start_updating

        self.batch_size = batch_size
        self.actor_optimizer = tf.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=critic_lr)

    def act(self, state, deterministic=False):
        output = self.mu(state.reshape(1, -1))[0]
        if not deterministic:
            output += np.random.normal(0, self.sigma, output.numpy().shape)

        mins, maxs = self.action_bounds[:, 0], self.action_bounds[:, 1]

        action = tf.clip_by_value(
            output, clip_value_min=mins, clip_value_max=maxs)

        # if not deterministic:
        #     mean = (clipped_output - mins) / (maxs - mins)
        #     alpha = mean * self.kappa + 1
        #     beta = (1 - mean) * self.kappa + 1
        #     dist = tfp.distributions.Beta(alpha, beta)

        #     action = dist.sample(1).numpy()
        #     assert action.shape == output.shape
        # else:
        #     action = output.numpy()

        return action.numpy()

    def simulate(self, env, render=False, evaluation=False):
        state = env.reset()
        done = False

        iters = 0
        total_rwd = 0

        while not done:
            if render:
                env.render()

            action = self.act(state, deterministic=evaluation)
            new_state, reward, done, _ = env.step(action)

            record = Experience(state, action, reward, done, new_state)
            self.replay.append(record)

            state = new_state

            iters += 1
            total_rwd += reward

        return iters, total_rwd

    def _update_weights(self, states, actions, rewards, dones, new_states, gamma):
        # Update the Critic
        with tf.GradientTape() as tape:
            # Calculate the predicted Q values
            this_Q = self.q(np.concatenate((states, actions), axis=-1))
            # Calculate the Q values in the next state
            next_actions = self.mu(new_states).numpy()
            next_Q = self.q(np.concatenate(
                (states, next_actions), axis=-1)).numpy()
            # Calculate the actual Q values
            y = np.where(dones, rewards, np.array(rewards) + gamma * next_Q)

            critic_loss = tf.reduce_mean(tf.math.square(y - this_Q))
            critic_grad = tape.gradient(
                critic_loss, self.q.trainable_variables)

        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.q.trainable_variables))

        # Update the Actor
        with tf.GradientTape() as tape2:
            # Calculate the current optimal actions
            actions_now = self.mu(states)
            # Predict Q values
            q_values = self.q(
                tf.concat([states, actions_now], axis=-1))

            actor_loss = -tf.reduce_mean(q_values)
            actor_grad = tape2.gradient(
                actor_loss, self.mu.trainable_variables)

        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.mu.trainable_variables))

        # Update target networks
        # --- TARGET NETWORKS NOT IMPLEMENTED ---

        return actor_loss, critic_loss

    def _sigma(self, step, total_steps):
        return self.sigma_init * np.power(self.sigma_decay, step / total_steps)

    def train(self, env, episodes):
        for episode in range(episodes):
            print(f'Episode: {episode}')

            self.sigma = self._sigma(episode, episodes)
            iters, total_rwd = self.simulate(env)
            if len(self.replay) >= self.start_updating:
                actor_loss, critic_loss = self._update_weights(
                    *self.replay.sample(self.batch_size), self.gamma)
                print(
                    f'iters: {iters}, tot_rwd: {total_rwd:.3f}, actr_lss: {actor_loss:.3e}, crtc_lss: {critic_loss:3e}')
            else:
                print(f'iters: {iters}, tot_rwd: {total_rwd:.3f}')

    def save(self, path):
        self.mu.save_weights(os.path.join(path, 'actor/'))
        self.q.save_weights(os.path.join(path, 'critic/'))
        with open(path + 'experience.pk', 'wb+') as file:
            pickle.dump(self.replay, file)

    def load(self, path):
        self.mu.load_weights(os.path.join(path, 'actor/'))
        self.q.load_weights(os.path.join(path, 'critic/'))
        with open(path + 'experience.pk', 'rb') as file:
            self.replay = pickle.load(file)

    def clear_experience(self,):
        self.replay = ExperienceReplay(self.memory)
