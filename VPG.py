import numpy as np
import tensorflow as tf

from nn import create_mlp

class VanillaPolicyGradient(tf.keras.Model):
    def __init__(self, state_space_dims, action_space_dims, actor_shape, critic_shape, lr, actor_activation='relu', critic_activation='relu', output_mode='Discrete') -> None:
        super().__init__()

        if isinstance(actor_shape, int):
            actor_shape = [actor_shape]
        if isinstance(critic_shape, int):
            critic_shape = [critic_shape]

        if output_mode == 'Continuous':
            final_activation = 'linear'
        elif output_mode == 'Discrete':
            final_activation = 'log_softmax'

        self.actor = create_mlp([state_space_dims, *actor_shape, action_space_dims],
                                activation=actor_activation, final_activation=final_activation)
        self.critic = create_mlp([state_space_dims, *critic_shape, 1],
                                 activation=critic_activation, final_activation='linear')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.output_mode = output_mode

    def act(self, state):
        state = state.reshape((1, -1))
        log_probs = self.actor(state)
        action = tf.random.categorical(log_probs, 1)[0][0]
        log_prob = log_probs[0][action]

        return action.numpy(), log_prob

    def _simulation_step(self, env, render):
        done = False
        iters = 0
        rewards = []
        log_probs = []
        actions = []
        values = []
        obs = env.reset()

        while not done:
            if render:
                env.render()

            action, log_prob = self.act(obs)
            value = self.critic(obs.reshape(1, -1))[0]
            obs, rwd, done, _ = env.step(action)

            iters += 1
            rewards.append(rwd)
            log_probs.append(log_prob)
            values.append(value)
            actions.append(action)

        return iters, rewards, actions, log_probs, values

    def _discounted_rewards(self, rewards, gamma):
        rewards = np.array(rewards)
        discounts = np.array([gamma ** pwr for pwr in range(len(rewards))])

        discounted_rewards = []

        for i in range(len(rewards)):
            if i == 0:
                discounted_rewards.append(rewards @ discounts)
            else:
                discounted_rewards.append(rewards[i:] @ discounts[:-i])

        return discounted_rewards

    def _train_step(self, tape, rewards, log_probs, values, gamma):
        log_probs = tf.convert_to_tensor(log_probs)
        discounted_rewards = self._discounted_rewards(rewards, gamma)
        discounted_rewards = tf.convert_to_tensor(discounted_rewards,
                                                  dtype=tf.float32)
        values = tf.convert_to_tensor(values, dtype=tf.float32)

        advantages = discounted_rewards - values
        actor_loss = -log_probs * advantages.numpy()
        critic_loss = tf.pow(advantages, 2)
        loss = actor_loss + critic_loss

        gradients = tape.gradient(
            loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables))

        return actor_loss, critic_loss, loss

    def train(self, env, epochs=10, gamma=0.99, render=True):
        for epoch in range(epochs):
            print(f'Epoch: {epoch}')
            with tf.GradientTape() as tape:
                iters, rewards, _, log_probs, values = self._simulation_step(
                    env, render)
                actor_losses, critic_losses, losses = self._train_step(
                    tape, rewards, log_probs, values, gamma)

            total_reward = tf.reduce_sum(rewards).numpy()
            loss = tf.reduce_sum(losses).numpy()
            actor_loss = tf.reduce_sum(actor_losses).numpy()
            critic_loss = tf.reduce_sum(critic_losses).numpy()

            print(
                f'Iters: {iters}, Rewards: {total_reward:.3f}, Loss: {loss:.3e}, Actor: {actor_loss:.3e}, Critic: {critic_loss:.3e}')
