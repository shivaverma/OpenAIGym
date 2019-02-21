#  The inverted pendulum swingup problem is a classic problem in the control literature.
#  In this version of the problem, the pendulum starts in a random position,
#  and the goal is to swing it up so it stays upright.

import gym
import copy
import random
import numpy as np
from keras import Model
import keras.backend as K
import matplotlib.pyplot as plt
from keras.optimizers import adam, Adam
from collections import deque, namedtuple
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, add, Input, Activation, Lambda

np.random.seed(0)
env = gym.make('Pendulum-v0')


class Actor:

    def __init__(self, tau, lr, state_size, action_size):

        self.tau = tau
        self.lr = lr
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):

        # input layer + hidden layer
        inp = Input((self.state_size,))

        state = Dense(128)(inp)
        state = BatchNormalization()(state)
        state = Activation('relu')(state)

        # hidden layer
        state = Dense(256)(state)
        state = BatchNormalization()(state)
        state = Activation('relu')(state)

        # output layer
        state = Dense(self.action_size)(state)
        state = Activation('tanh')(state)
        actions = Lambda(lambda x: x * 2.0)(state)      # because action range is (-2, 2)
        self.model = Model(inp, actions)

        action_gradients = Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = Adam(lr=self.lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:

    def __init__(self, tau, lr, state_size, action_size):

        self.tau = tau
        self.lr = lr
        self.state_size = state_size
        self.action_size = action_size
        self.build_model()

    def build_model(self):

        inp_state = Input(shape=(self.state_size, ))
        inp_action = Input(shape=(self.action_size,))

        # hidden layer for state
        state = Dense(128)(inp_state)
        state = BatchNormalization()(state)
        state = Activation('relu')(state)

        # hidden layer, also equaling the metrics dimensions
        action = Dense(256)(inp_action)
        state = Dense(256)(state)

        # adding both layers
        merged = add([action, state])
        merged = Activation('relu')(merged)

        # output layer
        out = Dense(1, activation='tanh')(merged)

        self.model = Model([inp_state, inp_action], out)
        self.model.compile(loss='mse', optimizer=adam(lr=self.lr))
        action_gradients = K.gradients(out, inp_action)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()],
                                               outputs=action_gradients)


class OUNoise:

    def __init__(self, size,  mu=0):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = 0.07
        self.sigma = 0.1
        self.dt = 1e-2
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""

        # x = self.state + self.theta * (self.mu - self.state) * self.dt + \
        # self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        # self.state = x
        # return x

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, action_size):

        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.action_size = action_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):

        batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        states = np.vstack([e.state for e in batch if e is not None])
        actions = np.array([e.action for e in batch if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in batch if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in batch if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in batch if e is not None])
        return states, actions, rewards, dones, next_states


class My_noise:

    def __init__(self):
        self.ep = 1
        self.decay = .996

    def sample(self):
        return np.random.randn()*self.ep

    def update_noise(self):
        self.ep *= self.decay


my_noise = My_noise()


class DDPG:

    def __init__(self):

        self.action_size = env.action_space.shape[0]
        self.state_size = env.observation_space.shape[0]

        self.exploration_mu = 0
        self.exploration_theta = 0.15
        self.exploration_sigma = 0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu)

        self.buffer_size = 500000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.action_size)

        self.actor_lr = 0.001     # learning rate
        self.critic_lr = 0.001
        self.gamma = 0.99         # discount factor
        self.tau = 0.001          # for soft update of target parameters

        self.actor_local = Actor(self.tau, self.actor_lr, self.state_size, self.action_size)
        self.actor_target = Actor(self.tau, self.actor_lr, self.state_size, self.action_size)

        self.critic_local = Critic(self.tau, self.critic_lr, self.state_size, self.action_size)
        self.critic_target = Critic(self.tau, self.critic_lr, self.state_size, self.action_size)

    def step(self, state, action, reward, next_state, done):

        self.memory.add(state, action, reward, next_state, done)
        self.learn()

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + np.array(my_noise.sample()).reshape(1,))   # add some noise for exploration

    def learn(self):

        states, actions, rewards, dones, next_states = self.memory.sample()

        # Get predicted next-state actions and Q values from target models
        # Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        q_targets = rewards + self.gamma * q_targets_next*(1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]),
                                     (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

        my_noise.update_noise()

    def soft_update(self, local_model, target_model):

        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)


if __name__ == '__main__':

    print(env.action_space.shape[0], env.observation_space.shape[0])
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    loss = []
    episodes = 500
    max_steps = 1000
    agent = DDPG()
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, (1, state_size))
        score = 0
        for i in range(max_steps):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, (1, state_size))
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                print("episode: {}/{}, score: {}".format(e, episodes, score))
                break
        loss.append(score)
        #my_noise.update_noise()
    plt.plot([i + 1 for i in range(episodes)], loss)
    plt.show()