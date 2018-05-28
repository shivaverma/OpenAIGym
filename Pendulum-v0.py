#  The inverted pendulum swingup problem is a classic problem in the control literature.
#  In this version of the problem, the pendulum starts in a random position,
#  and the goal is to swing it up so it stays upright.

import gym
import random
import numpy as np
from keras import Sequential
from keras.layers import add
from keras.layers import Dense
from keras.optimizers import adam
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization

from collections import deque

env = gym.make('Pendulum-v0')


class ReplayBuffer:

    def __init__(self, batch_size, buffer_size):

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, s, a, r, t, s2):

        experience = (s, a, r, t, s2)
        self.buffer.append(experience)

    def sample_batch(self):

        batch = random.sample(self.buffer, min(len(self.buffer)-1, self.batch_size))
        return batch


class Actor:

    def __init__(self):

        self.action_space = env.action_space.shape[0]
        self.state_space = env.observation_space.shape[0]

    def create_actor_network(self):

        # input layer + hidden layer
        model = Sequential()
        model.add(Dense(400, input_dim=self.state_space))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # hidden layer
        model.add(Dense(300))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        # output layer
        model.add(Dense(self.action_space))
        model.add(Activation('tanh'))

        model.compile(loss='mse', optimizer=adam())
        return model


class Critic:

    def __init__(self):

        self.action_space = env.action_space.shape[0]
        self.state_space = env.observation_space.shape[0]

    def create_critic_network(self):

        state = Sequential()
        action = Sequential()

        # hidden layer for state
        state.add(Dense(400, input_dim=self.state_space))
        state.add(BatchNormalization())
        state.add(Activation('relu'))

        # hidden layer, also equaling the metrics dimensions
        action.add(Dense(300, state_dim=self.action_space))
        state.add(Dense(300))

        # adding both layers
        state = add([action, state])
        state.add(Activation('relu'))

        # output layer
        state.add(Dense(self.action_space))
        return state


if __name__ == '__main__':

    print(env.action_space.shape[0], env.observation_space)