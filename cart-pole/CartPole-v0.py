#  A pole is attached by an un-actuated joint to a cart,
#  which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart.
#  The pendulum starts upright, and the goal is to prevent it from falling over.
#  A reward of +1 is provided for every timestep that the pole remains upright.
#  The episode ends when the pole is more than 15 degrees from vertical,
#  or the cart moves more than 2.4 units from the center.


import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt

import numpy as np
env = gym.make('CartPole-v0')
env.seed(0)
np.random.seed(0)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.epsilon_decay = .995
        self.learning_rate = 0.001
        self.memory = deque(maxlen=4000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(24, input_shape=(self.state_space,), activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):

        minibatch = random.sample(self.memory, min(len(self.memory)-1, self.batch_size))
        x = []
        y = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma*(np.amax(self.model.predict(next_state)[0]))
            target_full = self.model.predict(state)
            target_full[0][action] = target
            x.append(state)
            y.append(target_full)

        x = np.array(x)
        x = x.reshape(x.shape[0], x.shape[2])
        y = np.array(y)
        y = y.reshape(y.shape[0], y.shape[2])
        self.model.fit(x, y, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):

    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 4))
        score = 0
        for t in range(500000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            score += reward
            next_state = np.reshape(next_state, (1, 4))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        agent.replay()
        loss.append(score)
    return loss


def random_policy(episode, step):

    for i_episode in range(episode):
        env.reset()
        for t in range(step):
            env.render()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
            print("Starting next episode")


if __name__ == '__main__':

    ep = 1000
    loss = train_dqn(ep)
    plt.plot([i+1 for i in range(ep)], loss)
    plt.show()
