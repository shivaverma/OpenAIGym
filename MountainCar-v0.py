#  A car is on a one-dimensional track, positioned between two "mountains".
#  The goal is to drive up the mountain on the right; however,
#  the car's engine is not strong enough to scale the mountain in a single pass. Therefore,
#  the only way to succeed is to drive back and forth to build up momentum.


import gym
import random
from keras import Sequential
from collections import deque
from keras.layers import Dense
from keras.optimizers import adam
import matplotlib.pyplot as plt
from keras.activations import relu, linear

import numpy as np
env = gym.make('MountainCar-v0')
env.seed(0)
np.random.seed(0)


class DQN:

    """ Implementation of deep q learning algorithm """

    def __init__(self, action_space, state_space):

        self.action_space = action_space
        self.state_space = state_space
        self.epsilon = 1.0
        self.gamma = .95
        self.batch_size = 64
        self.epsilon_min = .01
        self.buffer_size = 32
        self.lr = 0.01
        self.epsilon_decay = .995
        self.memory = deque(maxlen=20000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Dense(14, input_dim=self.state_space, activation=relu))
        model.add(Dense(24, activation=relu))
        model.add(Dense(self.action_space, activation=linear))
        model.compile(loss='mse', optimizer=adam(lr=self.lr))
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


def get_reward(state):

    if state[0] >= 0.5:
        print("Car has reached the goal")
        return 50
    elif state[0] >= 0.4:
        return 40
    elif state[0] >= 0.3:
        return 20
    if state[0] > -0.4:
        return (1+state[0])**2
    return 0


def train_dqn(episode):

    loss = []
    agent = DQN(env.action_space.n, env.observation_space.shape[0])
    for e in range(episode):
        state = env.reset()
        state = np.reshape(state, (1, 2))
        score = 0
        for t in range(500000):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = get_reward(next_state)
            score += reward
            next_state = np.reshape(next_state, (1, 2))
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

    print(env.observation_space)
    print(env.action_space)
    episodes = 1000
    loss = train_dqn(episodes)
    plt.plot([i+1 for i in range(episodes)], loss)
    plt.show()
