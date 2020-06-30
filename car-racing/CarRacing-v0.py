import gym
import random
import numpy as np
from keras import Sequential
from collections import deque
import matplotlib.pyplot as plt
from keras.optimizers import adam
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dropout

env = gym.make('CarRacing-v0')
env.seed(0)
np.random.seed(0)

rgb_weights = [0.2125, 0.7154, 0.0721]


def encode_action(action):
    return [-action[0] + action[1], action[2], action[3]]


def decode_action(action):
    return [int(action[0] < 0), int(action[0] > 0), action[1], action[2]]


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
        self.memory = deque(maxlen=100000)
        self.model = self.build_model()

    def build_model(self):

        model = Sequential()
        model.add(Conv2D(50, kernel_size=3, strides=1, activation='relu', input_shape=self.state_space))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(BatchNormalization())

        model.add(Conv2D(100, kernel_size=3, strides=1, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(.3))

        model.add(Conv2D(100, kernel_size=3, strides=1, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(BatchNormalization())

        model.add(Conv2D(200, kernel_size=3, strides=2, activation='relu'))
        model.add(Dropout(.3))

        model.add(Flatten())
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))

        model.compile(loss='mse', optimizer=adam(lr=self.learning_rate))
        # print(model.summary())
        # exit()
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state.reshape(1,48,48,1))
        return np.argmax(act_values[0])

    def replay(self):

        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])

        targets = rewards + self.gamma*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
        targets_full = self.model.predict_on_batch(states)

        ind = np.array([i for i in range(self.batch_size)])

        targets_full[[ind], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        # print(self.epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episode):

    loss = []
    agent = DQN(4, (48, 48, 1))

    action_repeat = 3

    for e in range(episode):
        state = env.reset()
        # for i in range(200):
        #     state, _, _, _ = env.step([1,1,1])
        state = resize(state, (48, 48))
        state = rgb2gray(state).reshape((48,48,1))
        # # state = np.dot(state[..., :3], rgb_weights)
        # plt.imshow(state)
        # plt.show()
        # print(state.shape)
        # exit()
        # state = np.dot(state[..., :3], rgb_weights).reshape(96, 96, 1)
        score = 0
        max_steps = 1000
        for i in range(max_steps):
            env.render()

            action_one = agent.act(state)
            action = np.zeros(4)
            action[action_one] = 1
            action = encode_action(action)

            # for j in range(action_repeat):
            next_state, reward, done, _ = env.step(action)

            next_state = resize(next_state, (48, 48))
            next_state = rgb2gray(next_state).reshape((48, 48, 1))

            score += reward
            agent.remember(state, action_one, reward, next_state, done)
            state = next_state

            if i % 20 ==0 :
                agent.replay()

            if done:
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break
        loss.append(score)
        if e%50==0:
            agent.model.save('model.pth')
    return loss


if __name__ == '__main__':

    ep = 1000
    loss = train_dqn(ep)
    plt.plot([i+1 for i in range(0, ep, 2)], loss[::2])
    plt.show()
