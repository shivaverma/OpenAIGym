#  There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location
#  and drop him off in another. You receive +20 points for a successful dropoff,
#  and lose 1 point for every timestep it takes.
#  There is also a 10 point penalty for illegal pick-up and drop-off actions.


import gym
import numpy as np
env = gym.make('Taxi-v2')


def q_learning(episode):

    alpha = .8
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(episode):
        done = False
        G, reward = 0,0
        state = env.reset()

        while not done:
            env.render()
            action = np.argmax(Q[state])
            state2, reward, done, info = env.step(action)
            Q[state, action] += alpha*(reward + np.max(Q[state2]) - Q[state, action])
            G += reward
            state = state2

        print("episode {}, total reward = {}".format(i, G))


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

    q_learning(1000)