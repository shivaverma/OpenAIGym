import gym
import numpy as np
from ddpg import Agent
import matplotlib.pyplot as plt


if __name__ == '__main__':

    env = gym.make('LunarLanderContinuous-v2')

    env.seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high

    print(state_dim)
    print(action_dim)
    print(action_bound)

    agent = Agent(action_dim, state_dim, -action_bound, action_bound)

    max_steps = 3000
    episodes = 5000
    score_list = []

    for i in range(episodes):

        state = env.reset()
        score = 0

        for j in range(max_steps):

            # env.render()

            action = agent.act(state, i)
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state

            if done:
                print('Reward: {} | Episode: {}/{}'.format(int(score), i, episodes))
                break

        score_list.append(score)

    plt.plot([i + 1 for i in range(0, episodes, 3)], score_list[::3])
    plt.show()
