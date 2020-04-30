# Project 1: Cart-Pole

<img src=cart-pole/cart_pole.gif width="400">

### Introduction

- In this task we have to balance a rod on top of a cart. Number of action spaces is 2. Action space is discrete here.
- **`0`** - move cart to the left
- **`1`** - move cart to the right

- I solved this problem using DQN in around 60 episodes. Following is a graph of score vs episodes.

<img src=cart-pole/cart_pole.png width="400">

# Project 2: Mountain-Car

<img src=mountain-car/mountain_car.gif width="400">

### Introduction

- In this task we have to teach the car to reach at the goal position which is at the top of mountain. Number of action spaces is 3. Action space is descrete in this environment.
- **`0`** - move car to left
- **`1`** - do nothing
- **`2`** - move car to right

- I solved this problem using DQN in around 15 episodes. Following is a graph of score vs episodes.

<img src=mountain-car/mountain_car.png width="400">

# Project 3: Pendulam

<img src=pendulam/pendulam.gif width="400">

### Introduction

- In this task we have to balance the pendulam upside down. Number of action spaces is 1 which is torque applied on the joint. Action space is continuous here.
- **`0`** - torque [-2, 2]

- I solved this problem using DDPG in around 100 episodes. Following is a graph of score vs episodes.

<img src=pendulam/pendulam.png width="400">

# Project 4: Lunar-Lander

<img src=lunar-lander/discrete/images/training/after_training.gif width="400">

- The task is to land the space-ship between the flags smoothly. The ship has 3 throttles in it. One throttle points downward and other 2 points in the left and right direction. With the help of these, you have to control the Ship. There are 2 version for this task. One is discrete version which has discrete action space and other is continuous which has continuous action space.

- In order to solve the episode you have to get a reward of +200 for 100 consecutive episodes. I solved both the version under 400 episodes.

### Discrete Version Plot
<img src=lunar-lander/discrete/lunar_lander.png width="400">

### Continuous Version Plot
<img src=lunar-lander/continuous/lunar_lander.png width="400">

# Project 5: Bipedal-Walker

- BipedalWalker has 2 legs. Each leg has 2 joints. You have to teach the Bipedal-walker to walk by applying the torque on these joints. You can apply the torque in the range of (-1, 1). Positive reward is given for moving forward and small negative reward is given on applying torque on the motors.

### Smooth Terrain

- In the beginning, AI is behaving very randomly. It does not know how to control and balance the legs.

<img src=bipedal-walker/training/1.gif width="400">

- After 300 episodes, it learns to crawl on one knee and one leg. This AI is playing safe now because if it tumbles then it gets -100 reward.

<img src=bipedal-walker/training/2.gif width="400">

- After 500 episodes it started to balance on both of the legs. But It still needs to learn how to walk properly.

<img src=bipedal-walker/training/3.gif width="400">

- After 600 episodes, it learns to maximize the rewards. It is walking in some different style. After all, it’s an AI not a Human. This is just one of the way to walk in order to get maximum reward. If I train it again, it might learn some other optimal way to walk.

<img src=bipedal-walker/training/4.gif width="400">

### Hardcore Terrain

- I saved my weight from the previous training on simple terrain and resumed my training on the hardcore terrain. I did it because the agent already knew how to walk on simple terrain and now it needs to learn how to cross obstacles while walking.

<img src=bipedal-walker/training/5.gif width="400">
