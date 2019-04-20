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
