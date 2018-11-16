# 1. Introduction
This document describes the training of a yellow banana collecting robot that is operating in a (bounded) world where yellow and blue bananas are spawned randomly. For each collected yellow banana the robot receives a reward of +1; each blue banana gives a reward of -1. The training task is episodic - this means that the environment is reset after the robot has performed 1000 consecutive actions.
The robot can select from 4 actions (move forward, move backwards, turn left, turn right) and observes a total of states (robot velovity and ray based measurement of objects in the forward fild of view).
The training task is considered solved when the robot is able to achieve an average score of +13 for 100 consecutive episodes. The project contains the following files with the following functions:

- Navigation.ipynb: Jupyter Notebook to load the requried python modules, perform the training process and display the results
- dqn_agent.py: contains the classes Agent and ReplayBuffer. In the ReplayBuffer (state, action, reward, next_state)-tuples are recorded during robot operation and are stored to be used in the training process.
  The Agent class defines (among others) the functions to select the next action for the robot and to learn the Q-table that is required in the process of finding the best robot action. It is explained in greater detail in the next section of this document.
- model.py: here the architecture of the neural networks that are used to predict the Q-values for the state-action combinations are defined
- checkpoint.pth: Snapshot of a trained network that can be used to select the next action for a given state value.

# 2. Learning Algorithm
In this section the Learning Algorithm that is defined in dqn_agent.py is described in greater detail. The goal of Deep-Q-Learning is to learn a so called Q-Table which gives the expected future (discounted) reward for each state-action pair that can be reached. The formula how to compute the so called Q-value for a distinct state-action pair is shown in the following:

As can be seen from the formula the Q-value for a state-action pair (s|a) can be computed recursively as the immediate reward r added to the discounted future reward when following the optimal policy from state s' (which is reached by performing action a in state s). To account for the stochasticity of the process a weighted sum over all states that can be reached by performing action a in state s is computed; the weighting is performed according to the possibility to reach state s' by performing action a. The discount factor gamma (<1) is used to motivate the agent to accumulate reward as quickly as possible.
If this process is performed for all possible state action combinations the already mentioned Q-table can be computed and can be used to pick the optimal action from a certain state by selecting the action with the highest Q-value.
The Q-table is not known at the beginning of the training process but instead has to be learned over the training process. There exist a variety of training algorithms to do so - one that is well known is called SARSAmax. It can be seen in the following:


# 3. Model Architecture

# 4. Training Progress

# 5. Ideas for Future Work