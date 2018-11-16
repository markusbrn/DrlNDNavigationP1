# 1. Introduction
The goal of this project is to develop and test a (Deep-Q-) Reinforcement Learning Algorithm that is able to successfully steer a yellow banana collecting robot in a (bounded) environment where yellow and blue bananas are spawned randomly.
The goal of the robot is to collect as many yellow bananas as possible while avoiding to collect the blue bananas. For each successfully collected banana the robot receives a reward of +1 while each collected blue banana gives a reward of -1. The task is episodic which means that the "playing field" is reset after a certain amount of robot actions. The task is considered to be solved if the robot is able to achieve an average reward of +13 over 100 consecutive episodes.
To steer the robot, four discrete actions are availabe (move forward, move backward, turn left and turn right). The state space consists of the robot velocity as well as a ray based of the objects that are located in the robot's forward field of view.

# 2. Installation
To install the project, the project repository has to be cloned with the command "git clone https://github.com/markusbrn/DrlNDNavigationP1.git". Next please set up your python environment correctly. You can find the instructions how to do so here (https://github.com/udacity/deep-reinforcement-learning#dependencies).
Finally you have to download the environment that is used to simulate the robot world. Please follow the links below (depending on the operating system you are using):

- Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
- Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
- Windows (32-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip
- Windows (64-bit): https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip

The unzipped environment files have to be copied in the project repository folder.

#3. Training and Testing the ML Agent
In order to train and test the agent please start and run the jupyter notebook file from the project repository.