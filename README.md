# ReinforceAI

The ReinforceAI is a reinforcement learning model designed to solve tasks using the REINFORCE algorithm. The primary environment it targets is the InvertedPendulum-v4 from OpenAI Gym. The model aims to balance an inverted pendulum by applying forces to its base.

## Overview

- **Environment**: The AI is trained on the InvertedPendulum-v4 environment from OpenAI Gym. This environment simulates an inverted pendulum that the agent must balance by applying forces.
- **Model**: The core of the AI is the REINFORCE algorithm, a foundational policy gradient method. The model learns a policy that maximizes the expected cumulative reward.
- **Policy Network**: The policy is represented using a neural network that outputs the mean and standard deviation of a normal distribution. Actions are sampled from this distribution.
- **Training**: The agent interacts with the environment, collecting trajectories of states, actions, and rewards. After each episode, the policy is updated using the collected data.

## Files

- `main.py`: The main script that initializes the environment, trains the agent, and evaluates its performance.
- `policy_network.py`: Contains the neural network architecture that represents the policy. It outputs the parameters of a normal distribution from which actions are sampled.
- `reinforce.py` : Contains the implementation of the REINFORCE algorithm, including the policy update mechanism.

## Getting Started

- Clone the repository.
- Install the required libraries and dependencies.
- Run the main.py script to train and evaluate the agent.

## Training Process

The training process involves the following steps:

- **Interaction**: The agent interacts with the environment, sampling actions from its policy and observing the resulting states and rewards.
- **Policy Update**: After each episode, the agent updates its policy using the REINFORCE algorithm. The goal is to increase the probability of actions that resulted in high rewards and decrease the probability of actions that resulted in low rewards.
- **Evaluation**: The agent's performance can be evaluated by running it in the environment and observing the cumulative reward.
