from __future__ import annotations

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """"Parameterized Policy Network"""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation of a normal distribution
        from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation spaces
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        # Arbitrary values, can be changed
        hidden_layer1 = 16
        hidden_layer2 = 32

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_layer1),
            nn.Tanh(),
            nn.Linear(hidden_layer1, hidden_layer2),
            nn.Tanh()
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_layer2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_layer2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation of a normal distribution
        from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
