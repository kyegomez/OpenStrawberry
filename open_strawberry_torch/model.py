import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from loguru import logger
from typing import List, Tuple

# Set up logging
logger.add("training.log", rotation="500 MB")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerPolicyNetwork(nn.Module):
    """
    Transformer-based Policy Network that outputs action probabilities given a state sequence.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerPolicyNetwork, self).__init__()
        self.model_type = "Transformer"

        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers
        )
        self.fc_out = nn.Linear(dim_feedforward, action_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the policy network.

        Args:
            src (torch.Tensor): Input tensor of shape (sequence_length, batch_size, input_dim).

        Returns:
            torch.Tensor: Action probabilities of shape (batch_size, action_dim).
        """
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        # Take the output from the last time step
        output = output[-1, :, :]
        action_logits = self.fc_out(output)
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs


class TransformerValueNetwork(nn.Module):
    """
    Transformer-based Value Network that estimates the value of a given state sequence.
    """

    def __init__(
        self,
        input_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerValueNetwork, self).__init__()
        self.model_type = "Transformer"

        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers
        )
        self.fc_out = nn.Linear(dim_feedforward, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the value network.

        Args:
            src (torch.Tensor): Input tensor of shape (sequence_length, batch_size, input_dim).

        Returns:
            torch.Tensor: State value of shape (batch_size, 1).
        """
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        # Take the output from the last time step
        output = output[-1, :, :]
        state_value = self.fc_out(output)
        return state_value


class TransformerRewardModel(nn.Module):
    """
    Transformer-based Reward Model that assigns rewards to thought branches.
    """

    def __init__(
        self,
        input_dim: int,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super(TransformerRewardModel, self).__init__()
        self.model_type = "Transformer"

        self.embedding = nn.Linear(input_dim, dim_feedforward)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=dim_feedforward, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers
        )
        self.fc_out = nn.Linear(dim_feedforward, 1)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the reward model.

        Args:
            src (torch.Tensor): Input tensor of shape (sequence_length, batch_size, input_dim).

        Returns:
            torch.Tensor: Reward estimate of shape (batch_size, 1).
        """
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        # Take the output from the last time step
        output = output[-1, :, :]
        reward = self.fc_out(output)
        return reward


class ThoughtTree:
    """
    Class representing a tree of thoughts.
    """

    def __init__(self, root_state: torch.Tensor):
        self.root = {"state": root_state, "children": [], "reward": 0}

    def add_child(
        self, parent: dict, child_state: torch.Tensor, reward: float
    ):
        child = {
            "state": child_state,
            "children": [],
            "reward": reward,
        }
        parent["children"].append(child)
        return child


def monte_carlo_rollout(
    policy_net: TransformerPolicyNetwork,
    state_sequence: torch.Tensor,
    depth: int,
    max_depth: int,
    sequence_length: int,
) -> List[Tuple[torch.Tensor, float]]:
    """
    Perform a Monte Carlo rollout to simulate future thoughts.

    Args:
        policy_net (TransformerPolicyNetwork): The policy network.
        state_sequence (torch.Tensor): The current state sequence.
        depth (int): Current depth in the thought tree.
        max_depth (int): Maximum depth for rollouts.
        sequence_length (int): The length of the input sequence.

    Returns:
        List[Tuple[torch.Tensor, float]]: A list of (state_sequence, reward) tuples.
    """
    trajectory = []
    current_sequence = state_sequence.clone()
    for _ in range(depth, max_depth):
        action_probs = policy_net(current_sequence)
        m = Categorical(action_probs)
        action = m.sample()
        next_state = transition(current_sequence[-1], action)
        # Update the sequence by appending the new state
        next_sequence = torch.cat(
            [current_sequence, next_state.unsqueeze(0)], dim=0
        )
        # Ensure the sequence length does not exceed the maximum
        if next_sequence.size(0) > sequence_length:
            next_sequence = next_sequence[1:, :]
        reward = reward_function(next_state)
        trajectory.append((next_sequence, reward))
        current_sequence = next_sequence
    return trajectory


def transition(
    state: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    """
    State transition function (placeholder).

    Args:
        state (torch.Tensor): Current state tensor.
        action (torch.Tensor): Action tensor.

    Returns:
        torch.Tensor: Next state tensor.
    """
    # Implement your state transition logic here
    next_state = state + action.float()  # Simplified example
    return next_state


def reward_function(state: torch.Tensor) -> float:
    """
    Reward function (placeholder).

    Args:
        state (torch.Tensor): State tensor.

    Returns:
        float: Reward value.
    """
    # Implement your reward logic here
    reward = -torch.sum(state**2).item()  # Simplified example
    return reward


def train(
    policy_net: TransformerPolicyNetwork,
    value_net: TransformerValueNetwork,
    reward_model: TransformerRewardModel,
    num_iterations: int = 1000,
    episodes_per_iteration: int = 10,
    max_depth: int = 5,
    sequence_length: int = 10,
    gamma: float = 0.99,
    clip_epsilon: float = 0.2,
    policy_lr: float = 1e-4,
    value_lr: float = 1e-3,
):
    """
    Train the policy and value networks using PPO.

    Args:
        policy_net (TransformerPolicyNetwork): The policy network.
        value_net (TransformerValueNetwork): The value network.
        reward_model (TransformerRewardModel): The reward model.
        num_iterations (int): Number of training iterations.
        episodes_per_iteration (int): Episodes per iteration.
        max_depth (int): Maximum depth for Monte Carlo rollouts.
        sequence_length (int): Maximum sequence length for the transformer.
        gamma (float): Discount factor.
        clip_epsilon (float): Clipping epsilon for PPO.
        policy_lr (float): Learning rate for the policy optimizer.
        value_lr (float): Learning rate for the value optimizer.
    """
    policy_optimizer = optim.Adam(
        policy_net.parameters(), lr=policy_lr
    )
    value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)

    for iteration in range(num_iterations):
        logger.info(
            f"Starting iteration {iteration + 1}/{num_iterations}"
        )
        memory = []

        for episode in range(episodes_per_iteration):
            logger.debug(
                f"Starting episode {episode + 1}/{episodes_per_iteration}"
            )
            # Initialize state sequence with zeros
            state = torch.zeros(policy_net.embedding.in_features).to(
                device
            )
            state_sequence = state.unsqueeze(
                0
            )  # Shape: (1, input_dim)
            thought_tree = ThoughtTree(state_sequence)
            trajectory = []

            # Generate thought branches
            for depth in range(max_depth):
                # Expand dimensions to match (sequence_length, batch_size, input_dim)
                src = state_sequence.unsqueeze(
                    1
                )  # Shape: (sequence_length, 1, input_dim)
                action_probs = policy_net(src)
                m = Categorical(action_probs)
                actions = m.sample((5,))  # Generate multiple branches
                rewards = []

                for action in actions:
                    next_state = transition(
                        state_sequence[-1], action
                    )
                    # Update the sequence by appending the new state
                    next_sequence = torch.cat(
                        [state_sequence, next_state.unsqueeze(0)],
                        dim=0,
                    )
                    # Ensure the sequence length does not exceed the maximum
                    if next_sequence.size(0) > sequence_length:
                        next_sequence = next_sequence[1:, :]
                    rollout = monte_carlo_rollout(
                        policy_net,
                        next_sequence,
                        depth + 1,
                        max_depth,
                        sequence_length,
                    )
                    total_reward = sum([r for _, r in rollout])
                    # Expand dimensions for reward model input
                    reward_input = next_sequence.unsqueeze(1)
                    reward_estimate = reward_model(reward_input)
                    reward = reward_estimate.item() + total_reward
                    rewards.append(reward)

                    # Update thought tree
                    thought_tree.add_child(
                        thought_tree.root, next_sequence, reward
                    )

                # Select the best action based on rewards
                best_action_index = (
                    torch.tensor(rewards).argmax().item()
                )
                best_action = actions[best_action_index]
                best_reward = rewards[best_action_index]

                # Log the selected action and reward
                logger.debug(
                    f"Selected action {best_action.item()} with reward {best_reward}"
                )

                # Store the experience
                trajectory.append(
                    (state_sequence.clone(), best_action, best_reward)
                )

                # Move to the next state sequence
                next_state = transition(
                    state_sequence[-1], best_action
                )
                state_sequence = torch.cat(
                    [state_sequence, next_state.unsqueeze(0)], dim=0
                )
                if state_sequence.size(0) > sequence_length:
                    state_sequence = state_sequence[1:, :]

            # Compute returns and advantages
            returns = []
            advantages = []
            Gt = 0
            for state_seq_t, action_t, reward_t in reversed(
                trajectory
            ):
                Gt = reward_t + gamma * Gt
                returns.insert(0, Gt)
                # Expand dimensions for value network input
                value_input = state_seq_t.unsqueeze(1)
                state_value = value_net(value_input)
                advantage = Gt - state_value.item()
                advantages.insert(0, advantage)

            # Normalize advantages
            advantages_tensor = torch.tensor(
                advantages, dtype=torch.float32
            ).to(device)
            advantages_tensor = (
                advantages_tensor - advantages_tensor.mean()
            ) / (advantages_tensor.std() + 1e-8)

            # Update policy network using PPO
            for i, (state_seq_t, action_t, _) in enumerate(
                trajectory
            ):
                # Expand dimensions to match (sequence_length, batch_size, input_dim)
                src = state_seq_t.unsqueeze(1)
                action_probs = policy_net(src)
                m = Categorical(action_probs)
                log_prob = m.log_prob(action_t)
                old_log_prob = log_prob.detach()
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantages_tensor[i]
                surr2 = (
                    torch.clamp(
                        ratio, 1 - clip_epsilon, 1 + clip_epsilon
                    )
                    * advantages_tensor[i]
                )
                policy_loss = -torch.min(surr1, surr2)

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()

                # Log the policy loss
                logger.debug(
                    f"Policy loss at step {i}: {policy_loss.item()}"
                )

            # Update value network
            returns_tensor = (
                torch.tensor(returns, dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
            )
            # Prepare inputs for the value network
            value_inputs = torch.stack(
                [s for s, _, _ in trajectory]
            ).transpose(0, 1)
            value_inputs = value_inputs.to(device)
            values = value_net(value_inputs)
            value_loss = nn.MSELoss()(values, returns_tensor)

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # Log the value loss
            logger.debug(f"Value loss: {value_loss.item()}")

        logger.info(
            f"Completed iteration {iteration + 1}/{num_iterations}"
        )


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 10  # Dimension of the input state
    action_dim = 4  # Number of possible actions
    num_iterations = 10
    episodes_per_iteration = 5
    sequence_length = (
        10  # Maximum sequence length for the transformer
    )

    # Initialize networks
    policy_net = TransformerPolicyNetwork(input_dim, action_dim).to(
        device
    )
    value_net = TransformerValueNetwork(input_dim).to(device)
    reward_model = TransformerRewardModel(input_dim).to(device)

    # Start training
    train(
        policy_net,
        value_net,
        reward_model,
        num_iterations=num_iterations,
        episodes_per_iteration=episodes_per_iteration,
        sequence_length=sequence_length,
    )
