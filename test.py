from __future__ import annotations
from typing import List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from loguru import logger
from copy import deepcopy

# Define the end-of-sequence token ID (you should set this according to your tokenizer)
eos_token_id = 2  # Example EOS token ID


class Node:
    """
    A class representing a node in the tree of thoughts.

    Attributes:
        sequence (List[int]): The sequence of tokens from the root to this node.
        children (List[Node]): The list of child nodes.
    """

    def __init__(self, sequence: List[int]):
        self.sequence: List[int] = sequence
        self.children: List[Node] = []

    def add_child(self, child_node: Node):
        """Adds a child node to the current node."""
        self.children.append(child_node)


class PolicyModel(nn.Module):
    """
    Policy model π_θ to generate sequences.

    Args:
        vocab_size (int): Vocabulary size.
        hidden_size (int): Hidden layer size.
        num_layers (int): Number of transformer layers.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super(PolicyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Forward pass of the policy model.

        Args:
            src (Tensor): Source sequence tensor (prompt), shape (S, N).
            tgt (Tensor): Target sequence tensor (continuation), shape (T, N).

        Returns:
            Tensor: Logits over the vocabulary, shape (T, N, V).
        """
        src_emb = self.embedding(src)  # (S, N, E)
        tgt_emb = self.embedding(tgt)  # (T, N, E)
        memory = self.transformer.encoder(src_emb)
        output = self.transformer.decoder(tgt_emb, memory)
        logits = self.fc_out(output)  # (T, N, V)
        return logits


class RewardModel(nn.Module):
    """
    Reward model R(s) to compute rewards for sequences.

    Args:
        vocab_size (int): Vocabulary size.
        hidden_size (int): Hidden layer size.
        num_layers (int): Number of transformer layers.
    """

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super(RewardModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, sequence: Tensor) -> Tensor:
        """
        Forward pass of the reward model.

        Args:
            sequence (Tensor): Sequence tensor, shape (S, N).

        Returns:
            Tensor: Scalar reward value, shape (N).
        """
        emb = self.embedding(sequence)  # (S, N, E)
        output = self.transformer(emb)  # (S, N, E)
        # Take the mean over the sequence length
        pooled_output = output.mean(dim=0)  # (N, E)
        reward = self.fc_out(pooled_output)  # (N, 1)
        return reward.squeeze(-1)  # (N)


def sample_sequence(
    model: PolicyModel, context: List[int], max_length: int, eos_token_id: int
) -> List[int]:
    """
    Samples a continuation from the model given the context.

    Args:
        model (PolicyModel): The policy model.
        context (List[int]): The context sequence (list of token ids).
        max_length (int): Maximum length of the continuation.
        eos_token_id (int): End-of-sequence token ID.

    Returns:
        List[int]: Sampled continuation tokens.
    """
    model.eval()
    generated = context.copy()
    with torch.no_grad():
        for _ in range(max_length):
            src = torch.tensor(context).unsqueeze(1)  # (S, 1)
            tgt_input = torch.tensor(generated).unsqueeze(1)  # (T, 1)
            logits = model(src, tgt_input)
            next_token_logits = logits[-1, 0, :]  # (V)
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            generated.append(next_token_id)
            if next_token_id == eos_token_id:
                break
    continuation = generated[len(context) :]
    return continuation


def compute_log_probability(
    model: PolicyModel, sequence: List[int], requires_grad: bool = False
) -> Tensor:
    """
    Computes the total log probability of the sequence under the model.

    Args:
        model (PolicyModel): The model (policy or reference).
        sequence (List[int]): The sequence of token ids.
        requires_grad (bool): Whether to compute gradients.

    Returns:
        Tensor: Total log probability of the sequence.
    """
    sequence_tensor = torch.tensor(sequence).unsqueeze(1)  # (S, 1)
    src = sequence_tensor[:-1, :]  # (S-1, 1)
    tgt = sequence_tensor[:-1, :]  # (S-1, 1)
    target_ids = sequence_tensor[1:, 0]  # (S-1)
    if not requires_grad:
        model.eval()
        with torch.no_grad():
            logits = model(src, tgt)  # (T, N, V)
            logits = logits.squeeze(1)  # (S-1, V)
            log_probs = torch.log_softmax(logits, dim=-1)  # (S-1, V)
            token_logprobs = log_probs[range(len(target_ids)), target_ids]  # (S-1)
            total_logprob = token_logprobs.sum()  # Scalar
    else:
        model.train()
        logits = model(src, tgt)
        logits = logits.squeeze(1)
        log_probs = torch.log_softmax(logits, dim=-1)
        token_logprobs = log_probs[range(len(target_ids)), target_ids]
        total_logprob = token_logprobs.sum()
    return total_logprob


def compute_reward(reward_model: RewardModel, sequence: List[int]) -> float:
    """
    Computes the reward of a sequence using the reward model.

    Args:
        reward_model (RewardModel): The reward model.
        sequence (List[int]): The sequence of token ids.

    Returns:
        float: Reward value.
    """
    reward_model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(sequence).unsqueeze(1)  # (S, 1)
        reward = reward_model(input_ids)  # (1)
        return reward.item()


def train_policy_model(
    policy_model: PolicyModel,
    reward_model: RewardModel,
    prompts: List[List[int]],
    vocab_size: int,
    eos_token_id: int,
    beta: float = 0.1,
    D: int = 3,
    B: int = 2,
    max_length: int = 10,
    learning_rate: float = 1e-4,
    T_max: int = 1000,
    update_reference_model_every: Optional[int] = None,
) -> None:
    """
    Trains the policy model using RLHF with DPO and Monte Carlo Tree of Thoughts.

    Args:
        policy_model (PolicyModel): The policy model to train.
        reward_model (RewardModel): The reward model.
        prompts (List[List[int]]): The training dataset of prompts.
        vocab_size (int): Vocabulary size.
        eos_token_id (int): End-of-sequence token id.
        beta (float, optional): Beta parameter for DPO loss. Defaults to 0.1.
        D (int, optional): Maximum depth of the tree. Defaults to 3.
        B (int, optional): Number of branches per node. Defaults to 2.
        max_length (int, optional): Maximum length of continuations. Defaults to 10.
        learning_rate (float, optional): Learning rate for optimizer. Defaults to 1e-4.
        T_max (int, optional): Maximum number of training iterations. Defaults to 1000.
        update_reference_model_every (Optional[int], optional): Number of iterations after which to update the reference model. If None, keep fixed.

    """
    optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)
    reference_model = deepcopy(policy_model)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    for t in range(1, T_max + 1):
        logger.info(f"Starting iteration {t}/{T_max}")
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {prompt_idx + 1}/{len(prompts)}")
            # Initialize tree with root node
            root_node = Node(sequence=prompt)
            frontier = [root_node]
            # Build the tree up to depth D
            for d in range(1, D + 1):
                logger.debug(f"Depth {d}/{D}")
                new_frontier = []
                for node in frontier:
                    for b in range(B):
                        continuation = sample_sequence(
                            policy_model, node.sequence, max_length, eos_token_id
                        )
                        child_sequence = node.sequence + continuation
                        child_node = Node(sequence=child_sequence)
                        node.add_child(child_node)
                        new_frontier.append(child_node)
                frontier = new_frontier
                if not frontier:
                    logger.debug("Frontier is empty. Breaking out of depth loop.")
                    break
            # Collect leaf nodes
            leaf_nodes = frontier
            if not leaf_nodes:
                logger.warning("No leaf nodes generated for this prompt.")
                continue
            # Evaluate leaf nodes
            rewards = []
            for node in leaf_nodes:
                sequence = node.sequence
                reward = compute_reward(reward_model, sequence)
                rewards.append((node, reward))
            # Rank sequences based on rewards
            rewards.sort(key=lambda x: x[1], reverse=True)
            ranked_nodes = [node for node, reward in rewards]
            # Create preference pairs
            preference_pairs = []
            M = len(ranked_nodes)
            for i in range(M - 1):
                for j in range(i + 1, M):
                    preferred_node = ranked_nodes[i]
                    unpreferred_node = ranked_nodes[j]
                    preference_pairs.append((preferred_node, unpreferred_node))
            # Compute DPO loss
            losses = []
            for preferred_node, unpreferred_node in preference_pairs:
                s_i = preferred_node.sequence
                s_j = unpreferred_node.sequence
                # Compute log probabilities
                policy_logprob_s_i = compute_log_probability(
                    policy_model, s_i, requires_grad=True
                )
                policy_logprob_s_j = compute_log_probability(
                    policy_model, s_j, requires_grad=True
                )
                ref_logprob_s_i = compute_log_probability(
                    reference_model, s_i, requires_grad=False
                )
                ref_logprob_s_j = compute_log_probability(
                    reference_model, s_j, requires_grad=False
                )
                # Compute log ratios
                policy_log_ratio = policy_logprob_s_i - policy_logprob_s_j  # tensor
                ref_log_ratio = ref_logprob_s_i - ref_logprob_s_j  # tensor
                # Compute loss
                loss = -torch.log(
                    torch.sigmoid(beta * (policy_log_ratio - ref_log_ratio))
                )
                losses.append(loss)
            if losses:
                total_loss = torch.stack(losses).mean()
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                logger.info(f"Loss: {total_loss.item():.4f}")
            else:
                logger.info("No preference pairs generated.")
        # Optionally update the reference model
        if update_reference_model_every is not None and t % update_reference_model_every == 0:
            logger.info("Updating reference model.")
            reference_model = deepcopy(policy_model)
            reference_model.eval()
            for param in reference_model.parameters():
                param.requires_grad = False


# Example usage (you need to define your own data and models)
if __name__ == "__main__":
    # Initialize models with example hyperparameters
    vocab_size = 5000
    hidden_size = 256
    num_layers = 2

    policy_model = PolicyModel(vocab_size, hidden_size, num_layers)
    reward_model = RewardModel(vocab_size, hidden_size, num_layers)

    # Example prompts (list of token IDs)
    prompts = [
        [1, 5, 20],
        [1, 15, 30],
        [1, 25, 40],
    ]

    # Training parameters
    beta = 0.1
    D = 3
    B = 2
    max_length = 10
    learning_rate = 1e-4
    T_max = 10
    update_reference_model_every = 5

    # Train the policy model
    train_policy_model(
        policy_model,
        reward_model,
        prompts,
        vocab_size,
        eos_token_id,
        beta=beta,
        D=D,
        B=B,
        max_length=max_length,
        learning_rate=learning_rate,
        T_max=T_max,
        update_reference_model_every=update_reference_model_every,
    )
