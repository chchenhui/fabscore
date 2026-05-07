"""
Implementation of the Feasibility-Guided Fair Adaptive Reinforcement Learning (FCAF-RL) algorithm.

This module defines PyTorch models and training routines for offline reinforcement
learning with safety and fairness constraints. It includes:

    - DiffusionModel: a simple feed-forward network used to generate synthetic
      state-action pairs within a feasible region.
    - QNetwork: a neural network approximator for Q-functions.
    - FCAFTrainer: encapsulates training of conservative Q-functions with
      fairness penalties and adaptive policy switching.

The algorithm follows these steps:
    1. Augment the dataset using a diffusion model to sample feasible
       transitions (states and actions) that satisfy clinician-defined
       constraints.
    2. Train multiple Q-networks with different fairness weights using the
       conservative Bellman objective with an equalised-odds penalty.
    3. Deploy the policy by selecting the Q-network whose realised fairness
       disparity over a sliding window is below a user-specified threshold.

Note: This implementation is simplified for demonstration purposes. In
practice, the diffusion model and Q-networks would require more complex
architectures and careful tuning.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Define a simple dataset wrapper
class TransitionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, action_to_idx: Dict[str, int]):
        # filter out last week for next-state mapping
        self.states = df[[
            "age", "sex", "race", "comorbidity_count", "prior_ed_visits",
            "mental_health_flag", "substance_use_flag", "social_need_score",
        ]].values.astype(np.float32)
        # convert categorical variables to numeric codes
        # sex: Female=0, Male=1; race: White=0, Black=1, Hispanic=2, Other=3
        sex_map = {"Female": 0.0, "Male": 1.0}
        race_map = {"White": 0.0, "Black": 1.0, "Hispanic": 2.0, "Other": 3.0}
        self.states[:, 1] = [sex_map[x] for x in df["sex"]]
        self.states[:, 2] = [race_map[x] for x in df["race"]]

        self.actions = torch.tensor([action_to_idx[a] for a in df["intervention"]], dtype=torch.long)
        self.rewards = torch.tensor(1.0 - df["acute_event"].values.astype(np.float32))
        # for simplicity we treat next_state equal to current state (stationary)
        self.next_states = self.states
        # mask for sensitive attributes: sex and race indices in state vector
        self.protected_indices = [1, 2]

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
        return (
            torch.tensor(self.states[idx], dtype=torch.float32),
            self.actions[idx],
            self.rewards[idx],
            torch.tensor(self.next_states[idx], dtype=torch.float32),
        )


class DiffusionModel(nn.Module):
    """Simplified diffusion model for generating feasible state-action pairs."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim + action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sample(self, states: torch.Tensor, num_samples: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic state-action pairs given current states.

        Args:
            states: batch of state vectors.
            num_samples: number of samples per input state.

        Returns:
            synthetic_states, synthetic_actions: tensors of shape
            (batch_size * num_samples, state_dim) and (batch_size * num_samples, action_dim).
        """
        batch_size, state_dim = states.shape
        device = states.device
        states_rep = states.repeat_interleave(num_samples, dim=0)
        # sample random actions uniformly
        actions_onehot = F.one_hot(
            torch.randint(0, 9, (batch_size * num_samples,), device=device), num_classes=9
        ).float()
        inputs = torch.cat([states_rep, actions_onehot], dim=1)
        output = self.forward(inputs)
        new_states = output[:, :state_dim]
        new_actions = output[:, state_dim:]
        # clamp new_states to feasible range (0–80 for age, etc.)
        new_states = torch.clamp(new_states, 0.0, 80.0)
        # convert logits to one-hot for actions
        new_actions = F.one_hot(new_actions.argmax(dim=1), num_classes=9).float()
        return new_states, new_actions


class QNetwork(nn.Module):
    """Simple multilayer perceptron Q-network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor, actions_onehot: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions_onehot], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q.squeeze(-1)


@dataclass
class FCAFConfig:
    state_dim: int
    action_dim: int = 9
    discount: float = 0.99
    alpha: float = 0.1  # conservative regularisation weight
    fairness_weights: List[float] = None
    batch_size: int = 256
    q_lr: float = 1e-3
    diff_lr: float = 1e-4
    num_epochs: int = 10
    device: str = "cpu"

    def __post_init__(self):
        if self.fairness_weights is None:
            self.fairness_weights = [0.0, 0.5, 1.0, 2.0]


class FCAFTrainer:
    """Trainer for the FCAF-RL algorithm."""

    def __init__(self, config: FCAFConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.diff_model = DiffusionModel(config.state_dim, config.action_dim).to(self.device)
        self.q_networks = [
            QNetwork(config.state_dim, config.action_dim).to(self.device) for _ in config.fairness_weights
        ]
        self.diff_opt = torch.optim.Adam(self.diff_model.parameters(), lr=config.diff_lr)
        self.q_opts = [torch.optim.Adam(q.parameters(), lr=config.q_lr) for q in self.q_networks]

    def _fairness_penalty(self, states: torch.Tensor, actions_onehot: torch.Tensor, rewards: torch.Tensor) -> float:
        """Compute equalised-odds penalty on the batch.

        We approximate true positive and false positive rates by grouping samples by
        protected attributes and comparing reward distributions (assuming reward=1
        for non-acute event and 0 for acute event). This is a simplified
        implementation.
        """
        # Protected group membership: group 0 if sex==0 & race==0 (reference), group 1 otherwise
        group_mask = (states[:, 1] > 0) | (states[:, 2] > 0)
        tpr_group = rewards[group_mask].mean() if group_mask.any() else torch.tensor(0.0, device=self.device)
        tpr_ref = rewards[~group_mask].mean() if (~group_mask).any() else torch.tensor(0.0, device=self.device)
        # approximate false positive rate using inverted rewards
        fpr_group = (1 - rewards[group_mask]).mean() if group_mask.any() else torch.tensor(0.0, device=self.device)
        fpr_ref = (1 - rewards[~group_mask]).mean() if (~group_mask).any() else torch.tensor(0.0, device=self.device)
        penalty = (tpr_group - tpr_ref) ** 2 + (fpr_group - fpr_ref) ** 2
        return penalty.item()

    def train(self, dataset: TransitionDataset) -> None:
        """Train diffusion model and Q-networks on the provided dataset."""
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        # Train diffusion model to generate synthetic samples (simplified objective)
        for epoch in range(self.config.num_epochs):
            for states, actions, rewards, next_states in loader:
                states = states.to(self.device)
                # one-hot encode actions
                actions_onehot = F.one_hot(actions, num_classes=self.config.action_dim).float().to(self.device)
                # train diffusion model to reconstruct inputs (autoencoder-style)
                inputs = torch.cat([states, actions_onehot], dim=1)
                recon = self.diff_model(inputs)
                target = inputs
                loss = F.mse_loss(recon, target)
                self.diff_opt.zero_grad()
                loss.backward()
                self.diff_opt.step()
        # Augment data using the diffusion model
        with torch.no_grad():
            all_states = torch.tensor(dataset.states, dtype=torch.float32).to(self.device)
            synthetic_states, synthetic_actions = self.diff_model.sample(all_states, num_samples=1)
        # Create augmented dataset combining real and synthetic transitions
        aug_states = torch.cat([torch.tensor(dataset.states, dtype=torch.float32), synthetic_states.cpu()], dim=0)
        aug_actions = torch.cat([
            F.one_hot(dataset.actions, num_classes=self.config.action_dim).float(), synthetic_actions.cpu()
        ], dim=0)
        aug_rewards = torch.cat([
            torch.tensor(dataset.rewards, dtype=torch.float32),
            torch.tensor(dataset.rewards, dtype=torch.float32)  # reuse rewards for synthetic for simplicity
        ], dim=0)
        aug_dataset = list(zip(aug_states, aug_actions, aug_rewards))
        # Train Q-networks with different fairness weights
        for q_net, q_opt, lam in zip(self.q_networks, self.q_opts, self.config.fairness_weights):
            for epoch in range(self.config.num_epochs):
                random.shuffle(aug_dataset)
                for batch_start in range(0, len(aug_dataset), self.config.batch_size):
                    batch = aug_dataset[batch_start : batch_start + self.config.batch_size]
                    if not batch:
                        continue
                    states_b, actions_b, rewards_b = zip(*batch)
                    states_b = torch.stack(list(states_b)).to(self.device)
                    actions_b = torch.stack(list(actions_b)).to(self.device)
                    rewards_b = torch.stack(list(rewards_b)).to(self.device)
                    # Q-value targets using fixed policy (no target network for simplicity)
                    q_values = q_net(states_b, actions_b)
                    target_q = rewards_b  # single-step reward; no next-state bootstrap
                    bellman_loss = F.mse_loss(q_values, target_q)
                    conservative_loss = self.config.alpha * q_values.mean()
                    fairness_penalty = lam * self._fairness_penalty(states_b, actions_b, rewards_b)
                    loss = bellman_loss + conservative_loss + fairness_penalty
                    q_opt.zero_grad()
                    loss.backward()
                    q_opt.step()

    def select_action(self, state: np.ndarray, fairness_threshold: float) -> int:
        """Select an action for a single state based on fairness threshold.

        Args:
            state: array representing the patient state.
            fairness_threshold: maximum allowable disparity (in TPR/FPR differences).

        Returns:
            action index (0–8) corresponding to the chosen intervention.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        # initial realised disparity set to zero for demonstration
        current_disparity = 0.0
        # iterate through Q-networks in increasing fairness weight until disparity below threshold
        for q_net, lam in sorted(zip(self.q_networks, self.config.fairness_weights), key=lambda x: x[1]):
            with torch.no_grad():
                # evaluate fairness penalty on singleton state; approximate as zero
                if current_disparity <= fairness_threshold:
                    # compute Q-values for all actions
                    actions_onehot = torch.eye(self.config.action_dim).to(self.device)
                    states_rep = state_tensor.repeat(self.config.action_dim, 1)
                    q_vals = q_net(states_rep, actions_onehot)
                    return int(torch.argmax(q_vals).item())
                else:
                    continue
        # default to action 0 if threshold never satisfied
        return 0
