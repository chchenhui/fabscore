
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as e:
    raise RuntimeError("PyTorch is required for dqn_agent.py. Please install torch.") from e

@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    lr: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 64
    buffer_size: int = 10000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_steps: int = 5000
    target_update_freq: int = 250
    seed: int = 42

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, s2, d = zip(*[self.buffer[i] for i in idx])
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32),
                np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        self.q = QNetwork(cfg.state_dim, cfg.action_dim)
        self.q_target = QNetwork(cfg.state_dim, cfg.action_dim)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size)

        self.epsilon = cfg.epsilon_start
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        self.step_count += 1
        # epsilon decay
        eps_decay = max(0.0, min(1.0, 1.0 - self.step_count / max(1, self.cfg.epsilon_decay_steps)))
        self.epsilon = self.cfg.epsilon_end + (self.cfg.epsilon_start - self.cfg.epsilon_end) * eps_decay

        if random.random() < self.epsilon:
            return random.randrange(self.cfg.action_dim)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            qv = self.q(s)
            return int(torch.argmax(qv, dim=1).item())

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return None

        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)
        s = torch.tensor(s)
        a = torch.tensor(a)
        r = torch.tensor(r)
        s2 = torch.tensor(s2)
        d = torch.tensor(d)

        q_pred = self.q(s).gather(1, a.view(-1,1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1)[0]
            target = r + self.cfg.gamma * (1 - d) * q_next

        loss = (q_pred - target).pow(2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        if self.step_count % self.cfg.target_update_freq == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def remember(self, s, a, r, s2, d):
        self.buffer.push(s, a, r, s2, d)

    def save(self, path: str):
        torch.save(self.q.state_dict(), path)

    def load(self, path: str):
        self.q.load_state_dict(torch.load(path))
        self.q_target.load_state_dict(self.q.state_dict())
