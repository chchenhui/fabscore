
import random
from collections import deque, defaultdict
from typing import Dict, Tuple, List
import math

# Tiny MLP (no external deps)
class MLP:
    def __init__(self, in_dim, hidden=256, out_dim=16, seed=0):
        self.rng = random.Random(seed)
        def init(m,n):  # simple Xavier
            scale = math.sqrt(2.0/(m+n))
            return [[self.rng.uniform(-scale, scale) for _ in range(n)] for _ in range(m)]
        self.W1 = init(in_dim, hidden); self.b1=[0.0]*hidden
        self.W2 = init(hidden, hidden); self.b2=[0.0]*hidden
        self.W3 = init(hidden, out_dim); self.b3=[0.0]*out_dim
        self.lr = 1e-3

    def _relu(self, v): return [max(0.0, x) for x in v]
    def _matmul(self, A, x): return [sum(a*x_i for a,x_i in zip(row, x)) for row in A]
    def _add(self, v, b): return [x+y for x,y in zip(v,b)]

    def forward(self, x):
        h1 = self._relu(self._add(self._matmul(self.W1, x), self.b1))
        h2 = self._relu(self._add(self._matmul(self.W2, h1), self.b2))
        y  = self._add(self._matmul(self.W3, h2), self.b3)
        return y

    def train_step_mse(self, X, Y):
        # Very crude SGD with finite-diff gradient (just for runnable demo; replace with PyTorch in real use)
        eps = 1e-4
        for i in range(len(X)):
            x, y_true = X[i], Y[i]
            y_pred = self.forward(x)
            # compute gradients numerically (expensive but OK for tiny demos)
            def loss(weights=None):
                return sum((yp - yt)**2 for yp,yt in zip(self.forward(x), y_true)) / len(y_true)
            L0 = loss()
            # skip actual updates to keep runtime short and code compact
            # (intended as skeleton; plug a real optimizer in practice)
            pass

# Simple featurizer
def obs_to_vec(obs) -> List[float]:
    v = []
    street_map = {"pre":0, "flop":1, "turn":2, "river":3, "showdown":4}
    v.append(street_map.get(obs["street"],0))
    v.append(obs["to_act"])
    v.append(obs["pot"]/200.0)  # normalize
    v.append(obs["equity"])     # Monte Carlo equity
    # bucket encoding (hash to int)
    v.append(hash(obs["bucket"])%997/997.0)
    # recent history length
    v.append(len(obs["history"])/10.0)
    return v

class DeepCFR:
    def __init__(self, seed=0):
        self.regret_net = MLP(in_dim=6, hidden=128, out_dim=3, seed=seed)
        self.policy_net = MLP(in_dim=6, hidden=128, out_dim=3, seed=seed+1)
        self.regret_mem = deque(maxlen=20000)
        self.policy_mem = deque(maxlen=20000)
        self.actions = ["check/call","bet/raise","fold"]

    def regret_matching_from_net(self, obs):
        x = obs_to_vec(obs)
        r = self.regret_net.forward(x)
        pos = [max(0.0, ri) for ri in r]
        s = sum(pos)
        if s<=1e-9:
            return [1/3,1/3,1/3]
        return [ri/s for ri in pos]

    def collect_traversals(self, env, n=1000):
        for _ in range(n):
            obs = env.observe()
            pi = self.regret_matching_from_net(obs)
            # placeholder counterfactual regret target (random) for demo purposes
            self.regret_mem.append((obs_to_vec(obs), [random.uniform(0,1) for _ in range(3)]))
            self.policy_mem.append((obs_to_vec(obs), pi))

    def train(self, steps=100):
        # skeleton: call dummy train steps
        Xr, Yr = zip(*list(self.regret_mem)[-min(512, len(self.regret_mem)):] or [([0]*6,[0,0,0])])
        Xp, Yp = zip(*list(self.policy_mem)[-min(512, len(self.policy_mem)):] or [([0]*6,[1/3,1/3,1/3])])
        self.regret_net.train_step_mse(list(Xr), list(Yr))
        self.policy_net.train_step_mse(list(Xp), list(Yp))

    def action_probs(self, obs):
        # use policy net output as logits -> softmax
        import math
        x = obs_to_vec(obs)
        y = self.policy_net.forward(x)
        m = max(y); ex = [math.exp(z-m) for z in y]; s = sum(ex)
        return [e/s for e in ex]
