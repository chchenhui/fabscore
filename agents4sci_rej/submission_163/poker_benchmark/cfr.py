
import random
from collections import defaultdict, namedtuple
from typing import Dict, Tuple, List

# An "infoset" key: we abstract by (street, bucket_idx, position, action_hist_hash)
def infoset_key(obs) -> Tuple:
    street = obs["street"]
    bucket = obs["bucket"]
    # bucket is (hand_cls, texture); compress to str
    bkey = f"{bucket[0]}|{bucket[1]}"
    pos = obs["to_act"]
    # history signature (very compact)
    hist = tuple((h[0], h[2]) for h in obs["history"][-4:])
    return (street, bkey, pos, hist)

def regret_matching(R: Dict[Tuple, Dict[str, float]]) -> Dict[Tuple, Dict[str, float]]:
    """Turn regrets into action probs per infoset."""
    pi = {}
    for I, reg in R.items():
        pos_vals = {a: max(0.0, v) for a, v in reg.items()}
        s = sum(pos_vals.values())
        if s<=1e-9:
            # uniform over actions observed for this I
            n = max(1, len(reg))
            pi[I] = {a: 1.0/n for a in reg}
        else:
            pi[I] = {a: v/s for a, v in pos_vals.items()}
    return pi

class CFRTabular:
    def __init__(self, actions: List[str]=["check/call","bet/raise","fold"], seed: int=42):
        self.R = defaultdict(lambda: defaultdict(float))   # cumulative regrets
        self.S = defaultdict(lambda: defaultdict(float))   # strategy sums
        self.actions = actions
        self.rng = random.Random(seed)
        self.iter = 0

    def traverse(self, env) -> None:
        """Dummy traversal to illustrate structure; integrate with a full tree for production."""
        obs = env.observe()
        I = infoset_key(obs)
        # simulate regrets update by random pseudo-advantages
        # In a production system, compute counterfactual values by recursion.
        # Here we just create a placeholder to show the interface.
        for a in self.actions:
            adv = self.rng.uniform(-1, 1)
            self.R[I][a] += adv
        # update strategy with regret-matching policy
        pi = regret_matching(self.R)
        for a, p in pi[I].items():
            self.S[I][a] += p * max(1, self.iter)

    def iterate(self, env, T: int=10000):
        for t in range(1, T+1):
            self.iter = t
            self.traverse(env)

    def average_policy(self) -> Dict[Tuple, Dict[str, float]]:
        pi = {}
        for I, S_I in self.S.items():
            s = sum(S_I.values())
            if s<=1e-9:
                n = max(1, len(S_I))
                pi[I] = {a:1.0/n for a in S_I}
            else:
                pi[I] = {a:v/s for a, v in S_I.items()}
        return pi

class MCCFRExternal(CFRTabular):
    """Skeleton identical API; in a full version you'd sample actions and backpropagate IS-corrected regrets."""
    pass
