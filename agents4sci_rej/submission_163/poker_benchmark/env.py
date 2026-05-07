
import random
import math
import itertools
from collections import namedtuple, defaultdict
from typing import List, Tuple, Dict, Optional

# ----------- Card utilities -----------

RANKS = list("23456789TJQKA")
SUITS = list("cdhs")
DECK = [r+s for r in RANKS for s in SUITS]

def card_int(card: str) -> int:
    return RANKS.index(card[0])*4 + SUITS.index(card[1])

def hand_rank_strength_approx(cards5: List[str]) -> int:
    """
    Super-simplified 5-card rank score (NOT a full evaluator).
    Returns an integer where higher is better.
    For speed & demo purposes only. For research, replace with a real evaluator.
    """
    # Count ranks
    ranks = [c[0] for c in cards5]
    suits = [c[1] for c in cards5]
    rc = defaultdict(int)
    sc = defaultdict(int)
    for r in ranks: rc[r]+=1
    for s in suits: sc[s]+=1

    counts = sorted(rc.values(), reverse=True)
    is_flush = max(sc.values())==5
    # Straight check (very rough; treats A as high only)
    order = ''.join(RANKS)
    uniq = sorted(set(ranks), key=lambda x: RANKS.index(x))
    is_straight = False
    if len(uniq)>=5:
        s = ''.join(uniq)
        # slide window of length 5
        for i in range(len(order)-4):
            if set(order[i:i+5]).issubset(set(uniq)):
                is_straight = True
                break

    # Score layers
    if is_straight and is_flush: base = 8
    elif counts[0]==4: base = 7
    elif counts[0]==3 and counts[1]==2: base = 6
    elif is_flush: base = 5
    elif is_straight: base = 4
    elif counts[0]==3: base = 3
    elif counts[0]==2 and counts[1]==2: base = 2
    elif counts[0]==2: base = 1
    else: base = 0
    # Tiebreak with high cards (very rough)
    kicker = sum(RANKS.index(r) for r in ranks)
    return base*100 + kicker

def best5_from7(cards7: List[str]) -> int:
    best = -1
    for comb in itertools.combinations(cards7, 5):
        s = hand_rank_strength_approx(list(comb))
        if s>best: best=s
    return best

def monte_carlo_equity(hole: List[str], opp_known: Optional[List[str]], board: List[str], n_sims: int=800) -> float:
    """
    Approximate win-rate from current situation by Monte Carlo.
    - hole: our 2 cards
    - opp_known: [] or None if unknown
    - board: 0..5 community cards
    Returns probability of winning at showdown (ties count as 0.5).
    """
    used = set(hole + (opp_known or []) + board)
    deck = [c for c in DECK if c not in used]
    wins = 0.0
    trials = 0
    for _ in range(n_sims):
        d = deck[:]
        random.shuffle(d)
        # Opp hole
        if opp_known and len(opp_known)==2:
            opp = opp_known
            left = d
        else:
            opp = [d.pop(), d.pop()]
            left = d
        # Complete board to 5
        b = board[:]
        while len(b)<5:
            b.append(left.pop())
        # showdown
        ours = best5_from7(hole + b)
        theirs = best5_from7(opp + b)
        if ours>theirs: wins += 1.0
        elif ours==theirs: wins += 0.5
        trials += 1
    return wins/max(1, trials)

# ----------- Abstractions -----------

def texture_features(board: List[str]) -> Dict[str, int]:
    """Very light board texture flags."""
    feats = {"paired":0, "monotone":0, "two_tone":0, "straighty":0}
    if len(board)>=3:
        ranks = [c[0] for c in board]
        suits = [c[1] for c in board]
        # Paired
        feats["paired"] = int(len(set(ranks))<len(ranks))
        # Suits
        sc = defaultdict(int)
        for s in suits: sc[s]+=1
        if max(sc.values())>=3: feats["monotone"]=1
        elif max(sc.values())==2: feats["two_tone"]=1
        # Straighty rough check
        order = ''.join(RANKS)
        uniq = sorted(set(ranks), key=lambda x: RANKS.index(x))
        if len(uniq)>=3:
            for i in range(len(order)-2):
                if len(set(order[i:i+3]).intersection(set(uniq)))==3:
                    feats["straighty"]=1
                    break
    return feats

def card_bucket(hole: List[str], board: List[str]) -> Tuple[str,str]:
    """Bucket by (hand class, board texture)."""
    # Hand class by equity approx (pre or post flop)
    eq = monte_carlo_equity(hole, None, board, n_sims=200 if len(board)<3 else 400)
    if eq>=0.8: hcls="very_strong"
    elif eq>=0.65: hcls="strong"
    elif eq>=0.5: hcls="marginal"
    elif eq>=0.35: hcls="drawish_or_weak"
    else: hcls="air"
    tex = texture_features(board)
    tkey = []
    for k in ["paired","monotone","two_tone","straighty"]:
        if tex[k]: tkey.append(k)
    tkey = '+'.join(tkey) if tkey else "dry"
    return hcls, tkey

# ----------- Randomized Action Abstraction -----------

def randomized_bet_sizes(pot: float, street: str, seed: Optional[int]=None) -> List[float]:
    """
    Generate a randomized set of legal bet/raise sizes around common anchors.
    Ensures variability across episodes while staying sensible.
    """
    rng = random.Random(seed)
    anchors = [0.25, 0.33, 0.5, 0.75, 1.0, 1.5]
    # street-dependent jitter scale
    jitter = 0.05 if street in ("turn","river") else 0.08
    k = rng.randint(3, 5)  # number of sizes to expose this node
    cand = []
    for a in anchors:
        j = max(0.01, a + rng.uniform(-jitter, jitter))
        cand.append(j)
    # sample k sizes
    sizes = sorted(rng.sample(cand, k))
    return [round(pot*s, 2) for s in sizes]

# ----------- Minimal HU environment (one betting round per street, no multi-raises to keep tractable) -----------

class HUEnv:
    """
    Minimal heads-up NLHE environment with simplified betting:
    - One bet opportunity per player per street (check/call/fold or bet among abstract sizes).
    - Not a full rules engine; intended for benchmarking learning algorithms consistently.
    """
    def __init__(self, stack_bb: float=100.0, sb: float=1.0, bb: float=2.0, seed: Optional[int]=None):
        self.stack_bb = stack_bb
        self.sb = sb
        self.bb = bb
        self.rng = random.Random(seed)
        self.reset()

    def deal(self):
        d = DECK[:]
        self.rng.shuffle(d)
        self.p1 = [d.pop(), d.pop()]
        self.p2 = [d.pop(), d.pop()]
        self.board = [d.pop(), d.pop(), d.pop(), d.pop(), d.pop()]

    def reset(self):
        self.deal()
        self.street_idx = 0  # 0 pre,1 flop,2 turn,3 river,4 showdown
        self.pot = self.sb + self.bb
        self.eff_stacks = [self.stack_bb - self.bb, self.stack_bb]  # SB posted 1, BB posted 2 (simplified)
        self.to_act = 0  # SB acts first preflop in heads-up
        self.folded = None
        self.history = []  # (street, player, action, size)
        return self.observe()

    def street_name(self):
        return ["pre","flop","turn","river","showdown"][self.street_idx]

    def legal_actions(self) -> List[Tuple[str, float]]:
        if self.street_idx>=4: return [("terminal", 0.0)]
        street = self.street_name()
        actions = [("check/call", 0.0)]
        # expose randomized bet sizes this node
        sizes = randomized_bet_sizes(self.pot, street)
        for s in sizes:
            if s>0.0:
                actions.append(("bet/raise", min(s, self.stack_bb*2)))  # cap
        actions.append(("fold", 0.0))
        return actions

    def step(self, action: Tuple[str,float]):
        """Apply action and advance state; returns (obs, reward_p1, done, info)."""
        a_type, a_size = action
        street = self.street_name()
        player = self.to_act
        self.history.append((street, player, a_type, a_size))

        # Simplified: if bet/raise, opponent chooses to call or fold randomly (env is not adversary; learning happens in self-play layer)
        # Here we just move the environment; outer trainer handles policies.
        # For env-only stepping (no opponent), we don't resolve opponent response—left to self-play drivers.
        # We'll mark done only at showdown externally.
        # To keep env stateless regarding opponent decisions, we return observation now.
        return self.observe(), 0.0, False, {}

    def showdown(self) -> Tuple[float, float]:
        """Compute terminal payoffs (bb units) using showdown equity approximation vs revealed opponent (proxy)."""
        # For simplicity use full-board best5 comparison
        v1 = best5_from7(self.p1 + self.board)
        v2 = best5_from7(self.p2 + self.board)
        # Pot distribution (toy): winner takes pot, tie splits
        if v1>v2: return self.pot, -self.pot
        elif v2>v1: return -self.pot, self.pot
        else: return self.pot/2.0, self.pot/2.0

    def observe(self):
        street = self.street_name()
        board_cards = self.board[:0 if self.street_idx==0 else 3 if self.street_idx==1 else 4 if self.street_idx==2 else 5]
        # Build a compact observation dict
        obs = {
            "street": street,
            "to_act": self.to_act,
            "pot": self.pot,
            "stack_bb": self.stack_bb,
            "history": self.history[-10:],
            "board": board_cards,
            "bucket": card_bucket(self.p1 if self.to_act==0 else self.p2, board_cards),
            "equity": monte_carlo_equity(self.p1 if self.to_act==0 else self.p2, None, board_cards, n_sims=200)
        }
        return obs
