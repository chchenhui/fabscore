
import random
from collections import deque
from typing import List
import math

def softmax(z):
    m = max(z); ex=[math.exp(x-m) for x in z]; s=sum(ex); return [e/s for e in ex]

class TinyMLP:
    def __init__(self, in_dim, hidden=64, out_dim=3, seed=0):
        self.in_dim=in_dim; self.hidden=hidden; self.out_dim=out_dim
        random.seed(seed)
        self.W1=[[random.uniform(-0.1,0.1) for _ in range(in_dim)] for _ in range(hidden)]
        self.b1=[0.0]*hidden
        self.W2=[[random.uniform(-0.1,0.1) for _ in range(hidden)] for _ in range(out_dim)]
        self.b2=[0.0]*out_dim
    def forward(self,x):
        h=[max(0.0, sum(wi*xi for wi,xi in zip(w,x))+b) for w,b in zip(self.W1,self.b1)]
        y=[sum(wi*hi for wi,hi in zip(w,h))+b for w,b in zip(self.W2,self.b2)]
        return y

class NFSP:
    def __init__(self, in_dim=6, eta=0.1, seed=0):
        self.avg_net = TinyMLP(in_dim,hidden=64,out_dim=3,seed=seed)
        self.br_net  = TinyMLP(in_dim,hidden=64,out_dim=3,seed=seed+1)
        self.sl_buf = deque(maxlen=50000)
        self.rl_buf = deque(maxlen=50000)
        self.eta = eta

    def avg_action_probs(self, x): return softmax(self.avg_net.forward(x))
    def br_action(self, x): 
        q = self.br_net.forward(x)
        a = max(range(len(q)), key=lambda i:q[i])
        pi = [0.0]*len(q); pi[a]=1.0
        return pi

    def behavior_policy(self, x):
        if random.random()<self.eta:
            return self.br_action(x)
        else:
            return self.avg_action_probs(x)
