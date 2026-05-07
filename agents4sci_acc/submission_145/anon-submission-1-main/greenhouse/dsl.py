from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class Rule:
    bm: float
    bv: float
    am: float
    av: float
    a1: float
    a2: float
    p: float
    eta: float
    eps: float

    def to_dict(self):
        return asdict(self)

    def pretty(self) -> str:
        return (
            f"m_t = {self.bm}*m + {1-self.bm}*g^{self.am}; "
            f"v_t = {self.bv}*v + {1-self.bv}*(g^2)^{self.av}; "
            f"Δθ = {self.eta} * ({self.a1}*g + {self.a2}*m_t) / (sqrt(v_t)+{self.eps})^{self.p}"
        )

# Discrete token space (v0.2)
BM_CHOICES   = [0.0, 0.5, 0.9]
BV_CHOICES   = [0.0, 0.9, 0.99]
AM_CHOICES   = [1.0]
AV_CHOICES   = [1.0]
A1_CHOICES   = [0.0, 0.5, 1.0, 1.5]
A2_CHOICES   = [0.0, 0.5, 1.0]
P_CHOICES    = [0.0, 0.5, 1.0]
ETA_CHOICES  = [0.0005, 0.001, 0.002, 0.005]
EPS_CHOICES  = [1e-8, 1e-6, 1e-4]
