from .dsl import Rule

def baseline_rules():
    """Canonical baselines within the DSL."""
    sgd = Rule(bm=0.0, bv=0.0, am=1.0, av=1.0, a1=1.0, a2=0.0, p=0.0, eta=0.002, eps=1e-8)
    momentum = Rule(bm=0.9, bv=0.0, am=1.0, av=1.0, a1=0.0, a2=1.0, p=0.0, eta=0.002, eps=1e-8)
    adamish = Rule(bm=0.9, bv=0.99, am=1.0, av=1.0, a1=0.0, a2=1.0, p=1.0, eta=0.001, eps=1e-8)
    return {"SGD": sgd, "Momentum": momentum, "Adam-ish": adamish}
