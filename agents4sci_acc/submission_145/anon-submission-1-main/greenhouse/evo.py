import json, random, time
from typing import List, Dict, Tuple
import numpy as np
from .dsl import Rule, BM_CHOICES, BV_CHOICES, AM_CHOICES, AV_CHOICES, A1_CHOICES, A2_CHOICES, P_CHOICES, ETA_CHOICES, EPS_CHOICES
from .runners import eval_rule

def random_rule() -> Rule:
    import random as R
    return Rule(
        bm=R.choice(BM_CHOICES),
        bv=R.choice(BV_CHOICES),
        am=R.choice(AM_CHOICES),
        av=R.choice(AV_CHOICES),
        a1=R.choice(A1_CHOICES),
        a2=R.choice(A2_CHOICES),
        p=R.choice(P_CHOICES),
        eta=R.choice(ETA_CHOICES),
        eps=R.choice(EPS_CHOICES),
    )

def mutate_rule(rule: Rule, prob=0.25) -> Rule:
    def pick(old, choices):
        import random as R
        if R.random() < prob:
            options = [c for c in choices if c != old]
            return R.choice(options) if options else old
        return old
    return Rule(
        bm=pick(rule.bm, BM_CHOICES),
        bv=pick(rule.bv, BV_CHOICES),
        am=pick(rule.am, AM_CHOICES),
        av=pick(rule.av, AV_CHOICES),
        a1=pick(rule.a1, A1_CHOICES),
        a2=pick(rule.a2, A2_CHOICES),
        p=pick(rule.p, P_CHOICES),
        eta=pick(rule.eta, ETA_CHOICES),
        eps=pick(rule.eps, EPS_CHOICES),
    )

def evolve(bench_train: str, bench_test: str, dim=10, steps=300, seeds=(0,1),
           pop_size=32, elites=6, gens=20, mutate_prob=0.30):
    # Initialize population with a few randoms (baselines can be inserted externally if desired)
    pop: List[Rule] = [random_rule() for _ in range(pop_size)]
    history = {"gen": [], "best_loss_train": [], "mean_loss_train": []}
    archive = []

    for gen in range(gens):
        fitness: List[Tuple[float, Rule]] = []
        for r in pop:
            train_loss, _ = eval_rule(r, bench_train, dim=dim, steps=steps, seeds=seeds)
            fitness.append((train_loss, r))
        fitness.sort(key=lambda x: x[0])
        best_train = fitness[0][0]
        mean_train = float(np.mean([ft for ft, _ in fitness]))
        history["gen"].append(gen)
        history["best_loss_train"].append(best_train)
        history["mean_loss_train"].append(mean_train)

        top_rules = [r for _, r in fitness[:elites]]
        # log elites with cross-bench
        for r in top_rules:
            test_loss, _ = eval_rule(r, bench_test, dim=dim, steps=steps, seeds=seeds)
            archive.append({
                "gen": gen,
                "rule": r.to_dict(),
                "pretty": r.pretty(),
                "rule_train_loss": float(eval_rule(r, bench_train, dim=dim, steps=steps, seeds=seeds)[0]),
                "test_loss": float(test_loss)
            })

        # next gen
        new_pop = top_rules[:]
        import random as R
        while len(new_pop) < pop_size:
            parent = R.choice(top_rules)
            child = mutate_rule(parent, prob=mutate_prob)
            new_pop.append(child)
        pop = new_pop

    # return the best rule at the end
    final_fit = []
    for r in pop:
        tloss, _ = eval_rule(r, bench_train, dim=dim, steps=steps, seeds=seeds)
        final_fit.append((tloss, r))
    final_fit.sort(key=lambda x: x[0])
    best_rule = final_fit[0][1]
    best_train_loss = final_fit[0][0]
    best_test_loss, _ = eval_rule(best_rule, bench_test, dim=dim, steps=steps, seeds=seeds)

    return best_rule, best_train_loss, best_test_loss, history, archive
