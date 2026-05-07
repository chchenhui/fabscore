# DualRay-TopK+Weights Repair (Condition C)

## Original Experiment (K=5, Greedy, 7B)

Ablation testing whether adding normalized Farkas multiplier weights improves LP repair.
Result: 1/31 (3.23%), same as Conditions A and B. Weights alone insufficient in greedy K=5 setting.

## Optimized Experiment (K=10, N=16, 2 Rounds, 72B)

Combined four improvements: stronger model (72B), higher sampling (N=16), iterative repair (2 rounds), and increased K=10.

| Method | Model | Repair (on 31) | Pass@1 Overall | Pass@1 ComplexLP |
|--------|-------|----------------|----------------|-----------------|
| DualRay+Wt K=5 greedy (C) | 7B | 1/31 (3.23%) | 58.17% | 17.54% |
| DualRay+Wt K=10 best-of-6 (C+) | 7B | 3/31 (9.68%) | 58.40% | 17.54% |
| DualRay+Wt K=10 N=16 R=2 | Qwen3-32B | 6/31 (19.35%) | 58.75% | 18.48% |
| **DualRay+Wt K=10 N=16 R=2** | **72B** | **7/31 (22.58%)** | **58.86%** | **18.96%** |

72B repaired: EasyLP_246, EasyLP_425, EasyLP_620, EasyLP_645, ComplexLP_62, ComplexLP_111, ComplexLP_86.
Three ComplexLP repairs achieved for the first time. EasyLP_246 went from 0% pass (7B) to 100% pass (72B, 17/17).
