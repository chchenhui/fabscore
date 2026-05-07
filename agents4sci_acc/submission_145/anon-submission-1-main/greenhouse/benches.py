import math
import numpy as np
from typing import Callable, Tuple, Dict

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return A * x.shape[0] + np.sum(x**2 - A * np.cos(2 * np.pi * x))

def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    A = 10.0
    return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)

def rosenbrock(x: np.ndarray) -> float:
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(x)
    grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
    grad[1:] += 200 * (x[1:] - x[:-1]**2)
    return grad

def ackley(x: np.ndarray) -> float:
    a, b, c = 20, 0.2, 2 * np.pi
    d = x.shape[0]
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + math.e

def ackley_grad(x: np.ndarray) -> np.ndarray:
    a, b, c = 20, 0.2, 2 * np.pi
    d = x.shape[0]
    sum_sq = np.sum(x**2)
    sum_cos = np.sum(np.cos(c * x))
    sqrt_term = np.sqrt(sum_sq / d) + 1e-12
    common1 = a * b / (d * sqrt_term) * np.exp(-b * sqrt_term)
    grad1 = common1 * x
    grad2 = (np.exp(sum_cos / d) * c / d) * np.sin(c * x) * (-1)
    return grad1 + grad2

BENCHES: Dict[str, Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]] = {
    "rastrigin": (rastrigin, rastrigin_grad),
    "rosenbrock": (rosenbrock, rosenbrock_grad),
    "ackley": (ackley, ackley_grad),
}
