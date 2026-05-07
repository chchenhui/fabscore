"""
UnitMath: Unit-Aware Numerical Reasoning and Dimensional Consistency
A framework for enhancing table reasoning with unit-aware symbolic computation
"""

from .unit_parser import UnitParser, QuantityExtractor
from .symbolic_calculator import SymbolicCalculator, UnitConverter
from .operation_sketch import OperationSketch, SketchExecutor
from .neural_symbolic_model import NeuralSymbolicTableReasoner
from .evaluation import UnitMathEvaluator
from .data_augmentation import UnitAugmentor

__version__ = "0.1.0"
__all__ = [
    "UnitParser",
    "QuantityExtractor", 
    "SymbolicCalculator",
    "UnitConverter",
    "OperationSketch",
    "SketchExecutor",
    "NeuralSymbolicTableReasoner",
    "UnitMathEvaluator",
    "UnitAugmentor"
]