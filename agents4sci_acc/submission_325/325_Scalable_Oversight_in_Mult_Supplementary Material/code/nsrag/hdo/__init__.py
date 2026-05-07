"""
Hierarchical Delegated Oversight (HDO) - A scalable framework for multi-agent alignment
"""
from .debate_tree import DebateTree, DebateNode, ClaimType
from .verifiers import BaseVerifier, NLIVerifier, CodeVerifier, RuleVerifier, RetrievalVerifier
from .routing import CostAwareRouter, RoutingStrategy
from .aggregation import EntailmentAggregator, AggregationMethod
from .risk_bounds import PAC_BayesianRiskBound
from .hdo_system import HDOSystem, HDOConfig, HDOMode

__all__ = [
    'DebateTree', 'DebateNode', 'ClaimType',
    'BaseVerifier', 'NLIVerifier', 'CodeVerifier', 'RuleVerifier', 'RetrievalVerifier',
    'CostAwareRouter', 'RoutingStrategy',
    'EntailmentAggregator', 'AggregationMethod',
    'PAC_BayesianRiskBound',
    'HDOSystem', 'HDOConfig', 'HDOMode'
]
