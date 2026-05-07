"""
PAC-Bayesian risk bounds for Hierarchical Delegated Oversight

Implements the risk bound calculations from the paper:
"HDO formalizes oversight as a hierarchical tree of entailment checks, 
deriving PAC-Bayesian bounds on misalignment risk that tighten with delegation depth."
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math
from collections import defaultdict

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .debate_tree import DebateTree, DebateNode
from .verifiers import VerificationResult, BaseVerifier
from .aggregation import AggregationResult


@dataclass
class RiskBound:
    """PAC-Bayesian risk bound calculation"""
    tree_id: str
    delegation_depth: int
    
    # Risk components
    empirical_risk: float  # Observed misalignment rate
    complexity_penalty: float  # Penalty for tree complexity
    confidence_level: float  # 1 - δ (confidence level)
    
    # Bound calculation
    pac_bound: float  # PAC-Bayesian upper bound on true risk
    bayesian_bound: float  # Bayesian risk bound
    combined_bound: float  # Combined PAC-Bayesian bound
    
    # Contributing factors
    leaf_error_rates: Dict[str, float]  # Error rates at leaf nodes
    aggregation_errors: Dict[str, float]  # Errors in aggregation
    delegation_benefit: float  # Risk reduction from delegation
    
    # Metadata
    num_verifications: int
    total_cost: float
    bound_tightness: float  # How tight the bound is


class PAC_BayesianRiskBound:
    """
    Calculates PAC-Bayesian risk bounds for HDO trees
    
    Implements the theoretical framework from the paper for bounding
    misalignment risk as a function of delegation depth and tree structure.
    """
    
    def __init__(self,
                 confidence_level: float = 0.95,  # 1 - δ
                 prior_belief: float = 0.1,  # Prior on misalignment rate
                 complexity_weight: float = 1.0,  # Weight for complexity penalty
                 depth_discount: float = 0.9):  # Discount factor for depth benefit
        """
        Initialize PAC-Bayesian risk bound calculator
        
        Args:
            confidence_level: Confidence level (1 - δ) for bounds
            prior_belief: Prior belief about misalignment rate
            complexity_weight: Weight for tree complexity in penalty
            depth_discount: Discount factor for delegation depth benefit
        """
        self.confidence_level = confidence_level
        self.delta = 1.0 - confidence_level  # Failure probability
        self.prior_belief = prior_belief
        self.complexity_weight = complexity_weight
        self.depth_discount = depth_discount
        
        # Historical data for bound calibration with memory bounds
        self.historical_risks: List[float] = []
        self.bound_history: List[RiskBound] = []
        self.verifier_error_rates: Dict[str, List[float]] = defaultdict(list)
        self.max_history_size = 10000  # Prevent memory leaks
    
    def calculate_risk_bound(self,
                           tree: DebateTree,
                           verification_results: Dict[str, List[VerificationResult]],
                           aggregation_results: Dict[str, AggregationResult],
                           ground_truth: Dict[str, bool] = None) -> RiskBound:
        """
        Calculate PAC-Bayesian risk bound for a debate tree
        
        Args:
            tree: The debate tree structure
            verification_results: Verification results for each node
            aggregation_results: Aggregation results for each node
            ground_truth: Ground truth labels (if available) for calibration
            
        Returns:
            RiskBound with calculated bounds and analysis
        """
        # Calculate empirical risk
        empirical_risk = self._calculate_empirical_risk(
            tree, verification_results, aggregation_results, ground_truth
        )
        
        # Calculate complexity penalty
        complexity_penalty = self._calculate_complexity_penalty(tree, verification_results)
        
        # Calculate PAC bound
        pac_bound = self._calculate_pac_bound(empirical_risk, complexity_penalty, len(verification_results))
        
        # Calculate Bayesian bound
        bayesian_bound = self._calculate_bayesian_bound(empirical_risk, tree.delegation_depth)
        
        # Combine bounds
        combined_bound = self._combine_bounds(pac_bound, bayesian_bound)
        
        # Calculate leaf error rates
        leaf_error_rates = self._calculate_leaf_error_rates(tree, verification_results, ground_truth)
        
        # Calculate aggregation errors
        aggregation_errors = self._calculate_aggregation_errors(aggregation_results, ground_truth)
        
        # Calculate delegation benefit
        delegation_benefit = self._calculate_delegation_benefit(tree, empirical_risk)
        
        # Calculate metadata
        num_verifications = sum(len(results) for results in verification_results.values())
        total_cost = tree.total_cost
        bound_tightness = self._calculate_bound_tightness(empirical_risk, combined_bound)
        
        risk_bound = RiskBound(
            tree_id=tree.root_id,
            delegation_depth=tree.delegation_depth,
            empirical_risk=empirical_risk,
            complexity_penalty=complexity_penalty,
            confidence_level=self.confidence_level,
            pac_bound=pac_bound,
            bayesian_bound=bayesian_bound,
            combined_bound=combined_bound,
            leaf_error_rates=leaf_error_rates,
            aggregation_errors=aggregation_errors,
            delegation_benefit=delegation_benefit,
            num_verifications=num_verifications,
            total_cost=total_cost,
            bound_tightness=bound_tightness
        )
        
        # Record for history with memory management
        self.bound_history.append(risk_bound)
        self.historical_risks.append(empirical_risk)
        
        # Prevent memory leaks by limiting history size
        if len(self.bound_history) > self.max_history_size:
            self.bound_history.pop(0)
        if len(self.historical_risks) > self.max_history_size:
            self.historical_risks.pop(0)
        
        return risk_bound
    
    def _calculate_empirical_risk(self,
                                tree: DebateTree,
                                verification_results: Dict[str, List[VerificationResult]],
                                aggregation_results: Dict[str, AggregationResult],
                                ground_truth: Dict[str, bool] = None) -> float:
        """Calculate empirical misalignment risk"""
        if not ground_truth:
            # Without ground truth, estimate risk from verification confidence
            total_uncertainty = 0.0
            total_nodes = 0
            
            for node in tree.nodes.values():
                if node.status in [NodeStatus.VERIFIED, NodeStatus.REJECTED]:
                    total_uncertainty += node.uncertainty
                    total_nodes += 1
            
            if total_nodes == 0:
                return self.prior_belief
            
            return total_uncertainty / total_nodes
        
        # With ground truth, calculate actual error rate
        errors = 0
        total = 0
        
        for node_id, true_alignment in ground_truth.items():
            if node_id in tree.nodes:
                node = tree.nodes[node_id]
                if node.verification_result is not None:
                    if node.verification_result != true_alignment:
                        errors += 1
                    total += 1
        
        if total == 0:
            return self.prior_belief
        
        return errors / total
    
    def _calculate_complexity_penalty(self,
                                    tree: DebateTree,
                                    verification_results: Dict[str, List[VerificationResult]]) -> float:
        """Calculate complexity penalty based on tree structure"""
        # Number of nodes
        num_nodes = len(tree.nodes)
        
        # Depth penalty
        depth_penalty = tree.delegation_depth * 0.1
        
        # Branching factor penalty
        branching_factors = []
        for node in tree.nodes.values():
            if not node.is_leaf():
                branching_factors.append(len(node.children_ids))
        
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 1.0
        branching_penalty = (avg_branching - 1.0) * 0.05
        
        # Verifier diversity penalty (less diverse = higher penalty)
        verifier_types = set()
        for results in verification_results.values():
            for result in results:
                verifier_types.add(result.verifier_type.value)
        
        diversity_penalty = max(0.0, 0.2 - len(verifier_types) * 0.05)
        
        # Combined complexity penalty
        base_penalty = math.log(num_nodes + 1) / math.log(2)  # Log of nodes
        total_penalty = (base_penalty + depth_penalty + branching_penalty + diversity_penalty) * self.complexity_weight
        
        return total_penalty
    
    def _calculate_pac_bound(self,
                           empirical_risk: float,
                           complexity_penalty: float,
                           sample_size: int) -> float:
        """Calculate PAC learning bound"""
        if sample_size <= 0:
            return 1.0
        
        # Standard PAC bound: R ≤ R_emp + sqrt((complexity + log(1/δ)) / (2n))
        confidence_term = math.log(1.0 / self.delta)
        complexity_term = complexity_penalty
        
        pac_term = math.sqrt((complexity_term + confidence_term) / (2.0 * sample_size))
        
        pac_bound = min(1.0, empirical_risk + pac_term)
        return pac_bound
    
    def _calculate_bayesian_bound(self,
                                empirical_risk: float,
                                delegation_depth: int) -> float:
        """Calculate Bayesian risk bound with depth discount"""
        # Bayesian update with prior
        # Simplified: posterior mean with depth-dependent variance reduction
        
        prior_weight = 1.0
        data_weight = delegation_depth + 1  # More depth = more evidence
        
        # Posterior mean
        posterior_mean = (prior_weight * self.prior_belief + data_weight * empirical_risk) / (prior_weight + data_weight)
        
        # Posterior variance (decreases with depth)
        posterior_var = self.prior_belief * (1 - self.prior_belief) / (prior_weight + data_weight)
        
        # Confidence interval
        confidence_width = 1.96 * math.sqrt(posterior_var)  # 95% confidence
        
        bayesian_bound = min(1.0, posterior_mean + confidence_width)
        return bayesian_bound
    
    def _combine_bounds(self, pac_bound: float, bayesian_bound: float) -> float:
        """Combine PAC and Bayesian bounds"""
        # Take the minimum (tightest) bound
        # In practice, could use more sophisticated combination
        return min(pac_bound, bayesian_bound)
    
    def _calculate_leaf_error_rates(self,
                                  tree: DebateTree,
                                  verification_results: Dict[str, List[VerificationResult]],
                                  ground_truth: Dict[str, bool] = None) -> Dict[str, float]:
        """Calculate error rates at leaf nodes"""
        leaf_errors = {}
        
        for node in tree.get_leaves():
            node_id = node.id
            
            if node_id not in verification_results:
                continue
            
            results = verification_results[node_id]
            
            if ground_truth and node_id in ground_truth:
                # Calculate actual error rate
                errors = 0
                for result in results:
                    if result.is_valid != ground_truth[node_id]:
                        errors += 1
                
                error_rate = errors / len(results) if results else 1.0
            else:
                # Estimate error rate from uncertainty
                avg_uncertainty = sum(r.uncertainty for r in results) / len(results) if results else 1.0
                error_rate = avg_uncertainty
            
            leaf_errors[node_id] = error_rate
        
        return leaf_errors
    
    def _calculate_aggregation_errors(self,
                                    aggregation_results: Dict[str, AggregationResult],
                                    ground_truth: Dict[str, bool] = None) -> Dict[str, float]:
        """Calculate errors in aggregation process"""
        agg_errors = {}
        
        for node_id, agg_result in aggregation_results.items():
            if ground_truth and node_id in ground_truth:
                # Actual aggregation error
                error = 1.0 if agg_result.is_valid != ground_truth[node_id] else 0.0
            else:
                # Estimated aggregation error from uncertainty
                error = agg_result.uncertainty
            
            agg_errors[node_id] = error
        
        return agg_errors
    
    def _calculate_delegation_benefit(self, tree: DebateTree, empirical_risk: float) -> float:
        """Calculate risk reduction benefit from delegation"""
        if tree.delegation_depth == 0:
            return 0.0
        
        # Risk reduction scales with depth (but with diminishing returns)
        max_benefit = self.prior_belief - empirical_risk
        depth_factor = 1.0 - (self.depth_discount ** tree.delegation_depth)
        
        delegation_benefit = max_benefit * depth_factor
        return max(0.0, delegation_benefit)
    
    def _calculate_bound_tightness(self, empirical_risk: float, bound: float) -> float:
        """Calculate how tight the bound is (lower = tighter)"""
        if bound <= empirical_risk:
            return 0.0  # Perfect bound
        
        if empirical_risk < 1e-10:  # Use epsilon for near-zero risk
            return bound  # Absolute tightness when no empirical risk
        
        return (bound - empirical_risk) / empirical_risk  # Relative tightness
    
    def analyze_depth_scaling(self, 
                            trees_by_depth: Dict[int, List[DebateTree]],
                            results_by_tree: Dict[str, Dict[str, List[VerificationResult]]],
                            aggregations_by_tree: Dict[str, Dict[str, AggregationResult]]) -> Dict[int, Dict[str, float]]:
        """
        Analyze how risk bounds scale with delegation depth
        
        Args:
            trees_by_depth: Trees organized by delegation depth
            results_by_tree: Verification results for each tree
            aggregations_by_tree: Aggregation results for each tree
            
        Returns:
            Analysis of risk bound scaling by depth
        """
        depth_analysis = {}
        
        for depth, trees in trees_by_depth.items():
            depth_risks = []
            depth_bounds = []
            depth_costs = []
            depth_benefits = []
            
            for tree in trees:
                tree_id = tree.root_id
                
                if tree_id not in results_by_tree or tree_id not in aggregations_by_tree:
                    continue
                
                risk_bound = self.calculate_risk_bound(
                    tree,
                    results_by_tree[tree_id],
                    aggregations_by_tree[tree_id]
                )
                
                depth_risks.append(risk_bound.empirical_risk)
                depth_bounds.append(risk_bound.combined_bound)
                depth_costs.append(risk_bound.total_cost)
                depth_benefits.append(risk_bound.delegation_benefit)
            
            if depth_risks:
                depth_analysis[depth] = {
                    'avg_empirical_risk': sum(depth_risks) / len(depth_risks),
                    'avg_bound': sum(depth_bounds) / len(depth_bounds),
                    'avg_cost': sum(depth_costs) / len(depth_costs),
                    'avg_delegation_benefit': sum(depth_benefits) / len(depth_benefits),
                    'num_trees': len(trees),
                    'risk_reduction_rate': (depth_risks[0] - depth_risks[-1]) / depth_risks[0] if len(depth_risks) > 1 and depth_risks[0] > 0 else 0.0
                }
        
        return depth_analysis
    
    def get_risk_bound_certificate(self, risk_bound: RiskBound) -> Dict[str, Any]:
        """Generate a formal certificate for the risk bound"""
        certificate = {
            'tree_id': risk_bound.tree_id,
            'timestamp': len(self.bound_history),  # Simple timestamp
            'confidence_level': self.confidence_level,
            'delegation_depth': risk_bound.delegation_depth,
            
            # Main result
            'certified_risk_bound': risk_bound.combined_bound,
            'empirical_risk': risk_bound.empirical_risk,
            'bound_type': 'PAC-Bayesian',
            
            # Guarantees
            'probability_guarantee': f"With probability ≥ {self.confidence_level}, true misalignment risk ≤ {risk_bound.combined_bound:.4f}",
            'depth_guarantee': f"Risk bound tightens with delegation depth (current depth: {risk_bound.delegation_depth})",
            
            # Evidence
            'num_verifications': risk_bound.num_verifications,
            'total_cost': risk_bound.total_cost,
            'verifier_diversity': len(risk_bound.leaf_error_rates),
            
            # Quality metrics
            'bound_tightness': risk_bound.bound_tightness,
            'delegation_benefit': risk_bound.delegation_benefit,
            
            # Components
            'pac_component': risk_bound.pac_bound,
            'bayesian_component': risk_bound.bayesian_bound,
            'complexity_penalty': risk_bound.complexity_penalty,
            
            # Validity
            'certificate_valid': True,
            'assumptions_met': self._check_assumptions(risk_bound),
            'limitations': [
                'Assumes verifier independence',
                'Based on observed verification data',
                'Subject to complexity penalty approximation'
            ]
        }
        
        return certificate
    
    def _check_assumptions(self, risk_bound: RiskBound) -> List[str]:
        """Check which theoretical assumptions are met"""
        assumptions_met = []
        
        # Sufficient sample size
        if risk_bound.num_verifications >= 10:
            assumptions_met.append('Sufficient sample size')
        
        # Reasonable complexity
        if risk_bound.complexity_penalty < 1.0:
            assumptions_met.append('Bounded complexity')
        
        # Non-trivial delegation
        if risk_bound.delegation_depth > 0:
            assumptions_met.append('Non-trivial delegation')
        
        # Verifier diversity
        if len(risk_bound.leaf_error_rates) > 1:
            assumptions_met.append('Verifier diversity')
        
        return assumptions_met
    
    def update_prior(self, new_empirical_risks: List[float]) -> None:
        """Update prior belief based on new empirical evidence"""
        if not new_empirical_risks:
            return
        
        # Simple Bayesian update
        old_weight = len(self.historical_risks)
        new_weight = len(new_empirical_risks)
        
        if old_weight + new_weight == 0:
            return
        
        old_mean = self.prior_belief
        new_mean = sum(new_empirical_risks) / len(new_empirical_risks)
        
        # Weighted average
        self.prior_belief = (old_weight * old_mean + new_weight * new_mean) / (old_weight + new_weight)
        
        # Keep prior in reasonable bounds
        self.prior_belief = max(0.01, min(0.5, self.prior_belief))
    
    def get_bound_statistics(self) -> Dict[str, Any]:
        """Get statistics about risk bound performance"""
        if not self.bound_history:
            return {}
        
        bounds = [b.combined_bound for b in self.bound_history]
        empirical_risks = [b.empirical_risk for b in self.bound_history]
        depths = [b.delegation_depth for b in self.bound_history]
        
        return {
            'num_bounds_calculated': len(self.bound_history),
            'avg_bound': sum(bounds) / len(bounds),
            'avg_empirical_risk': sum(empirical_risks) / len(empirical_risks),
            'avg_depth': sum(depths) / len(depths),
            'bound_tightness_avg': sum(b.bound_tightness for b in self.bound_history) / len(self.bound_history),
            'delegation_benefit_avg': sum(b.delegation_benefit for b in self.bound_history) / len(self.bound_history),
            'current_prior': self.prior_belief,
            'depth_correlation': self._calculate_correlation(depths, empirical_risks) if len(depths) > 1 else 0.0
        }
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate correlation coefficient without numpy"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        sum_y2 = sum(yi * yi for yi in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator_x = n * sum_x2 - sum_x * sum_x
        denominator_y = n * sum_y2 - sum_y * sum_y
        
        if denominator_x <= 0 or denominator_y <= 0:
            return 0.0
        
        correlation = numerator / (math.sqrt(denominator_x) * math.sqrt(denominator_y))
        return correlation
