"""
Cost-aware routing policy for Hierarchical Delegated Oversight

Implements the routing policy from the paper that selects minimal-cost competent
verifiers and triggers delegation based on uncertainty thresholds.
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import random
import math
from collections import defaultdict

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .debate_tree import DebateNode, ClaimType, NodeStatus
from .verifiers import BaseVerifier, VerifierType


class RoutingStrategy(Enum):
    """Routing strategies for verifier selection"""
    COST_MINIMAL = "cost_minimal"  # Minimize cost (default from paper)
    ACCURACY_OPTIMAL = "accuracy_optimal"  # Maximize accuracy
    BALANCED = "balanced"  # Balance cost and accuracy
    DIVERSE = "diverse"  # Maximize verifier diversity


@dataclass
class RoutingDecision:
    """Decision made by the routing policy"""
    node_id: str
    should_delegate: bool
    selected_verifier: Optional[str] = None
    delegation_reason: str = ""
    estimated_cost: float = 0.0
    expected_accuracy: float = 0.0
    redundancy_level: int = 1  # Number of parallel verifiers
    
    # Routing metadata
    uncertainty_score: float = 0.0
    available_verifiers: List[str] = None
    routing_strategy: RoutingStrategy = RoutingStrategy.COST_MINIMAL


class CostAwareRouter:
    """
    Cost-aware routing policy implementing the algorithm from the paper
    
    Routes disputes to cost-minimal verifiers as described in the paper:
    "selects V⋆= argmax_V Δu(q;V)/c(V,q) with stochastic tie-break to deter collusion"
    """
    
    def __init__(self,
                 tau_reject: float = 0.2,  # Lower uncertainty threshold
                 tau_accept: float = 0.8,  # Upper uncertainty threshold  
                 redundancy_prob: float = 0.1,  # Probability of redundant verification
                 max_redundancy: int = 3,  # Maximum parallel verifiers
                 diversity_weight: float = 0.1,  # Weight for verifier diversity
                 collusion_resistance: bool = True):
        """
        Initialize cost-aware router
        
        Args:
            tau_reject: Reject threshold - below this, don't delegate
            tau_accept: Accept threshold - above this, always delegate
            redundancy_prob: Probability of using redundant verification
            max_redundancy: Maximum number of parallel verifiers
            diversity_weight: Weight for promoting verifier diversity
            collusion_resistance: Enable anti-collusion measures
        """
        self.tau_reject = tau_reject
        self.tau_accept = tau_accept
        self.redundancy_prob = redundancy_prob
        self.max_redundancy = max_redundancy
        self.diversity_weight = diversity_weight
        self.collusion_resistance = collusion_resistance
        
        # Verifier registry and performance tracking
        self.verifiers: Dict[str, BaseVerifier] = {}
        self.verifier_history: Dict[str, List[float]] = defaultdict(list)  # Performance history
        self.routing_history: List[RoutingDecision] = []
        self.cost_budget_remaining = float('inf')
        
        # Anti-collusion measures
        self.verifier_usage_counts: Dict[str, int] = defaultdict(int)
        self.recent_assignments: List[Tuple[str, str]] = []  # (node_id, verifier_id)
        self.randomization_factor = 0.2  # Amount of randomization for tie-breaking
    
    def register_verifier(self, verifier: BaseVerifier) -> None:
        """Register a verifier with the router"""
        self.verifiers[verifier.verifier_id] = verifier
        self.verifier_usage_counts[verifier.verifier_id] = 0
    
    def route(self, 
             node: DebateNode, 
             context: Dict[str, any] = None,
             budget_constraint: float = None) -> RoutingDecision:
        """
        Make routing decision for a debate node
        
        Args:
            node: The debate node to route
            context: Additional context for routing
            budget_constraint: Remaining budget constraint
            
        Returns:
            RoutingDecision with routing outcome
        """
        context = context or {}
        
        # Check if delegation should be triggered based on uncertainty
        should_delegate = self._should_delegate(node, context)
        
        if not should_delegate:
            return RoutingDecision(
                node_id=node.id,
                should_delegate=False,
                delegation_reason="Uncertainty below delegation threshold",
                uncertainty_score=node.uncertainty
            )
        
        # Find compatible verifiers
        compatible_verifiers = self._find_compatible_verifiers(node.claim_type)
        
        if not compatible_verifiers:
            return RoutingDecision(
                node_id=node.id,
                should_delegate=False,
                delegation_reason="No compatible verifiers available",
                uncertainty_score=node.uncertainty
            )
        
        # Select optimal verifier(s) based on cost-benefit analysis
        selected_verifiers = self._select_verifiers(
            node, compatible_verifiers, context, budget_constraint
        )
        
        if not selected_verifiers:
            return RoutingDecision(
                node_id=node.id,
                should_delegate=False,
                delegation_reason="Budget constraints prevent delegation",
                uncertainty_score=node.uncertainty,
                available_verifiers=[v.verifier_id for v in compatible_verifiers]
            )
        
        # Determine redundancy level
        redundancy_level = self._determine_redundancy(node, context)
        final_verifiers = selected_verifiers[:redundancy_level]
        
        # Create routing decision
        primary_verifier = final_verifiers[0]
        estimated_cost = sum(v.estimate_cost(node.claim, context) for v in final_verifiers)
        expected_accuracy = self._estimate_accuracy(final_verifiers, node, context)
        
        decision = RoutingDecision(
            node_id=node.id,
            should_delegate=True,
            selected_verifier=primary_verifier.verifier_id,
            delegation_reason=f"Uncertainty {node.uncertainty:.3f} above threshold {self.tau_reject}",
            estimated_cost=estimated_cost,
            expected_accuracy=expected_accuracy,
            redundancy_level=len(final_verifiers),
            uncertainty_score=node.uncertainty,
            available_verifiers=[v.verifier_id for v in compatible_verifiers],
            routing_strategy=RoutingStrategy.COST_MINIMAL
        )
        
        # Record decision
        self.routing_history.append(decision)
        for verifier in final_verifiers:
            self.verifier_usage_counts[verifier.verifier_id] += 1
        
        # Update recent assignments for anti-collusion
        if self.collusion_resistance:
            self.recent_assignments.append((node.id, primary_verifier.verifier_id))
            if len(self.recent_assignments) > 100:  # Keep recent history bounded
                self.recent_assignments.pop(0)
        
        return decision
    
    def _should_delegate(self, node: DebateNode, context: Dict[str, any]) -> bool:
        """
        Determine if delegation should be triggered based on uncertainty
        
        From paper: "HDO triggers delegation when u(q) ∈ (τ_rej, τ_acc)"
        """
        uncertainty = node.uncertainty
        
        # Always delegate if uncertainty is very high
        if uncertainty > self.tau_accept:
            return True
        
        # Never delegate if uncertainty is very low
        if uncertainty < self.tau_reject:
            return False
        
        # In the middle range, use additional factors
        # Consider claim importance, available budget, etc.
        importance_weight = context.get('importance', 1.0)
        budget_factor = min(1.0, self.cost_budget_remaining / 100.0) if self.cost_budget_remaining != float('inf') else 1.0
        
        # Weighted decision
        delegation_score = uncertainty * importance_weight * budget_factor
        delegation_threshold = (self.tau_reject + self.tau_accept) / 2
        
        return delegation_score > delegation_threshold
    
    def _find_compatible_verifiers(self, claim_type: ClaimType) -> List[BaseVerifier]:
        """Find verifiers that can handle the given claim type"""
        compatible = []
        
        for verifier in self.verifiers.values():
            if verifier.can_verify(claim_type):
                compatible.append(verifier)
        
        return compatible
    
    def _select_verifiers(self, 
                         node: DebateNode,
                         compatible_verifiers: List[BaseVerifier],
                         context: Dict[str, any],
                         budget_constraint: float = None) -> List[BaseVerifier]:
        """
        Select optimal verifier(s) based on cost-benefit analysis
        
        Implements: V⋆= argmax_V Δu(q;V)/c(V,q) from the paper
        """
        if not compatible_verifiers:
            return []
        
        # Calculate cost-benefit scores for each verifier
        scores = []
        
        for verifier in compatible_verifiers:
            # Estimate cost
            cost = verifier.estimate_cost(node.claim, context)
            
            # Skip if over budget
            if budget_constraint and cost > budget_constraint:
                continue
            
            # Estimate uncertainty reduction (Δu)
            uncertainty_reduction = self._estimate_uncertainty_reduction(verifier, node, context)
            
            # Calculate cost-benefit ratio with safe division
            if cost > 1e-10:  # Use small epsilon to avoid numerical issues
                cost_benefit_ratio = uncertainty_reduction / cost
            else:
                cost_benefit_ratio = float('inf')  # Infinite benefit for zero/near-zero cost
            
            # Add diversity bonus to prevent over-reliance on single verifiers
            diversity_bonus = self._calculate_diversity_bonus(verifier)
            
            # Anti-collusion randomization
            randomization = 0.0
            if self.collusion_resistance:
                randomization = random.uniform(-self.randomization_factor, self.randomization_factor)
            
            final_score = cost_benefit_ratio + diversity_bonus + randomization
            
            scores.append((final_score, verifier, cost))
        
        if not scores:
            return []
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[0], reverse=True)
        
        # Return top verifier(s)
        selected = [score[1] for score in scores]
        return selected
    
    def _estimate_uncertainty_reduction(self, 
                                      verifier: BaseVerifier,
                                      node: DebateNode, 
                                      context: Dict[str, any]) -> float:
        """
        Estimate how much uncertainty the verifier will reduce
        
        Based on verifier's historical performance and claim characteristics
        """
        # Base reduction based on verifier's false negative rate
        base_reduction = 1.0 - verifier.false_negative_rate
        
        # Adjust based on historical performance
        if verifier.verifier_id in self.verifier_history:
            recent_performance = self.verifier_history[verifier.verifier_id][-10:]  # Last 10 verifications
            if recent_performance:
                performance_factor = sum(recent_performance) / len(recent_performance)
                base_reduction *= performance_factor
        
        # Adjust based on claim complexity
        claim_complexity = len(node.claim.split()) / 20.0  # Rough complexity measure
        complexity_penalty = max(0.1, 1.0 - claim_complexity * 0.1)
        
        # Adjust based on verifier specialization
        specialization_bonus = 1.0
        if node.claim_type in verifier.supported_claim_types:
            # More specialized verifiers get bonus
            num_supported = len(verifier.supported_claim_types)
            specialization_bonus = 1.0 + (1.0 / max(1, num_supported))
        
        return base_reduction * complexity_penalty * specialization_bonus
    
    def _calculate_diversity_bonus(self, verifier: BaseVerifier) -> float:
        """Calculate diversity bonus to encourage verifier variety"""
        if not self.diversity_weight:
            return 0.0
        
        # Bonus inversely related to recent usage
        usage_count = self.verifier_usage_counts[verifier.verifier_id]
        total_usage = sum(self.verifier_usage_counts.values())
        
        if total_usage == 0:
            return 0.0
        
        usage_ratio = usage_count / total_usage
        diversity_bonus = self.diversity_weight * (1.0 - usage_ratio)
        
        return diversity_bonus
    
    def _determine_redundancy(self, node: DebateNode, context: Dict[str, any]) -> int:
        """Determine how many parallel verifiers to use"""
        # Base redundancy decision
        use_redundancy = random.random() < self.redundancy_prob
        
        if not use_redundancy:
            return 1
        
        # Determine redundancy level based on importance and uncertainty
        importance = context.get('importance', 1.0)
        uncertainty = node.uncertainty
        
        # Higher importance and uncertainty -> more redundancy
        redundancy_score = importance * uncertainty
        
        if redundancy_score > 0.8:
            return min(3, self.max_redundancy)
        elif redundancy_score > 0.6:
            return min(2, self.max_redundancy)
        else:
            return 1
    
    def _estimate_accuracy(self, 
                          verifiers: List[BaseVerifier],
                          node: DebateNode,
                          context: Dict[str, any]) -> float:
        """Estimate expected accuracy of verification"""
        if not verifiers:
            return 0.0
        
        # For single verifier, use historical performance
        if len(verifiers) == 1:
            verifier = verifiers[0]
            if verifier.accuracy_history:
                return sum(verifier.accuracy_history[-10:]) / min(10, len(verifier.accuracy_history))
            else:
                return 1.0 - verifier.false_negative_rate
        
        # For multiple verifiers, estimate ensemble accuracy
        individual_accuracies = []
        for verifier in verifiers:
            if verifier.accuracy_history:
                acc = sum(verifier.accuracy_history[-10:]) / min(10, len(verifier.accuracy_history))
            else:
                acc = 1.0 - verifier.false_negative_rate
            individual_accuracies.append(acc)
        
        # Simple ensemble accuracy estimate (majority vote)
        if len(individual_accuracies) == 2:
            p1, p2 = individual_accuracies
            # Probability both correct OR one correct and other incorrect
            ensemble_acc = p1 * p2 + p1 * (1 - p2) + (1 - p1) * p2
        else:
            # For more than 2, approximate as best individual + diversity bonus
            ensemble_acc = max(individual_accuracies) + 0.1
        
        return min(1.0, ensemble_acc)
    
    def update_verifier_performance(self, 
                                  verifier_id: str, 
                                  accuracy: float) -> None:
        """Update performance tracking for a verifier"""
        if verifier_id in self.verifiers:
            self.verifier_history[verifier_id].append(accuracy)
            
            # Keep history bounded
            if len(self.verifier_history[verifier_id]) > 100:
                self.verifier_history[verifier_id].pop(0)
    
    def set_budget(self, budget: float) -> None:
        """Set remaining cost budget"""
        self.cost_budget_remaining = budget
    
    def get_routing_stats(self) -> Dict[str, any]:
        """Get routing statistics"""
        if not self.routing_history:
            return {}
        
        total_decisions = len(self.routing_history)
        delegated = sum(1 for d in self.routing_history if d.should_delegate)
        
        avg_cost = sum(d.estimated_cost for d in self.routing_history) / total_decisions
        avg_accuracy = sum(d.expected_accuracy for d in self.routing_history) / total_decisions
        
        verifier_usage = dict(self.verifier_usage_counts)
        
        return {
            'total_decisions': total_decisions,
            'delegation_rate': delegated / total_decisions,
            'average_cost': avg_cost,
            'average_expected_accuracy': avg_accuracy,
            'verifier_usage': verifier_usage,
            'redundancy_usage': sum(1 for d in self.routing_history if d.redundancy_level > 1),
            'budget_remaining': self.cost_budget_remaining
        }
