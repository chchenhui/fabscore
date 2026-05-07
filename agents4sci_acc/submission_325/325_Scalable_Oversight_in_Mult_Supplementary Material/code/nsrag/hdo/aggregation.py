"""
Entailment aggregation for Hierarchical Delegated Oversight

Implements aggregation logic for combining verifier results as described in the paper:
"Parents aggregate via logic-specific rules: conjunctions for conjunctive premises, 
max for risk contributions, or learned aggregators calibrated on held-out oversight episodes."
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
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
from .verifiers import VerificationResult


class AggregationMethod(Enum):
    """Methods for aggregating verification results"""
    CONJUNCTION = "conjunction"  # All must be true (AND)
    DISJUNCTION = "disjunction"  # At least one must be true (OR)
    MAJORITY = "majority"  # Majority vote
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted by confidence
    MAX_RISK = "max_risk"  # Take maximum risk (for safety)
    MIN_CONFIDENCE = "min_confidence"  # Take minimum confidence
    LEARNED = "learned"  # Learned aggregator
    EVIDENCE_BALANCE = "evidence_balance"  # Based on evidence strength


@dataclass
class AggregationResult:
    """Result of aggregating multiple verifications"""
    node_id: str
    method: AggregationMethod
    
    # Aggregated result
    is_valid: bool
    confidence: float
    uncertainty: float
    
    # Input verifications
    input_results: List[VerificationResult]
    child_results: List['AggregationResult'] = None
    
    # Aggregation details
    weights: Dict[str, float] = None
    method_details: Dict[str, Any] = None
    explanation: str = ""
    
    # Meta information
    total_cost: float = 0.0
    aggregation_confidence: float = 1.0  # Confidence in aggregation itself


class EntailmentAggregator:
    """
    Aggregates verification results using various methods
    
    Implements the aggregation logic described in the paper for combining
    leaf verifier results and propagating decisions up the debate tree.
    """
    
    def __init__(self,
                 default_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
                 risk_averse: bool = True,
                 calibration_data: Dict[str, List] = None):
        """
        Initialize entailment aggregator
        
        Args:
            default_method: Default aggregation method
            risk_averse: Whether to be conservative in aggregation
            calibration_data: Historical data for calibrating aggregators
        """
        self.default_method = default_method
        self.risk_averse = risk_averse
        self.calibration_data = calibration_data or {}
        
        # Method-specific configurations
        self.method_configs = {
            AggregationMethod.CONJUNCTION: {'threshold': 0.9},
            AggregationMethod.DISJUNCTION: {'threshold': 0.1},
            AggregationMethod.MAJORITY: {'tie_break': 'conservative'},
            AggregationMethod.WEIGHTED_AVERAGE: {'confidence_power': 2.0},
            AggregationMethod.MAX_RISK: {'risk_amplification': 1.2},
            AggregationMethod.MIN_CONFIDENCE: {'safety_margin': 0.1}
        }
        
        # Learned aggregator parameters (would be trained in practice)
        self.learned_weights = defaultdict(lambda: 1.0)
        self.aggregation_history: List[AggregationResult] = []
    
    def aggregate_verifications(self,
                              node: DebateNode,
                              verification_results: List[VerificationResult],
                              method: AggregationMethod = None) -> AggregationResult:
        """
        Aggregate multiple verification results for a single node
        
        Args:
            node: The debate node being verified
            verification_results: List of verification results to aggregate
            method: Aggregation method to use (defaults to node-specific or global default)
            
        Returns:
            AggregationResult with combined decision
        """
        if not verification_results:
            return AggregationResult(
                node_id=node.id,
                method=AggregationMethod.CONJUNCTION,
                is_valid=False,
                confidence=0.0,
                uncertainty=1.0,
                input_results=[],
                explanation="No verification results to aggregate"
            )
        
        # Determine aggregation method
        aggregation_method = method or self._select_method(node, verification_results)
        
        # Perform aggregation based on method
        if aggregation_method == AggregationMethod.CONJUNCTION:
            result = self._aggregate_conjunction(node, verification_results)
        elif aggregation_method == AggregationMethod.DISJUNCTION:
            result = self._aggregate_disjunction(node, verification_results)
        elif aggregation_method == AggregationMethod.MAJORITY:
            result = self._aggregate_majority(node, verification_results)
        elif aggregation_method == AggregationMethod.WEIGHTED_AVERAGE:
            result = self._aggregate_weighted_average(node, verification_results)
        elif aggregation_method == AggregationMethod.MAX_RISK:
            result = self._aggregate_max_risk(node, verification_results)
        elif aggregation_method == AggregationMethod.MIN_CONFIDENCE:
            result = self._aggregate_min_confidence(node, verification_results)
        elif aggregation_method == AggregationMethod.LEARNED:
            result = self._aggregate_learned(node, verification_results)
        elif aggregation_method == AggregationMethod.EVIDENCE_BALANCE:
            result = self._aggregate_evidence_balance(node, verification_results)
        else:
            # Fallback to weighted average
            result = self._aggregate_weighted_average(node, verification_results)
        
        # Post-process result
        result.method = aggregation_method
        result.total_cost = sum(vr.cost for vr in verification_results)
        result.input_results = verification_results
        
        # Record for learning
        self.aggregation_history.append(result)
        
        return result
    
    def aggregate_tree(self, 
                      root_node: DebateNode,
                      node_results: Dict[str, List[VerificationResult]],
                      tree_structure: Dict[str, List[str]]) -> AggregationResult:
        """
        Aggregate results across an entire debate tree
        
        Args:
            root_node: Root node of the debate tree
            node_results: Map from node_id to verification results
            tree_structure: Map from parent_id to list of child_ids
            
        Returns:
            AggregationResult for the root node
        """
        # Bottom-up aggregation
        aggregated_results = {}
        
        def aggregate_node(node_id: str, node: DebateNode) -> AggregationResult:
            # If already processed, return cached result
            if node_id in aggregated_results:
                return aggregated_results[node_id]
            
            # Get direct verification results for this node
            direct_results = node_results.get(node_id, [])
            
            # Get child results
            child_ids = tree_structure.get(node_id, [])
            child_results = []
            
            for child_id in child_ids:
                # Recursively aggregate child (assuming we have child nodes)
                # In practice, would need access to child nodes
                child_result = aggregate_node(child_id, node)  # Simplified
                child_results.append(child_result)
            
            # Combine direct results with child results
            all_results = direct_results.copy()
            
            # Convert child aggregation results to verification results for combination
            for child_result in child_results:
                synthetic_verification = VerificationResult(
                    claim_id=child_result.node_id,
                    verifier_id="child_aggregator",
                    verifier_type="aggregation",
                    is_valid=child_result.is_valid,
                    confidence=child_result.confidence,
                    uncertainty=child_result.uncertainty,
                    cost=child_result.total_cost,
                    latency=0.0,
                    evidence=[],
                    explanation=f"Child node aggregation: {child_result.explanation}",
                    method_details=child_result.method_details or {}
                )
                all_results.append(synthetic_verification)
            
            # Aggregate all results for this node
            result = self.aggregate_verifications(node, all_results)
            result.child_results = child_results
            
            aggregated_results[node_id] = result
            return result
        
        return aggregate_node(root_node.id, root_node)
    
    def _select_method(self, 
                      node: DebateNode, 
                      verification_results: List[VerificationResult]) -> AggregationMethod:
        """Select appropriate aggregation method based on context"""
        
        # Safety-critical claims use conservative methods
        if node.claim_type in [ClaimType.SAFETY, ClaimType.CONSTRAINT_SATISFACTION]:
            return AggregationMethod.MIN_CONFIDENCE if self.risk_averse else AggregationMethod.CONJUNCTION
        
        # Factual claims use evidence-based methods
        if node.claim_type == ClaimType.TRUTHFULNESS:
            return AggregationMethod.EVIDENCE_BALANCE
        
        # Logical consistency uses conjunction
        if node.claim_type == ClaimType.LOGICAL_CONSISTENCY:
            return AggregationMethod.CONJUNCTION
        
        # For alignment and goal adherence, use weighted average
        if node.claim_type in [ClaimType.ALIGNMENT, ClaimType.GOAL_ADHERENCE]:
            return AggregationMethod.WEIGHTED_AVERAGE
        
        return self.default_method
    
    def _aggregate_conjunction(self, 
                             node: DebateNode,
                             results: List[VerificationResult]) -> AggregationResult:
        """Aggregate using conjunction (all must be valid)"""
        config = self.method_configs[AggregationMethod.CONJUNCTION]
        threshold = config['threshold']
        
        # All must be valid and above threshold
        all_valid = all(r.is_valid for r in results)
        all_confident = all(r.confidence >= threshold for r in results)
        
        is_valid = all_valid and all_confident
        
        # Confidence is minimum of all confidences
        confidence = min(r.confidence for r in results) if results else 0.0
        
        explanation = f"Conjunction aggregation: {sum(r.is_valid for r in results)}/{len(results)} valid, min confidence: {confidence:.3f}"
        
        return AggregationResult(
            node_id=node.id,
            method=AggregationMethod.CONJUNCTION,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=explanation,
            method_details={'threshold': threshold, 'all_valid': all_valid, 'all_confident': all_confident}
        )
    
    def _aggregate_disjunction(self,
                             node: DebateNode,
                             results: List[VerificationResult]) -> AggregationResult:
        """Aggregate using disjunction (at least one must be valid)"""
        config = self.method_configs[AggregationMethod.DISJUNCTION]
        threshold = config['threshold']
        
        # At least one must be valid and confident
        any_valid = any(r.is_valid for r in results)
        any_confident = any(r.confidence >= threshold for r in results)
        
        is_valid = any_valid and any_confident
        
        # Confidence is maximum of all confidences
        confidence = max(r.confidence for r in results) if results else 0.0
        
        explanation = f"Disjunction aggregation: {sum(r.is_valid for r in results)}/{len(results)} valid, max confidence: {confidence:.3f}"
        
        return AggregationResult(
            node_id=node.id,
            method=AggregationMethod.DISJUNCTION,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=explanation,
            method_details={'threshold': threshold, 'any_valid': any_valid, 'any_confident': any_confident}
        )
    
    def _aggregate_majority(self,
                          node: DebateNode,
                          results: List[VerificationResult]) -> AggregationResult:
        """Aggregate using majority vote"""
        config = self.method_configs[AggregationMethod.MAJORITY]
        tie_break = config['tie_break']
        
        valid_votes = sum(1 for r in results if r.is_valid)
        total_votes = len(results)
        
        if valid_votes > total_votes / 2:
            is_valid = True
        elif valid_votes < total_votes / 2:
            is_valid = False
        else:
            # Tie - use tie breaking strategy
            is_valid = False if tie_break == 'conservative' else True
        
        # Confidence based on vote strength and individual confidences
        vote_strength = abs(valid_votes - (total_votes - valid_votes)) / total_votes
        avg_confidence = sum(r.confidence for r in results) / total_votes if total_votes > 0 else 0.0
        confidence = vote_strength * avg_confidence
        
        explanation = f"Majority vote: {valid_votes}/{total_votes} valid, vote strength: {vote_strength:.3f}"
        
        return AggregationResult(
            node_id=node.id,
            method=AggregationMethod.MAJORITY,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=explanation,
            method_details={
                'valid_votes': valid_votes,
                'total_votes': total_votes,
                'vote_strength': vote_strength,
                'tie_break': tie_break
            }
        )
    
    def _aggregate_weighted_average(self,
                                  node: DebateNode,
                                  results: List[VerificationResult]) -> AggregationResult:
        """Aggregate using weighted average by confidence"""
        config = self.method_configs[AggregationMethod.WEIGHTED_AVERAGE]
        confidence_power = config['confidence_power']
        
        if not results:
            return AggregationResult(
                node_id=node.id,
                method=AggregationMethod.WEIGHTED_AVERAGE,
                is_valid=False,
                confidence=0.0,
                uncertainty=1.0,
                explanation="No results to aggregate"
            )
        
        # Calculate weights based on confidence
        weights = [r.confidence ** confidence_power for r in results]
        total_weight = sum(weights)
        
        if total_weight == 0:
            # All confidences are 0, use uniform weighting
            weights = [1.0] * len(results)
            total_weight = len(results)
        
        # Weighted average of validity (treating boolean as 0/1)
        weighted_validity = sum(w * (1.0 if r.is_valid else 0.0) for w, r in zip(weights, results)) / total_weight
        
        # Weighted average of confidence
        weighted_confidence = sum(w * r.confidence for w, r in zip(weights, results)) / total_weight
        
        # Final validity decision
        is_valid = weighted_validity > 0.5
        
        # Final confidence combines weighted validity strength and weighted confidence
        final_confidence = (weighted_validity * weighted_confidence + weighted_confidence) / 2
        
        explanation = f"Weighted average: validity={weighted_validity:.3f}, confidence={weighted_confidence:.3f}"
        
        return AggregationResult(
            node_id=node.id,
            method=AggregationMethod.WEIGHTED_AVERAGE,
            is_valid=is_valid,
            confidence=final_confidence,
            uncertainty=1.0 - final_confidence,
            explanation=explanation,
            weights={r.verifier_id: w/total_weight for r, w in zip(results, weights)},
            method_details={
                'weighted_validity': weighted_validity,
                'weighted_confidence': weighted_confidence,
                'confidence_power': confidence_power
            }
        )
    
    def _aggregate_max_risk(self,
                          node: DebateNode,
                          results: List[VerificationResult]) -> AggregationResult:
        """Aggregate by taking maximum risk (minimum confidence)"""
        config = self.method_configs[AggregationMethod.MAX_RISK]
        risk_amplification = config['risk_amplification']
        
        # Find result with highest risk (lowest confidence)
        min_confidence_result = min(results, key=lambda r: r.confidence)
        max_risk = 1.0 - min_confidence_result.confidence
        
        # Amplify risk for safety
        amplified_risk = min(1.0, max_risk * risk_amplification)
        final_confidence = 1.0 - amplified_risk
        
        # Valid only if the riskiest result is valid and confident enough
        is_valid = min_confidence_result.is_valid and final_confidence > 0.5
        
        explanation = f"Max risk aggregation: highest risk={max_risk:.3f}, amplified={amplified_risk:.3f}"
        
        return AggregationResult(
            node_id=node.id,
            method=AggregationMethod.MAX_RISK,
            is_valid=is_valid,
            confidence=final_confidence,
            uncertainty=amplified_risk,
            explanation=explanation,
            method_details={
                'max_risk_verifier': min_confidence_result.verifier_id,
                'original_risk': max_risk,
                'amplified_risk': amplified_risk,
                'risk_amplification': risk_amplification
            }
        )
    
    def _aggregate_min_confidence(self,
                                node: DebateNode,
                                results: List[VerificationResult]) -> AggregationResult:
        """Aggregate by taking minimum confidence (most conservative)"""
        config = self.method_configs[AggregationMethod.MIN_CONFIDENCE]
        safety_margin = config['safety_margin']
        
        # Find minimum confidence
        min_confidence = min(r.confidence for r in results) if results else 0.0
        
        # Apply safety margin
        adjusted_confidence = max(0.0, min_confidence - safety_margin)
        
        # Valid only if all are valid and minimum confidence is sufficient
        all_valid = all(r.is_valid for r in results)
        is_valid = all_valid and adjusted_confidence > 0.5
        
        explanation = f"Min confidence aggregation: min={min_confidence:.3f}, adjusted={adjusted_confidence:.3f}"
        
        return AggregationResult(
            node_id=node.id,
            method=AggregationMethod.MIN_CONFIDENCE,
            is_valid=is_valid,
            confidence=adjusted_confidence,
            uncertainty=1.0 - adjusted_confidence,
            explanation=explanation,
            method_details={
                'original_min_confidence': min_confidence,
                'safety_margin': safety_margin,
                'all_valid': all_valid
            }
        )
    
    def _aggregate_learned(self,
                         node: DebateNode,
                         results: List[VerificationResult]) -> AggregationResult:
        """Aggregate using learned weights (simplified implementation)"""
        # In practice, this would use trained neural network or other ML model
        # Here we use a simple learned weighting scheme
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for result in results:
            # Get learned weight for this verifier type
            weight = self.learned_weights[result.verifier_type.value]
            
            # Score combines validity and confidence
            score = (1.0 if result.is_valid else 0.0) * result.confidence
            
            total_weighted_score += weight * score
            total_weight += weight
        
        if total_weight == 0:
            final_score = 0.0
        else:
            final_score = total_weighted_score / total_weight
        
        is_valid = final_score > 0.5
        confidence = final_score
        
        explanation = f"Learned aggregation: weighted score={final_score:.3f}"
        
        return AggregationResult(
            node_id=node.id,
            method=AggregationMethod.LEARNED,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=explanation,
            weights={r.verifier_type.value: self.learned_weights[r.verifier_type.value] for r in results},
            method_details={'weighted_score': final_score}
        )
    
    def _aggregate_evidence_balance(self,
                                  node: DebateNode,
                                  results: List[VerificationResult]) -> AggregationResult:
        """Aggregate based on balance of evidence"""
        supporting_weight = 0.0
        refuting_weight = 0.0
        
        for result in results:
            weight = result.confidence
            
            if result.is_valid:
                supporting_weight += weight
            else:
                refuting_weight += weight
        
        total_weight = supporting_weight + refuting_weight
        
        if total_weight == 0:
            evidence_balance = 0.0
            confidence = 0.0
        else:
            evidence_balance = (supporting_weight - refuting_weight) / total_weight
            confidence = abs(evidence_balance)
        
        is_valid = evidence_balance > 0
        
        explanation = f"Evidence balance: supporting={supporting_weight:.3f}, refuting={refuting_weight:.3f}, balance={evidence_balance:.3f}"
        
        return AggregationResult(
            node_id=node.id,
            method=AggregationMethod.EVIDENCE_BALANCE,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            explanation=explanation,
            method_details={
                'supporting_weight': supporting_weight,
                'refuting_weight': refuting_weight,
                'evidence_balance': evidence_balance
            }
        )
    
    def calibrate_aggregator(self, 
                           ground_truth_data: List[Tuple[List[VerificationResult], bool]]) -> None:
        """Calibrate aggregator using ground truth data"""
        # Simple calibration - adjust learned weights based on performance
        method_performance = defaultdict(list)
        
        for results, ground_truth in ground_truth_data:
            for method in AggregationMethod:
                # Create dummy node for testing
                dummy_node = DebateNode(
                    id="calibration",
                    claim="calibration claim",
                    claim_type=ClaimType.ALIGNMENT
                )
                
                try:
                    agg_result = self.aggregate_verifications(dummy_node, results, method)
                    accuracy = 1.0 if agg_result.is_valid == ground_truth else 0.0
                    method_performance[method].append(accuracy)
                except:
                    continue
        
        # Update learned weights based on performance
        for verifier_type in ['nli', 'code', 'rule', 'retrieval']:
            if verifier_type in method_performance:
                performance = method_performance[verifier_type]
                if performance:
                    avg_performance = sum(performance) / len(performance)
                    self.learned_weights[verifier_type] = avg_performance
    
    def get_aggregation_stats(self) -> Dict[str, Any]:
        """Get statistics about aggregation performance"""
        if not self.aggregation_history:
            return {}
        
        method_counts = defaultdict(int)
        avg_confidence = 0.0
        avg_cost = 0.0
        
        for result in self.aggregation_history:
            method_counts[result.method.value] += 1
            avg_confidence += result.confidence
            avg_cost += result.total_cost
        
        total_results = len(self.aggregation_history)
        
        return {
            'total_aggregations': total_results,
            'method_distribution': dict(method_counts),
            'average_confidence': avg_confidence / total_results,
            'average_cost': avg_cost / total_results,
            'learned_weights': dict(self.learned_weights)
        }
