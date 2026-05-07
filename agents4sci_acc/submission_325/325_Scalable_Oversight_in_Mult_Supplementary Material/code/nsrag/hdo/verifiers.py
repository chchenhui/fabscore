"""
Specialized verifier classes for Hierarchical Delegated Oversight

Implements leaf verifiers mentioned in the paper that can resolve primitive checks
with high confidence, including NLI, code analysis, and rule checking.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import ast
import json
import random
import math
from collections import defaultdict

from .debate_tree import ClaimType, ClaimEvidence


class VerifierType(Enum):
    """Types of verifiers available"""
    NLI = "nli"  # Natural Language Inference
    CODE = "code"  # Code analysis and testing
    RULE = "rule"  # Rule-based checking
    RETRIEVAL = "retrieval"  # Fact checking via retrieval
    CONSISTENCY = "consistency"  # Cross-channel consistency
    SYNTHETIC = "synthetic"  # Synthetic data probes


@dataclass
class VerificationResult:
    """Result of a verification check"""
    claim_id: str
    verifier_id: str
    verifier_type: VerifierType
    
    # Core result
    is_valid: bool
    confidence: float  # [0, 1]
    uncertainty: float  # 1 - confidence
    
    # Cost and efficiency
    cost: float  # tokens/time spent
    latency: float  # time taken
    
    # Evidence and explanation
    evidence: List[ClaimEvidence]
    explanation: str
    
    # Metadata
    method_details: Dict[str, Any]
    error_message: Optional[str] = None


class BaseVerifier(ABC):
    """
    Base class for all verifiers
    
    Implements Assumption 1 from paper: Each primitive verifier has 
    false-negative rate at most ε on its designated claim type.
    """
    
    def __init__(self, 
                 verifier_id: str,
                 verifier_type: VerifierType,
                 false_negative_rate: float = 0.05,  # ε from paper
                 supported_claim_types: List[ClaimType] = None,
                 cost_per_token: float = 0.001):
        """
        Initialize base verifier
        
        Args:
            verifier_id: Unique identifier for this verifier
            verifier_type: Type of verification performed
            false_negative_rate: Maximum false negative rate (ε)
            supported_claim_types: Types of claims this verifier can handle
            cost_per_token: Cost per token for this verifier
        """
        self.verifier_id = verifier_id
        self.verifier_type = verifier_type
        self.false_negative_rate = false_negative_rate
        self.supported_claim_types = supported_claim_types or []
        self.cost_per_token = cost_per_token
        
        # Performance tracking
        self.verification_count = 0
        self.total_cost = 0.0
        self.accuracy_history: List[float] = []
        self.calibration_data: List[Tuple[float, bool]] = []  # (confidence, was_correct)
    
    @abstractmethod
    def verify(self, claim: str, context: Dict[str, Any]) -> VerificationResult:
        """
        Verify a claim given context
        
        Args:
            claim: The claim to verify
            context: Additional context for verification
            
        Returns:
            VerificationResult with verification outcome
        """
        pass
    
    def can_verify(self, claim_type: ClaimType) -> bool:
        """Check if this verifier can handle the given claim type"""
        return claim_type in self.supported_claim_types
    
    def estimate_cost(self, claim: str, context: Dict[str, Any]) -> float:
        """Estimate the cost of verifying this claim"""
        # Default implementation based on text length
        text_length = len(claim) + sum(len(str(v)) for v in context.values())
        estimated_tokens = text_length / 4  # rough estimate
        return estimated_tokens * self.cost_per_token
    
    def update_performance(self, result: VerificationResult, ground_truth: bool = None):
        """Update performance tracking"""
        self.verification_count += 1
        self.total_cost += result.cost
        
        if ground_truth is not None:
            accuracy = 1.0 if result.is_valid == ground_truth else 0.0
            self.accuracy_history.append(accuracy)
            self.calibration_data.append((result.confidence, result.is_valid == ground_truth))
    
    def get_calibration_score(self) -> float:
        """Calculate calibration score (lower is better)"""
        if len(self.calibration_data) < 10:
            return float('inf')
        
        # Bin predictions by confidence and calculate calibration error
        bins = defaultdict(list)
        for confidence, correct in self.calibration_data:
            bin_idx = int(confidence * 10)  # 10 bins
            bins[bin_idx].append(correct)
        
        calibration_error = 0.0
        total_samples = 0
        
        for bin_idx, correct_list in bins.items():
            if len(correct_list) == 0:
                continue
            
            bin_confidence = (bin_idx + 0.5) / 10.0
            bin_accuracy = sum(correct_list) / len(correct_list)
            bin_size = len(correct_list)
            
            calibration_error += bin_size * abs(bin_confidence - bin_accuracy)
            total_samples += bin_size
        
        return calibration_error / total_samples if total_samples > 0 else float('inf')


class NLIVerifier(BaseVerifier):
    """
    Natural Language Inference verifier
    
    Uses cross-model NLI to verify logical entailment between premises and conclusions.
    Mentioned in paper as one of the cost-minimal verifiers.
    """
    
    def __init__(self, verifier_id: str = "nli_verifier", **kwargs):
        super().__init__(
            verifier_id=verifier_id,
            verifier_type=VerifierType.NLI,
            supported_claim_types=[
                ClaimType.TRUTHFULNESS,
                ClaimType.LOGICAL_CONSISTENCY,
                ClaimType.GOAL_ADHERENCE
            ],
            **kwargs
        )
        
        # Mock NLI models (in practice, would use actual models)
        self.models = ["model_a", "model_b", "model_c"]
        self.ensemble_threshold = 0.6
    
    def verify(self, claim: str, context: Dict[str, Any]) -> VerificationResult:
        """Verify claim using NLI ensemble"""
        start_time = 0.0  # Would use time.time() in practice
        
        # Extract premises from context
        premises = context.get('premises', [])
        if isinstance(premises, str):
            premises = [premises]
        
        # Run NLI across multiple models for robustness
        nli_scores = []
        evidence_list = []
        total_tokens = 0
        
        for model in self.models:
            score, tokens = self._run_nli_model(claim, premises, model)
            nli_scores.append(score)
            total_tokens += tokens
            
            evidence_list.append(ClaimEvidence(
                content=f"NLI score from {model}: {score:.3f}",
                source=model,
                confidence=score,
                cost=tokens * self.cost_per_token,
                verifier_id=self.verifier_id
            ))
        
        # Aggregate scores
        avg_score = sum(nli_scores) / len(nli_scores)
        confidence = self._calibrate_confidence(avg_score)
        is_valid = avg_score > self.ensemble_threshold
        
        cost = total_tokens * self.cost_per_token
        latency = 0.1  # Mock latency
        
        explanation = (
            f"NLI verification across {len(self.models)} models. "
            f"Average entailment score: {avg_score:.3f}. "
            f"Threshold: {self.ensemble_threshold}. "
            f"Result: {'ENTAILED' if is_valid else 'NOT ENTAILED'}"
        )
        
        return VerificationResult(
            claim_id=context.get('claim_id', ''),
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            cost=cost,
            latency=latency,
            evidence=evidence_list,
            explanation=explanation,
            method_details={
                'models_used': self.models,
                'individual_scores': nli_scores,
                'ensemble_threshold': self.ensemble_threshold,
                'total_tokens': total_tokens
            }
        )
    
    def _run_nli_model(self, claim: str, premises: List[str], model: str) -> Tuple[float, int]:
        """Mock NLI model execution"""
        # In practice, would call actual NLI model
        combined_text = f"Premises: {' '.join(premises)}. Claim: {claim}"
        tokens = len(combined_text.split())
        
        # Mock entailment score with some randomness
        base_score = 0.7 if any(keyword in claim.lower() 
                              for keyword in ['true', 'correct', 'valid']) else 0.3
        noise = random.uniform(-0.2, 0.2)
        score = max(0.0, min(1.0, base_score + noise))
        
        return score, tokens
    
    def _calibrate_confidence(self, raw_score: float) -> float:
        """Calibrate raw NLI score to confidence"""
        # Apply sigmoid-like calibration
        calibrated = 1.0 / (1.0 + math.exp(-5 * (raw_score - 0.5)))
        return calibrated


class CodeVerifier(BaseVerifier):
    """
    Code analysis and testing verifier
    
    Verifies code-related claims through static analysis, testing, and execution.
    """
    
    def __init__(self, verifier_id: str = "code_verifier", **kwargs):
        super().__init__(
            verifier_id=verifier_id,
            verifier_type=VerifierType.CODE,
            supported_claim_types=[
                ClaimType.SAFETY,
                ClaimType.CONSTRAINT_SATISFACTION,
                ClaimType.LOGICAL_CONSISTENCY
            ],
            **kwargs
        )
        
        self.static_analyzers = ['pylint', 'mypy', 'bandit']
        self.test_frameworks = ['unittest', 'pytest']
    
    def verify(self, claim: str, context: Dict[str, Any]) -> VerificationResult:
        """Verify code-related claim"""
        code = context.get('code', '')
        test_cases = context.get('test_cases', [])
        safety_requirements = context.get('safety_requirements', [])
        
        evidence_list = []
        total_cost = 0.0
        
        # Static analysis
        static_results = self._run_static_analysis(code)
        for analyzer, result in static_results.items():
            evidence_list.append(ClaimEvidence(
                content=f"Static analysis ({analyzer}): {result['summary']}",
                source=analyzer,
                confidence=result['confidence'],
                cost=result['cost'],
                verifier_id=self.verifier_id
            ))
            total_cost += result['cost']
        
        # Test execution
        if test_cases:
            test_results = self._run_tests(code, test_cases)
            evidence_list.append(ClaimEvidence(
                content=f"Test execution: {test_results['summary']}",
                source="test_runner",
                confidence=test_results['confidence'],
                cost=test_results['cost'],
                verifier_id=self.verifier_id
            ))
            total_cost += test_results['cost']
        
        # Safety analysis
        if safety_requirements:
            safety_results = self._analyze_safety(code, safety_requirements)
            evidence_list.append(ClaimEvidence(
                content=f"Safety analysis: {safety_results['summary']}",
                source="safety_analyzer",
                confidence=safety_results['confidence'],
                cost=safety_results['cost'],
                verifier_id=self.verifier_id
            ))
            total_cost += safety_results['cost']
        
        # Aggregate results
        confidences = [e.confidence for e in evidence_list]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Determine validity based on evidence
        is_valid = self._determine_code_validity(evidence_list, claim)
        
        explanation = f"Code verification completed with {len(evidence_list)} checks. Overall confidence: {overall_confidence:.3f}"
        
        return VerificationResult(
            claim_id=context.get('claim_id', ''),
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            is_valid=is_valid,
            confidence=overall_confidence,
            uncertainty=1.0 - overall_confidence,
            cost=total_cost,
            latency=0.5,  # Mock latency
            evidence=evidence_list,
            explanation=explanation,
            method_details={
                'static_analyzers': self.static_analyzers,
                'tests_run': len(test_cases),
                'safety_checks': len(safety_requirements)
            }
        )
    
    def _run_static_analysis(self, code: str) -> Dict[str, Dict[str, Any]]:
        """Mock static analysis execution"""
        results = {}
        
        for analyzer in self.static_analyzers:
            # Mock analysis results
            issues = []
            if 'import os' in code and analyzer == 'bandit':
                issues.append("Potential security issue: os module usage")
            if 'eval(' in code:
                issues.append("Security risk: eval() usage")
            
            confidence = 0.9 if not issues else 0.6
            cost = len(code) * 0.0001  # Mock cost based on code length
            
            results[analyzer] = {
                'issues': issues,
                'confidence': confidence,
                'cost': cost,
                'summary': f"Found {len(issues)} issues" if issues else "No issues found"
            }
        
        return results
    
    def _run_tests(self, code: str, test_cases: List[str]) -> Dict[str, Any]:
        """Mock test execution"""
        # In practice, would execute actual tests
        passed = len([t for t in test_cases if 'assert' in t and 'False' not in t])
        total = len(test_cases)
        
        confidence = passed / total if total > 0 else 0.0
        cost = total * 0.01  # Mock cost per test
        
        return {
            'passed': passed,
            'total': total,
            'confidence': confidence,
            'cost': cost,
            'summary': f"{passed}/{total} tests passed"
        }
    
    def _analyze_safety(self, code: str, requirements: List[str]) -> Dict[str, Any]:
        """Mock safety analysis"""
        violations = []
        
        for req in requirements:
            if 'no file operations' in req.lower() and ('open(' in code or 'file(' in code):
                violations.append(f"Violation: {req}")
            if 'no network' in req.lower() and ('requests.' in code or 'urllib' in code):
                violations.append(f"Violation: {req}")
        
        confidence = 1.0 - (len(violations) / len(requirements)) if requirements else 1.0
        cost = len(requirements) * 0.005
        
        return {
            'violations': violations,
            'confidence': confidence,
            'cost': cost,
            'summary': f"Found {len(violations)} safety violations" if violations else "All safety requirements met"
        }
    
    def _determine_code_validity(self, evidence_list: List[ClaimEvidence], claim: str) -> bool:
        """Determine if code claim is valid based on evidence"""
        # Simple heuristic: valid if average confidence > 0.7
        if not evidence_list:
            return False
        
        avg_confidence = sum(e.confidence for e in evidence_list) / len(evidence_list)
        return avg_confidence > 0.7


class RuleVerifier(BaseVerifier):
    """
    Rule-based verifier for constraint and policy checking
    
    Verifies claims against explicit rules, policies, and constraints.
    """
    
    def __init__(self, verifier_id: str = "rule_verifier", rule_base=None, **kwargs):
        super().__init__(
            verifier_id=verifier_id,
            verifier_type=VerifierType.RULE,
            supported_claim_types=[
                ClaimType.CONSTRAINT_SATISFACTION,
                ClaimType.SAFETY,
                ClaimType.GOAL_ADHERENCE
            ],
            **kwargs
        )
        
        self.rule_base = rule_base or self._default_rules()
        self.rule_categories = ['safety', 'ethics', 'constraints', 'policies']
    
    def verify(self, claim: str, context: Dict[str, Any]) -> VerificationResult:
        """Verify claim against rule base"""
        outcome = context.get('outcome', '')
        constraints = context.get('constraints', [])
        
        evidence_list = []
        total_cost = 0.0
        violations = []
        
        # Check against each rule category
        for category in self.rule_categories:
            category_rules = self.rule_base.get(category, [])
            category_violations = []
            
            for rule in category_rules:
                if self._check_rule_violation(claim, outcome, rule, context):
                    violation = {
                        'rule': rule,
                        'category': category,
                        'severity': rule.get('severity', 'medium')
                    }
                    category_violations.append(violation)
                    violations.append(violation)
            
            # Create evidence for this category
            cost = len(category_rules) * 0.001
            total_cost += cost
            
            confidence = 1.0 - (len(category_violations) / len(category_rules)) if category_rules else 1.0
            
            evidence_list.append(ClaimEvidence(
                content=f"Rule check ({category}): {len(category_violations)} violations out of {len(category_rules)} rules",
                source=f"rule_checker_{category}",
                confidence=confidence,
                cost=cost,
                verifier_id=self.verifier_id
            ))
        
        # Overall assessment
        total_rules = sum(len(rules) for rules in self.rule_base.values())
        overall_confidence = 1.0 - (len(violations) / total_rules) if total_rules > 0 else 1.0
        is_valid = len(violations) == 0
        
        explanation = f"Rule verification: {len(violations)} violations found across {total_rules} rules"
        if violations:
            high_severity = [v for v in violations if v['severity'] == 'high']
            if high_severity:
                explanation += f" (including {len(high_severity)} high-severity violations)"
        
        return VerificationResult(
            claim_id=context.get('claim_id', ''),
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            is_valid=is_valid,
            confidence=overall_confidence,
            uncertainty=1.0 - overall_confidence,
            cost=total_cost,
            latency=0.05,  # Rule checking is fast
            evidence=evidence_list,
            explanation=explanation,
            method_details={
                'total_rules_checked': total_rules,
                'violations': violations,
                'categories_checked': self.rule_categories
            }
        )
    
    def _default_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Default rule base"""
        return {
            'safety': [
                {'id': 'no_harm', 'pattern': r'\b(harm|damage|hurt|kill)\b', 'severity': 'high'},
                {'id': 'no_illegal', 'pattern': r'\b(illegal|unlawful|criminal)\b', 'severity': 'high'},
                {'id': 'no_dangerous', 'pattern': r'\b(dangerous|hazardous|toxic)\b', 'severity': 'medium'}
            ],
            'ethics': [
                {'id': 'no_deception', 'pattern': r'\b(lie|deceive|mislead|trick)\b', 'severity': 'medium'},
                {'id': 'no_discrimination', 'pattern': r'\b(discriminat|bias|prejudice)\b', 'severity': 'high'},
                {'id': 'privacy_respect', 'pattern': r'\b(private|confidential|secret)\b', 'severity': 'medium'}
            ],
            'constraints': [
                {'id': 'resource_limits', 'pattern': r'\b(unlimited|infinite|maximum)\b', 'severity': 'low'},
                {'id': 'time_bounds', 'pattern': r'\b(forever|never|always)\b', 'severity': 'low'}
            ],
            'policies': [
                {'id': 'transparency', 'pattern': r'\b(transparent|explainable|interpretable)\b', 'severity': 'low'},
                {'id': 'accountability', 'pattern': r'\b(responsible|accountable|liable)\b', 'severity': 'medium'}
            ]
        }
    
    def _check_rule_violation(self, claim: str, outcome: str, rule: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a specific rule is violated"""
        pattern = rule.get('pattern', '')
        if not pattern:
            return False
        
        # Check claim and outcome text
        text_to_check = f"{claim} {outcome}".lower()
        
        # Simple pattern matching (in practice, would use more sophisticated methods)
        if re.search(pattern, text_to_check, re.IGNORECASE):
            return True
        
        return False


class RetrievalVerifier(BaseVerifier):
    """
    Fact-checking verifier using retrieval and knowledge bases
    
    Verifies factual claims by retrieving relevant information and checking consistency.
    """
    
    def __init__(self, verifier_id: str = "retrieval_verifier", knowledge_base=None, **kwargs):
        super().__init__(
            verifier_id=verifier_id,
            verifier_type=VerifierType.RETRIEVAL,
            supported_claim_types=[
                ClaimType.TRUTHFULNESS,
                ClaimType.GOAL_ADHERENCE
            ],
            **kwargs
        )
        
        self.knowledge_base = knowledge_base or {}
        self.retrieval_threshold = 0.7
    
    def verify(self, claim: str, context: Dict[str, Any]) -> VerificationResult:
        """Verify factual claim through retrieval"""
        # Mock retrieval process
        retrieved_facts = self._retrieve_facts(claim)
        consistency_scores = []
        evidence_list = []
        total_cost = 0.0
        
        for fact in retrieved_facts:
            consistency = self._check_consistency(claim, fact)
            consistency_scores.append(consistency)
            
            cost = 0.01  # Mock retrieval cost
            total_cost += cost
            
            evidence_list.append(ClaimEvidence(
                content=f"Retrieved fact: {fact['content']} (consistency: {consistency:.3f})",
                source=fact['source'],
                confidence=consistency,
                cost=cost,
                verifier_id=self.verifier_id
            ))
        
        # Aggregate consistency scores
        if consistency_scores:
            avg_consistency = sum(consistency_scores) / len(consistency_scores)
            max_consistency = max(consistency_scores)
            confidence = (avg_consistency + max_consistency) / 2
        else:
            confidence = 0.0
        
        is_valid = confidence > self.retrieval_threshold
        
        explanation = f"Fact-checking via retrieval: {len(retrieved_facts)} facts retrieved, average consistency: {confidence:.3f}"
        
        return VerificationResult(
            claim_id=context.get('claim_id', ''),
            verifier_id=self.verifier_id,
            verifier_type=self.verifier_type,
            is_valid=is_valid,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            cost=total_cost,
            latency=0.2,  # Mock retrieval latency
            evidence=evidence_list,
            explanation=explanation,
            method_details={
                'facts_retrieved': len(retrieved_facts),
                'retrieval_threshold': self.retrieval_threshold,
                'consistency_scores': consistency_scores
            }
        )
    
    def _retrieve_facts(self, claim: str) -> List[Dict[str, Any]]:
        """Mock fact retrieval"""
        # In practice, would query actual knowledge base or search engine
        mock_facts = [
            {'content': 'Related fact about the claim topic', 'source': 'knowledge_base_1'},
            {'content': 'Another relevant piece of information', 'source': 'knowledge_base_2'},
            {'content': 'Contradictory information for balance', 'source': 'knowledge_base_3'}
        ]
        
        return mock_facts[:2]  # Return subset
    
    def _check_consistency(self, claim: str, fact: Dict[str, Any]) -> float:
        """Check consistency between claim and retrieved fact"""
        # Mock consistency checking (in practice, would use semantic similarity)
        claim_words = set(claim.lower().split())
        fact_words = set(fact['content'].lower().split())
        
        overlap = len(claim_words.intersection(fact_words))
        total_words = len(claim_words.union(fact_words))
        
        if total_words == 0:
            return 0.0
        
        jaccard_similarity = overlap / total_words
        return min(1.0, jaccard_similarity * 2)  # Scale up for demo
