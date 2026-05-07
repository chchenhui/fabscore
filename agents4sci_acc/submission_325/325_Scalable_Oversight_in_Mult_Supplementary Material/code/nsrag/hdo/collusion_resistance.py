"""
Collusion resistance mechanisms for Hierarchical Delegated Oversight

Implements anti-collusion measures from the paper:
"HDO mitigates via (i) randomized routing (unpredictable auditors), 
(ii) verifier diversity (architectures/data), (iii) cross-channel consistency checks, 
and (iv) periodic audits of verifiers themselves"
"""

from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import random
import hashlib
import time
import math
from collections import defaultdict, Counter

from .debate_tree import DebateNode, ClaimType
from .verifiers import BaseVerifier, VerificationResult, VerifierType
from .routing import RoutingDecision


class CollusionThreat(Enum):
    """Types of collusion threats"""
    STEGANOGRAPHIC = "steganographic"  # Hidden communication in text
    COORDINATED_BIAS = "coordinated_bias"  # Systematic bias across verifiers
    SYBIL_ATTACK = "sybil_attack"  # Multiple fake identities
    ADVERSARIAL_COOPERATION = "adversarial_cooperation"  # Explicit cooperation
    REWARD_HACKING = "reward_hacking"  # Gaming the oversight system


@dataclass
class CollusionDetection:
    """Detection result for potential collusion"""
    threat_type: CollusionThreat
    confidence: float  # [0, 1] confidence in detection
    involved_verifiers: List[str]
    evidence: List[str]
    severity: str  # "low", "medium", "high"
    recommended_action: str
    detection_timestamp: float = field(default_factory=time.time)


@dataclass
class VerifierProfile:
    """Profile for tracking verifier behavior"""
    verifier_id: str
    verifier_type: VerifierType
    
    # Behavioral patterns
    agreement_patterns: Dict[str, float] = field(default_factory=dict)  # Agreement rates with other verifiers
    confidence_distribution: List[float] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)
    linguistic_features: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    accuracy_over_time: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, accuracy)
    calibration_scores: List[float] = field(default_factory=list)
    
    # Suspicious indicators
    anomaly_scores: List[float] = field(default_factory=list)
    collusion_alerts: List[CollusionDetection] = field(default_factory=list)
    
    # Metadata
    creation_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


class CollusionResistance:
    """
    Anti-collusion system implementing the mechanisms from the paper
    
    Provides randomized routing, verifier diversity, consistency checks,
    and periodic auditing to detect and prevent collusion.
    """
    
    def __init__(self,
                 randomization_strength: float = 0.3,  # Amount of randomization in routing
                 diversity_threshold: float = 0.7,  # Minimum diversity requirement
                 audit_frequency: int = 50,  # Audit every N verifications
                 detection_sensitivity: float = 0.8,  # Sensitivity for collusion detection
                 steganography_detection: bool = True):
        """
        Initialize collusion resistance system
        
        Args:
            randomization_strength: How much randomization to add to routing
            diversity_threshold: Minimum required verifier diversity
            audit_frequency: How often to audit verifiers
            detection_sensitivity: Sensitivity threshold for detection
            steganography_detection: Enable steganographic communication detection
        """
        self.randomization_strength = randomization_strength
        self.diversity_threshold = diversity_threshold
        self.audit_frequency = audit_frequency
        self.detection_sensitivity = detection_sensitivity
        self.steganography_detection = steganography_detection
        
        # Verifier tracking with memory bounds
        self.verifier_profiles: Dict[str, VerifierProfile] = {}
        self.verification_history: List[Tuple[str, VerificationResult]] = []
        self.routing_history: List[RoutingDecision] = []
        self.max_history_size = 10000  # Prevent memory leaks
        
        # Collusion detection with bounds
        self.collusion_detections: List[CollusionDetection] = []
        self.max_detections_history = 1000  # Prevent memory leaks
        self.blacklisted_verifiers: Set[str] = set()
        self.suspicious_pairs: Dict[Tuple[str, str], float] = {}  # Suspicious verifier pairs
        
        # Randomization state
        self.random_seed = int(time.time())
        self.routing_entropy_pool: List[float] = []
        
        # Audit tracking
        self.audit_count = 0
        self.last_audit_time = time.time()
    
    def apply_randomized_routing(self,
                               candidate_verifiers: List[BaseVerifier],
                               node: DebateNode,
                               base_scores: Dict[str, float]) -> List[BaseVerifier]:
        """
        Apply randomized routing to prevent predictable verifier assignment
        
        Args:
            candidate_verifiers: List of candidate verifiers
            node: Node being routed
            base_scores: Base routing scores for each verifier
            
        Returns:
            Reordered list of verifiers with randomization applied
        """
        if not candidate_verifiers:
            return []
        
        # Generate routing entropy
        routing_entropy = self._generate_routing_entropy(node)
        
        # Apply randomization to scores
        randomized_scores = {}
        for verifier in candidate_verifiers:
            base_score = base_scores.get(verifier.verifier_id, 0.0)
            
            # Add randomization noise
            random_noise = (routing_entropy - 0.5) * 2 * self.randomization_strength
            randomized_score = base_score + random_noise
            
            # Apply diversity bonus
            diversity_bonus = self._calculate_diversity_bonus(verifier, node)
            randomized_score += diversity_bonus
            
            # Penalize suspicious verifiers
            suspicion_penalty = self._calculate_suspicion_penalty(verifier.verifier_id)
            randomized_score -= suspicion_penalty
            
            randomized_scores[verifier.verifier_id] = randomized_score
        
        # Sort by randomized scores
        sorted_verifiers = sorted(
            candidate_verifiers,
            key=lambda v: randomized_scores[v.verifier_id],
            reverse=True
        )
        
        return sorted_verifiers
    
    def enforce_verifier_diversity(self,
                                 selected_verifiers: List[BaseVerifier],
                                 all_candidates: List[BaseVerifier]) -> List[BaseVerifier]:
        """
        Ensure sufficient verifier diversity to prevent collusion
        
        Args:
            selected_verifiers: Initially selected verifiers
            all_candidates: All available candidate verifiers
            
        Returns:
            Diversified list of verifiers
        """
        if not selected_verifiers:
            return selected_verifiers
        
        # Calculate current diversity
        current_diversity = self._calculate_verifier_diversity(selected_verifiers)
        
        if current_diversity >= self.diversity_threshold:
            return selected_verifiers
        
        # Need to increase diversity
        diversified_verifiers = selected_verifiers.copy()
        
        # Add verifiers from underrepresented types
        type_counts = Counter(v.verifier_type for v in selected_verifiers)
        
        for candidate in all_candidates:
            if candidate in diversified_verifiers:
                continue
            
            # Check if this verifier type is underrepresented
            if type_counts[candidate.verifier_type] == 0:
                diversified_verifiers.append(candidate)
                type_counts[candidate.verifier_type] += 1
                
                # Recheck diversity
                new_diversity = self._calculate_verifier_diversity(diversified_verifiers)
                if new_diversity >= self.diversity_threshold:
                    break
        
        return diversified_verifiers
    
    def perform_consistency_check(self,
                                node: DebateNode,
                                verification_results: List[VerificationResult],
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform cross-channel consistency checks to detect collusion
        
        Args:
            node: Node being verified
            verification_results: Results from different verifiers
            context: Additional context for consistency checking
            
        Returns:
            Consistency analysis results
        """
        consistency_results = {
            'overall_consistency': 0.0,
            'pairwise_agreements': {},
            'outliers': [],
            'suspicious_patterns': [],
            'confidence_correlation': 0.0
        }
        
        if len(verification_results) < 2:
            consistency_results['overall_consistency'] = 1.0
            return consistency_results
        
        # Check pairwise agreement patterns
        agreements = []
        for i, result1 in enumerate(verification_results):
            for j, result2 in enumerate(verification_results[i+1:], i+1):
                agreement = self._calculate_pairwise_agreement(result1, result2)
                agreements.append(agreement)
                
                pair_key = f"{result1.verifier_id}_{result2.verifier_id}"
                consistency_results['pairwise_agreements'][pair_key] = agreement
                
                # Track suspicious pairs
                verifier_pair = tuple(sorted([result1.verifier_id, result2.verifier_id]))
                if verifier_pair not in self.suspicious_pairs:
                    self.suspicious_pairs[verifier_pair] = 0.0
                
                # Update suspicion if agreement is too high or too low
                if agreement > 0.95 or agreement < 0.05:
                    self.suspicious_pairs[verifier_pair] += 0.1
        
        # Overall consistency
        consistency_results['overall_consistency'] = sum(agreements) / len(agreements) if agreements else 1.0
        
        # Detect outliers
        confidences = [r.confidence for r in verification_results]
        mean_confidence = sum(confidences) / len(confidences)
        std_confidence = math.sqrt(sum((c - mean_confidence)**2 for c in confidences) / len(confidences))
        
        for result in verification_results:
            z_score = abs(result.confidence - mean_confidence) / (std_confidence + 1e-6)
            if z_score > 2.0:  # More than 2 standard deviations
                consistency_results['outliers'].append({
                    'verifier_id': result.verifier_id,
                    'confidence': result.confidence,
                    'z_score': z_score
                })
        
        # Check for suspicious patterns
        consistency_results['suspicious_patterns'] = self._detect_suspicious_patterns(verification_results)
        
        # Confidence correlation (detect artificial correlation)
        if len(confidences) > 1:
            consistency_results['confidence_correlation'] = self._calculate_confidence_correlation(confidences)
        
        return consistency_results
    
    def audit_verifiers(self,
                       verifiers: List[BaseVerifier],
                       recent_results: Dict[str, List[VerificationResult]]) -> List[CollusionDetection]:
        """
        Perform periodic audit of verifiers to detect collusion
        
        Args:
            verifiers: List of verifiers to audit
            recent_results: Recent verification results for each verifier
            
        Returns:
            List of collusion detections
        """
        self.audit_count += 1
        self.last_audit_time = time.time()
        
        detections = []
        
        # Update verifier profiles
        for verifier in verifiers:
            if verifier.verifier_id not in self.verifier_profiles:
                self.verifier_profiles[verifier.verifier_id] = VerifierProfile(
                    verifier_id=verifier.verifier_id,
                    verifier_type=verifier.verifier_type
                )
            
            profile = self.verifier_profiles[verifier.verifier_id]
            
            # Update profile with recent results
            if verifier.verifier_id in recent_results:
                self._update_verifier_profile(profile, recent_results[verifier.verifier_id])
        
        # Detect various types of collusion
        detections.extend(self._detect_steganographic_collusion(verifiers, recent_results))
        detections.extend(self._detect_coordinated_bias(verifiers, recent_results))
        detections.extend(self._detect_sybil_attacks(verifiers))
        detections.extend(self._detect_reward_hacking(verifiers, recent_results))
        
        # Record detections with memory management
        self.collusion_detections.extend(detections)
        
        # Prevent memory leaks by limiting detection history
        if len(self.collusion_detections) > self.max_detections_history:
            excess = len(self.collusion_detections) - self.max_detections_history
            self.collusion_detections = self.collusion_detections[excess:]
        
        # Update blacklist based on severe detections
        for detection in detections:
            if detection.severity == "high":
                for verifier_id in detection.involved_verifiers:
                    self.blacklisted_verifiers.add(verifier_id)
        
        return detections
    
    def _generate_routing_entropy(self, node: DebateNode) -> float:
        """Generate cryptographically secure entropy for routing randomization"""
        # Use node-specific information to generate entropy
        entropy_input = f"{node.id}_{node.claim}_{time.time()}_{self.random_seed}"
        hash_bytes = hashlib.sha256(entropy_input.encode()).digest()
        
        # Convert to float [0, 1]
        entropy = int.from_bytes(hash_bytes[:4], byteorder='big') / (2**32 - 1)
        
        # Add to entropy pool
        self.routing_entropy_pool.append(entropy)
        if len(self.routing_entropy_pool) > 1000:
            self.routing_entropy_pool.pop(0)
        
        return entropy
    
    def _calculate_diversity_bonus(self, verifier: BaseVerifier, node: DebateNode) -> float:
        """Calculate diversity bonus for verifier selection"""
        # Count recent usage of this verifier type
        recent_usage = 0
        recent_limit = 20
        
        for _, routing in enumerate(self.routing_history[-recent_limit:]):
            if routing.selected_verifier and routing.selected_verifier == verifier.verifier_id:
                recent_usage += 1
        
        # Bonus inversely proportional to recent usage
        usage_ratio = recent_usage / recent_limit if recent_limit > 0 else 0.0
        diversity_bonus = (1.0 - usage_ratio) * 0.2  # Max bonus of 0.2
        
        return diversity_bonus
    
    def _calculate_suspicion_penalty(self, verifier_id: str) -> float:
        """Calculate penalty based on suspicion level"""
        if verifier_id in self.blacklisted_verifiers:
            return float('inf')  # Effectively exclude blacklisted verifiers
        
        # Calculate suspicion score from various factors
        suspicion_score = 0.0
        
        # Recent collusion detections involving this verifier
        recent_detections = [
            d for d in self.collusion_detections[-50:]  # Last 50 detections
            if verifier_id in d.involved_verifiers
        ]
        
        for detection in recent_detections:
            if detection.severity == "high":
                suspicion_score += 0.3
            elif detection.severity == "medium":
                suspicion_score += 0.2
            else:
                suspicion_score += 0.1
        
        # Suspicious pair associations
        for (v1, v2), suspicion in self.suspicious_pairs.items():
            if verifier_id in (v1, v2):
                suspicion_score += suspicion * 0.1
        
        return min(suspicion_score, 1.0)  # Cap at 1.0
    
    def _calculate_verifier_diversity(self, verifiers: List[BaseVerifier]) -> float:
        """Calculate diversity score for a set of verifiers"""
        if not verifiers:
            return 0.0
        
        # Type diversity
        types = set(v.verifier_type for v in verifiers)
        type_diversity = len(types) / len(VerifierType)
        
        # Architecture diversity (simplified - in practice would check actual architectures)
        architectures = set(v.verifier_id.split('_')[0] for v in verifiers)  # Assume ID format
        arch_diversity = len(architectures) / len(verifiers)
        
        # Performance diversity (different accuracy ranges)
        if all(hasattr(v, 'accuracy_history') and v.accuracy_history for v in verifiers):
            accuracies = [sum(v.accuracy_history[-10:]) / min(10, len(v.accuracy_history)) for v in verifiers]
            acc_std = math.sqrt(sum((a - sum(accuracies)/len(accuracies))**2 for a in accuracies) / len(accuracies))
            perf_diversity = min(1.0, acc_std * 4)  # Scale to [0, 1]
        else:
            perf_diversity = 0.5  # Default
        
        # Combined diversity score
        overall_diversity = (type_diversity + arch_diversity + perf_diversity) / 3
        return overall_diversity
    
    def _calculate_pairwise_agreement(self,
                                    result1: VerificationResult,
                                    result2: VerificationResult) -> float:
        """Calculate agreement between two verification results"""
        # Boolean agreement
        bool_agreement = 1.0 if result1.is_valid == result2.is_valid else 0.0
        
        # Confidence similarity
        conf_diff = abs(result1.confidence - result2.confidence)
        conf_agreement = 1.0 - conf_diff
        
        # Evidence similarity (simplified)
        evidence_agreement = 0.5  # Default - would need semantic comparison
        
        # Weighted average
        agreement = (bool_agreement * 0.5 + conf_agreement * 0.3 + evidence_agreement * 0.2)
        return agreement
    
    def _detect_suspicious_patterns(self, results: List[VerificationResult]) -> List[str]:
        """Detect suspicious patterns in verification results"""
        patterns = []
        
        # All results identical (suspicious)
        if len(set((r.is_valid, round(r.confidence, 3)) for r in results)) == 1:
            patterns.append("identical_results")
        
        # Confidence values too similar
        confidences = [r.confidence for r in results]
        if len(confidences) > 1:
            conf_std = math.sqrt(sum((c - sum(confidences)/len(confidences))**2 for c in confidences) / len(confidences))
            if conf_std < 0.01:  # Very low variance
                patterns.append("artificial_confidence_correlation")
        
        # Response times too similar (if available)
        latencies = [r.latency for r in results if hasattr(r, 'latency')]
        if len(latencies) > 1:
            lat_std = math.sqrt(sum((l - sum(latencies)/len(latencies))**2 for l in latencies) / len(latencies))
            if lat_std < 0.001:  # Very low variance in response time
                patterns.append("synchronized_responses")
        
        return patterns
    
    def _calculate_confidence_correlation(self, confidences: List[float]) -> float:
        """Calculate artificial correlation in confidence values"""
        if len(confidences) < 2:
            return 0.0
        
        # Simple correlation measure - in practice would use more sophisticated methods
        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf)**2 for c in confidences) / len(confidences)
        
        # High correlation indicated by low variance relative to mean
        if mean_conf > 0:
            correlation_indicator = 1.0 - (variance / mean_conf)
        else:
            correlation_indicator = 0.0
        
        return max(0.0, min(1.0, correlation_indicator))
    
    def _update_verifier_profile(self, profile: VerifierProfile, results: List[VerificationResult]) -> None:
        """Update verifier profile with new results"""
        profile.last_updated = time.time()
        
        # Update confidence distribution
        for result in results:
            profile.confidence_distribution.append(result.confidence)
            if hasattr(result, 'latency'):
                profile.response_times.append(result.latency)
        
        # Keep distributions bounded
        if len(profile.confidence_distribution) > 1000:
            profile.confidence_distribution = profile.confidence_distribution[-1000:]
        if len(profile.response_times) > 1000:
            profile.response_times = profile.response_times[-1000:]
        
        # Update anomaly scores
        recent_confidences = profile.confidence_distribution[-20:]  # Last 20
        if len(recent_confidences) > 5:
            mean_conf = sum(recent_confidences) / len(recent_confidences)
            std_conf = math.sqrt(sum((c - mean_conf)**2 for c in recent_confidences) / len(recent_confidences))
            
            # Anomaly if standard deviation is too low (artificial consistency)
            anomaly_score = 1.0 - min(1.0, std_conf * 10)  # Scale appropriately
            profile.anomaly_scores.append(anomaly_score)
            
            if len(profile.anomaly_scores) > 100:
                profile.anomaly_scores = profile.anomaly_scores[-100:]
    
    def _detect_steganographic_collusion(self,
                                       verifiers: List[BaseVerifier],
                                       recent_results: Dict[str, List[VerificationResult]]) -> List[CollusionDetection]:
        """Detect steganographic communication between verifiers"""
        if not self.steganography_detection:
            return []
        
        detections = []
        
        # Look for hidden patterns in explanations
        for verifier_id, results in recent_results.items():
            explanations = [r.explanation for r in results if hasattr(r, 'explanation')]
            
            if len(explanations) < 5:
                continue
            
            # Check for repeated phrases or patterns
            phrase_counts = defaultdict(int)
            for explanation in explanations:
                words = explanation.split()
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    phrase_counts[phrase] += 1
            
            # Detect unusually repeated phrases
            max_repetition = max(phrase_counts.values()) if phrase_counts else 0
            if max_repetition > len(explanations) * 0.5:  # More than 50% repetition
                detections.append(CollusionDetection(
                    threat_type=CollusionThreat.STEGANOGRAPHIC,
                    confidence=0.7,
                    involved_verifiers=[verifier_id],
                    evidence=[f"Repeated phrase pattern detected: {max_repetition} repetitions"],
                    severity="medium",
                    recommended_action="Investigate verifier for steganographic communication"
                ))
        
        return detections
    
    def _detect_coordinated_bias(self,
                               verifiers: List[BaseVerifier],
                               recent_results: Dict[str, List[VerificationResult]]) -> List[CollusionDetection]:
        """Detect coordinated bias across verifiers"""
        detections = []
        
        if len(recent_results) < 2:
            return detections
        
        # Check for systematic bias patterns
        verifier_bias_scores = {}
        
        for verifier_id, results in recent_results.items():
            if not results:
                continue
            
            # Calculate bias towards positive/negative results
            positive_rate = sum(1 for r in results if r.is_valid) / len(results)
            avg_confidence = sum(r.confidence for r in results) / len(results)
            
            # Bias score combines extreme positive/negative rates with high confidence
            if positive_rate > 0.9 or positive_rate < 0.1:
                bias_score = abs(positive_rate - 0.5) * 2 * avg_confidence
                verifier_bias_scores[verifier_id] = bias_score
        
        # Look for coordinated bias (multiple verifiers with similar bias)
        if len(verifier_bias_scores) >= 2:
            bias_values = list(verifier_bias_scores.values())
            mean_bias = sum(bias_values) / len(bias_values)
            
            if mean_bias > 0.7:  # High coordinated bias
                involved_verifiers = list(verifier_bias_scores.keys())
                detections.append(CollusionDetection(
                    threat_type=CollusionThreat.COORDINATED_BIAS,
                    confidence=0.8,
                    involved_verifiers=involved_verifiers,
                    evidence=[f"Coordinated bias detected: mean bias score {mean_bias:.3f}"],
                    severity="high",
                    recommended_action="Investigate verifiers for coordinated bias"
                ))
        
        return detections
    
    def _detect_sybil_attacks(self, verifiers: List[BaseVerifier]) -> List[CollusionDetection]:
        """Detect Sybil attacks (multiple fake identities)"""
        detections = []
        
        # Group verifiers by behavioral similarity
        similar_groups = []
        
        for i, verifier1 in enumerate(verifiers):
            if verifier1.verifier_id not in self.verifier_profiles:
                continue
            
            profile1 = self.verifier_profiles[verifier1.verifier_id]
            
            for j, verifier2 in enumerate(verifiers[i+1:], i+1):
                if verifier2.verifier_id not in self.verifier_profiles:
                    continue
                
                profile2 = self.verifier_profiles[verifier2.verifier_id]
                
                # Calculate behavioral similarity
                similarity = self._calculate_behavioral_similarity(profile1, profile2)
                
                if similarity > 0.9:  # Very high similarity
                    # Check if already in a group
                    added_to_group = False
                    for group in similar_groups:
                        if verifier1.verifier_id in group or verifier2.verifier_id in group:
                            group.add(verifier1.verifier_id)
                            group.add(verifier2.verifier_id)
                            added_to_group = True
                            break
                    
                    if not added_to_group:
                        similar_groups.append({verifier1.verifier_id, verifier2.verifier_id})
        
        # Report groups with high similarity as potential Sybil attacks
        for group in similar_groups:
            if len(group) >= 2:
                detections.append(CollusionDetection(
                    threat_type=CollusionThreat.SYBIL_ATTACK,
                    confidence=0.6,
                    involved_verifiers=list(group),
                    evidence=[f"High behavioral similarity detected among {len(group)} verifiers"],
                    severity="medium",
                    recommended_action="Investigate for potential Sybil attack"
                ))
        
        return detections
    
    def _detect_reward_hacking(self,
                             verifiers: List[BaseVerifier],
                             recent_results: Dict[str, List[VerificationResult]]) -> List[CollusionDetection]:
        """Detect reward hacking behavior"""
        detections = []
        
        for verifier_id, results in recent_results.items():
            if len(results) < 10:
                continue
            
            # Check for patterns indicating reward hacking
            
            # 1. Artificially high confidence with poor calibration
            confidences = [r.confidence for r in results]
            avg_confidence = sum(confidences) / len(confidences)
            
            if verifier_id in self.verifier_profiles:
                profile = self.verifier_profiles[verifier_id]
                if profile.calibration_scores:
                    avg_calibration = sum(profile.calibration_scores) / len(profile.calibration_scores)
                    
                    # High confidence but poor calibration = potential reward hacking
                    if avg_confidence > 0.9 and avg_calibration > 0.3:
                        detections.append(CollusionDetection(
                            threat_type=CollusionThreat.REWARD_HACKING,
                            confidence=0.7,
                            involved_verifiers=[verifier_id],
                            evidence=[f"High confidence ({avg_confidence:.3f}) with poor calibration ({avg_calibration:.3f})"],
                            severity="medium",
                            recommended_action="Review verifier calibration and incentives"
                        ))
        
        return detections
    
    def _calculate_behavioral_similarity(self,
                                       profile1: VerifierProfile,
                                       profile2: VerifierProfile) -> float:
        """Calculate behavioral similarity between two verifier profiles"""
        similarities = []
        
        # Confidence distribution similarity
        if profile1.confidence_distribution and profile2.confidence_distribution:
            # Simple similarity based on means and standard deviations
            mean1 = sum(profile1.confidence_distribution) / len(profile1.confidence_distribution)
            mean2 = sum(profile2.confidence_distribution) / len(profile2.confidence_distribution)
            
            std1 = math.sqrt(sum((c - mean1)**2 for c in profile1.confidence_distribution) / len(profile1.confidence_distribution))
            std2 = math.sqrt(sum((c - mean2)**2 for c in profile2.confidence_distribution) / len(profile2.confidence_distribution))
            
            mean_similarity = 1.0 - abs(mean1 - mean2)
            std_similarity = 1.0 - abs(std1 - std2)
            conf_similarity = (mean_similarity + std_similarity) / 2
            similarities.append(conf_similarity)
        
        # Response time similarity
        if profile1.response_times and profile2.response_times:
            mean_time1 = sum(profile1.response_times) / len(profile1.response_times)
            mean_time2 = sum(profile2.response_times) / len(profile2.response_times)
            
            time_similarity = 1.0 - min(1.0, abs(mean_time1 - mean_time2) / max(mean_time1, mean_time2, 1e-6))
            similarities.append(time_similarity)
        
        # Return average similarity
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and metrics"""
        return {
            'collusion_detections': len(self.collusion_detections),
            'blacklisted_verifiers': len(self.blacklisted_verifiers),
            'suspicious_pairs': len(self.suspicious_pairs),
            'audit_count': self.audit_count,
            'last_audit_time': self.last_audit_time,
            'randomization_strength': self.randomization_strength,
            'diversity_threshold': self.diversity_threshold,
            'detection_sensitivity': self.detection_sensitivity,
            'recent_detections': [
                {
                    'threat_type': d.threat_type.value,
                    'severity': d.severity,
                    'involved_verifiers': d.involved_verifiers,
                    'confidence': d.confidence
                }
                for d in self.collusion_detections[-10:]  # Last 10 detections
            ]
        }
