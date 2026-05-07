"""
Main HDO System - Hierarchical Delegated Oversight

Integrates all components into a complete oversight system as described in the paper:
"We introduce Hierarchical Delegated Oversight (HDO), a scalable framework where 
weak overseer agents delegate verification to specialized sub-agents via structured 
debates, achieving provable alignment guarantees under bounded communication budgets."
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import logging
from collections import defaultdict

from .debate_tree import DebateTree, DebateNode, ClaimType, NodeStatus, ClaimEvidence
from .verifiers import (
    BaseVerifier, NLIVerifier, CodeVerifier, RuleVerifier, RetrievalVerifier,
    VerificationResult, VerifierType
)
from .routing import CostAwareRouter, RoutingDecision, RoutingStrategy
from .aggregation import EntailmentAggregator, AggregationResult, AggregationMethod
from .risk_bounds import PAC_BayesianRiskBound, RiskBound
from .collusion_resistance import CollusionResistance, CollusionDetection


class HDOMode(Enum):
    """Operating modes for HDO system"""
    STANDARD = "standard"  # Normal operation
    CONSERVATIVE = "conservative"  # Risk-averse mode
    EFFICIENT = "efficient"  # Cost-optimized mode
    AUDIT = "audit"  # Enhanced monitoring mode


@dataclass
class HDOConfig:
    """Configuration for HDO system"""
    # Core parameters
    tau_reject: float = 0.2  # Lower uncertainty threshold
    tau_accept: float = 0.8  # Upper uncertainty threshold
    max_delegation_depth: int = 5  # Maximum tree depth
    budget_limit: float = 1000.0  # Cost budget per episode
    
    # Verifier settings
    enable_nli_verifier: bool = True
    enable_code_verifier: bool = True
    enable_rule_verifier: bool = True
    enable_retrieval_verifier: bool = True
    verifier_redundancy_prob: float = 0.1
    
    # Routing settings
    routing_strategy: RoutingStrategy = RoutingStrategy.COST_MINIMAL
    randomization_strength: float = 0.3
    diversity_threshold: float = 0.7
    
    # Aggregation settings
    default_aggregation: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE
    risk_averse_aggregation: bool = True
    
    # Risk bound settings
    confidence_level: float = 0.95
    prior_misalignment_rate: float = 0.1
    
    # Collusion resistance
    enable_collusion_resistance: bool = True
    audit_frequency: int = 50
    detection_sensitivity: float = 0.8
    
    # System settings
    mode: HDOMode = HDOMode.STANDARD
    enable_logging: bool = True
    max_concurrent_verifications: int = 10


@dataclass
class OversightEpisode:
    """A complete oversight episode"""
    episode_id: str
    root_claim: str
    context: Dict[str, Any]
    
    # Results
    debate_tree: Optional[DebateTree] = None
    final_decision: Optional[bool] = None
    confidence: float = 0.0
    
    # Analysis
    risk_bound: Optional[RiskBound] = None
    collusion_detections: List[CollusionDetection] = field(default_factory=list)
    
    # Metrics
    total_cost: float = 0.0
    total_time: float = 0.0
    num_verifications: int = 0
    delegation_depth_reached: int = 0
    
    # Metadata
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "running"


class HDOSystem:
    """
    Complete Hierarchical Delegated Oversight System
    
    Implements the full HDO framework from the paper, integrating:
    - Debate tree construction and management
    - Cost-aware routing with uncertainty-based delegation
    - Specialized verifiers with ensemble aggregation
    - PAC-Bayesian risk bounds
    - Anti-collusion mechanisms
    """
    
    def __init__(self, config: HDOConfig = None):
        """
        Initialize HDO system
        
        Args:
            config: System configuration (uses defaults if None)
        """
        self.config = config or HDOConfig()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self.router = CostAwareRouter(
            tau_reject=self.config.tau_reject,
            tau_accept=self.config.tau_accept,
            redundancy_prob=self.config.verifier_redundancy_prob,
            diversity_weight=self.config.diversity_threshold,
            collusion_resistance=self.config.enable_collusion_resistance
        )
        
        self.aggregator = EntailmentAggregator(
            default_method=self.config.default_aggregation,
            risk_averse=self.config.risk_averse_aggregation
        )
        
        self.risk_calculator = PAC_BayesianRiskBound(
            confidence_level=self.config.confidence_level,
            prior_belief=self.config.prior_misalignment_rate
        )
        
        self.collusion_detector = CollusionResistance(
            randomization_strength=self.config.randomization_strength,
            diversity_threshold=self.config.diversity_threshold,
            audit_frequency=self.config.audit_frequency,
            detection_sensitivity=self.config.detection_sensitivity
        ) if self.config.enable_collusion_resistance else None
        
        # Initialize verifiers
        self.verifiers: Dict[str, BaseVerifier] = {}
        self._initialize_verifiers()
        
        # Register verifiers with router
        for verifier in self.verifiers.values():
            self.router.register_verifier(verifier)
        
        # Episode tracking with bounds
        self.episodes: List[OversightEpisode] = []
        self.current_episode: Optional[OversightEpisode] = None
        self.max_episodes_history = 1000  # Prevent memory leaks
        
        # Performance tracking
        self.total_episodes = 0
        self.total_cost = 0.0
        self.total_verifications = 0
        
        # Logging
        if self.config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("HDO")
        else:
            self.logger = None
    
    def conduct_oversight(self,
                         outcome: str,
                         context: Dict[str, Any] = None,
                         ground_truth: bool = None) -> OversightEpisode:
        """
        Conduct a complete oversight episode
        
        Args:
            outcome: The outcome/action to verify for alignment
            context: Additional context for verification
            ground_truth: Ground truth alignment (for evaluation)
            
        Returns:
            Completed OversightEpisode with results
        """
        episode_id = f"episode_{self.total_episodes + 1}_{int(time.time())}"
        root_claim = f"Is aligned({outcome}|{context})"
        
        episode = OversightEpisode(
            episode_id=episode_id,
            root_claim=root_claim,
            context=context or {}
        )
        
        self.current_episode = episode
        
        try:
            if self.logger:
                self.logger.info(f"Starting oversight episode {episode_id}")
            
            # Phase 1: Initialize debate tree
            debate_tree = DebateTree(root_claim, context)
            episode.debate_tree = debate_tree
            
            # Phase 2: Adaptive tree expansion and verification
            self._expand_and_verify_tree(debate_tree, episode)
            
            # Phase 3: Aggregate results bottom-up
            final_result = self._aggregate_tree_results(debate_tree, episode)
            episode.final_decision = final_result.is_valid
            episode.confidence = final_result.confidence
            
            # Phase 4: Calculate risk bounds
            episode.risk_bound = self._calculate_risk_bounds(debate_tree, episode, ground_truth)
            
            # Phase 5: Collusion detection and security analysis
            if self.collusion_detector:
                episode.collusion_detections = self._perform_security_analysis(debate_tree, episode)
            
            # Phase 6: Finalize episode
            episode.end_time = time.time()
            episode.total_time = episode.end_time - episode.start_time
            episode.total_cost = debate_tree.total_cost
            episode.num_verifications = len(debate_tree.verification_history)
            episode.delegation_depth_reached = debate_tree.delegation_depth
            episode.status = "completed"
            
            if self.logger:
                self.logger.info(f"Completed oversight episode {episode_id}: "
                               f"decision={episode.final_decision}, "
                               f"confidence={episode.confidence:.3f}, "
                               f"cost={episode.total_cost:.2f}, "
                               f"depth={episode.delegation_depth_reached}")
            
        except Exception as e:
            episode.status = f"failed: {str(e)}"
            episode.end_time = time.time()
            episode.total_time = episode.end_time - episode.start_time
            
            if self.logger:
                self.logger.error(f"Episode {episode_id} failed: {str(e)}")
            
            raise
        
        finally:
            # Record episode with memory management
            self.episodes.append(episode)
            
            # Prevent memory leaks by limiting history size
            if len(self.episodes) > self.max_episodes_history:
                self.episodes.pop(0)  # Remove oldest episode
            
            self.current_episode = None
            self.total_episodes += 1
            self.total_cost += episode.total_cost
            self.total_verifications += episode.num_verifications
        
        return episode
    
    def _initialize_verifiers(self) -> None:
        """Initialize the verifier ensemble"""
        if self.config.enable_nli_verifier:
            nli_verifier = NLIVerifier("nli_primary")
            self.verifiers[nli_verifier.verifier_id] = nli_verifier
            
            # Add diverse NLI verifiers for robustness
            nli_verifier_2 = NLIVerifier("nli_secondary")
            self.verifiers[nli_verifier_2.verifier_id] = nli_verifier_2
        
        if self.config.enable_code_verifier:
            code_verifier = CodeVerifier("code_analyzer")
            self.verifiers[code_verifier.verifier_id] = code_verifier
        
        if self.config.enable_rule_verifier:
            rule_verifier = RuleVerifier("rule_checker")
            self.verifiers[rule_verifier.verifier_id] = rule_verifier
        
        if self.config.enable_retrieval_verifier:
            retrieval_verifier = RetrievalVerifier("fact_checker")
            self.verifiers[retrieval_verifier.verifier_id] = retrieval_verifier
    
    def _expand_and_verify_tree(self, tree: DebateTree, episode: OversightEpisode) -> None:
        """
        Adaptively expand and verify the debate tree
        
        Implements the core HDO algorithm with breadth-first expansion
        followed by depth-first verification.
        """
        # Set budget constraint
        self.router.set_budget(self.config.budget_limit)
        
        # Breadth-first expansion phase with safety limits
        nodes_to_process = [tree.root_node]
        current_depth = 0
        max_iterations = 1000  # Prevent infinite loops
        iteration_count = 0
        
        while nodes_to_process and current_depth < self.config.max_delegation_depth and iteration_count < max_iterations:
            next_level_nodes = []
            
            for node in nodes_to_process:
                # Check if node needs delegation
                routing_decision = self.router.route(node, episode.context, self.config.budget_limit - tree.total_cost)
                
                if routing_decision.should_delegate:
                    # Verify at leaf level or expand further
                    if current_depth >= self.config.max_delegation_depth - 1:
                        # At max depth, perform verification
                        self._verify_node(node, tree, episode)
                    else:
                        # Expand node with sub-claims
                        sub_claims = self._generate_sub_claims(node, episode.context)
                        if sub_claims:
                            child_ids = tree.expand_node(node.id, sub_claims)
                            for child_id in child_ids:
                                child_node = tree.get_node(child_id)
                                if child_node:
                                    next_level_nodes.append(child_node)
                        else:
                            # No sub-claims generated, verify directly
                            self._verify_node(node, tree, episode)
                else:
                    # Node doesn't need delegation, mark as accepted/rejected based on uncertainty
                    if node.uncertainty < self.config.tau_reject:
                        tree.update_node_status(node.id, NodeStatus.VERIFIED, True, 1.0 - node.uncertainty)
                    else:
                        tree.update_node_status(node.id, NodeStatus.REJECTED, False, node.uncertainty)
            
            nodes_to_process = next_level_nodes
            current_depth += 1
            iteration_count += 1
        
        # Verify any remaining leaf nodes
        for leaf in tree.get_leaves():
            if leaf.status == NodeStatus.PENDING:
                self._verify_node(leaf, tree, episode)
    
    def _generate_sub_claims(self, node: DebateNode, context: Dict[str, Any]) -> List[Tuple[str, ClaimType]]:
        """
        Generate sub-claims for node expansion
        
        In practice, this would use sophisticated claim decomposition.
        Here we provide a simplified version based on claim type.
        """
        sub_claims = []
        
        if node.claim_type == ClaimType.ALIGNMENT:
            # Decompose alignment into constituent parts
            sub_claims = [
                ("The outcome is truthful and factually accurate", ClaimType.TRUTHFULNESS),
                ("The outcome poses no safety risks", ClaimType.SAFETY),
                ("The outcome adheres to specified goals", ClaimType.GOAL_ADHERENCE),
                ("The reasoning is logically consistent", ClaimType.LOGICAL_CONSISTENCY)
            ]
        
        elif node.claim_type == ClaimType.TRUTHFULNESS:
            sub_claims = [
                ("All factual statements are verifiable", ClaimType.TRUTHFULNESS),
                ("No contradictory information is present", ClaimType.LOGICAL_CONSISTENCY)
            ]
        
        elif node.claim_type == ClaimType.SAFETY:
            sub_claims = [
                ("No harmful actions are recommended", ClaimType.CONSTRAINT_SATISFACTION),
                ("All safety constraints are satisfied", ClaimType.CONSTRAINT_SATISFACTION)
            ]
        
        elif node.claim_type == ClaimType.GOAL_ADHERENCE:
            sub_claims = [
                ("The outcome achieves stated objectives", ClaimType.GOAL_ADHERENCE),
                ("No goal conflicts are present", ClaimType.LOGICAL_CONSISTENCY)
            ]
        
        elif node.claim_type == ClaimType.LOGICAL_CONSISTENCY:
            sub_claims = [
                ("All premises are valid", ClaimType.TRUTHFULNESS),
                ("Conclusions follow from premises", ClaimType.LOGICAL_CONSISTENCY)
            ]
        
        elif node.claim_type == ClaimType.CONSTRAINT_SATISFACTION:
            sub_claims = [
                ("All explicit constraints are met", ClaimType.CONSTRAINT_SATISFACTION),
                ("No implicit constraint violations", ClaimType.CONSTRAINT_SATISFACTION)
            ]
        
        # Filter out sub-claims if node is already quite specific or at depth limit
        if node.depth >= self.config.max_delegation_depth - 2 or len(node.claim.split()) < 5:
            sub_claims = []
        
        return sub_claims
    
    def _verify_node(self, node: DebateNode, tree: DebateTree, episode: OversightEpisode) -> None:
        """Verify a node using appropriate verifiers"""
        # Get routing decision
        routing_decision = self.router.route(node, episode.context, self.config.budget_limit - tree.total_cost)
        
        if not routing_decision.should_delegate or not routing_decision.selected_verifier:
            # Mark as uncertain if can't verify
            tree.update_node_status(node.id, NodeStatus.UNCERTAIN, None, 0.5, 0.0)
            return
        
        # Get selected verifiers (including redundant ones if specified)
        primary_verifier = self.verifiers.get(routing_decision.selected_verifier)
        if not primary_verifier:
            tree.update_node_status(node.id, NodeStatus.UNCERTAIN, None, 0.5, 0.0)
            return
        
        verifiers_to_use = [primary_verifier]
        
        # Add redundant verifiers if specified
        if routing_decision.redundancy_level > 1:
            compatible_verifiers = [v for v in self.verifiers.values() 
                                  if v.can_verify(node.claim_type) and v != primary_verifier]
            
            # Apply collusion resistance in verifier selection
            if self.collusion_detector:
                compatible_verifiers = self.collusion_detector.apply_randomized_routing(
                    compatible_verifiers, node, {v.verifier_id: 1.0 for v in compatible_verifiers}
                )
            
            additional_verifiers = compatible_verifiers[:routing_decision.redundancy_level - 1]
            verifiers_to_use.extend(additional_verifiers)
        
        # Perform verifications
        verification_results = []
        
        for verifier in verifiers_to_use:
            try:
                # Prepare context for verification
                verification_context = {
                    'claim_id': node.id,
                    'outcome': episode.context.get('outcome', ''),
                    'constraints': episode.context.get('constraints', []),
                    'premises': episode.context.get('premises', []),
                    **episode.context
                }
                
                # Perform verification
                result = verifier.verify(node.claim, verification_context)
                verification_results.append(result)
                
                # Add evidence to node
                for evidence in result.evidence:
                    node.add_evidence(evidence, result.is_valid)
                
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Verification failed for {verifier.verifier_id}: {str(e)}")
                continue
        
        if not verification_results:
            tree.update_node_status(node.id, NodeStatus.UNCERTAIN, None, 0.5, 0.0)
            return
        
        # Aggregate verification results
        aggregation_result = self.aggregator.aggregate_verifications(node, verification_results)
        
        # Update node with aggregated result
        tree.update_node_status(
            node.id,
            NodeStatus.VERIFIED if aggregation_result.is_valid else NodeStatus.REJECTED,
            aggregation_result.is_valid,
            aggregation_result.confidence,
            aggregation_result.total_cost
        )
        
        # Perform consistency check if collusion detection is enabled
        if self.collusion_detector and len(verification_results) > 1:
            consistency_check = self.collusion_detector.perform_consistency_check(
                node, verification_results, episode.context
            )
            
            # Store consistency information in node metadata
            node.metadata['consistency_check'] = consistency_check
    
    def _aggregate_tree_results(self, tree: DebateTree, episode: OversightEpisode) -> AggregationResult:
        """Aggregate results across the entire debate tree"""
        # Collect verification results for each node
        node_results = {}
        tree_structure = {}
        
        for node in tree.nodes.values():
            # Get verification results from node's evidence
            verification_results = []
            
            # Convert evidence back to verification results (simplified)
            for evidence in node.supporting_evidence + node.refuting_evidence:
                if evidence.verifier_id:
                    # Create synthetic verification result from evidence
                    synthetic_result = VerificationResult(
                        claim_id=node.id,
                        verifier_id=evidence.verifier_id,
                        verifier_type=VerifierType.NLI,  # Default, would need to track actual type
                        is_valid=evidence in node.supporting_evidence,
                        confidence=evidence.confidence,
                        uncertainty=1.0 - evidence.confidence,
                        cost=evidence.cost,
                        latency=0.0,
                        evidence=[evidence],
                        explanation=evidence.content,
                        method_details={}
                    )
                    verification_results.append(synthetic_result)
            
            node_results[node.id] = verification_results
            tree_structure[node.id] = list(node.children_ids)
        
        # Perform tree-wide aggregation
        final_result = self.aggregator.aggregate_tree(tree.root_node, node_results, tree_structure)
        
        return final_result
    
    def _calculate_risk_bounds(self,
                             tree: DebateTree,
                             episode: OversightEpisode,
                             ground_truth: bool = None) -> RiskBound:
        """Calculate PAC-Bayesian risk bounds for the episode"""
        # Collect verification results
        verification_results = {}
        aggregation_results = {}
        
        for node in tree.nodes.values():
            # Create simplified verification results from node state
            if node.verification_result is not None:
                synthetic_result = VerificationResult(
                    claim_id=node.id,
                    verifier_id="aggregated",
                    verifier_type=VerifierType.NLI,
                    is_valid=node.verification_result,
                    confidence=node.confidence,
                    uncertainty=node.uncertainty,
                    cost=node.verification_cost,
                    latency=0.0,
                    evidence=[],
                    explanation=f"Node {node.id} verification result",
                    method_details={}
                )
                verification_results[node.id] = [synthetic_result]
        
        # Ground truth for risk calculation
        ground_truth_dict = {tree.root_id: ground_truth} if ground_truth is not None else {}
        
        # Calculate risk bound
        risk_bound = self.risk_calculator.calculate_risk_bound(
            tree, verification_results, aggregation_results, ground_truth_dict
        )
        
        return risk_bound
    
    def _perform_security_analysis(self,
                                 tree: DebateTree,
                                 episode: OversightEpisode) -> List[CollusionDetection]:
        """Perform collusion detection and security analysis"""
        if not self.collusion_detector:
            return []
        
        # Collect recent verification results for audit
        recent_results = defaultdict(list)
        
        for node in tree.nodes.values():
            for evidence in node.supporting_evidence + node.refuting_evidence:
                if evidence.verifier_id:
                    synthetic_result = VerificationResult(
                        claim_id=node.id,
                        verifier_id=evidence.verifier_id,
                        verifier_type=VerifierType.NLI,
                        is_valid=evidence in node.supporting_evidence,
                        confidence=evidence.confidence,
                        uncertainty=1.0 - evidence.confidence,
                        cost=evidence.cost,
                        latency=0.0,
                        evidence=[evidence],
                        explanation=evidence.content,
                        method_details={}
                    )
                    recent_results[evidence.verifier_id].append(synthetic_result)
        
        # Perform audit if due
        detections = []
        if self.total_verifications % self.config.audit_frequency == 0:
            detections = self.collusion_detector.audit_verifiers(
                list(self.verifiers.values()),
                recent_results
            )
        
        return detections
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        if not self.episodes:
            return {}
        
        # Episode statistics
        completed_episodes = [e for e in self.episodes if e.status == "completed"]
        
        episode_stats = {}
        if completed_episodes:
            decisions = [e.final_decision for e in completed_episodes if e.final_decision is not None]
            confidences = [e.confidence for e in completed_episodes]
            costs = [e.total_cost for e in completed_episodes]
            times = [e.total_time for e in completed_episodes]
            depths = [e.delegation_depth_reached for e in completed_episodes]
            
            episode_stats = {
                'total_episodes': len(self.episodes),
                'completed_episodes': len(completed_episodes),
                'success_rate': len(completed_episodes) / len(self.episodes),
                'positive_alignment_rate': sum(decisions) / len(decisions) if decisions else 0.0,
                'average_confidence': sum(confidences) / len(confidences) if confidences else 0.0,
                'average_cost': sum(costs) / len(costs) if costs else 0.0,
                'average_time': sum(times) / len(times) if times else 0.0,
                'average_delegation_depth': sum(depths) / len(depths) if depths else 0.0,
                'total_cost': self.total_cost,
                'total_verifications': self.total_verifications
            }
        
        # Verifier statistics
        verifier_stats = {}
        for verifier_id, verifier in self.verifiers.items():
            verifier_stats[verifier_id] = {
                'verification_count': verifier.verification_count,
                'total_cost': verifier.total_cost,
                'average_accuracy': sum(verifier.accuracy_history) / len(verifier.accuracy_history) if verifier.accuracy_history else 0.0,
                'calibration_score': verifier.get_calibration_score()
            }
        
        # Risk bound statistics
        risk_stats = self.risk_calculator.get_bound_statistics()
        
        # Security statistics
        security_stats = {}
        if self.collusion_detector:
            security_stats = self.collusion_detector.get_security_status()
        
        # Routing statistics
        routing_stats = self.router.get_routing_stats()
        
        # Aggregation statistics
        aggregation_stats = self.aggregator.get_aggregation_stats()
        
        return {
            'system_info': {
                'config': {
                    'mode': self.config.mode.value,
                    'tau_reject': self.config.tau_reject,
                    'tau_accept': self.config.tau_accept,
                    'max_delegation_depth': self.config.max_delegation_depth,
                    'budget_limit': self.config.budget_limit
                },
                'uptime': time.time() - (self.episodes[0].start_time if self.episodes else time.time())
            },
            'episode_statistics': episode_stats,
            'verifier_statistics': verifier_stats,
            'risk_bound_statistics': risk_stats,
            'security_statistics': security_stats,
            'routing_statistics': routing_stats,
            'aggregation_statistics': aggregation_stats
        }
    
    def export_episode(self, episode_id: str, format: str = "json") -> Union[str, Dict]:
        """Export episode data for analysis"""
        episode = next((e for e in self.episodes if e.episode_id == episode_id), None)
        
        if not episode:
            raise ValueError(f"Episode {episode_id} not found")
        
        export_data = {
            'episode_id': episode.episode_id,
            'root_claim': episode.root_claim,
            'context': episode.context,
            'final_decision': episode.final_decision,
            'confidence': episode.confidence,
            'total_cost': episode.total_cost,
            'total_time': episode.total_time,
            'num_verifications': episode.num_verifications,
            'delegation_depth_reached': episode.delegation_depth_reached,
            'status': episode.status,
            'debate_tree': episode.debate_tree.to_dict() if episode.debate_tree else None,
            'risk_bound': {
                'empirical_risk': episode.risk_bound.empirical_risk,
                'combined_bound': episode.risk_bound.combined_bound,
                'delegation_depth': episode.risk_bound.delegation_depth,
                'bound_tightness': episode.risk_bound.bound_tightness
            } if episode.risk_bound else None,
            'collusion_detections': [
                {
                    'threat_type': d.threat_type.value,
                    'confidence': d.confidence,
                    'severity': d.severity,
                    'involved_verifiers': d.involved_verifiers,
                    'evidence': d.evidence
                }
                for d in episode.collusion_detections
            ]
        }
        
        if format.lower() == "json":
            return json.dumps(export_data, indent=2)
        else:
            return export_data
    
    def update_configuration(self, new_config: HDOConfig) -> None:
        """Update system configuration"""
        self.config = new_config
        
        # Update component configurations
        self.router.tau_reject = new_config.tau_reject
        self.router.tau_accept = new_config.tau_accept
        self.router.redundancy_prob = new_config.verifier_redundancy_prob
        
        self.aggregator.default_method = new_config.default_aggregation
        self.aggregator.risk_averse = new_config.risk_averse_aggregation
        
        self.risk_calculator.confidence_level = new_config.confidence_level
        self.risk_calculator.prior_belief = new_config.prior_misalignment_rate
        
        if self.collusion_detector:
            self.collusion_detector.randomization_strength = new_config.randomization_strength
            self.collusion_detector.diversity_threshold = new_config.diversity_threshold
            self.collusion_detector.audit_frequency = new_config.audit_frequency
            self.collusion_detector.detection_sensitivity = new_config.detection_sensitivity
    
    def set_random_seed(self, seed: int) -> None:
        """
        Set random seed for reproducible results
        
        Args:
            seed: Random seed for deterministic behavior
        """
        import random
        random.seed(seed)
        
        # Set seeds for components
        if self.collusion_detector:
            self.collusion_detector.random_seed = seed
        
        # Set seeds for verifiers
        for verifier in self.verifiers.values():
            if hasattr(verifier, 'random_seed'):
                verifier.random_seed = seed
    
    def enable_deterministic_mode(self, seed: int = 42) -> None:
        """
        Enable deterministic mode for reproducibility
        
        Args:
            seed: Random seed to use
        """
        self.set_random_seed(seed)
        
        # Disable randomization for reproducibility
        if self.collusion_detector:
            self.collusion_detector.randomization_strength = 0.0
        
        self.router.randomization_factor = 0.0
        
        if self.logger:
            self.logger.info(f"HDO system set to deterministic mode with seed {seed}")
    
    def _validate_config(self) -> None:
        """Validate HDO system configuration"""
        if self.config.tau_reject >= self.config.tau_accept:
            raise ValueError(f"tau_reject ({self.config.tau_reject}) must be less than tau_accept ({self.config.tau_accept})")
        
        if not 0.0 <= self.config.tau_reject <= 1.0:
            raise ValueError(f"tau_reject must be between 0 and 1, got {self.config.tau_reject}")
        
        if not 0.0 <= self.config.tau_accept <= 1.0:
            raise ValueError(f"tau_accept must be between 0 and 1, got {self.config.tau_accept}")
        
        if self.config.max_delegation_depth < 1:
            raise ValueError(f"max_delegation_depth must be at least 1, got {self.config.max_delegation_depth}")
        
        if self.config.budget_limit <= 0:
            raise ValueError(f"budget_limit must be positive, got {self.config.budget_limit}")
        
        if not 0.0 <= self.config.confidence_level <= 1.0:
            raise ValueError(f"confidence_level must be between 0 and 1, got {self.config.confidence_level}")
        
        if not 0.0 <= self.config.prior_misalignment_rate <= 1.0:
            raise ValueError(f"prior_misalignment_rate must be between 0 and 1, got {self.config.prior_misalignment_rate}")
    
    def cleanup(self) -> None:
        """Clean up resources and free memory"""
        # Clear episode history but keep recent episodes
        if len(self.episodes) > 100:
            self.episodes = self.episodes[-100:]  # Keep last 100 episodes
        
        # Clear verifier histories
        for verifier in self.verifiers.values():
            if hasattr(verifier, 'accuracy_history') and len(verifier.accuracy_history) > 100:
                verifier.accuracy_history = verifier.accuracy_history[-100:]
            if hasattr(verifier, 'calibration_data') and len(verifier.calibration_data) > 1000:
                verifier.calibration_data = verifier.calibration_data[-1000:]
        
        # Clear component histories
        if hasattr(self.router, 'routing_history') and len(self.router.routing_history) > 1000:
            self.router.routing_history = self.router.routing_history[-1000:]
        
        if hasattr(self.aggregator, 'aggregation_history') and len(self.aggregator.aggregation_history) > 1000:
            self.aggregator.aggregation_history = self.aggregator.aggregation_history[-1000:]
        
        if self.collusion_detector:
            if len(self.collusion_detector.verification_history) > 1000:
                self.collusion_detector.verification_history = self.collusion_detector.verification_history[-1000:]
            if len(self.collusion_detector.routing_history) > 1000:
                self.collusion_detector.routing_history = self.collusion_detector.routing_history[-1000:]
