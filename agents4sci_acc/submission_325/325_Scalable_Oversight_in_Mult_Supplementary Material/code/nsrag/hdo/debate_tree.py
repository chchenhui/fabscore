"""
Debate Tree implementation for Hierarchical Delegated Oversight

The debate tree represents a hierarchical decomposition of oversight claims,
where each node represents a claim to be verified and edges represent
entailment relationships.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict

# Optional networkx import
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    nx = None


class ClaimType(Enum):
    """Types of claims that can be verified"""
    ALIGNMENT = "alignment"  # Is aligned(O|C)?
    TRUTHFULNESS = "truthfulness"  # Is the claim factually correct?
    SAFETY = "safety"  # Does the action pose safety risks?
    GOAL_ADHERENCE = "goal_adherence"  # Does the outcome satisfy the goal?
    LOGICAL_CONSISTENCY = "logical_consistency"  # Are premises logically sound?
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"  # Are constraints met?


class NodeStatus(Enum):
    """Status of a debate node"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass
class ClaimEvidence:
    """Evidence supporting or refuting a claim"""
    content: str
    source: str
    confidence: float
    cost: float = 0.0
    verifier_id: Optional[str] = None


@dataclass
class DebateNode:
    """
    A node in the debate tree representing a claim to be verified
    
    Based on Definition 1 from the paper: A debate tree T=(N,E) roots at 
    the query q0: "Is aligned(O|C)?" and expands via sub-claims.
    """
    id: str
    claim: str
    claim_type: ClaimType
    parent_id: Optional[str] = None
    children_ids: Set[str] = field(default_factory=set)
    
    # Verification state
    status: NodeStatus = NodeStatus.PENDING
    uncertainty: float = 1.0  # u(q) ∈ [0,1] from paper
    confidence: float = 0.0
    
    # Evidence and verification results
    supporting_evidence: List[ClaimEvidence] = field(default_factory=list)
    refuting_evidence: List[ClaimEvidence] = field(default_factory=list)
    verification_result: Optional[bool] = None
    verification_cost: float = 0.0
    
    # Debate participants
    proposer_agent: Optional[str] = None
    critic_agent: Optional[str] = None
    assigned_verifier: Optional[str] = None
    
    # Metadata
    depth: int = 0
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child_id: str) -> None:
        """Add a child node ID"""
        self.children_ids.add(child_id)
    
    def remove_child(self, child_id: str) -> None:
        """Remove a child node ID"""
        self.children_ids.discard(child_id)
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)"""
        return len(self.children_ids) == 0
    
    def is_root(self) -> bool:
        """Check if this is the root node (no parent)"""
        return self.parent_id is None
    
    def add_evidence(self, evidence: ClaimEvidence, supporting: bool = True) -> None:
        """Add evidence supporting or refuting the claim"""
        if supporting:
            self.supporting_evidence.append(evidence)
        else:
            self.refuting_evidence.append(evidence)
    
    def get_evidence_balance(self) -> float:
        """Get the balance of evidence (positive = supporting, negative = refuting)"""
        support_weight = sum(e.confidence for e in self.supporting_evidence)
        refute_weight = sum(e.confidence for e in self.refuting_evidence)
        
        if support_weight + refute_weight == 0:
            return 0.0
        
        return (support_weight - refute_weight) / (support_weight + refute_weight)


class DebateTree:
    """
    Hierarchical debate tree for delegated oversight
    
    Implements the tree structure from Definition 1 in the paper,
    with adaptive expansion and cost tracking.
    """
    
    def __init__(self, root_claim: str, context: Dict[str, Any] = None):
        """
        Initialize debate tree with root claim
        
        Args:
            root_claim: The root oversight question (e.g., "Is aligned(O|C)?")
            context: Additional context including transcript, constraints, etc.
        """
        self.nodes: Dict[str, DebateNode] = {}
        self.graph = nx.DiGraph() if HAS_NETWORKX else None
        self.context = context or {}
        
        # Create root node
        root_id = str(uuid.uuid4())
        self.root_id = root_id
        self.root_node = DebateNode(
            id=root_id,
            claim=root_claim,
            claim_type=ClaimType.ALIGNMENT,
            depth=0
        )
        
        self.nodes[root_id] = self.root_node
        if self.graph is not None:
            self.graph.add_node(root_id)
        
        # Tracking
        self.total_cost = 0.0
        self.delegation_depth = 0
        self.verification_history: List[Dict[str, Any]] = []
    
    def add_node(self, 
                 claim: str, 
                 claim_type: ClaimType,
                 parent_id: str,
                 proposer_agent: str = None,
                 critic_agent: str = None) -> str:
        """
        Add a new node to the debate tree
        
        Args:
            claim: The claim to be verified
            claim_type: Type of the claim
            parent_id: ID of the parent node
            proposer_agent: Agent proposing this claim
            critic_agent: Agent critiquing this claim
            
        Returns:
            ID of the newly created node
        """
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} not found")
        
        parent_node = self.nodes[parent_id]
        node_id = str(uuid.uuid4())
        
        new_node = DebateNode(
            id=node_id,
            claim=claim,
            claim_type=claim_type,
            parent_id=parent_id,
            depth=parent_node.depth + 1,
            proposer_agent=proposer_agent,
            critic_agent=critic_agent
        )
        
        # Update tracking
        self.delegation_depth = max(self.delegation_depth, new_node.depth)
        
        # Add to structures
        self.nodes[node_id] = new_node
        if self.graph is not None:
            self.graph.add_node(node_id)
            self.graph.add_edge(parent_id, node_id)
        
        # Update parent
        parent_node.add_child(node_id)
        
        return node_id
    
    def expand_node(self, 
                   node_id: str, 
                   sub_claims: List[Tuple[str, ClaimType]],
                   proposer_agent: str = None,
                   critic_agent: str = None) -> List[str]:
        """
        Expand a node with sub-claims (children)
        
        Args:
            node_id: ID of node to expand
            sub_claims: List of (claim_text, claim_type) tuples
            proposer_agent: Agent proposing the expansion
            critic_agent: Agent critiquing the expansion
            
        Returns:
            List of IDs of newly created child nodes
        """
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        child_ids = []
        for claim_text, claim_type in sub_claims:
            child_id = self.add_node(
                claim=claim_text,
                claim_type=claim_type,
                parent_id=node_id,
                proposer_agent=proposer_agent,
                critic_agent=critic_agent
            )
            child_ids.append(child_id)
        
        return child_ids
    
    def get_node(self, node_id: str) -> Optional[DebateNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def get_children(self, node_id: str) -> List[DebateNode]:
        """Get all children of a node"""
        if node_id not in self.nodes:
            return []
        
        node = self.nodes[node_id]
        return [self.nodes[child_id] for child_id in node.children_ids 
                if child_id in self.nodes]
    
    def get_parent(self, node_id: str) -> Optional[DebateNode]:
        """Get parent of a node"""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes[node_id]
        if node.parent_id is None:
            return None
        
        return self.nodes.get(node.parent_id)
    
    def get_leaves(self) -> List[DebateNode]:
        """Get all leaf nodes (nodes with no children)"""
        return [node for node in self.nodes.values() if node.is_leaf()]
    
    def get_path_to_root(self, node_id: str) -> List[DebateNode]:
        """Get path from node to root with cycle detection"""
        path = []
        current_id = node_id
        visited = set()  # Prevent infinite loops from cycles
        
        while current_id is not None:
            if current_id not in self.nodes:
                break
            
            # Cycle detection
            if current_id in visited:
                raise ValueError(f"Cycle detected in tree structure at node {current_id}")
            visited.add(current_id)
            
            node = self.nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        
        return path
    
    def get_subtree(self, node_id: str) -> List[DebateNode]:
        """Get all nodes in subtree rooted at given node"""
        if node_id not in self.nodes:
            return []
        
        subtree = []
        stack = [node_id]
        
        while stack:
            current_id = stack.pop()
            if current_id not in self.nodes:
                continue
            
            current_node = self.nodes[current_id]
            subtree.append(current_node)
            stack.extend(current_node.children_ids)
        
        return subtree
    
    def update_node_status(self, node_id: str, status: NodeStatus, 
                          verification_result: bool = None,
                          confidence: float = None,
                          cost: float = 0.0) -> None:
        """Update the verification status of a node"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        node = self.nodes[node_id]
        node.status = status
        
        if verification_result is not None:
            node.verification_result = verification_result
        
        if confidence is not None:
            node.confidence = confidence
            node.uncertainty = 1.0 - confidence
        
        node.verification_cost += cost
        self.total_cost += cost
        
        # Record in history
        self.verification_history.append({
            'node_id': node_id,
            'status': status.value,
            'result': verification_result,
            'confidence': confidence,
            'cost': cost,
            'timestamp': len(self.verification_history)
        })
    
    def get_uncertain_nodes(self, threshold: float = 0.5) -> List[DebateNode]:
        """Get nodes with uncertainty above threshold"""
        return [node for node in self.nodes.values() 
                if node.uncertainty > threshold and node.status == NodeStatus.PENDING]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the debate tree"""
        total_nodes = len(self.nodes)
        leaf_nodes = len(self.get_leaves())
        
        status_counts = defaultdict(int)
        for node in self.nodes.values():
            status_counts[node.status.value] += 1
        
        return {
            'total_nodes': total_nodes,
            'leaf_nodes': leaf_nodes,
            'delegation_depth': self.delegation_depth,
            'total_cost': self.total_cost,
            'status_distribution': dict(status_counts),
            'avg_uncertainty': sum(n.uncertainty for n in self.nodes.values()) / total_nodes,
            'verification_episodes': len(self.verification_history)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary representation"""
        return {
            'root_id': self.root_id,
            'nodes': {
                node_id: {
                    'id': node.id,
                    'claim': node.claim,
                    'claim_type': node.claim_type.value,
                    'parent_id': node.parent_id,
                    'children_ids': list(node.children_ids),
                    'status': node.status.value,
                    'uncertainty': node.uncertainty,
                    'confidence': node.confidence,
                    'verification_result': node.verification_result,
                    'verification_cost': node.verification_cost,
                    'depth': node.depth,
                    'proposer_agent': node.proposer_agent,
                    'critic_agent': node.critic_agent,
                    'assigned_verifier': node.assigned_verifier,
                    'supporting_evidence': len(node.supporting_evidence),
                    'refuting_evidence': len(node.refuting_evidence)
                }
                for node_id, node in self.nodes.items()
            },
            'context': self.context,
            'stats': self.get_stats()
        }
