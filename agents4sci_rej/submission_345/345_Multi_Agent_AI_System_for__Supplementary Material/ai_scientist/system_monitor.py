"""
System Monitor for Phase 4 Implementation Pipeline
Following Linus principles: Track every decision for reproducibility
"""

import hashlib
import subprocess
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Decision:
    """Single decision made by an agent"""
    timestamp: str
    agent: str
    decision: str
    reasoning: str
    confidence: float
    git_hash: str
    input_hash: str
    output_hash: str

@dataclass 
class APICall:
    """Single API call for reproducibility"""
    timestamp: str
    model: str
    prompt_hash: str
    response_hash: str
    tokens_input: int
    tokens_output: int
    cost: float
    latency_ms: float

class SystemMonitor:
    """
    Track every decision for reproducibility.
    No more "it just works" - full provenance.
    Phase 4 requirement from MASSIVE_OVERHAUL_PLAN.md
    """
    
    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.decisions: List[Decision] = []
        self.api_calls: List[APICall] = []
        self.data_sources: List[str] = []
        
        # Cache git info
        self._git_hash = self._get_git_hash()
        
        # Start timing
        self.start_time = datetime.now()
    
    def _get_git_hash(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else 'unknown'
        except:
            return 'unknown'
    
    def _hash_content(self, content: str) -> str:
        """Create hash of content for reproducibility"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def log_decision(self, agent: str, decision: str, reasoning: str, 
                    confidence: float, input_data: Any = None, output_data: Any = None):
        """
        Log a decision made by an agent
        
        Args:
            agent: Name of the agent making the decision
            decision: What decision was made
            reasoning: Why this decision was made
            confidence: Confidence level (0-1)
            input_data: Input data that led to this decision
            output_data: Output data from this decision
        """
        
        # Hash input/output for reproducibility
        input_hash = self._hash_content(str(input_data)) if input_data else 'none'
        output_hash = self._hash_content(str(output_data)) if output_data else 'none'
        
        decision_record = Decision(
            timestamp=datetime.now().isoformat(),
            agent=agent,
            decision=decision,
            reasoning=reasoning,
            confidence=confidence,
            git_hash=self._git_hash,
            input_hash=input_hash,
            output_hash=output_hash
        )
        
        self.decisions.append(decision_record)
        
        # Also log to console for immediate feedback
        print(f"[DECISION] {agent}: {decision} (confidence: {confidence:.2f})")
    
    def log_api_call(self, model: str, prompt: str, response: str, 
                     tokens_input: int, tokens_output: int, cost: float, latency_ms: float):
        """
        Log an API call for cost tracking and reproducibility
        """
        
        api_call = APICall(
            timestamp=datetime.now().isoformat(),
            model=model,
            prompt_hash=self._hash_content(prompt),
            response_hash=self._hash_content(response),
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            latency_ms=latency_ms
        )
        
        self.api_calls.append(api_call)
    
    def log_data_source(self, source: str):
        """Log a data source being accessed"""
        if source not in self.data_sources:
            self.data_sources.append(source)
    
    def get_total_cost(self) -> float:
        """Get total cost of API calls"""
        return sum(call.cost for call in self.api_calls)
    
    def get_total_tokens(self) -> int:
        """Get total tokens used"""
        return sum(call.tokens_input + call.tokens_output for call in self.api_calls)
    
    def get_execution_time(self) -> float:
        """Get total execution time in seconds"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def generate_audit_trail(self) -> Dict[str, Any]:
        """
        Complete reproducibility package
        Phase 4 requirement: Full provenance
        """
        
        return {
            'execution_summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'execution_time_seconds': self.get_execution_time(),
                'total_cost_usd': self.get_total_cost(),
                'total_tokens': self.get_total_tokens(),
                'total_decisions': len(self.decisions),
                'total_api_calls': len(self.api_calls),
                'data_sources_accessed': len(self.data_sources)
            },
            'decisions': [asdict(d) for d in self.decisions],
            'api_calls': [asdict(c) for c in self.api_calls],
            'data_sources': self.data_sources,
            'system_info': {
                'git_hash': self._git_hash,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def save_audit_trail(self, filename: str = None) -> Path:
        """Save complete audit trail to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_audit_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.generate_audit_trail(), f, indent=2)
        
        print(f"[MONITOR] Audit trail saved to: {filepath}")
        return filepath
    
    def print_summary(self):
        """Print execution summary"""
        
        print("\n" + "="*60)
        print("SYSTEM MONITOR SUMMARY")
        print("="*60)
        print(f"Execution time: {self.get_execution_time():.1f} seconds")
        print(f"Total cost: ${self.get_total_cost():.4f}")
        print(f"Total tokens: {self.get_total_tokens():,}")
        print(f"Decisions logged: {len(self.decisions)}")
        print(f"API calls: {len(self.api_calls)}")
        print(f"Data sources: {len(self.data_sources)}")
        
        if self.decisions:
            print(f"\nKey decisions:")
            for decision in self.decisions[-3:]:  # Show last 3 decisions
                print(f"  - {decision.agent}: {decision.decision}")
        
        print("="*60)

# Global monitor instance for easy access
_global_monitor = None

def get_system_monitor() -> SystemMonitor:
    """Get global system monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SystemMonitor()
    return _global_monitor

def reset_system_monitor():
    """Reset global monitor (for testing)"""
    global _global_monitor
    _global_monitor = SystemMonitor()