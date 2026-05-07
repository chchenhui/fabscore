#!/usr/bin/env python3
"""
Token Usage Tracker
Tracks and reports token usage across different operations
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TokenTracker:
    """Tracks token usage across different operations"""
    
    def __init__(self, log_dir: str = "results/token_usage"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.usage = {
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cost": 0.0,
            "operations": []
        }
        
        # Token costs per 1K tokens (as of 2025)
        self.cost_per_1k = {
            "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},  # $0.15/$0.60 per 1M tokens
            "gpt-4o": {"prompt": 0.0025, "completion": 0.01}  # $2.50/$10.00 per 1M tokens
        }
    
    def track_usage(self, operation_type: str, model: str, 
                   prompt_tokens: int, completion_tokens: int,
                   cached: bool = False) -> None:
        """Track token usage for an operation"""
        
        # Calculate cost
        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        
        # Record operation
        operation = {
            "timestamp": datetime.now().isoformat(),
            "operation_type": operation_type,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "cached": cached
        }
        
        # Update totals
        if not cached:  # Only count towards totals if not cached
            self.usage["total_prompt_tokens"] += prompt_tokens
            self.usage["total_completion_tokens"] += completion_tokens
            self.usage["total_cost"] += cost
        
        self.usage["operations"].append(operation)
        
        # Log the usage
        if not cached:
            logger.info(f"💰 API Usage - {operation_type}: {prompt_tokens} prompt + "
                       f"{completion_tokens} completion tokens = ${cost:.5f}")
        else:
            logger.info(f"💾 Cached Usage - {operation_type}: {prompt_tokens} prompt + "
                       f"{completion_tokens} completion tokens (cost saved: ${cost:.5f})")
    
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage"""
        if model not in self.cost_per_1k:
            logger.warning(f"Unknown model {model}, using gpt-4o pricing")
            model = "gpt-4o"
            
        prompt_cost = (prompt_tokens / 1000) * self.cost_per_1k[model]["prompt"]
        completion_cost = (completion_tokens / 1000) * self.cost_per_1k[model]["completion"]
        return prompt_cost + completion_cost
    
    def save_report(self) -> None:
        """Save usage report to file"""
        report_file = self.log_dir / f"token_usage_{self.current_session}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(self.usage, f, indent=2)
            logger.info(f"Token usage report saved to {report_file}")
            
            # Log summary
            total_tokens = self.usage["total_prompt_tokens"] + self.usage["total_completion_tokens"]
            logger.info(f"\n=== Token Usage Summary ===\n"
                       f"Total Tokens: {total_tokens:,}\n"
                       f"- Prompt Tokens: {self.usage['total_prompt_tokens']:,}\n"
                       f"- Completion Tokens: {self.usage['total_completion_tokens']:,}\n"
                       f"Total Cost: ${self.usage['total_cost']:.5f}\n"
                       f"Operations: {len(self.usage['operations'])}")
        except Exception as e:
            logger.exception(f"Failed to save token usage report: {e}")

# Global instance
_token_tracker: Optional[TokenTracker] = None

def get_token_tracker() -> TokenTracker:
    """Get or create the global token tracker instance"""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker()
    return _token_tracker