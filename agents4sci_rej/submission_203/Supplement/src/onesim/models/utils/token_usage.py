"""
Token usage tracking utilities for OneSim.

This module provides functionality for tracking token usage across different models
and calculating estimated costs.
"""

import json
import time
from threading import Lock
from typing import Dict, Any, Optional
from loguru import logger

class TokenUsageTracker:
    """
    Tracks token usage statistics across different model calls.
    
    This class provides methods to record token usage and generate statistics
    about token consumption by different models.
    """
    
    def __init__(self):
        """Initialize the token usage tracker with zero counts."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.model_usage = {}  # 按模型名称统计
        self.request_count = 0
        self.start_time = time.time()
        
    def track(self, model_name: str, prompt_tokens: int, completion_tokens: int, total_tokens: Optional[int] = None):
        """
        Track token usage for a single model call.
        
        Args:
            model_name: Name of the model used
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            total_tokens: Total tokens used (if not provided, will be calculated as sum)
        """
        if total_tokens is None:
            total_tokens = prompt_tokens + completion_tokens
            
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.request_count += 1
        
        # 按模型统计
        if model_name not in self.model_usage:
            self.model_usage[model_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0
            }
        
        self.model_usage[model_name]["prompt_tokens"] += prompt_tokens
        self.model_usage[model_name]["completion_tokens"] += completion_tokens
        self.model_usage[model_name]["total_tokens"] += total_tokens
        self.model_usage[model_name]["request_count"] += 1
        
        # Log summary every 10 requests
        if self.request_count % 10 == 0:
            logger.debug(f"Token usage after {self.request_count} requests: {self.total_tokens} tokens")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage statistics.
        
        Returns:
            Dictionary containing token usage statistics
        """
        elapsed_time = time.time() - self.start_time
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
            "model_usage": self.model_usage,
            "elapsed_time_seconds": elapsed_time,
            "tokens_per_second": self.total_tokens / elapsed_time if elapsed_time > 0 else 0
        }
    
    def estimate_cost(self, price_config: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
        """
        Estimate the cost of token usage based on provided price configuration.
        
        Args:
            price_config: Dictionary mapping model names to price configurations
                          (price per 1000 tokens for prompt and completion)
                          
        Returns:
            Dictionary containing cost estimates
        """
        # 默认价格配置（每1000个token的美元价格）
        default_price = {
            "gpt-4": {"prompt": 0.005, "completion": 0.015},
            # "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002}
        }
        
        price_config = price_config or default_price
        
        cost = 0.0
        model_costs = {}
        
        for model, usage in self.model_usage.items():
            model_price = None
            
            # 寻找匹配的价格配置
            for price_model, price in price_config.items():
                if price_model in model:
                    model_price = price
                    break
            
            if model_price:
                prompt_cost = usage["prompt_tokens"] * model_price["prompt"] / 1000
                completion_cost = usage["completion_tokens"] * model_price["completion"] / 1000
                total_cost = prompt_cost + completion_cost
                
                model_costs[model] = {
                    "prompt_cost": prompt_cost,
                    "completion_cost": completion_cost,
                    "total_cost": total_cost
                }
                
                cost += total_cost
        
        return {
            "total_cost_usd": cost,
            "model_costs": model_costs
        }
        
    def reset(self):
        """Reset all counters to zero."""
        self.__init__()

    def export_to_file(self, filepath: Optional[str] = None) -> str:
        """
        Export token usage statistics to a JSON file.
        
        Args:
            filepath: Path to save the file (if None, generates timestamp-based filename)
            
        Returns:
            Path to the saved file
        """
        stats = self.get_usage_stats()
        cost_estimate = self.estimate_cost()
        
        # Combine stats and cost estimate
        export_data = {
            **stats,
            "cost_estimate": cost_estimate
        }
        
        # Generate timestamp-based filename if not provided
        if filepath is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"token_usage_{timestamp}.json"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Token usage statistics exported to {filepath}")
        return filepath


# Singleton instance and access functions
_token_tracker_instance = None
_token_tracker_lock = Lock()

def get_token_tracker() -> TokenUsageTracker:
    """
    Get the global token tracker instance.
    
    Returns:
        The singleton TokenUsageTracker instance
    """
    global _token_tracker_instance
    
    if _token_tracker_instance is None:
        with _token_tracker_lock:
            if _token_tracker_instance is None:
                _token_tracker_instance = TokenUsageTracker()
    
    return _token_tracker_instance

def reset_token_stats():
    """Reset all token usage statistics."""
    tracker = get_token_tracker()
    tracker.reset()
    logger.info("Token usage statistics have been reset")

def get_token_usage_stats() -> Dict[str, Any]:
    """
    Get current token usage statistics.
    
    Returns:
        Dictionary containing token usage statistics
    """
    tracker = get_token_tracker()
    return tracker.get_usage_stats()

def estimate_token_cost(price_config: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Any]:
    """
    Estimate cost of token usage based on provided price configuration.
    
    Args:
        price_config: Dictionary mapping model names to price configurations
        
    Returns:
        Dictionary containing cost estimates
    """
    tracker = get_token_tracker()
    return tracker.estimate_cost(price_config)

def export_token_usage_stats(filepath: Optional[str] = None) -> str:
    """
    Export token usage statistics to a JSON file.
    
    Args:
        filepath: Path to save the file
        
    Returns:
        Path to the saved file
    """
    tracker = get_token_tracker()
    return tracker.export_to_file(filepath)

def log_token_usage():
    """Log current token usage statistics."""
    stats = get_token_usage_stats()
    logger.info(f"Token usage: {stats['total_tokens']} total tokens "
                f"({stats['total_prompt_tokens']} prompt, {stats['total_completion_tokens']} completion) "
                f"in {stats['request_count']} requests")
    
    # Log model-specific usage
    for model, usage in stats['model_usage'].items():
        logger.info(f"  - {model}: {usage['total_tokens']} tokens in {usage['request_count']} requests")
    
    # Estimate and log cost
    cost_estimate = estimate_token_cost()
    if cost_estimate['total_cost_usd'] > 0:
        logger.info(f"Estimated cost: ${cost_estimate['total_cost_usd']:.4f}") 