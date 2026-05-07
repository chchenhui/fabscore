#!/usr/bin/env python
"""
Example script demonstrating the caching mechanism of load balancers.
This script shows how the load balancer instances are cached for better performance.
"""

import time
import asyncio
import os

from loguru import logger
from onesim.models import (
    configure_load_balancer, get_model, get_embedding_model, get_model_manager,
    SystemMessage, UserMessage
)


async def test_cached_load_balancer():
    """Test the load balancer caching mechanism."""
    # Set up logger to show debug messages
    logger.remove()
    logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    
    # Get API key from environment or set it here
    api_key = os.environ.get("OPENAI_API_KEY", "dummy-api-key")
    
    # Get the model manager
    manager = get_model_manager()
    
    # Configure some model configs for testing
    manager.load_model_configs([
        {
            "config_name": "gpt-4",
            "model_type": "openai_chat",
            "model_name": "gpt-4",
            "api_key": api_key
        },
        {
            "config_name": "gpt-3.5-turbo",
            "model_type": "openai_chat",
            "model_name": "gpt-3.5-turbo",
            "api_key": api_key
        },
        {
            "config_name": "ada-002",
            "model_type": "openai_embedding",
            "model_name": "text-embedding-ada-002",
            "api_key": api_key
        }
    ])
    
    # Configure load balancers
    configure_load_balancer(
        model_configs=["gpt-4", "gpt-3.5-turbo"],
        strategy="round_robin",
        config_name="llm_balancer",
        model_type="chat"
    )
    
    configure_load_balancer(
        model_configs=["ada-002"],
        strategy="round_robin",
        config_name="embedding_balancer",
        model_type="embedding"
    )
    
    # Test LLM load balancer caching
    print("\n=== Testing LLM Load Balancer Caching ===")
    
    # First request - should create a new instance
    start_time = time.time()
    llm_balancer1 = get_model("llm_balancer")
    time1 = time.time() - start_time
    print(f"First load balancer request took: {time1:.6f} seconds")
    
    # Second request - should use cached instance
    start_time = time.time()
    llm_balancer2 = get_model("llm_balancer")
    time2 = time.time() - start_time
    print(f"Second load balancer request took: {time2:.6f} seconds")
    
    # Check if we got the same instance
    print(f"Same instance: {llm_balancer1 is llm_balancer2}")
    print(f"Speed improvement: {time1/time2:.2f}x faster")

    # Test embedding load balancer
    print("\n=== Testing Embedding Load Balancer Caching ===")
    
    # First request - should create a new instance
    start_time = time.time()
    emb_balancer1 = get_model("embedding_balancer")
    time1 = time.time() - start_time
    print(f"First embedding balancer request took: {time1:.6f} seconds")
    
    # Second request - should use cached instance
    start_time = time.time()
    emb_balancer2 = get_model("embedding_balancer")
    time2 = time.time() - start_time
    print(f"Second embedding balancer request took: {time2:.6f} seconds")
    
    # Check if we got the same instance
    print(f"Same instance: {emb_balancer1 is emb_balancer2}")
    print(f"Speed improvement: {time1/time2:.2f}x faster")

    # Test convenience functions
    print("\n=== Testing Convenience Functions ===")
    
    # Test get_model() default
    start_time = time.time()
    default_llm = get_model()  # Should return the default "load_balancer"
    time1 = time.time() - start_time
    print(f"Default get_model() took: {time1:.6f} seconds")
    
    # Test get_embedding_model() default
    start_time = time.time()
    default_emb = get_embedding_model()  # Should return the "embedding_load_balancer"
    time2 = time.time() - start_time
    print(f"Default get_embedding_model() took: {time2:.6f} seconds")
    
    # Test reconfiguration clears cache
    print("\n=== Testing Cache Invalidation ===")
    old_balancer = get_model("llm_balancer")
    
    # Reconfigure load balancer
    configure_load_balancer(
        model_configs=["gpt-4"],  # Changed model list
        strategy="random",      # Changed strategy
        config_name="llm_balancer",
        model_type="chat"
    )
    
    # Get new balancer
    new_balancer = get_model("llm_balancer")
    
    # Check if new instance is created
    print(f"Same instance after reconfiguration: {old_balancer is new_balancer}")
    print(f"Old strategy: {old_balancer._strategy_name}, New strategy: {new_balancer._strategy_name}")


if __name__ == "__main__":
    asyncio.run(test_cached_load_balancer()) 