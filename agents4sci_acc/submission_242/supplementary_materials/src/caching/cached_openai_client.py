#!/usr/bin/env python3
"""
Cached OpenAI Client
Wraps the OpenAI client to cache all API responses for cost savings
"""

import logging
from typing import Dict, List
from .api_cache import get_api_cache
from .token_tracker import get_token_tracker
from config.llm_config import DEFAULT_MODEL

logger = logging.getLogger(__name__)

class CachedChatCompletion:
    """Mock response object that mimics OpenAI's ChatCompletion response"""
    def __init__(self, content: str, input_tokens: int = 0, output_tokens: int = 0):
        self.choices = [type('Choice', (), {
            'message': type('Message', (), {
                'content': content
            })()
        })()]
        self.usage = type('Usage', (), {
            'prompt_tokens': input_tokens,
            'completion_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        })

class CachedOpenAIClient:
    """OpenAI client wrapper with comprehensive caching"""
    
    def __init__(self, original_client):
        self.original_client = original_client
        self.cache = get_api_cache()
        self.chat = type('Chat', (), {
            'completions': self
        })()
    
    def create(self, model: str = DEFAULT_MODEL, messages: List[Dict[str, str]] = None, 
               temperature: float = 0.7, max_completion_tokens: int = 1000, **kwargs):
        """Create a chat completion with caching"""
        
        # Extract the user prompt from messages
        if messages:
            # Combine all messages into a single prompt for caching
            prompt_parts = []
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_parts.append(f"{role}: {content}")
            prompt = "\n".join(prompt_parts)
        else:
            prompt = ""
        
        # Check cache first
        cached_response = self.cache.get_cached_response(
            prompt, model, temperature=temperature, max_tokens=max_completion_tokens
        )
        
        if cached_response:
            # Validate cached response before using it
            if isinstance(cached_response, dict):
                content = cached_response.get('response', '')
            else:
                content = cached_response
            
            # Try to validate if the content contains valid JSON (for structured responses)
            if self._is_structured_response(content):
                is_valid = self._validate_json_response(content)
                if not is_valid:
                    logger.warning("⚠️ Cached response contains invalid JSON, clearing cache entry and retrying...")
                    logger.warning(f"🔍 CORRUPTED CONTENT (first 200 chars): {content[:200]}")
                    self._remove_cached_response(prompt, model, temperature, max_completion_tokens)
                    # Continue to make fresh API call
                else:
                    logger.info("🎯 Using cached API response ($$$ saved!)")
                    # Track cached usage with exact token counts
                    token_tracker = get_token_tracker()
                    token_tracker.track_usage(
                        operation_type="chat_completion",
                        model=model,
                        prompt_tokens=cached_response.get('input_tokens', 0),
                        completion_tokens=cached_response.get('output_tokens', 0),
                        cached=True
                    )
                    return CachedChatCompletion(
                        content=content,
                        input_tokens=cached_response.get('input_tokens', 0),
                        output_tokens=cached_response.get('output_tokens', 0)
                    )
            else:
                logger.info("🎯 Using cached API response ($$$ saved!)")
                # Track cached usage with exact token counts
                token_tracker = get_token_tracker()
                token_tracker.track_usage(
                    operation_type="chat_completion",
                    model=model,
                    prompt_tokens=cached_response.get('input_tokens', 0),
                    completion_tokens=cached_response.get('output_tokens', 0),
                    cached=True
                )
                return CachedChatCompletion(
                    content=content,
                    input_tokens=cached_response.get('input_tokens', 0),
                    output_tokens=cached_response.get('output_tokens', 0)
                )
        
        # Make actual API call if not cached
        logger.info("🌐 Making new API request (costs money)")
        try:
            # Use appropriate parameters based on model
            api_kwargs = self._get_api_kwargs(model, messages, temperature, max_completion_tokens, **kwargs)
            
            response = self.original_client.chat.completions.create(**api_kwargs)
            
            # Extract response content and token usage
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            # Track token usage
            token_tracker = get_token_tracker()
            token_tracker.track_usage(
                operation_type="chat_completion",
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                cached=False
            )
            
            # Cache response with token information
            self.cache.cache_response(
                prompt=prompt,
                response=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                temperature=temperature,
                max_tokens=max_completion_tokens
            )
            
            return response
            
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
    
    def _get_api_kwargs(self, model: str, messages, temperature: float, max_completion_tokens: int, **kwargs) -> dict:
        """Get appropriate API parameters based on model type"""
        if "gpt-5" in model.lower():
            # GPT-5 models use max_completion_tokens and only support temperature=1.0
            return {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_completion_tokens,
                **kwargs
            }
        else:
            # GPT-4 and earlier models use max_tokens and support custom temperature
            return {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_completion_tokens,
                **kwargs
            }
    
    def _is_structured_response(self, content: str) -> bool:
        """Check if response appears to contain structured data (JSON)"""
        return ("```json" in content or 
                (content.strip().startswith('[') and content.strip().endswith(']')) or
                (content.strip().startswith('{') and content.strip().endswith('}')))
    
    def _validate_json_response(self, content: str) -> bool:
        """Validate that cached response contains parseable JSON"""
        try:
            import json
            
            # Simple validation - just try to parse the JSON directly
            content = content.strip()
            
            # Handle markdown JSON blocks
            if content.startswith('```json') and content.endswith('```'):
                content = content[7:-3].strip()
            elif content.startswith('```') and content.endswith('```'):
                content = content[3:-3].strip()
            
            # Try to parse as JSON
            json.loads(content)
            return True
            
        except (json.JSONDecodeError, Exception):
            return False
    
    def _remove_cached_response(self, prompt: str, model: str, temperature: float, max_completion_tokens: int):
        """Remove a corrupted cache entry"""
        try:
            # Generate the same cache key that would be used
            cache_key = self.cache._generate_cache_key(prompt, model, temperature=temperature, max_tokens=max_completion_tokens)
            with self.cache._lock:
                if hasattr(self.cache, 'cache') and cache_key in self.cache.cache:
                    del self.cache.cache[cache_key]
            self.cache._save_cache()  # Persist the change (this already has its own lock)
            logger.info(f"🗑️ Removed corrupted cache entry")
        except Exception as e:
            logger.warning(f"Failed to remove corrupted cache entry: {e}")
    
    def get_cache_stats(self):
        """Get caching statistics"""
        return self.cache.get_cache_stats()
    
    def clear_cache(self):
        """Clear the API cache"""
        self.cache.clear_cache()

def create_cached_openai_client(original_client):
    """Create a cached version of an OpenAI client"""
    return CachedOpenAIClient(original_client)
