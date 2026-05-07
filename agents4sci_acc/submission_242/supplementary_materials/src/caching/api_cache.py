#!/usr/bin/env python3
"""
API Request Caching System
Caches all OpenAI API responses to avoid repeated costs during experimentation
"""

import json
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class APICache:
    def __init__(self, cache_dir: str = "cache/api_responses"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "api_responses.json"
        self._lock = threading.Lock()  # Thread safety lock
        self.cache = self._load_cache()
        
    def _load_cache(self) -> Dict[str, Any]:
        """Load existing cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded API cache with {len(cache)} entries")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk with thread safety"""
        with self._lock:
            try:
                # Create a copy of the cache to avoid iteration issues
                cache_copy = dict(self.cache)
                with open(self.cache_file, 'w') as f:
                    json.dump(cache_copy, f, indent=2)
            except Exception as e:
                logger.exception(f"Failed to save cache: {e}")
    
    def _generate_cache_key(self, prompt: str, model: str = "gpt-4o-mini", **kwargs) -> str:
        """Generate a unique cache key for the request"""
        # Create a hash of the prompt and key parameters
        cache_input = {
            'prompt': prompt,
            'model': model,
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1000)
        }
        cache_string = json.dumps(cache_input, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def get_cached_response(self, prompt: str, model: str = "gpt-4o-mini", **kwargs) -> Optional[Dict]:
        """Get cached response if it exists"""
        cache_key = self._generate_cache_key(prompt, model, **kwargs)
        with self._lock:
            return self.cache.get(cache_key)
    
    def cache_response(self, prompt: str, response: str, model: str = "gpt-4o-mini", 
                     input_tokens: int = 0, output_tokens: int = 0, **kwargs):
        """Cache a new response"""
        cache_key = self._generate_cache_key(prompt, model, **kwargs)
        with self._lock:
            self.cache[cache_key] = {
                'response': response,
                'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt,
                'model': model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cached_at': __import__('datetime').datetime.now().isoformat()
            }
        # Save cache outside the lock to avoid holding it too long
        self._save_cache()
        logger.debug(f"Cached response for key: {cache_key}")
    
    def clear_cache(self):
        """Clear all cached responses"""
        with self._lock:
            self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        logger.info("API cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            cache_size = len(self.cache)
        return {
            'total_entries': cache_size,
            'cache_file_size': self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            'cache_location': str(self.cache_file)
        }

# Global cache instance
_api_cache = APICache()

def get_api_cache() -> APICache:
    """Get the global API cache instance"""
    return _api_cache
