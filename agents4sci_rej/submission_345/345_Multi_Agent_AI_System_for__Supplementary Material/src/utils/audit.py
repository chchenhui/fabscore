"""
Audit logging for full reproducibility and cost tracking.
Following Linus principle: Log everything that matters, nothing that doesn't.
Enhanced with timeout/backoff logic and JSON schema validation per GPT-5 guidance.
"""

import json
import hashlib
import subprocess
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
import os
import sys
import jsonschema
from functools import wraps

class AuditLogger:
    """
    Unified audit logger for usage tracking and provenance.
    Writes to:
    - results/usage_log.jsonl: API calls and costs
    - results/run_provenance.json: Git hash, seeds, configs
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path(__file__).parent.parent.parent / "results"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.usage_log_path = self.output_dir / "usage_log.jsonl"
        self.provenance_path = self.output_dir / "run_provenance.json"
        
        # Initialize provenance
        self.provenance = self._init_provenance()
        
        # Track totals
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0
    
    def _init_provenance(self) -> Dict[str, Any]:
        """Initialize provenance with system info."""
        return {
            'timestamp': datetime.now().isoformat(),
            'git_commit': self._get_git_commit(),
            'git_branch': self._get_git_branch(),
            'git_dirty': self._is_git_dirty(),
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd(),
            'env_vars': {
                k: v for k, v in os.environ.items() 
                if any(x in k for x in ['OPENAI', 'ANTHROPIC', 'GOOGLE', 'DEEPSEEK', 'PERPLEXITY'])
            },
            'runs': []
        }
    
    def _get_git_commit(self) -> str:
        """Get current git commit hash."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_git_branch(self) -> str:
        """Get current git branch."""
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _is_git_dirty(self) -> bool:
        """Check if git working directory has uncommitted changes."""
        try:
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            return len(result.stdout.strip()) > 0
        except:
            return True
    
    def log_api_call(self, 
                     provider: str,
                     model: str,
                     prompt: str,
                     response: str,
                     tokens: int,
                     cost: float,
                     task_type: Optional[str] = None,
                     metadata: Optional[Dict] = None) -> None:
        """
        Log an API call for usage tracking.
        
        Args:
            provider: API provider (openai, anthropic, etc.)
            model: Model name
            prompt: The prompt sent
            response: The response received
            tokens: Token count
            cost: Estimated cost in USD
            task_type: Optional task classification
            metadata: Optional additional metadata
        """
        
        # Create log entry
        entry = {
            'timestamp': datetime.now().isoformat(),
            'provider': provider,
            'model': model,
            'prompt_hash': hashlib.sha256(prompt.encode()).hexdigest(),
            'response_hash': hashlib.sha256(response.encode()).hexdigest(),
            'prompt_length': len(prompt),
            'response_length': len(response),
            'tokens': tokens,
            'cost': cost,
            'task_type': task_type,
            'metadata': metadata or {}
        }
        
        # Append to usage log
        with open(self.usage_log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
        
        # Update totals
        self.total_tokens += tokens
        self.total_cost += cost
        self.api_calls += 1
        
        # Log to console if expensive
        if cost > 0.10:
            print(f"[AUDIT] Expensive call: ${cost:.3f} for {model} ({tokens} tokens)")
    
    def log_run(self,
                experiment_name: str,
                config: Dict[str, Any],
                seed: int,
                results: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an experimental run for provenance.
        
        Args:
            experiment_name: Name of the experiment
            config: Configuration used
            seed: Random seed
            results: Optional results summary
        """
        
        run_entry = {
            'timestamp': datetime.now().isoformat(),
            'experiment': experiment_name,
            'seed': seed,
            'config': config,
            'results': results or {},
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost
        }
        
        # Add to provenance
        self.provenance['runs'].append(run_entry)
        
        # Save provenance
        self.save_provenance()
        
        print(f"[AUDIT] Logged run: {experiment_name} (seed={seed}, cost=${self.total_cost:.3f})")
    
    def save_provenance(self) -> None:
        """Save provenance to file."""
        with open(self.provenance_path, 'w') as f:
            json.dump(self.provenance, f, indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        return {
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_cost_per_call': self.total_cost / max(1, self.api_calls),
            'experiments_run': len(self.provenance['runs'])
        }
    
    def reset_counters(self) -> None:
        """Reset counters for new experiment."""
        self.total_tokens = 0
        self.total_cost = 0.0
        self.api_calls = 0


# Global singleton instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    """Get or create global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


def audit_api_call(provider: str, model: str, prompt: str, response: str,
                   tokens: int, cost: float, **kwargs) -> None:
    """Convenience function to log API call."""
    logger = get_audit_logger()
    logger.log_api_call(provider, model, prompt, response, tokens, cost, **kwargs)


def audit_run(experiment_name: str, config: Dict, seed: int, results: Dict = None) -> None:
    """Convenience function to log experimental run."""
    logger = get_audit_logger()
    logger.log_run(experiment_name, config, seed, results)


def audit_summary() -> Dict[str, Any]:
    """Get audit summary."""
    logger = get_audit_logger()
    return logger.get_summary()


class APIOrchestrator:
    """
    API orchestration with timeout/backoff and schema validation per GPT-5 guidance.
    Handles retries, rate limiting, and structured output validation.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 timeout: float = 120.0):
        """
        Initialize orchestrator.
        
        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
            timeout: Request timeout (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout
        self.audit_logger = get_audit_logger()
    
    def with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            Function result
        
        Raises:
            Exception: Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Log successful call
                elapsed = time.time() - start_time
                self.audit_logger.log_api_call(
                    provider="orchestrator",
                    model="retry_wrapper",
                    prompt=f"Function: {func.__name__}",
                    response="SUCCESS",
                    tokens=0,
                    cost=0.0,
                    task_type="orchestration",
                    metadata={
                        'attempt': attempt + 1,
                        'elapsed_seconds': elapsed,
                        'function': func.__name__
                    }
                )
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    # Calculate delay with exponential backoff + jitter
                    delay = min(
                        self.base_delay * (2 ** attempt) + random.uniform(0, 1),
                        self.max_delay
                    )
                    
                    print(f"[ORCHESTRATOR] Attempt {attempt + 1} failed: {e}")
                    print(f"[ORCHESTRATOR] Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    print(f"[ORCHESTRATOR] All {self.max_retries + 1} attempts failed")
        
        # Log failed call
        self.audit_logger.log_api_call(
            provider="orchestrator",
            model="retry_wrapper",
            prompt=f"Function: {func.__name__}",
            response=f"FAILED: {last_exception}",
            tokens=0,
            cost=0.0,
            task_type="orchestration_failed",
            metadata={
                'attempts': self.max_retries + 1,
                'function': func.__name__,
                'final_error': str(last_exception)
            }
        )
        
        raise last_exception
    
    def validate_json_schema(self, 
                           data: Union[str, Dict], 
                           schema: Dict[str, Any],
                           strict: bool = True) -> Dict[str, Any]:
        """
        Validate JSON data against schema with retry logic for LLM responses.
        
        Args:
            data: JSON string or dict to validate
            schema: JSON schema to validate against
            strict: If True, raise on validation failure. If False, log and return data.
        
        Returns:
            Validated and parsed JSON data
        
        Raises:
            jsonschema.ValidationError: If validation fails and strict=True
        """
        # Parse JSON if string
        if isinstance(data, str):
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e}"
                if strict:
                    raise ValueError(error_msg)
                else:
                    print(f"[SCHEMA] Warning: {error_msg}")
                    return {}
        else:
            parsed_data = data
        
        # Validate against schema
        try:
            jsonschema.validate(parsed_data, schema)
            
            # Log successful validation
            self.audit_logger.log_api_call(
                provider="orchestrator",
                model="json_validator",
                prompt=f"Schema: {json.dumps(schema, default=str)}",
                response="VALID",
                tokens=len(json.dumps(parsed_data)),
                cost=0.0,
                task_type="schema_validation",
                metadata={
                    'schema_type': schema.get('title', 'unknown'),
                    'data_size': len(str(parsed_data))
                }
            )
            
            return parsed_data
            
        except jsonschema.ValidationError as e:
            error_msg = f"Schema validation failed: {e.message}"
            
            # Log validation failure
            self.audit_logger.log_api_call(
                provider="orchestrator",
                model="json_validator",
                prompt=f"Schema: {json.dumps(schema, default=str)}",
                response=f"INVALID: {error_msg}",
                tokens=len(json.dumps(parsed_data)),
                cost=0.0,
                task_type="schema_validation_failed",
                metadata={
                    'schema_type': schema.get('title', 'unknown'),
                    'validation_error': str(e),
                    'invalid_path': list(e.absolute_path) if e.absolute_path else []
                }
            )
            
            if strict:
                raise
            else:
                print(f"[SCHEMA] Warning: {error_msg}")
                return parsed_data
    
    def llm_call_with_schema(self,
                           llm_func: Callable,
                           schema: Dict[str, Any],
                           max_schema_retries: int = 2,
                           **llm_kwargs) -> Dict[str, Any]:
        """
        Make LLM call with automatic schema validation and retry.
        
        Args:
            llm_func: Function that makes LLM call and returns response
            schema: JSON schema for expected response
            max_schema_retries: Retries for schema validation failures
            **llm_kwargs: Arguments for LLM function
        
        Returns:
            Validated JSON response
        """
        for attempt in range(max_schema_retries + 1):
            try:
                # Make LLM call with retry logic
                response = self.with_retry(llm_func, **llm_kwargs)
                
                # Validate response against schema
                validated_data = self.validate_json_schema(
                    response, schema, strict=True
                )
                
                return validated_data
                
            except (jsonschema.ValidationError, ValueError) as e:
                if attempt < max_schema_retries:
                    print(f"[ORCHESTRATOR] Schema validation failed, retry {attempt + 1}/{max_schema_retries}")
                    # Add schema reminder to prompt for next attempt
                    if 'prompt' in llm_kwargs:
                        llm_kwargs['prompt'] += f"\n\nIMPORTANT: Response must be valid JSON matching this schema: {json.dumps(schema)}"
                else:
                    print(f"[ORCHESTRATOR] Schema validation failed after {max_schema_retries + 1} attempts")
                    raise


# Create common JSON schemas for Phase 5
REVENUE_EXTRACTION_SCHEMA = {
    "type": "object",
    "title": "Revenue Extraction",
    "properties": {
        "drug_name": {"type": "string"},
        "year_since_launch": {"type": "integer", "minimum": 0},
        "revenue_usd": {"type": "number", "minimum": 0},
        "source": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["drug_name", "year_since_launch", "revenue_usd", "source"]
}

FORECAST_RESULT_SCHEMA = {
    "type": "object", 
    "title": "Forecast Result",
    "properties": {
        "forecast": {
            "type": "array",
            "items": {"type": "number", "minimum": 0}
        },
        "confidence_intervals": {
            "type": "object",
            "properties": {
                "lower": {"type": "array", "items": {"type": "number"}},
                "upper": {"type": "array", "items": {"type": "number"}}
            }
        },
        "metadata": {"type": "object"}
    },
    "required": ["forecast"]
}


# Global orchestrator instance
_orchestrator = None

def get_orchestrator() -> APIOrchestrator:
    """Get or create global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = APIOrchestrator()
    return _orchestrator


def with_retry(func: Callable) -> Callable:
    """Decorator for automatic retry with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        orchestrator = get_orchestrator()
        return orchestrator.with_retry(func, *args, **kwargs)
    return wrapper


def validate_schema(schema: Dict[str, Any], strict: bool = True):
    """Decorator for automatic JSON schema validation."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            orchestrator = get_orchestrator()
            return orchestrator.validate_json_schema(result, schema, strict)
        return wrapper
    return decorator