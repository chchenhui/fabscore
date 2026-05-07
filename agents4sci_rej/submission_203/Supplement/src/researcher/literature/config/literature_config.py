"""
Configuration management for the literature query system.

This module provides comprehensive configuration management including ArXiv API settings,
LLM configuration, caching options, and domain-specific mappings.
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger


@dataclass
class ArXivConfig:
    """ArXiv API configuration settings."""
    base_url: str = "https://export.arxiv.org/api/query"
    rate_limit_seconds: float = 3.0
    max_results_per_query: int = 2000
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Validate ArXiv configuration."""
        if self.rate_limit_seconds < 1.0:
            logger.warning("rate_limit_seconds < 1.0 may violate ArXiv usage policy")
        
        if self.max_results_per_query > 2000:
            raise ValueError("ArXiv API limits max_results to 2000 per query")
        
        if self.timeout_seconds < 5:
            raise ValueError("timeout_seconds should be at least 5 seconds")


@dataclass  
class LLMConfig:
    """LLM service configuration for keyword extraction."""
    model_name: str = "gpt-4"
    max_keywords: int = 15
    confidence_threshold: float = 0.7
    temperature: float = 0.3
    max_tokens: int = 1000
    
    # Fallback options
    enable_fallback: bool = True
    fallback_model: Optional[str] = "gpt-3.5-turbo"
    
    def __post_init__(self):
        """Validate LLM configuration."""
        if self.confidence_threshold < 0.0 or self.confidence_threshold > 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        if self.max_keywords < 1 or self.max_keywords > 50:
            raise ValueError("max_keywords must be between 1 and 50")


@dataclass
class CacheConfig:
    """Caching system configuration."""
    enabled: bool = True
    ttl_hours: int = 24
    max_size_mb: int = 500
    redis_url: Optional[str] = None
    local_cache_dir: str = "cache/literature"
    
    # Cache strategies
    cache_keyword_extractions: bool = True
    cache_arxiv_responses: bool = True
    cache_paper_summaries: bool = True
    
    def __post_init__(self):
        """Validate cache configuration."""
        if self.ttl_hours < 1:
            raise ValueError("ttl_hours must be at least 1 hour")
        
        if self.max_size_mb < 10:
            raise ValueError("max_size_mb must be at least 10 MB")


@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    max_concurrent_requests: int = 3
    request_timeout: int = 30
    enable_request_batching: bool = True
    batch_size: int = 5
    
    # Result processing
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be at least 1")
        
        if self.max_worker_threads < 1:
            raise ValueError("max_worker_threads must be at least 1")


class LiteratureConfig:
    """
    Main configuration class for the literature query system.
    
    Provides centralized configuration management with support for file-based
    configuration, environment variable overrides, and domain-specific settings.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize configuration sections
        self.arxiv = ArXivConfig()
        self.llm = LLMConfig()  
        self.cache = CacheConfig()
        self.performance = PerformanceConfig()
        
        # Domain and category mappings
        self._init_category_mappings()
        self._init_search_field_mappings()
        self._init_prompt_templates()
        
        # Load configuration from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _init_category_mappings(self):
        """Initialize ArXiv category mappings for different research domains."""
        self.category_mappings = {
            # Computer Science domains
            "artificial intelligence": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.NE"],
            "machine learning": ["cs.LG", "stat.ML", "cs.AI", "cs.NE"],
            "computer vision": ["cs.CV", "cs.LG", "cs.AI", "eess.IV"],
            "natural language processing": ["cs.CL", "cs.AI", "cs.LG", "cs.IR"],
            "robotics": ["cs.RO", "cs.AI", "cs.SY"],
            "software engineering": ["cs.SE", "cs.PL", "cs.DC"],
            "computer science": ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.SE", "cs.DC"],
            
            # Agent-based modeling and simulation domains
            "agent-based modeling": ["cs.AI", "cs.MA", "cs.CY", "physics.soc-ph", "econ.GN"],
            "llm-based agent simulation": ["cs.AI", "cs.CL", "cs.LG", "cs.MA", "cs.HC"],
            "multi-agent systems": ["cs.MA", "cs.AI", "cs.GT", "cs.DC"],
            "social simulation": ["cs.CY", "physics.soc-ph", "cs.AI", "cs.MA", "econ.GN"],
            "computational social science": ["cs.CY", "physics.soc-ph", "cs.AI", "stat.AP"],
            "agent-based social simulation": ["cs.CY", "cs.AI", "cs.MA", "physics.soc-ph", "econ.GN"],
            "cultural dynamics": ["cs.CY", "physics.soc-ph", "cs.AI", "q-bio.PE"],
            "social dynamics modeling": ["cs.CY", "physics.soc-ph", "cs.AI", "nlin.AO"],
            
            # Mathematics domains
            "mathematics": ["math.NA", "math.OC", "math.ST", "stat.TH", "math.PR"],
            "optimization": ["math.OC", "cs.LG", "stat.ML"],
            "statistics": ["stat.TH", "stat.ML", "stat.ME", "math.ST"],
            "probability": ["math.PR", "stat.TH", "math.ST"],
            
            # Physics domains
            "physics": ["physics.comp-ph", "cond-mat", "quant-ph", "physics.data-an"],
            "quantum physics": ["quant-ph", "physics.atom-ph", "cond-mat.mes-hall"],
            "computational physics": ["physics.comp-ph", "cond-mat", "physics.data-an"],
            
            # Biology and Medicine
            "biology": ["q-bio.QM", "q-bio.GN", "q-bio.NC", "q-bio.MN"],
            "bioinformatics": ["q-bio.QM", "q-bio.GN", "cs.LG", "stat.AP"],
            "neuroscience": ["q-bio.NC", "cs.NE", "cs.AI"],
            
            # Economics and Finance
            "economics": ["econ.EM", "q-fin.CP", "q-fin.EC", "stat.AP"],
            "finance": ["q-fin.CP", "q-fin.RM", "q-fin.TR", "stat.AP"],
            
            # Engineering
            "signal processing": ["eess.SP", "cs.LG", "stat.ML"],
            "control systems": ["cs.SY", "math.OC", "eess.SY"]
        }
    
    def _init_search_field_mappings(self):
        """Initialize ArXiv search field prefix mappings."""
        self.search_fields = {
            "title": "ti",
            "abstract": "abs",
            "authors": "au", 
            "comments": "co",
            "categories": "cat",
            "journal": "jr",
            "report_number": "rn",
            "id": "id",
            "all": "all"
        }
    
    def _init_prompt_templates(self):
        """Initialize LLM prompt templates for different extraction modes."""
        self.prompt_templates = {
            "focused": """You are an expert research assistant. Extract 5-8 highly specific, technical keywords that precisely target the core research question: "{query}"

Focus on:
- Technical terminology that would appear in paper titles
- Specific methods, algorithms, or approaches
- Domain-specific concepts and terminology

Respond in JSON format:
{{"primary_keywords": ["keyword1", "keyword2"], "secondary_keywords": ["keyword3"], "domain_category": "field", "search_queries": ["query1"], "confidence_score": 0.8, "reasoning": "explanation"}}""",
            
            "comprehensive": """You are an expert research assistant. Extract 10-15 keywords for comprehensive literature coverage on: "{query}"

Include:
- Core technical terms (primary keywords)
- Related concepts and methods (secondary keywords)  
- Alternative phrasings and synonyms
- Interdisciplinary connections

Respond in JSON format with primary_keywords, secondary_keywords, domain_category, search_queries, confidence_score, and reasoning.""",
            
            "exploratory": """You are an expert research assistant. Extract 15-20 keywords for exploratory literature discovery on: "{query}"

Include diverse terms for discovery:
- Emerging and cutting-edge terminology
- Cross-disciplinary connections
- Alternative approaches and methodologies
- Related applications and use cases

Respond in JSON format with comprehensive keyword categorization."""
        }
    
    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file contains invalid JSON
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            if "literature" in config_data:
                self._update_from_dict(config_data["literature"])
                logger.info(f"Loaded literature configuration from {config_path}")
            else:
                logger.warning(f"No 'literature' section found in {config_path}")
                
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        # Update ArXiv config
        if "arxiv" in config_dict:
            self._update_dataclass(self.arxiv, config_dict["arxiv"])
        
        # Update LLM config  
        if "llm" in config_dict:
            self._update_dataclass(self.llm, config_dict["llm"])
        
        # Update cache config
        if "cache" in config_dict:
            self._update_dataclass(self.cache, config_dict["cache"])
        
        # Update performance config
        if "performance" in config_dict:
            self._update_dataclass(self.performance, config_dict["performance"])
        
        # Update category mappings if provided
        if "category_mappings" in config_dict:
            self.category_mappings.update(config_dict["category_mappings"])
        
        # Update prompt templates if provided
        if "prompt_templates" in config_dict:
            self.prompt_templates.update(config_dict["prompt_templates"])
    
    def _update_dataclass(self, target_obj: object, update_dict: Dict[str, Any]) -> None:
        """Update dataclass object with dictionary values."""
        for key, value in update_dict.items():
            if hasattr(target_obj, key):
                setattr(target_obj, key, value)
            else:
                logger.warning(f"Unknown configuration option: {key}")
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # ArXiv configuration overrides
        if os.getenv("ARXIV_RATE_LIMIT"):
            self.arxiv.rate_limit_seconds = float(os.getenv("ARXIV_RATE_LIMIT"))
        
        if os.getenv("ARXIV_MAX_RESULTS"):
            self.arxiv.max_results_per_query = int(os.getenv("ARXIV_MAX_RESULTS"))
        
        # LLM configuration overrides
        if os.getenv("LLM_MODEL"):
            self.llm.model_name = os.getenv("LLM_MODEL")
        
        if os.getenv("LLM_TEMPERATURE"):
            self.llm.temperature = float(os.getenv("LLM_TEMPERATURE"))
        
        # Cache configuration overrides
        if os.getenv("REDIS_URL"):
            self.cache.redis_url = os.getenv("REDIS_URL")
        
        if os.getenv("CACHE_DISABLED"):
            self.cache.enabled = os.getenv("CACHE_DISABLED").lower() != "true"
    
    def get_domain_categories(self, domain: str) -> List[str]:
        """
        Get ArXiv categories for a research domain.
        
        Args:
            domain: Research domain name
            
        Returns:
            List of relevant ArXiv categories
        """
        domain_lower = domain.lower().strip()
        
        # Try exact match first
        if domain_lower in self.category_mappings:
            return self.category_mappings[domain_lower]
        
        # Try partial matches
        for key, categories in self.category_mappings.items():
            if key in domain_lower or domain_lower in key:
                return categories
        
        # No match found
        logger.warning(f"No category mapping found for domain: {domain}")
        return []
    
    def get_search_prefix(self, field: str) -> str:
        """
        Get ArXiv search field prefix.
        
        Args:
            field: Search field name
            
        Returns:
            ArXiv API field prefix
        """
        return self.search_fields.get(field.lower(), "all")
    
    def get_prompt_template(self, mode: str, query: str = "", domain: str = "") -> str:
        """
        Get formatted prompt template for keyword extraction.
        
        Args:
            mode: Extraction mode (focused, comprehensive, exploratory)
            query: Research query to insert into template
            domain: Optional domain context
            
        Returns:
            Formatted prompt template
        """
        template = self.prompt_templates.get(mode, self.prompt_templates["comprehensive"])
        
        # Add domain context if provided
        domain_context = ""
        if domain:
            categories = self.get_domain_categories(domain)
            if categories:
                domain_context = f"\nResearch Domain: {domain}\nRelevant ArXiv Categories: {', '.join(categories)}\n"
        
        # Format template
        formatted_template = template.format(query=query) + domain_context
        return formatted_template
    
    def save_to_file(self, config_path: str) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path where to save configuration
        """
        config_dict = {
            "literature": {
                "arxiv": asdict(self.arxiv),
                "llm": asdict(self.llm),
                "cache": asdict(self.cache),
                "performance": asdict(self.performance),
                "category_mappings": self.category_mappings,
                "prompt_templates": self.prompt_templates
            }
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def validate(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Validate dataclass constraints through __post_init__
            ArXivConfig(**asdict(self.arxiv))
            LLMConfig(**asdict(self.llm))
            CacheConfig(**asdict(self.cache))
            PerformanceConfig(**asdict(self.performance))
        except ValueError as e:
            errors.append(f"Configuration validation error: {e}")
        
        # Additional validation
        if not self.category_mappings:
            errors.append("No category mappings defined")
        
        if not self.prompt_templates:
            errors.append("No prompt templates defined")
        
        return errors
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"""LiteratureConfig(
    arxiv_rate_limit={self.arxiv.rate_limit_seconds}s,
    llm_model={self.llm.model_name},
    cache_enabled={self.cache.enabled},
    domains={len(self.category_mappings)}
)"""


# Global configuration instance
_global_config: Optional[LiteratureConfig] = None


def get_config() -> LiteratureConfig:
    """
    Get the global configuration instance.
    
    Returns:
        Global LiteratureConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = LiteratureConfig()
    return _global_config


def load_config(config_path: str) -> LiteratureConfig:
    """
    Load configuration from file and set as global instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded LiteratureConfig instance
    """
    global _global_config
    _global_config = LiteratureConfig(config_path)
    return _global_config


def reset_config() -> None:
    """Reset global configuration to default."""
    global _global_config
    _global_config = None


# Example configuration file template
DEFAULT_CONFIG_TEMPLATE = {
    "literature": {
        "arxiv": {
            "rate_limit_seconds": 3.0,
            "max_results_per_query": 100,
            "timeout_seconds": 30
        },
        "llm": {
            "model_name": "gpt-4",
            "max_keywords": 15,
            "temperature": 0.3
        },
        "cache": {
            "enabled": True,
            "ttl_hours": 24,
            "max_size_mb": 500
        },
        "performance": {
            "max_concurrent_requests": 3,
            "enable_parallel_processing": True
        }
    }
}


def create_default_config(config_path: str) -> None:
    """
    Create a default configuration file.
    
    Args:
        config_path: Path where to create the configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CONFIG_TEMPLATE, f, indent=2)
    
    logger.info(f"Default configuration created at {config_path}")


if __name__ == "__main__":
    # Example usage and testing
    config = LiteratureConfig()
    
    print("Default Configuration:")
    print(config)
    
    print("\nDomain categories for 'machine learning':")
    print(config.get_domain_categories("machine learning"))
    
    print("\nValidation errors:")
    errors = config.validate()
    if errors:
        for error in errors:
            print(f"- {error}")
    else:
        print("Configuration is valid")