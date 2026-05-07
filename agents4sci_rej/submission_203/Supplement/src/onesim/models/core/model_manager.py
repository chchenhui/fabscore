"""
This module provides a manager for model configurations and instances.
"""

import json
import os
from typing import Any, Dict, List, Optional, Type, Union

from loguru import logger

from .model_base import ModelAdapterBase
from .load_balancer import LoadBalancer


class ModelManager:
    """
    Manager for model configurations and instances.
    
    This class acts as a registry and factory for model adapters, handling
    configuration loading and model instantiation.
    """

    _instance = None
    _model_types = {}

    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern for ModelManager."""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the model manager if not already initialized."""
        if not getattr(self, '_initialized', False):
            self.model_configs = {}
            self.load_balancer_configs = {}  # 使用字典存储多个负载均衡器配置
            self._load_balancer_instances = {}  # 缓存已创建的负载均衡器实例
            self._initialized = True

    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """
        Get or create the singleton instance of ModelManager.
        
        Returns:
            ModelManager: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, config_path: Optional[Union[str, Dict, List]] = None):
        """
        Initialize the model manager with configurations.
        
        Args:
            config_path: Path to config file, a config dict, or list of configs.
        """
        if config_path is not None:
            self.load_model_configs(config_path)

    def load_model_configs(
        self, 
        configs: Union[str, Dict, List], 
        clear_existing: bool = False
    ):
        """
        Load model configurations from various sources.
        
        Args:
            configs: Path to config file, config dict, or list of configs.
            clear_existing: Whether to clear existing configs before loading.
        """
        if clear_existing:
            self.model_configs.clear()

        # Parse configurations based on input type
        config_list = []

        if isinstance(configs, str):
            # Load from file path
            if not os.path.exists(configs):
                raise FileNotFoundError(f"Config file not found: {configs}")

            with open(configs, 'r', encoding='utf-8') as f:
                loaded_configs = json.load(f)

            # Process the loaded config based on its structure
            if isinstance(loaded_configs, dict):
                # Check if it's a categorized config (with chat, embedding keys)
                if "chat" in loaded_configs or "embedding" in loaded_configs:
                    # Extract configs from different categories
                    chat_configs = loaded_configs.get("chat", [])
                    embedding_configs = loaded_configs.get("embedding", [])

                    # For chat configs, add category field
                    for config in chat_configs:
                        config["category"] = "chat"

                    # For embedding configs, add category field
                    for config in embedding_configs:
                        config["category"] = "embedding"

                    # Combine all configs
                    config_list = []
                    config_list.extend(chat_configs)
                    config_list.extend(embedding_configs)

                    logger.info(f"Loaded categorized config with {len(chat_configs)} chat and "
                               f"{len(embedding_configs)} embedding models")
                else:
                    # It's a single model config
                    config_list = [loaded_configs]
            elif isinstance(loaded_configs, list):
                # It's already a list of configs
                config_list = loaded_configs
            else:
                raise ValueError(f"Invalid config format in {configs}")

        elif isinstance(configs, dict):
            # If it's a dictionary, check if it's categorized
            if "chat" in configs or "embedding" in configs:
                chat_configs = configs.get("chat", [])
                embedding_configs = configs.get("embedding", [])

                # For chat configs, add category field
                for config in chat_configs:
                    config["category"] = "chat"

                # For embedding configs, add category field
                for config in embedding_configs:
                    config["category"] = "embedding"

                config_list = []
                config_list.extend(chat_configs)
                config_list.extend(embedding_configs)
            else:
                config_list = [configs]

        elif isinstance(configs, list):
            config_list = configs

        else:
            raise TypeError(
                f"Expected str, dict, or list, got {type(configs)}"
            )

        # Validate and store configurations
        for config in config_list:
            if "config_name" not in config:
                raise ValueError("Each config must have a 'config_name'")

            # Backward compatibility - derive category from model_type if not present
            if "category" not in config:
                if "model_type" in config:
                    model_type = config["model_type"].lower()
                    if "embedding" in model_type:
                        config["category"] = "embedding"
                    else:
                        config["category"] = "chat"
                else:
                    raise ValueError(f"Config '{config['config_name']}' missing 'category' field")

            # Ensure provider is present
            if "provider" not in config:
                # For backward compatibility, derive provider from model_type
                if "model_type" in config:
                    model_type = config["model_type"].lower()
                    if "openai" in model_type:
                        config["provider"] = "openai"
                    else:
                        # Use model_type as provider if we can't determine
                        config["provider"] = model_type
                else:
                    raise ValueError(f"Config '{config['config_name']}' missing 'provider' field")

            # Check for duplicates and warn
            if config["config_name"] in self.model_configs:
                logger.warning(
                    f"Overwriting existing config '{config['config_name']}'"
                )

            self.model_configs[config["config_name"]] = config

        logger.info(
            f"Loaded {len(config_list)} model configs: {', '.join(c['config_name'] for c in config_list)}"
        )

    def get_model(self, config_name: Optional[str] = None, model_name: Optional[str] = None, model_type: Optional[str] = None) -> ModelAdapterBase:
        """
        Get a model instance by configuration name or automatically select based on criteria.
        
        Models can be accessed in several ways:
        1. By specific config_name - returns the exact model configuration
        2. By model_name - returns a load balancer for all models with the given name
        3. By model_type - returns a load balancer for all models of that type or provider
        4. With no parameters - returns the default load balancer (all chat models)
        
        Args:
            config_name: Name of the model configuration.
            model_type: Type of models to use ('chat', 'embedding', or specific provider).
            model_name: Name of the model (e.g., 'gpt-4') to load balance across providers.
            
        Returns:
            ModelAdapterBase: An initialized model adapter.
            
        Raises:
            ValueError: If the config doesn't exist or no models match the criteria.
        """
        if not self.model_configs:
            raise ValueError("No model configurations loaded")

        # Priority 1: Specific config_name provided
        if config_name is not None:
            # Check if it's a load balancer config
            if config_name in self.load_balancer_configs:
                return self._get_or_create_load_balancer(config_name)

            # Otherwise, it's a regular model config
            if config_name in self.model_configs:
                config = self.model_configs[config_name]
                provider = config["provider"]
                model_class = self._get_model_class(provider, config["category"])
                kwargs = {
                    k: v for k, v in config.items() if k not in ["provider", "category"]
                }
                return model_class(**kwargs)

        # Priority 2: Model name provided - load balance all providers with this model
        if model_name is not None:
            # Create a load balancer for all instances of this model name across providers
            lb_config_name = f"{model_name}_load_balancer"

            # Check if we already have this load balancer
            if lb_config_name in self.load_balancer_configs:
                return self._get_or_create_load_balancer(lb_config_name)

            # Otherwise create a new load balancer for this model name
            return self._get_or_create_load_balancer(config_name=None, model_type=model_type, model_name=model_name)

        # Priority 3: Model type provided - load balance specific model type or provider
        if model_type is not None:
            return self._get_or_create_load_balancer(config_name=None, model_type=model_type)

        # Priority 4: config_name was provided but not found. Try to use it as a model_type.
        if config_name is not None:
            try:
                # This handles cases like get_model("chat") where "chat" is not a specific config name
                return self._get_or_create_load_balancer(
                    config_name=None, model_type=config_name
                )
            except ValueError as e:
                raise ValueError(
                    f"Model config '{config_name}' not found as a specific configuration, "
                    f"and could not be used as a model_type."
                )

        # Priority 5: No parameters - return default chat load balancer
        return self._get_or_create_load_balancer(config_name="chat_load_balancer", model_type="chat")

    def _get_model_class(self, provider: str, category: str) -> Type[ModelAdapterBase]:
        """
        Get the model class for a given provider and category.
        
        Args:
            provider: Provider name (e.g., 'openai', 'vllm').
            category: Model category ('chat' or 'embedding').
            
        Returns:
            Model class that inherits from ModelAdapterBase.
            
        Raises:
            ValueError: If the provider is unknown.
        """
        # Map model providers to import paths and class names
        # FEAT: add ark, deepseek, aliyun, tencent
        model_registry = {
            "openai": {
                "chat": ("onesim.models.providers.openai", "OpenAIChatAdapter"),
                "embedding": ("onesim.models.providers.openai", "OpenAIEmbeddingAdapter")
            },
            "vllm": {
                "chat": ("onesim.models.providers.vllm", "VLLMChatAdapter"),
                "embedding": ("onesim.models.providers.vllm", "VLLMEmbeddingAdapter")
            },
            "ark": {
                "chat":      ("onesim.models.providers.ark",         "ArkChatAdapter"),
                "embedding": ("onesim.models.providers.ark",         "ArkEmbeddingAdapter"),
            },
            "deepseek": {
                "chat":      ("onesim.models.providers.deepseek",         "DeepSeekChatAdapter"),
                "embedding": ("onesim.models.providers.deepseek",         "DeepSeekEmbeddingAdapter"),
            },
            "aliyun": {
                "chat":      ("onesim.models.providers.aliyun",         "AliyunChatAdapter"),
                "embedding": ("onesim.models.providers.aliyun",         "AliyunEmbeddingAdapter"),
            },
            "tencent": {
                "chat":      ("onesim.models.providers.tencent",         "TencentChatAdapter"),
                "embedding": ("onesim.models.providers.tencent",         "TencentEmbeddingAdapter"),
            },
            # Add additional providers as needed
        }

        # Backward compatibility - handle model_type format
        if provider == "openai_chat":
            provider = "openai" 
            category = "chat"
        elif provider == "openai_embedding":
            provider = "openai"
            category = "embedding"
        elif provider == "ark_chat":
            provider = "ark"
            category = "chat"
        elif provider == "ark_embedding":
            provider = "ark"
            category = "embedding"
        elif provider == "deepseek_chat":
            provider = "deepseek"
            category = "chat"
        elif provider == "deepseek_embedding":
            provider = "deepseek"
            category = "embedding"
        elif provider == "aliyun_chat":
            provider = "aliyun"
            category = "chat"
        elif provider == "aliyun_embedding":
            provider = "aliyun"
            category = "embedding"
        elif provider == "tencent_chat":
            provider = "tencent"
            category = "chat"
        elif provider == "tencent_embedding":
            provider = "tencent"
            category = "embedding"

        # Check if the provider exists in our registry
        if provider not in model_registry:
            raise ValueError(f"Unknown provider: {provider}")

        # Check if the category exists for this provider
        if category not in model_registry[provider]:
            raise ValueError(f"Provider '{provider}' does not support '{category}' category")

        # Get module path and class name
        module_path, class_name = model_registry[provider][category]

        # Import the module dynamically
        try:
            import importlib
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return model_class
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Failed to import model for provider '{provider}' and category '{category}': {e}")

    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a model configuration by name.
        
        Args:
            config_name: Name of the model configuration.
            
        Returns:
            Dict or None: The model configuration if found, otherwise None.
        """
        return self.model_configs.get(config_name)

    def clear_configs(self):
        """Clear all loaded model configurations."""
        self.model_configs.clear()
        self.load_balancer_configs.clear()
        self._load_balancer_instances.clear()  # 清除负载均衡器实例缓存

    def get_configs_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all model configurations of a specific type.
        
        Args:
            model_type: Type of models to filter by ('chat', 'embedding', or specific provider).
            
        Returns:
            List of configuration dictionaries.
        """
        configs = []

        for config in self.model_configs.values():
            # If model_type is a category (chat/embedding)
            if model_type.lower() in ["chat", "embedding"]:
                if config.get("category", "").lower() == model_type.lower():
                    configs.append(config)
            # If model_type is a provider
            elif model_type.lower() == config.get("provider", "").lower():
                configs.append(config)

        return configs

    def get_model_names_by_type(self, model_type: str) -> List[str]:
        """
        Get all model configuration names of a specific type.
        
        Args:
            model_type: Type of models to filter by ('chat', 'embedding', or specific type).
            
        Returns:
            List of configuration names.
        """
        configs = self.get_configs_by_type(model_type)
        return [config["config_name"] for config in configs]

    def configure_load_balancer(
        self,
        model_configs=None,
        strategy: str = "round_robin",
        config_name: str = "chat_load_balancer",
        model_type: str = "chat",
        model_name: str = None
    ):
        """
        Configure a load balancer.
        
        Args:
            model_configs: List of model configuration names to load balance between.
                          If None, will use models based on model_type/model_name parameters.
            strategy: Load balancing strategy ('round_robin' or 'random').
            config_name: Name to assign to the load balancer configuration.
            model_type: Type of models to balance ('chat', 'embedding', or specific provider).
                       Used to filter eligible models when model_configs is None.
            model_name: Name of specific model to load balance across providers.
                       Used to filter eligible models when model_configs is None.
                       
        The load balancer can operate at three levels:
        1. At category level (model_type='chat'/'embedding'): uses all models of that category
        2. At provider level: uses all models from a specific provider
        3. At model_name level: uses all models with a specific model_name
        """
        # Determine the model configurations based on the specified filtering criteria
        if model_configs is None:
            # Case 1: Model name specified - filter by model_name
            if model_name is not None:
                model_configs = []
                for model_config_name, config in self.model_configs.items():
                    # Extract the actual model name from the configuration
                    # Some configurations use 'model' key, others 'model_name'
                    config_model_name = config.get("model_name", config.get("model"))
                    # Filter by model_name and optionally by category/provider
                    if config_model_name == model_name:
                        if model_type in ["chat", "embedding"]:
                            # Also check category
                            if config.get("category") == model_type:
                                model_configs.append(model_config_name)
                        elif model_type is not None:
                            # Also check provider
                            if config.get("provider") == model_type:
                                model_configs.append(model_config_name)
                        else:
                            # If no additional filter, include all matches
                            model_configs.append(model_config_name)

                if not model_configs:
                    logger.warning(f"No models found with model_name '{model_name}'")

            # Case 2: Provider specified - filter by provider
            elif model_type not in ["chat", "embedding"] and model_type is not None:
                model_configs = []
                for model_config_name, config in self.model_configs.items():
                    if config.get("provider") == model_type:
                        model_configs.append(model_config_name)

                if not model_configs:
                    logger.warning(f"No models found with provider '{model_type}'")

            # Case 3: Category specified (chat/embedding) - get all models of that category
            else:
                model_configs = self.get_model_names_by_type(model_type if model_type else "chat")

                if not model_configs:
                    logger.warning(f"No models found of category '{model_type if model_type else 'chat'}'")

        # If no models were found for the criteria, don't create an empty config
        if not model_configs:
            error_msg = f"Load balancer '{config_name}' could not be configured because no matching models were found."
            logger.warning(error_msg)
            # Prevent creation of an empty/invalid load balancer config
            raise ValueError(error_msg)

        # Store the configuration
        load_balancer_config = {
            "provider": "load_balancer",
            "category": model_type if model_type in ["chat", "embedding"] else "chat",
            "models": model_configs,
            "strategy": strategy,
            "target_model_type": model_type,
            "target_model_name": model_name
        }
        self.load_balancer_configs[config_name] = load_balancer_config
        # Clear the instance cache for this config if it exists
        if config_name in self._load_balancer_instances:
            del self._load_balancer_instances[config_name]

        logger.info(
            f"Configured load balancer '{config_name}' with strategy '{strategy}' "
            f"and {len(model_configs) if model_configs else 0} models"
        )

        # Log the models included in the load balancer for debugging
        if model_configs:
            logger.debug(f"Load balancer '{config_name}' includes models: {', '.join(model_configs)}")
        else:
            logger.warning(f"Load balancer '{config_name}' has no models configured")

    def _get_or_create_load_balancer(self, config_name: str=None, model_type: str=None, model_name: str=None) -> ModelAdapterBase:
        """
        Get an existing load balancer or create a new one.
        
        Args:
            config_name: The name of the load balancer configuration. If None, uses a default name.
            model_type: The type of models to balance ('chat' or 'embedding' or specific model_type).
            model_name: Specific model name to balance across providers.
            
        Returns:
            ModelAdapterBase: The load balancer instance.
            
        Raises:
            ValueError: If the config doesn't exist or can't be created.
        """
        # Handle default values and backward compatibility
        if model_type is None and model_name is None:
            model_type = "chat"  # Default to chat

        # Determine the configuration name to use
        if config_name is None:
            if model_name is not None:
                # Using model name level load balancer
                config_name = f"{model_name}_load_balancer"
                if model_type is not None:
                    # With model type specified, use a more specific name
                    config_name = f"{model_name}_{model_type}_load_balancer"
            else:
                # Using general category level load balancer
                config_name = f"{model_type}_load_balancer"

        # Check if we already have this load balancer cached
        if config_name in self._load_balancer_instances:
            return self._load_balancer_instances[config_name]

        # If the configuration doesn't exist yet, create it
        if config_name not in self.load_balancer_configs:
            # Auto-configure the load balancer based on available models
            self.configure_load_balancer(
                model_configs=None,  # Auto-detect based on type/name
                config_name=config_name,
                model_type=model_type,
                model_name=model_name
            )

            if config_name not in self.load_balancer_configs:
                raise ValueError(f"Failed to create load balancer configuration '{config_name}'")

        # Get the load balancer config
        config = self.load_balancer_configs[config_name]
        models = config.get("models", [])

        # Validate that there are models to load balance
        if not models:
            # More descriptive error message
            if model_name is not None:
                error_msg = f"No models found with name '{model_name}'"
                if model_type is not None:
                    error_msg += f" and type '{model_type}'"
            else:
                error_msg = f"No models found for load balancer '{config_name}'"

            raise ValueError(error_msg)

        # Create the load balancer
        from .load_balancer import LoadBalancer

        strategy = config.get("strategy", "round_robin")
        target_model_type = config.get("target_model_type", "chat")

        # Create the instance
        load_balancer = LoadBalancer(
            config_name=config_name,
            models=models,
            strategy=strategy,
            model_type=target_model_type
        )

        # Initialize the models
        load_balancer.initialize_models(self)

        # Cache the instance
        self._load_balancer_instances[config_name] = load_balancer

        return load_balancer

    def get_model_config_name(self, model_type: str) -> str:
        """
        Get the model config name for a given model type.
        """
        return model_type+"_load_balancer"
