import os
import yaml
import json
import uuid
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from loguru import logger


# Component configuration classes
@dataclass_json
@dataclass
class ModelConfig:
    """Model component configuration"""
    enabled: bool = True
    config_path: Optional[str] = None
    chat_configs: List[Dict] = field(default_factory=list)
    embedding_configs: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if self.config_path and os.path.exists(self.config_path):
            self.load_from_file(self.config_path)

    def load_from_file(self, config_path: str) -> bool:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)

            if "chat" in model_config:
                self.chat_configs = model_config.get("chat", [])
                # Add category field if missing
                for config in self.chat_configs:
                    config["category"] = "chat"

            self.embedding_configs = model_config.get("embedding", [])
            # Add category field if missing
            for config in self.embedding_configs:
                config["category"] = "embedding"

            self.enabled = True
            return True
        except Exception as e:
            logger.error(f"Error loading model config from {config_path}: {e}")
            return False

    def load_from_dict(self, config_dict: Dict[str, Any]) -> bool:
        """
        Load model configuration from a dictionary
        
        Args:
            config_dict: Dictionary containing model configuration
            
        Returns:
            bool: Success status
        """
        try:
            if "chat" in config_dict:
                self.chat_configs = config_dict.get("chat", [])
                # Add category field if missing
                for config in self.chat_configs:
                    config["category"] = "chat"

            self.embedding_configs = config_dict.get("embedding", [])
            # Add category field if missing
            for config in self.embedding_configs:
                config["category"] = "embedding"

            self.enabled = True
            return True
        except Exception as e:
            logger.error(f"Error loading model config from dictionary: {e}")
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "enabled": self.enabled,
            "config_path": self.config_path,
            "chat": self.chat_configs,
            "embedding": self.embedding_configs
        }

@dataclass_json
@dataclass
class MonitorConfig:
    """Monitor component configuration"""
    enabled: bool = False
    update_interval: Optional[int] = None  # Default update interval in seconds
    metrics_path: Optional[str] = None  # Path to metrics directory
    visualization_defaults: Dict[str, Any] = field(default_factory=lambda: {
        "line": {"max_points": 100},
        "bar": {"sort": True},
        "pie": {"sort": True, "merge_threshold": 0.05}
    })
    
    def set_env_path(self, env_path: str):
        """Set environment path and calculate metrics_path if not explicitly set"""
        if self.metrics_path is None and env_path:
            self.metrics_path = os.path.join(env_path, "code", "metrics")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "enabled": self.enabled,
            "update_interval": self.update_interval,
            "metrics_path": self.metrics_path,
            "visualization_defaults": self.visualization_defaults
        }

@dataclass_json
@dataclass
class DistributionConfig:
    """Distribution component configuration"""
    enabled: bool = False
    mode: str = "single"  # "single", "master", or "worker"
    node_id: Optional[str] = None
    master_address: str = "localhost"
    master_port: int = 10051
    worker_address: Optional[str] = None
    worker_port: int = 0  # 0 means auto-assign
    expected_workers: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "node_id": self.node_id,
            "master_address": self.master_address,
            "master_port": self.master_port,
            "worker_address": self.worker_address,
            "worker_port": self.worker_port,
            "expected_workers": self.expected_workers
        }

@dataclass_json
@dataclass
class DatabaseConfig:
    """Database component configuration"""
    enabled: bool = False
    host: str = "localhost"
    port: int = 5432
    dbname: str = "onesim"
    user: str = "postgres"
    password: str = "postgres"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "dbname": self.dbname,
            "user": self.user,
            "password": self.password
        }

@dataclass_json
@dataclass
class ObservationConfig:
    """Observation system configuration"""
    enabled: bool = False
    loki_url: Optional[str] = None
    prometheus_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "enabled": self.enabled,
            "loki_url": self.loki_url,
            "prometheus_url": self.prometheus_url
        }

@dataclass_json
@dataclass
class AgentMemoryConfig:
    """Configuration for agent memory system"""
    strategy: str = "ListStrategy"
    storages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metric_weights: Dict[str, float] = field(default_factory=dict)
    transfer_conditions: Dict[str, Any] = field(default_factory=dict)
    operations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "strategy": self.strategy,
            "storages": self.storages,
            "metric_weights": self.metric_weights,
            "transfer_conditions": self.transfer_conditions,
            "operations": self.operations,
            "metrics": self.metrics
        }

@dataclass_json
@dataclass
class AgentConfig:
    """Configuration for agent component"""
    profile: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    planning: Optional[str] = None
    memory: AgentMemoryConfig = field(default_factory=AgentMemoryConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "profile": self.profile,
            "planning": self.planning,
            "memory": self.memory.to_dict()
        }

@dataclass_json
@dataclass
class InterventionConfig:
    """Configuration for runtime interventions"""
    enabled: bool = False
    intervention_specs_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "enabled": self.enabled,
            "intervention_specs_path": self.intervention_specs_path
        }

@dataclass_json
@dataclass
class SimulatorConfig:
    """Configuration for simulator component"""
    environment: Dict[str, Any] = field(default_factory=dict)
    event_manager: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "environment": self.environment,
            "event_manager": self.event_manager
        }

class ComponentRegistry:
    """Registry to keep track of initialized components"""
    
    def __init__(self):
        self._initialized_components: Set[str] = set()
        self._component_instances = {}
    
    def register(self, component_name: str, instance=None):
        """Register a component as initialized"""
        self._initialized_components.add(component_name)
        if instance is not None:
            self._component_instances[component_name] = instance
    
    def is_initialized(self, component_name: str) -> bool:
        """Check if a component has been initialized"""
        return component_name in self._initialized_components
    
    def get_instance(self, component_name: str):
        """Get instance of initialized component"""
        return self._component_instances.get(component_name)
    
    def clear(self):
        """Clear registry (mainly for testing)"""
        self._initialized_components.clear()
        self._component_instances.clear()

# Global singleton instances
_config_instance = None
_component_registry = ComponentRegistry()

@dataclass_json
@dataclass
class OneSimConfig:
    """Main configuration class for OneSim framework"""

    # SimulationConfig attributes merged in
    env_name: Optional[str] = None
    env_path: Optional[str] = None
    base_dir: Optional[str] = None

    # Component configurations
    model_config: ModelConfig = field(default_factory=ModelConfig)
    distribution_config: DistributionConfig = field(default_factory=DistributionConfig)
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    observation_config: ObservationConfig = field(default_factory=ObservationConfig)
    monitor_config: MonitorConfig = field(default_factory=MonitorConfig)
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    simulator_config: SimulatorConfig = field(default_factory=SimulatorConfig)
    intervention_config: InterventionConfig = field(default_factory=InterventionConfig)

    # Component registry
    registry: ComponentRegistry = field(default_factory=lambda: _component_registry)

    def __post_init__(self):
        if not self.base_dir:
            self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Set metrics_path based on environment path if available
        if self.env_path:
            self.monitor_config.set_env_path(self.env_path)

    def load_from_file(self, config_path: str) -> bool:
        """Load configuration from a YAML or JSON file"""
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found. Using defaults.")
            return False

        try:
            ext = os.path.splitext(config_path)[1].lower()

            if ext == '.yaml' or ext == '.yml':
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
            elif ext == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
            else:
                logger.warning(f"Unsupported config file format: {ext}. Using defaults.")
                return False

            # Use common method to apply configuration
            return self._apply_loaded_config(loaded_config)

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return False

    def load_from_dict(self, config_dict: Dict[str, Any]) -> bool:
        """
        Load configuration from a dictionary
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            bool: Success status
        """
        try:
            # Apply configuration using common method
            return self._apply_loaded_config(config_dict)
        except Exception as e:
            logger.error(f"Error loading config from dictionary: {e}")
            return False

    def _apply_loaded_config(self, loaded_config: Dict[str, Any]) -> bool:
        """
        Apply loaded configuration from either file or dictionary
        
        Args:
            loaded_config: Configuration dictionary
            
        Returns:
            bool: Success status
        """
        try:
            # Apply configuration to component configs
            if "model" in loaded_config:
                self._apply_config(self.model_config, loaded_config["model"])

            # Handle simulator configuration
            if "simulator" in loaded_config:
                self._apply_config(self.simulator_config, loaded_config["simulator"])

            if "agent" in loaded_config:
                agent_config = loaded_config["agent"]
                # Handle agent profile configuration
                if "profile" in agent_config:
                    self.agent_config.profile = agent_config["profile"]

                # Handle agent planning configuration
                if "planning" in agent_config:
                    self.agent_config.planning = agent_config["planning"]

                # Handle agent memory configuration
                if "memory" in agent_config:
                    memory_config = agent_config["memory"]
                    if not memory_config:
                        self.agent_config.memory=None
                    else:
                        # Set the strategy
                        if "strategy" in memory_config:
                            self.agent_config.memory.strategy = memory_config["strategy"]

                        # Set other memory subsystems
                        for key in ["storages", "metric_weights", "transfer_conditions", "operations", "metrics"]:
                            if key in memory_config:
                                setattr(self.agent_config.memory, key, memory_config[key])
                else:
                    self.agent_config.memory = None

            # Handle monitor configuration
            if "monitor" in loaded_config:
                monitor_config = loaded_config["monitor"]
                self._apply_config(self.monitor_config, monitor_config)
                self.monitor_config.enabled = monitor_config.get("enabled", False)

            # Handle intervention configuration
            if "intervention_config" in loaded_config:
                intervention_config = loaded_config["intervention_config"]
                self._apply_config(self.intervention_config, intervention_config)

            if "distribution" in loaded_config:
                dist_config = loaded_config["distribution"]
                self._apply_config(self.distribution_config, dist_config)

                # Handle special case for distribution properties
                self.distribution_config.enabled = dist_config.get("enabled", False)
                if self.distribution_config.enabled:
                    self.distribution_config.mode = dist_config.get("mode", "single")

            if "database" in loaded_config:
                db_config = loaded_config["database"]
                self._apply_config(self.database_config, db_config)
                self.database_config.enabled = db_config.get("enabled", False)

                for key in ["host", "port", "dbname", "user", "password"]:
                    if key in db_config:
                        setattr(self.database_config, key, db_config[key])

            if "observation" in loaded_config:
                obs_config = loaded_config["observation"]
                self._apply_config(self.observation_config, obs_config)
                self.observation_config.enabled = obs_config.get("enabled", False)

            logger.info(f"Loaded configuration from dictionary")
            return True

        except Exception as e:
            logger.error(f"Error applying configuration: {e}")
            return False

    def _apply_config(self, target_obj, config_dict):
        """Apply configuration dictionary to a dataclass instance"""
        for key, value in config_dict.items():
            if hasattr(target_obj, key):
                setattr(target_obj, key, value)

    def update(self, **kwargs):
        """Update configuration with provided values"""
        # Update model config
        if "model" in kwargs:
            model_conf = kwargs.pop("model")
            self._apply_config(self.model_config, model_conf)

        # Update simulator config
        if "simulator" in kwargs:
            sim_conf = kwargs.pop("simulator")
            self._apply_config(self.simulator_config, sim_conf)

        # Update agent config
        if "agent" in kwargs:
            agent_conf = kwargs.pop("agent")
            self._apply_config(self.agent_config, agent_conf)

        # Update monitor config
        if "monitor" in kwargs:
            monitor_conf = kwargs.pop("monitor")
            self._apply_config(self.monitor_config, monitor_conf)

        # Update distribution config
        if "distribution" in kwargs:
            dist_conf = kwargs.pop("distribution")
            self._apply_config(self.distribution_config, dist_conf)

        # Update database config
        if "database" in kwargs:
            db_conf = kwargs.pop("database")
            self._apply_config(self.database_config, db_conf)

        # Update observation config
        if "observation" in kwargs:
            obs_conf = kwargs.pop("observation")
            self._apply_config(self.observation_config, obs_conf)

        # Handle top-level parameters
        for key, value in kwargs.items():
            # Handle environment-related attributes
            if key in ["env_name", "env_path", "base_dir"]:
                setattr(self, key, value)
                # If env_path was updated, update monitor config
                if key == "env_path" and value:
                    self.monitor_config.set_env_path(value)
            else:
                # Check if the parameter belongs to a specific config object
                for config_obj in [self.model_config, self.simulator_config, self.agent_config,
                                self.monitor_config, self.distribution_config, 
                                self.database_config, self.observation_config]:
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
                        break

    def load_simulation_config(self, config_path: str, model_config_path: str = None, env_name: str = None) -> 'OneSimConfig':
        """Load simulation configuration from files"""
        # Load main config
        with open(config_path, "r", encoding='utf-8') as f:
            config = json.load(f)

        # Create ModelConfig
        if model_config_path:
            self.model_config.load_from_file(model_config_path)

        # Extract environment information
        simulator_config_dict = config.get("simulator", {})
        env_config = simulator_config_dict.get("environment", {})
        # Override environment name if provided
        if env_name:
            env_config["name"] = env_name
        else:
            env_name = env_config.get("name")

        if not env_name:
            raise ValueError("Environment name must be provided either in config or as parameter")

        # Set environment name and path
        self.env_name = env_name
        self.env_path = os.path.join(self.base_dir, "envs", env_name)
        # Apply configuration from dictionary
        self.load_from_dict(config)

        # Set metrics_path for monitor config based on environment path
        self.monitor_config.set_env_path(self.env_path)

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert instance to a dictionary for JSON serialization"""
        return {
            "env_name": self.env_name,
            "env_path": self.env_path,
            "base_dir": self.base_dir,
            "simulator_config": self.simulator_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "distribution_config": self.distribution_config.to_dict(),
            "database_config": self.database_config.to_dict(),
            "observation_config": self.observation_config.to_dict(),
            "monitor_config": self.monitor_config.to_dict(),
            "agent_config": self.agent_config.to_dict()
        }

# Global function to get or create config
def get_config() -> OneSimConfig:
    """Get the global config instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = OneSimConfig()
    return _config_instance

def get_component_registry() -> ComponentRegistry:
    """Get the global component registry"""
    global _component_registry
    return _component_registry

def parse_json(config_path: str) -> dict:
    """Parse a JSON file and return the contents as a dictionary"""
    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    return config 
