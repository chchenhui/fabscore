from abc import ABC, abstractmethod
from typing import Dict, Any
from ..storage import *
from ..operation import *
from ..metric import *
from onesim.models import ModelManager
from onesim.profile import AgentProfile
from onesim.relationship import RelationshipManager

class AgentContext:
    def __init__(self, agent_id,profile: AgentProfile, relationship_manager: RelationshipManager):
        self.agent_id = agent_id
        self.profile = profile
        self.relationship_manager = relationship_manager

    def __str__(self) -> str:
        profile_str = self.profile.get_profile_str() if self.profile else "No profile information provided."
        relationship_str = self.relationship_manager.get_all_relationships_str() if self.relationship_manager else "No relationship information provided."
        return f"Pofile: {profile_str}\n\nRelationshipManager: {relationship_str}\n\n"

class MemoryStrategy(ABC):
    def __init__(self, config: Dict[str, Any],model_config_name: str) -> None:
        """
        Initialize memory strategy with dynamic component loading
        
        :param config: Configuration for storages, operations, and metrics
        """
        self.config = config 
        self._storage_map: Dict[str, MemoryStorage] = {}
        self._operations: Dict[str, MemoryOperation] = {}
        self._metrics: Dict[str, MemoryMetric] = {}
        if model_config_name is not None:
            model_manager = ModelManager.get_instance()
            self.model = model_manager.get_model(
                config_name=model_config_name,
            )
        
        self._initialize_components(config)

    def _initialize_components(self, config: Dict[str, Any]):
        """
        Dynamically initialize storage, operations, and metrics
        
        :param config: Configuration dictionary
        """
        # Initialize storages
        storage_configs = config.get('storages', {})
        for name, storage_config in storage_configs.items():
            self._storage_map[name] = self._create_storage(storage_config)

        # Initialize operations
        operation_configs = config.get('operations', {})
        for name, op_config in operation_configs.items():
            self._operations[name] = self._create_operation(op_config)

        # Initialize metrics
        metric_configs = config.get('metrics', {})
        for name, metric_config in metric_configs.items():
            self._metrics[name] = self._create_metric(metric_config)

    def _create_storage(self, config: Dict[str, Any]) -> MemoryStorage:
        """
        Create a storage instance based on configuration
        
        :param config: Storage configuration
        :return: Storage instance
        """
        storage_type = config.get('class')
        
        # Example storage type mapping
        storage_map = {
            'ListMemoryStorage': ListMemoryStorage,
            'VectorMemoryStorage': VectorMemoryStorage
        }
        
        storage_class = storage_map.get(storage_type)
        if not storage_class:
            raise ValueError(f"Unsupported storage type: {storage_type}")
        return storage_class(config)

    def _create_operation(self, config: Dict[str, Any]) -> MemoryOperation:
        """
        Create an operation instance based on configuration
        
        :param config: Operation configuration
        :return: Operation instance
        """
        operation_type = config.get('class')
        
        # Example operation type mapping
        operation_map = {
            'AddMemoryOperation': AddMemoryOperation,
            'RetrieveMemoryOperation': RetrieveMemoryOperation,
            'RemoveMemoryOperation': RemoveMemoryOperation,
        }
        
        operation_class = operation_map.get(operation_type)
        if not operation_class:
            raise ValueError(f"Unsupported operation type: {operation_type}")
        
        return operation_class()

    def _create_metric(self, config: Dict[str, Any]) -> MemoryMetric:
        """
        Create a metric instance based on configuration
        
        :param config: Metric configuration
        :return: Metric instance
        """
        metric_type = config.get('class')
        
        # Example metric type mapping
        metric_map = {
            'RelevanceMetric': RelevanceMetric,
            'RecencyMetric': RecencyMetric,
            'ImportanceMetric': ImportanceMetric
        }
        
        metric_class = metric_map.get(metric_type)
        if not metric_class:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        
        return metric_class(config)


    def set_agent_context(self, agent_context: AgentContext):
        """
        Set the agent context for the memory strategy

        :param agent_context: Agent context
        """
        self.agent_context = agent_context

    async def execute(self, operation_name: str, *args, **kwargs):
        """
        Execute a registered operation
        
        :param operation_name: Name of the operation
        """
        operation = self._operations.get(operation_name)
        if not operation:
            raise ValueError(f"Operation not found: {operation_name}")
        return await operation.execute(self, *args, **kwargs)

    async def get_all_memory(self):
        """
        Get all memories from all storages
        
        :return: List of all memories
        """
        return [await storage.get_all() for storage in self._storage_map.values()]

    @abstractmethod
    async def retrieve(self, query, top_k):
        """Abstract method for retrieving memories"""
        pass

    @abstractmethod
    async def add(self, query, top_k):
        """Abstract method for adding memories"""
        pass

    def select_storage(self, memory_item):
        """
        Select appropriate storage for the memory item.
        By default, returns the first storage in the storage map.
        Override this method in subclasses for more sophisticated selection logic.
        
        :param memory_item: The memory item to find storage for
        :return: Selected storage
        """
        # If no storages are available, raise error
        if not self._storage_map:
            raise ValueError("No storage available in the strategy")
            
        # Return the first storage by default
        # Subclasses should override this method for more sophisticated selection logic
        return next(iter(self._storage_map.values()))