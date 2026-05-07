import uuid
import time
from abc import ABC, abstractmethod
from threading import RLock
import json
from onesim.models import ModelManager
import importlib
from .strategy.strategy import MemoryStrategy


class MemoryManager:
    def __init__(self, config_file):
        self.config = self.load_config(config_file)
        # Initialize metrics
        self.metrics = self._initialize_metrics(self.config)
        # Initialize strategy
        self.strategy = self._initialize_strategy(self.config)
        # Initialize storages
        self.short_term_storage = self._initialize_storage(self.config.get('short_term_storage'), self.config)
        self.long_term_storage = self._initialize_storage(self.config.get('long_term_storage'), self.config)
        self.operations = {}
        self.lock = RLock() if self.config.get('thread_safety') else None

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def _initialize_strategy(self, config):
        try:
            strategy_name = self.config['strategy']
            # First try to import from the memory module directly
            try:
                module = importlib.import_module('onesim.memory')
                strategy_class = getattr(module, strategy_name)
            except (ImportError, AttributeError) as e:
                # Fall back to checking specific strategy files in the strategy directory
                try:
                    module = importlib.import_module('onesim.memory.strategy')
                    strategy_class = getattr(module, strategy_name)
                except (ImportError, AttributeError) as e2:
                    # Try importing specific strategy classes directly
                    if strategy_name == "ListStrategy":
                        from .strategy.list_strategy import ListStrategy
                        strategy_class = ListStrategy
                    elif strategy_name == "ShortLongStrategy":
                        from .strategy.short_long_strategy import ShortLongStrategy
                        strategy_class = ShortLongStrategy
                    else:
                        raise ImportError(f"Cannot import {strategy_name} from 'onesim.memory.strategy': {e2}")
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import {self.config['strategy']}: {e}")

        strategy = strategy_class(self.config, model_config_name=self.config.get('model_config_name'))
        return strategy

    def _initialize_metrics(self, config):
        metrics = {}
        metric_configs = config.get('metrics', {})
        for metric_name, metric_info in metric_configs.items():
            class_name = metric_info['class']
            init_args = metric_info.get('init_args', {})

            # 动态导入模块和类
            try:
                module = importlib.import_module('onesim.memory.metric')
                metric_class = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Cannot import {class_name} from memory.metric: {e}")

            metrics[metric_name] = metric_class(**init_args)
        return metrics

    def _initialize_storage(self, storage_class_name, config):
        if not storage_class_name:
            return None
        try:
            module = importlib.import_module('onesim.memory.storage')
            storage_class = getattr(module, storage_class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Cannot import {storage_class_name} from memory.storage: {e}")
        if storage_class_name == 'VectorMemoryStorage':
            embedding_model = config.get('embedding_model')
            return storage_class(embedding_model)
        else:
            return storage_class()

    def register_operation(self, operation_name, operation):
        self.operations[operation_name] = operation

    def execute_operation(self, operation_name, **kwargs):
        if operation_name in self.operations:
            operation = self.operations[operation_name]
            if self.lock:
                with self.lock:
                    return operation.execute(self,**kwargs)
            else:
                return operation.execute(self, **kwargs)
        else:
            raise ValueError(f"Operation '{operation_name}' not registered.")

# Export example operation classes
class AddMemoryOperation:
    def __init__(self, config=None):
        self.config = config or {}

    def execute(self, strategy, memory_item, storage_name=None):
        # Use select_storage if storage_name is not provided
        if storage_name:
            storage = strategy._storage_map.get(storage_name)
            if not storage:
                raise ValueError(f"Storage not found: {storage_name}")
        else:
            storage = strategy.select_storage(memory_item)
        
        storage.add(memory_item)

class RetrieveMemoryOperation:
    def __init__(self, config=None):
        self.config = config or {}

    def execute(self, strategy, query, top_k=5):
        return strategy.retrieve(query, top_k)
