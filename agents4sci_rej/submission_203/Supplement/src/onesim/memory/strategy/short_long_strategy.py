from .strategy import MemoryStrategy
from typing import Dict, Any, List
from .strategy import AgentContext
from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from loguru import logger
import json
from ..memory_item import MemoryItem
from ..operation.operation import ReflectMemoryOperation

class ShortLongStrategy(MemoryStrategy):
    def __init__(self, config: Dict[str, Any], model_config_name:str=None):
        """
        Initialize ShortLongStrategy with configurations.

        :param config: Configuration dictionary
        :param model_config_name: Name of the model configuration to use
        """
        super().__init__(config, model_config_name)
        self.transfer_conditions = config.get('transfer_conditions', {})
        self.cumulated_importance = 0
        self.reflec_threshold = config.get('reflect_threshold', 100)
        
        # Define storage names for clarity
        self.short_term_storage_name = 'short_term_storage'
        self.long_term_storage_name = 'long_term_storage'

        # Initialize storages if not already present
        if self.short_term_storage_name not in self._storage_map:
            self._storage_map[self.short_term_storage_name] = self._create_storage({
                'type': 'list',  # Assuming list storage for short-term
                'class': 'ListMemoryStorage',
                'name': self.short_term_storage_name,
                'capacity': config.get('short_term_capacity', 100),
                'eviction_policy': 'fifo'
            })

        if self.long_term_storage_name not in self._storage_map:
            self._storage_map[self.long_term_storage_name] = self._create_storage({
                'type': 'vector',  # Assuming vector storage for long-term
                'class': 'VectorMemoryStorage',
                'name': self.long_term_storage_name,
                'model_config_name': model_config_name
            })
        
        # Ensure reflect operation is registered
        if 'reflect' not in self._operations:
            self._operations['reflect'] = ReflectMemoryOperation()

    async def reflect(self):
        """
        Execute the reflection process using the ReflectMemoryOperation.
        """
        await self.execute('reflect')

    async def add(self, memory_item):
        """
        Add a memory item to the appropriate storage.

        :param memory_item: The memory item to add
        """
        
        await self.execute('add', storage_name=self.short_term_storage_name, memory_item=memory_item)
        
        # 更新累积重要性
        if 'importance' in memory_item.attributes:
            importance = memory_item.attributes.get('importance', 0)
        else:
            # 如果尚未计算重要性，计算它
            importance_metric = self._metrics.get('importance')
            if importance_metric:
                try:
                    importance = await importance_metric.calculate(memory_item)
                    memory_item.attributes['importance'] = importance
                except Exception as e:
                    logger.error(f"Error calculating importance: {e}")
                    importance = 0
            else:
                importance = 0
            
        self.cumulated_importance += importance
        
        # 检查是否触发反思
        if self.cumulated_importance > self.reflec_threshold:
            logger.info(f"Triggering reflection: accumulated importance {self.cumulated_importance} exceeds threshold {self.reflec_threshold}")
            await self.reflect()
            self.cumulated_importance = 0

    async def retrieve(self, query, top_k=5) -> List[Any]:
        """
        Retrieve memory items based on a query and ranking.

        :param query: Query for retrieval
        :param top_k: Number of top items to retrieve
        :param source: Source of memories ('short_term', 'long_term', or 'both')
        :return: List of retrieved memory items
        """
        memories = []

        memories.extend(await self._storage_map[self.short_term_storage_name].get_all())
        long_memories = await self.execute('retrieve', storage_name=self.long_term_storage_name, query=query, top_k=top_k)
        memories.extend(long_memories)
        
        # 获取每个metric的权重
        metric_weights = {name: metric_config.get('weight', 1.0) 
                         for name, metric_config in self.config.get('metrics', {}).items()}
        
        memories = await self.score(memories, query, metric_weights)
        # Sort memories based on score
        sorted_memories = sorted(memories, key=lambda m: m.attributes.get('score', 0), reverse=True)
        return sorted_memories[:top_k]

    async def should_transfer(self, memory_item) -> bool:
        """
        Decide if a memory item should be transferred to long-term storage.

        :param memory_item: The memory item to evaluate
        :return: True if it should be transferred, False otherwise
        """
        for metric_name, condition in self.transfer_conditions.items():
            metric = self._metrics.get(metric_name)
            if metric:
                try:
                    if metric_name in memory_item.attributes:
                        value = memory_item.attributes[metric_name]
                    else:
                        value = await metric.calculate(memory_item)
                        memory_item.attributes[metric_name] = value
                    
                    if eval(f"{value} {condition}"):
                        return True
                except Exception as e:
                    logger.error(f"Error evaluating condition '{condition}' for metric '{metric_name}': {e}")
        return False

    async def transfer_memories(self):
        """
        Transfer eligible memories from short-term to long-term storage.
        """
        short_term_storage = self._storage_map[self.short_term_storage_name]
        memories_to_transfer = []

        # Evaluate each memory item in short-term storage
        for memory_item in short_term_storage.get_all():
            if self.should_transfer(memory_item):
                memories_to_transfer.append(memory_item)

        # Transfer memories
        for memory_item in memories_to_transfer:
            await self.execute('remove', storage_name=self.short_term_storage_name, memory_item=memory_item)
            await self.execute('add', storage_name=self.long_term_storage_name, memory_item=memory_item)

    async def score(self, memories: List[Any], query: Any, weights: Dict[str, float]) -> List[Any]:
        """
        Score memories based on multiple metrics with weights.

        :param memories: List of memory items to score
        :param query: Query for relevance scoring
        :param weights: Weights for scoring metrics
        :return: List of scored memory items
        """
        # Score memories
        for memory_item in memories:
            score = 0
            for metric_name, weight in weights.items():
                metric = self._metrics.get(metric_name)
                if metric:
                    try:
                        metric_value = memory_item.attributes.get(metric_name)
                        if metric_value is None:
                            metric_value = await metric.calculate(memory_item, query)
                            memory_item.attributes[metric_name] = metric_value
                        score += metric_value * weight
                    except Exception as e:
                        logger.error(f"Error calculating metric {metric_name}: {e}")
            memory_item.attributes['score'] = score

        return memories

    def select_storage(self, memory_item):
        """
        Override the select_storage method to select the appropriate storage based on
        the memory item's properties. By default, it selects the short-term storage.
        Long-term storage is only used after transfer_memories() is called.
        
        :param memory_item: The memory item to store
        :return: The selected storage (short or long term)
        """
        # By default, new memories go to short-term storage
        return self._storage_map[self.short_term_storage_name]

