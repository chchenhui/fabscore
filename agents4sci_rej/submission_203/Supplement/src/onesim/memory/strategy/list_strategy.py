from .strategy import MemoryStrategy, AgentContext
from typing import Dict, Any, List
from ..memory_item import MemoryItem

class ListStrategy(MemoryStrategy):
    def __init__(self, config: Dict[str, Any], model_config_name: str = None):
        """
        Initialize a simple list-based memory strategy.
        
        :param config: Configuration dictionary
        :param model_config_name: Model configuration name
        """
        super().__init__(config, model_config_name)
        
        self.memory_storage_name = 'simple_memory_storage'
        

        if self.memory_storage_name not in self._storage_map:
            self._storage_map[self.memory_storage_name] = self._create_storage({
                'type': 'list',
                'class': 'ListMemoryStorage',
                'name': self.memory_storage_name
            })
    
    async def add(self, memory_item: MemoryItem):
        """
        Add a memory item to the storage.
        
        :param memory_item: The memory item to add
        """
        await self.execute('add', storage_name=self.memory_storage_name, memory_item=memory_item)
    
    async def retrieve(self, query, top_k=5) -> List[Any]:
        """
        Retrieve memory items based on the query.
        
        :param query: The retrieval query
        :param top_k: The number of top items to retrieve
        :return: The list of retrieved memory items
        """

        memories = await self._storage_map[self.memory_storage_name].get_all()
        

        relevance_metric = self._metrics.get('relevance')
        if relevance_metric and query:

            for memory_item in memories:
                relevance_score = await relevance_metric.calculate(memory_item, query)
                memory_item.attributes['relevance'] = relevance_score
            

            sorted_memories = sorted(memories, key=lambda m: m.attributes.get('relevance', 0), reverse=True)
            return sorted_memories[:top_k]
        

        return memories[-top_k:] if len(memories) > top_k else memories
    
    async def remove(self, memory_item):
        """
        Remove a memory item from the storage.
        
        :param memory_item: The memory item to remove
        """
        await self.execute('remove', storage_name=self.memory_storage_name, memory_item=memory_item)
    
    async def clear(self):
        """
        Clear all memories.
        """
        storage = self._storage_map[self.memory_storage_name]
        memories = await storage.get_all()
        for memory_item in memories:
            await self.execute('remove', storage_name=self.memory_storage_name, memory_item=memory_item)
    
    async def get_all(self):
        """
        Get all memory items.
        
        :return: The list of all memory items
        """
        return await self._storage_map[self.memory_storage_name].get_all()

    def select_storage(self, memory_item):
        """
        Override the select_storage method to always select the simple memory storage.
        
        :param memory_item: The memory item to store
        :return: The simple list memory storage
        """
        return self._storage_map[self.memory_storage_name]