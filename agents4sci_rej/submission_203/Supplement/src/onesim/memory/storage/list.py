from abc import ABC, abstractmethod
from .storage import MemoryStorage
from loguru import logger

class ListMemoryStorage(MemoryStorage):
    def __init__(self, config):
        self.config = config
        # 验证容量参数
        try:
            self.capacity = int(config.get('capacity', 100))
            if self.capacity <= 0:
                logger.warning(f"Invalid capacity value: {self.capacity}, using default 100")
                self.capacity = 100
        except (ValueError, TypeError):
            logger.warning(f"Invalid capacity format in config, using default 100")
            self.capacity = 100
            
        self.memory_list = []
        self.eviction_policy = config.get('eviction_policy', 'fifo')  # 'fifo', 'lru', 'importance'

    async def add(self, memory_item):
        # 检查容量并在必要时清除旧记忆
        if len(self.memory_list) >= self.capacity:
            await self._evict_memory()
            
        self.memory_list.append(memory_item)
        return memory_item.id  # 返回添加的项目ID以便追踪

    async def get_all(self):
        return self.memory_list.copy()

    async def delete(self, memory_item):
        try:
            self.memory_list.remove(memory_item)
        except ValueError:
            # 如果通过ID删除
            if hasattr(memory_item, 'id'):
                for idx, item in enumerate(self.memory_list):
                    if item.id == memory_item.id:
                        del self.memory_list[idx]
                        return
            logger.warning(f"Memory item not found for deletion: {memory_item}")

    async def query(self, query=None, top_k=None):
        try:
            if query is None:
                result = self.memory_list
            else:
                if callable(query):
                    # 如果query是函数，用作过滤器
                    result = [item for item in self.memory_list if query(item)]
                elif isinstance(query, list) and all(callable(q) for q in query):
                    # 如果query是可调用对象列表，应用所有过滤器
                    result = self.memory_list
                    for condition in query:
                        result = [item for item in result if condition(item)]
                else:
                    # 默认情况下，将所有内容返回
                    result = self.memory_list
            
            # 应用限制
            if top_k is not None and top_k > 0:
                return result[:min(top_k, len(result))]
            return result
        except Exception as e:
            logger.error(f"Error during query operation: {e}")
            return []
    
    async def get_size(self):
        return len(self.memory_list)
        
    async def clear(self):
        """清除所有内存项"""
        self.memory_list.clear()
        
    async def merge(self):
        """合并功能的存根实现 - 在实际应用中应该被覆盖"""
        logger.warning("Default merge operation called - no action taken")
        return self.memory_list.copy()
        
    async def forget(self, criteria):
        """根据条件忘记某些记忆"""
        try:
            if callable(criteria):
                # 删除满足条件的项
                self.memory_list = [item for item in self.memory_list if not criteria(item)]
            else:
                logger.warning(f"Invalid criteria for forget operation: {criteria}")
        except Exception as e:
            logger.error(f"Error during forget operation: {e}")
            
    async def batch_add(self, memory_items):
        """批量添加多个记忆项"""
        added_ids = []
        for item in memory_items:
            item_id = await self.add(item)
            added_ids.append(item_id)
        return added_ids
        
    async def _evict_memory(self):
        """根据淘汰策略移除记忆"""
        if not self.memory_list:
            return
            
        if self.eviction_policy == 'fifo':
            # 先进先出策略
            self.memory_list.pop(0)
        elif self.eviction_policy == 'lru':
            # 最近最少使用策略 - 需要跟踪访问时间
            # 这里我们简单地使用timestamp作为近似
            self.memory_list.sort(key=lambda x: x.timestamp)
            self.memory_list.pop(0)
        elif self.eviction_policy == 'importance':
            # 根据重要性移除最不重要的
            self.memory_list.sort(key=lambda x: x.attributes.get('importance', 0))
            self.memory_list.pop(0)
        else:
            # 默认为FIFO
            self.memory_list.pop(0)