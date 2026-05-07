from abc import ABC, abstractmethod
import time
import numpy as np
from datetime import datetime
from onesim.models import ModelManager
from onesim.models.core.message import Message
from onesim.models.parsers import TagParser

class MemoryMetric(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    async def calculate(self, memory_item, query=None):
        pass


class ImportanceMetric(MemoryMetric):
    def __init__(self, config):
        model_config_name = config.get("model_config_name")
        model_manager = ModelManager.get_instance()
        self.llm_model = model_manager.get_model(
            config_name=model_config_name,
        )  # LLM model instance
        self.parser = TagParser(
            tag_begin="[SCORE]",
            content_hint="the importance score",
            tag_end="[/SCORE]"
        )
        # 添加缓存以减少LLM调用
        self.cache = {}
        
  
    async def calculate(self, memory_item, query=None):
        # 检查缓存
        if memory_item.id in self.cache:
            return self.cache[memory_item.id]
            
        if 'importance' in memory_item.attributes and memory_item.attributes['importance'] is not None:
            return memory_item.attributes['importance']
            
        # Use LLM to compute importance
        prompt=f"Evaluate the importance of this memory based on its relevance, context, and potential impact. Provide a score from 1 to 10, where 1 is the least important and 10 is the most important.\nMemory Content: {memory_item.content}\n"+self.parser.format_instruction
        prompt = self.llm_model.format(
            Message("user",prompt, role="user")
        )
        try:
            response = await self.llm_model.acall(prompt)
            res = self.parser.parse(response)
            importance = float(res.parsed['score'])
            # 存入缓存
            self.cache[memory_item.id] = importance
            return importance
        except Exception as e:
            # 更详细的错误处理
            error_msg = f"Error parsing importance score: {e}\nResponse: {response if 'response' in locals() else 'No response'}"
            # 默认返回中等重要性而不是崩溃
            importance = 5.0
            return importance

class RecencyMetric(MemoryMetric):
    @staticmethod
    async def calculate(memory_item, query=None):
        # Calculate recency based on timestamp
        try:
            if isinstance(memory_item.timestamp, str):
                dt = datetime.strptime(memory_item.timestamp, "%Y-%m-%d %H:%M:%S")
                memory_timestamp = dt.timestamp()
            else:
                # 如果 timestamp 已经是 float 或 int，直接使用
                memory_timestamp = float(memory_item.timestamp)
            
            # 计算时间差
            return 1 / (time.time() - memory_timestamp + 1)
        except Exception as e:
            # 出错时返回低优先级
            return 0.1

class RelevanceMetric(MemoryMetric):
    def __init__(self, config):
        model_config_name = config.get("model_config_name")
        model_manager = ModelManager.get_instance()
        self.embedding_model = model_manager.get_model(
            config_name=model_config_name,
        )
        # 添加缓存
        self.embedding_cache = {}
           
    async def calculate(self, memory_item, query=None):
        if query is None or self.embedding_model is None:
            return 1.0
            
        # 使用缓存减少重复计算
        cache_key = f"{memory_item.id}:{query}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
            
        try:
            # 检查memory_item是否已有嵌入向量
            if hasattr(memory_item, 'embedding') and memory_item.embedding is not None:
                memory_embedding = memory_item.embedding
            else:
                # 计算并缓存
                embedding_result = await self.embedding_model.acall(memory_item.content)
                memory_embedding = embedding_result.embedding
                memory_item.embedding = memory_embedding
                
            query_embedding = await self.embedding_model.acall(query)
            query_embedding = query_embedding.embedding
            
            similarity = self.cosine_similarity(memory_embedding, query_embedding)
            # 保存到缓存
            self.embedding_cache[cache_key] = similarity
            return similarity
        except Exception as e:
            # 出错时返回默认相关性
            return 0.5
    
    def cosine_similarity(self, vec1, vec2):
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except:
            return 0.0
