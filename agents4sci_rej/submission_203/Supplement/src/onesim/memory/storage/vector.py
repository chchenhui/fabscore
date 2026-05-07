from abc import ABC, abstractmethod
import asyncio
import numpy as np
import faiss  # 假设使用 FAISS 作为向量检索库
from .storage import MemoryStorage
from onesim.models import ModelManager
from loguru import logger

class VectorMemoryStorage(MemoryStorage):
    def __init__(self, config):
        self.config = config
        model_config_name = config.get("model_config_name")
        model_manager = ModelManager.get_instance()
        self.embedding_model = model_manager.get_model(
            config_name=model_config_name, model_type="embedding"
        )
        self.memory_items = []
        self.embeddings = []
        self.index = None  # FAISS 索引
        self.batch_size = config.get("batch_size", 10)  # 批量更新大小
        self.pending_updates = 0  # 待更新计数
        self.index_dimension = None  # 动态确定的维度
        self.max_index_size = config.get("max_index_size", 10000)  # 最大索引大小

    async def add(self, memory_item):
        """
        添加记忆项并生成embedding（如果尚未提供）
        
        :param memory_item: 待添加的记忆项
        :return: 记忆项ID用于跟踪
        """
        try:
            # 检查是否已经有嵌入向量
            if not hasattr(memory_item, 'embedding') or memory_item.embedding is None:
                try:
                    logger.debug(f"Computing embedding for new memory item: {memory_item.id}")
                    embedding_result = await self.embedding_model.acall(memory_item.content)
                    embedding = embedding_result.embedding
                    memory_item.embedding = embedding
                except Exception as e:
                    logger.error(f"Error computing embedding for item {memory_item.id}: {e}")
                    # 创建一个零向量作为回退，避免完全失败
                    # 只有在我们已经有其他embedding的情况下才能确定维度
                    if self.embeddings and self.index_dimension:
                        embedding = np.zeros(self.index_dimension, dtype=np.float32)
                        memory_item.embedding = embedding
                    else:
                        # 如果我们无法创建零向量，则抛出异常
                        raise ValueError(f"Cannot create fallback embedding: no index dimension determined yet")
            else:
                embedding = memory_item.embedding

            # 确定维度并初始化索引（如果尚未初始化）
            if self.index is None and embedding is not None:
                self.index_dimension = len(embedding)
                self._initialize_index()

            # 添加到内存和embedding列表
            self.memory_items.append(memory_item)
            self.embeddings.append(embedding)

            # 增加待更新计数
            self.pending_updates += 1

            # 判断是否需要更新索引
            if self.pending_updates >= self.batch_size:
                await self._update_index_batch()

            return memory_item.id
        except Exception as e:
            logger.error(f"Error adding item to vector storage: {e}")
            raise

    async def get_all(self):
        return self.memory_items.copy()

    async def delete(self, memory_item):
        try:
            idx = -1
            # 根据ID或对象查找
            if hasattr(memory_item, 'id'):
                for i, item in enumerate(self.memory_items):
                    if item.id == memory_item.id:
                        idx = i
                        break
            else:
                idx = self.memory_items.index(memory_item)

            if idx >= 0:
                self.memory_items.pop(idx)
                self.embeddings.pop(idx)
                # 删除操作也需要更新索引
                self.pending_updates += 1

                # 判断是否需要更新索引
                if self.pending_updates >= self.batch_size:
                    await self._update_index_batch()
            else:
                logger.warning(f"Memory item not found for deletion: {memory_item}")
        except Exception as e:
            logger.error(f"Error deleting item from vector storage: {e}")
            raise

    async def update(self, memory_item):
        """
        更新记忆项，只在内容变化时重新计算embedding
        """
        try:
            idx = -1
            if hasattr(memory_item, 'id'):
                for i, item in enumerate(self.memory_items):
                    if item.id == memory_item.id:
                        idx = i
                        break

            if idx >= 0:
                old_item = self.memory_items[idx]
                content_changed = old_item.content != memory_item.content

                # 更新记忆项
                self.memory_items[idx] = memory_item

                # 只有在内容变化或embedding不存在时才重新计算
                if content_changed or not hasattr(memory_item, 'embedding') or memory_item.embedding is None:
                    logger.debug(f"Content changed or embedding missing, recalculating embedding for item {memory_item.id}")
                    embedding_result = await self.embedding_model.acall(memory_item.content)
                    embedding = embedding_result.embedding
                    memory_item.embedding = embedding
                    self.embeddings[idx] = embedding

                    # 标记需要更新索引
                    self.pending_updates += 1
                elif hasattr(old_item, 'embedding') and old_item.embedding is not None:
                    # 如果内容没变且旧项有embedding，保留旧embedding
                    memory_item.embedding = old_item.embedding
                    self.embeddings[idx] = old_item.embedding
                else:
                    # 使用新提供的embedding
                    self.embeddings[idx] = memory_item.embedding

                # 判断是否需要更新索引
                if self.pending_updates >= self.batch_size:
                    await self._update_index_batch()
            else:
                logger.warning(f"Memory item not found for update: {memory_item}")
        except Exception as e:
            logger.error(f"Error updating item in vector storage: {e}")
            raise

    def _initialize_index(self):
        """初始化FAISS索引"""
        try:
            if self.index_dimension is None:
                logger.warning("Cannot initialize index: dimension not yet determined")
                return

            self.index = faiss.IndexFlatL2(self.index_dimension)
            logger.info(f"Initialized FAISS index with dimension {self.index_dimension}")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise

    async def _update_index_batch(self):
        """批量更新索引而不是每次操作都重建"""
        try:
            if not self.embeddings:
                logger.debug("No embeddings to update index")
                self.index = None
                self.pending_updates = 0
                return

            if self.index is None and len(self.embeddings) > 0:
                # 如果索引不存在但有嵌入向量，则初始化索引
                self.index_dimension = len(self.embeddings[0])
                self._initialize_index()

            # 完全重建索引
            embeddings_array = np.array(self.embeddings).astype('float32')
            if self.index is not None:
                self.index.reset()  # 清空索引
                self.index.add(embeddings_array)  # 添加所有向量

            self.pending_updates = 0  # 重置待更新计数
            logger.debug(f"Updated FAISS index with {len(self.embeddings)} vectors")
        except Exception as e:
            logger.error(f"Error updating FAISS index: {e}")
            raise

    async def query(self, query, top_k=5):
        """查询最相似的记忆项"""
        try:
            # 如果没有数据或索引未初始化，则返回空列表
            if not self.memory_items or self.index is None:
                return []

            # 确保索引是最新的
            if self.pending_updates > 0:
                await self._update_index_batch()

            # 处理查询
            if query is None:
                return self.memory_items[:min(top_k, len(self.memory_items))]

            # 转换查询为字符串
            if isinstance(query, list):
                query_string = ".".join(query)
            else:
                query_string = str(query)

            # 获取查询的嵌入向量
            try:
                embedding_res = await self.embedding_model.acall(query_string[:500])
                query_vector = embedding_res.embedding

                # 确保查询向量是正确的形状和类型
                query_vector = np.array([query_vector]).astype('float32')

                # 执行搜索
                distances, indices = self.index.search(query_vector, min(top_k, len(self.memory_items)))

                # 收集结果
                retrieved_items = []
                for idx in indices[0]:
                    if 0 <= idx < len(self.memory_items):  # 确保索引有效
                        retrieved_items.append(self.memory_items[idx])

                return retrieved_items
            except Exception as e:
                logger.error(f"Error during vector search: {e}")
                return self.memory_items[:min(top_k, len(self.memory_items))]
        except Exception as e:
            logger.error(f"Error in query operation: {e}")
            return []

    async def get_size(self):
        return len(self.memory_items)

    async def clear(self):
        """清空存储"""
        self.memory_items = []
        self.embeddings = []
        if self.index is not None:
            self.index.reset()
        self.pending_updates = 0

    async def batch_add(self, memory_items):
        """
        批量添加多个记忆项，高效计算embeddings
        
        :param memory_items: 记忆项列表
        :return: 添加的记忆项ID列表
        """
        added_ids = []

        if not memory_items:
            logger.debug("No items to add in batch operation")
            return added_ids

        try:
            # 收集需要计算嵌入向量的项目
            items_to_embed = []

            for item in memory_items:
                if not hasattr(item, 'embedding') or item.embedding is None:
                    items_to_embed.append(item)

            # 批量计算嵌入向量
            if items_to_embed:
                logger.info(f"Computing embeddings for {len(items_to_embed)} items in batch")
                contents = [item.content for item in items_to_embed]

                # 为了避免过大的批次导致资源问题，我们限制并发数量
                max_concurrent = min(len(contents), 10)  # 并发不超过10个请求
                semaphore = asyncio.Semaphore(max_concurrent)

                async def get_embedding_with_semaphore(content, item_idx):
                    try:
                        async with semaphore:
                            result = await self.embedding_model.acall(content)
                            return result, item_idx
                    except Exception as e:
                        logger.error(f"Error computing embedding for item {items_to_embed[item_idx].id}: {e}")
                        return None, item_idx

                # 并发执行，但控制并发量
                batch_tasks = [get_embedding_with_semaphore(content, i) for i, content in enumerate(contents)]
                batch_results = await asyncio.gather(*batch_tasks)

                # 处理结果，包括错误处理
                for result, idx in batch_results:
                    if result is not None:
                        items_to_embed[idx].embedding = result.embedding
                    elif self.embeddings and self.index_dimension:
                        # 如果失败，使用零向量回退
                        items_to_embed[idx].embedding = np.zeros(self.index_dimension, dtype=np.float32)
                        logger.warning(f"Using zero vector fallback for item {items_to_embed[idx].id}")

            # 添加所有项目
            for item in memory_items:
                try:
                    if hasattr(item, 'embedding') and item.embedding is not None:
                        # 如果已有嵌入向量，直接添加
                        embedding = item.embedding

                        # 初始化索引（如果需要）
                        if self.index is None and embedding is not None:
                            self.index_dimension = len(embedding)
                            self._initialize_index()

                        self.memory_items.append(item)
                        self.embeddings.append(embedding)
                        added_ids.append(item.id)
                        self.pending_updates += 1
                    else:
                        logger.warning(f"Skipping item {item.id} with no embedding")
                except Exception as e:
                    logger.error(f"Error adding individual item in batch: {e}")
                    # 继续处理其他项目，而不是完全失败
                    continue

            # 更新索引
            if self.pending_updates > 0:
                await self._update_index_batch()

            return added_ids
        except Exception as e:
            logger.error(f"Error in batch_add operation: {e}")
            # 返回已成功添加的ID，而不是完全失败
            return added_ids
