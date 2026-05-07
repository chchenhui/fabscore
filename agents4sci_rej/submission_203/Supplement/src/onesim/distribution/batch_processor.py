import asyncio
import time
from typing import List, Dict, Any
from loguru import logger

class BatchProcessor:
    """
    批处理工具类，用于减少网络开销，提高吞吐量。
    在高并发多Agent模拟中特别有用，可以合并事件和决策记录的发送请求。
    """
    def __init__(self, batch_size=50, max_wait_time=0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.storage_events = []
        self.decision_records = []
        self.last_flush_time = time.time()
        self.flush_lock = asyncio.Lock()
        self.master_address = None
        self.master_port = None
        self._task = None
        self._stopped = False
    
    def start(self, master_address, master_port):
        """启动批处理器，开始后台刷新任务"""
        if self._task is not None:
            return
            
        self.master_address = master_address
        self.master_port = master_port
        self._stopped = False
        self._task = asyncio.create_task(self._periodic_flush())
        logger.info(f"Started batch processor for {master_address}:{master_port}")
    
    def stop(self):
        """停止批处理器并刷新所有剩余项目"""
        self._stopped = True
        if self._task:
            self._task.cancel()
            self._task = None
        
        # 最终刷新
        asyncio.create_task(self.flush())
    
    async def _periodic_flush(self):
        """定期刷新批处理的后台任务"""
        try:
            while not self._stopped:
                now = time.time()
                if now - self.last_flush_time >= self.max_wait_time:
                    await self.flush()
                await asyncio.sleep(0.05)  # 小睡眠避免忙等
        except asyncio.CancelledError:
            logger.debug("Batch processor task cancelled")
        except Exception as e:
            logger.error(f"Error in batch processor: {e}")
    
    async def add_storage_event(self, event_data):
        """添加存储事件到批处理"""
        # Avoid circular import by importing here
        from onesim.distribution.grpc_impl import send_storage_event_to_master
        
        if not self.master_address:
            logger.warning("Batch processor not started yet")
            # 如果未启动，立即发送
            return await send_storage_event_to_master(
                self.master_address, 
                self.master_port, 
                event_data
            )
        
        async with self.flush_lock:
            self.storage_events.append(event_data)
        
        # 达到批处理大小时刷新
        if len(self.storage_events) >= self.batch_size:
            await self.flush_storage_events()
            
        return True
    
    async def add_decision_record(self, decision_data):
        """添加决策记录到批处理"""
        # Avoid circular import by importing here
        from onesim.distribution.grpc_impl import send_decision_record_to_master
        
        if not self.master_address:
            logger.warning("Batch processor not started yet")
            # 如果未启动，立即发送
            return await send_decision_record_to_master(
                self.master_address, 
                self.master_port, 
                decision_data
            )
        
        async with self.flush_lock:
            self.decision_records.append(decision_data)
        
        # 达到批处理大小时刷新
        if len(self.decision_records) >= self.batch_size:
            await self.flush_decision_records()
            
        return True
    
    async def flush_storage_events(self):
        """刷新所有待处理的存储事件"""
        # Avoid circular import by importing here
        from onesim.distribution.grpc_impl import send_storage_event_batch_to_master
        
        async with self.flush_lock:
            if not self.storage_events:
                return True
                
            events_to_send = self.storage_events
            self.storage_events = []
        
        success = await send_storage_event_batch_to_master(
            self.master_address,
            self.master_port,
            events_to_send
        )
        
        return success
    
    async def flush_decision_records(self):
        """刷新所有待处理的决策记录"""
        # Avoid circular import by importing here
        from onesim.distribution.grpc_impl import send_decision_record_batch_to_master
        
        async with self.flush_lock:
            if not self.decision_records:
                return True
                
            records_to_send = self.decision_records
            self.decision_records = []
        
        success = await send_decision_record_batch_to_master(
            self.master_address,
            self.master_port,
            records_to_send
        )
        
        return success
    
    async def flush(self):
        """刷新所有待处理项目"""
        results = []
        if self.storage_events:
            results.append(await self.flush_storage_events())
        if self.decision_records:
            results.append(await self.flush_decision_records())
        
        self.last_flush_time = time.time()
        return all(results) if results else True

# 创建全局批处理器实例
batch_processor = BatchProcessor() 