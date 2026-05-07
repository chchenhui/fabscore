import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import uuid
from loguru import logger
import importlib
from onesim.events import Event, get_event_bus
from onesim.distribution.node import Node, NodeRole
from onesim.events import Event
import threading
import time
import grpc
from onesim.distribution.worker import WorkerNode
from onesim.distribution.connection_manager import initialize_connection_manager

@dataclass
class WorkerInfo:
    """Information about a connected worker node"""
    worker_id: str
    address: str
    port: int
    agent_count: int = 0
    agent_ids: List[str] = field(default_factory=list)
    status: str = "connected"
    last_heartbeat: float = 0.0

class MasterNode(Node):
    """Master node for distributed operation"""

    def __init__(self, 
                 node_id: str,
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(node_id, NodeRole.MASTER, config)
        self.listen_port = config.get("listen_port", 10051)
        self._server = None
        self.workers = {}  # worker_id -> WorkerInfo
        self.agent_locations = {}  # agent_id -> worker_id
        self.expected_worker_count = config.get("expected_workers", 1)
        self.initialized = asyncio.Event()  # 标记初始化完成
        self.sim_env = None  # Reference to simulation environment for data storage
        self.worker_lock = asyncio.Lock()
        self.event_bus = get_event_bus()
        self.health_check_interval = config.get("health_check_interval", 1800)  # 默认2分钟检查一次
        self.worker_timeout = config.get("worker_timeout", 3600)  # 默认3分钟超时
        self._health_check_task = None
        self.shutting_down = False

    async def initialize(self) -> None:
        """Initialize master node"""
        await super().initialize()

        # Start gRPC server
        if self._grpc_module:
            self._server = await self._grpc_module.create_master_server(self, self.listen_port)
            logger.info(f"Master node {self.node_id} started gRPC server on port {self.listen_port}")

        # 启动worker连接监视任务
        asyncio.create_task(self._wait_for_workers())

        # 启动健康检查任务
        self._health_check_task = asyncio.create_task(self._periodic_health_check())

    async def _wait_for_workers(self):
        """等待所有预期数量的worker连接"""
        logger.info(f"Waiting for {self.expected_worker_count} workers to connect...")

        while len(self.workers) < self.expected_worker_count:
            logger.info(f"Connected workers: {len(self.workers)}/{self.expected_worker_count}")
            await asyncio.sleep(2)  # 每2秒检查一次
        workers=[(worker.address, worker.port) for worker in self.workers.values()]
        await initialize_connection_manager(workers)
        logger.info(f"All {self.expected_worker_count} workers connected. Ready to initialize environment.")
        self.initialized.set()  # 设置事件，表示所有worker已连接

    async def _periodic_health_check(self):
        """定期执行健康检查，移除不响应的worker"""
        try:
            while True:
                await asyncio.sleep(self.health_check_interval)
                missing_workers = await self.check_workers_health(self.worker_timeout)

                if missing_workers:
                    logger.warning(f"Health check found {len(missing_workers)} missing workers: {missing_workers}")
        except asyncio.CancelledError:
            logger.info("Health check task cancelled")
        except Exception as e:
            logger.error(f"Error in health check task: {e}")

    async def register_worker(self, worker_id: str, address: str, port: int) -> Tuple[bool, str]:
        """
        注册Worker节点 (异步版本)
        
        Args:
            worker_id: Worker节点ID
            address: Worker节点地址
            port: Worker节点端口
            
        Returns:
            Tuple[bool, str]: (成功标志, 消息)
        """
        try:
            # logger.info(f"Received registration request from worker {worker_id} at {address}:{port}")

            async with self.worker_lock:  # 使用异步锁
                if worker_id in self.workers:
                    logger.warning(f"Worker {worker_id} already registered, updating address and port")
                    self.workers[worker_id].address = address
                    self.workers[worker_id].port = port
                    self.workers[worker_id].last_heartbeat = time.time()
                    self.workers[worker_id].status = "connected"
                    return True, f"Worker {worker_id} updated"
                else:
                    worker_info=WorkerInfo(
                        worker_id=worker_id, 
                        address=address, 
                        port=port,
                        last_heartbeat=time.time()
                    )
                    self.workers[worker_id] = worker_info
                    # logger.info(f"Worker {worker_id} registered at {address}:{port}")
                    return True, f"Worker {worker_id} registered successfully"
        except Exception as e:
            logger.error(f"Error registering worker {worker_id}: {e}")
            return False, f"Registration failed: {str(e)}"

    async def allocate_agent(self, agent_id: str) -> Optional[str]:
        """Allocate an agent to the least loaded worker"""
        if not self.workers:
            logger.error("No workers available to allocate agent")
            return None

        async with self.worker_lock:  # 使用异步锁
            # Find worker with lowest load
            target_worker = min(self.workers.values(), key=lambda w: w.agent_count)
            target_worker.agent_count += 1
            target_worker.agent_ids.append(agent_id)
            self.agent_locations[agent_id] = target_worker.worker_id

            logger.info(f"Allocated agent {agent_id} to worker {target_worker.worker_id}")
            return target_worker.worker_id

    async def forward_event(self, event: Event) -> bool:
        """Forward an event to the appropriate worker(s)"""
        success = True
        to_agent_id = event.to_agent_id
        # During shutdown, avoid forwarding events to workers to prevent expected UNAVAILABLE noise
        if self.shutting_down and to_agent_id != "ENV":
            return True
        if to_agent_id == "ENV":
            await self.handle_event(event)
            return success

        worker_id = None
        target_address = None
        target_port = None

        async with self.worker_lock:  # 使用异步锁
            # Find worker hosting this agent
            worker_id = self.agent_locations.get(to_agent_id)
            if not worker_id:
                logger.warning(f"Unknown agent {to_agent_id} for event {event.event_kind}")
                # success = False # No need to set success, just return
                return False

                # Get worker info
            worker = self.workers.get(worker_id)
            if not worker:
                logger.warning(f"Unknown worker {worker_id} for agent {to_agent_id}")
                # success = False # No need to set success, just return
                return False

            # Copy worker details needed for sending before releasing the lock
            target_address = worker.address
            target_port = worker.port

        # Lock is released. Now perform the gRPC call.
        try:
            if self._grpc_module:
                await self._grpc_module.send_event_to_worker(
                    target_address, 
                    target_port, 
                    event
                )
        except Exception as e:
            # Use the worker_id captured while the lock was held for logging context
            logger.error(f"Error forwarding event to worker {worker_id} (destination: {target_address}:{target_port}): {e}")
            success = False

        return success

    async def _forward_events(self):
        """Background task to forward events from the event bus"""
        from onesim.events import get_event_bus

        event_bus = get_event_bus()
        queue = event_bus.queue

        while True:
            try:
                event = await queue.get()

                # Handle event according to destinations
                await self.forward_event(event)

                queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event forwarder: {e}")

    async def handle_event(self, event: Event):
        """Handle an event received from the master"""
        # Forward to local event bus
        event_bus = get_event_bus()
        await event_bus.dispatch_event(event)

    def set_sim_env(self, sim_env):
        """Set reference to simulation environment for data forwarding"""
        self.sim_env = sim_env
        # Also set master_node reference in sim_env for token collection
        if hasattr(sim_env, '__dict__'):
            sim_env.master_node = self
        logger.info(f"Master node linked to simulation environment: {sim_env.name}")

    async def get_token_usage(self, worker_id=None):
        """
        Get token usage statistics from this node.
        
        Args:
            worker_id: Optional worker ID, ignored in this implementation
            
        Returns:
            Dict[str, Any]: Token usage statistics
        """
        try:
            from onesim.models.utils.token_usage import get_token_usage_stats
            return get_token_usage_stats()
        except ImportError:
            logger.warning("Token usage module not available")
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0,
                "model_usage": {}
            }
        except Exception as e:
            logger.error(f"Error getting token usage: {e}")
            return {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "request_count": 0,
                "model_usage": {},
                "error": str(e)
            }

    async def stop_server(self) -> None:
        """停止gRPC服务器"""
        if self._server:
            await self._server.stop(0)
            logger.info("Master server stopped")

    async def update_worker_heartbeat(self, worker_id: str) -> None:
        """更新Worker节点的最后心跳时间 (异步版本)"""
        try:
            async with self.worker_lock:  # 使用异步锁
                if worker_id in self.workers:
                    previous_time = self.workers[worker_id].last_heartbeat
                    current_time = time.time()
                    elapsed = current_time - previous_time

                    # 更新心跳时间和状态
                    self.workers[worker_id].last_heartbeat = current_time
                    self.workers[worker_id].status = "connected"

                    # 只在大于60秒间隔或首次心跳时记录日志，避免日志过多
                    if elapsed > 60.0 or previous_time == 0:
                        logger.info(f"Received heartbeat from worker {worker_id} after {elapsed:.1f}s")
                else:
                    logger.warning(f"Received heartbeat from unknown worker {worker_id}, suggesting registration")

                    # 返回一个特殊响应提示重新注册
                    return False
            return True
        except Exception as e:
            logger.error(f"Error processing heartbeat from {worker_id}: {e}")
            return False

    async def check_workers_health(self, timeout: int = 60) -> List[str]:
        """
        检查Worker节点健康状态，移除超时的节点 (异步版本)
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            List[str]: 已移除worker的ID列表
        """
        current_time = time.time()

        # 获取待移除的worker列表
        workers_to_remove = []

        async with self.worker_lock:  # 使用异步锁
            for worker_id, worker in list(self.workers.items()):
                time_since_heartbeat = current_time - worker.last_heartbeat

                if time_since_heartbeat > timeout:
                    logger.warning(f"Worker {worker_id} timed out ({time_since_heartbeat:.1f}s without heartbeat), removing")
                    workers_to_remove.append(worker_id)
                elif time_since_heartbeat > timeout * 0.7:  # 超过70%超时时间发出警告
                    logger.warning(f"Worker {worker_id} may be unhealthy, no heartbeat for {time_since_heartbeat:.1f}s")

            # 移除超时的worker
            for worker_id in workers_to_remove:
                worker_info = self.workers.pop(worker_id)
                logger.info(f"Removed worker {worker_id} from {worker_info.address}:{worker_info.port}")

                # 更新agent位置
                for agent_id in worker_info.agent_ids:
                    if agent_id in self.agent_locations and self.agent_locations[agent_id] == worker_id:
                        self.agent_locations.pop(agent_id)
                        logger.warning(f"Agent {agent_id} is no longer available (worker {worker_id} removed)")

        return workers_to_remove

    def find_worker_for_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Find the worker node hosting a specific agent"""
        if agent_id not in self.agent_locations:
            return None

        worker_id = self.agent_locations[agent_id]
        if worker_id not in self.workers:
            return None

        worker = self.workers[worker_id]
        return {
            "worker_id": worker.worker_id,
            "address": worker.address,
            "port": worker.port
        }

        # Check if agent exists locally on master
        if hasattr(self, "agents") and agent_id in self.agents:
            return {
                "worker_id": self.node_id,
                "address": "localhost",
                "port": 0  # Not applicable for local access
            }

        return None

    async def shutdown(self):
        """Clean shutdown of master node resources"""
        logger.info(f"Shutting down master node {self.node_id}")
        self.shutting_down = True

        # 取消健康检查任务
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()

        # 发送终止信号给所有worker
        async with self.worker_lock:
            for worker_id, worker in self.workers.items():
                try:
                    logger.info(f"Sending shutdown signal to worker {worker_id}")
                    await self._grpc_module.send_termination_to_worker(
                        worker.address, worker.port, "master_shutdown"
                    )
                except Exception as e:
                    logger.error(f"Error sending shutdown to worker {worker_id}: {e}")

        # 停止gRPC服务器
        if self._server:
            await self._server.stop(10)  # 10秒优雅停止

        # 调用父类shutdown方法
        await super().shutdown()

        logger.info(f"Master node {self.node_id} shutdown complete")