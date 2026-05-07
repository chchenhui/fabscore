import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
import uuid
import time
from datetime import datetime

from onesim.distribution.node import get_node, NodeRole
from onesim.distribution import grpc_impl
from onesim.events import Event, DataEvent, DataResponseEvent, DataUpdateEvent, DataUpdateResponseEvent, get_event_bus
from onesim.events import EndEvent
from onesim.distribution.grpc_impl import run_async_safely
from onesim.distribution.batch_processor import batch_processor

class ProxyEnv:
    """
    A proxy environment for worker nodes in distributed mode.
    
    This class mimics the interface of the simulation environment but forwards
    operations to the master node. This allows agents to work with
    the same interface regardless of whether they're in single-node or distributed mode.
    """
    
    def __init__(self, node_id: str, master_address: str, master_port: int, 
                 batch_size: int = 50, flush_interval: float = 10.0):
        """
        Initialize the proxy environment.
        
        Args:
            node_id: The ID of this worker node
            master_address: Address of the master node
            master_port: Port of the master node
            batch_size: Maximum number of items to batch before sending
            flush_interval: Seconds between forced batch flushes
        """
        self.node_id = node_id
        self.master_address = master_address
        self.master_port = master_port
        self.trail_id = None  # Will be set after connecting to master
        self._running = False
        self._tasks = []
        self._queue = asyncio.Queue()
        self._event_schema = {}
        self._data_futures: Dict[str, asyncio.Future] = {}
        self._data_update_futures: Dict[str, asyncio.Future] = {}
        self._sync_event = asyncio.Event()
        
        # 使用全局BatchProcessor替代内部批处理实现
        # 配置BatchProcessor
        batch_processor.batch_size = batch_size
        batch_processor.max_wait_time = flush_interval
        batch_processor.start(master_address, master_port)
        
        # Create a unique ID for this proxy environment
        self.env_id = f"{node_id}_ENV"
        
        # Register event handlers
        self.register_event("DataEvent", "handle_data_event")
        self.register_event("DataResponseEvent", "handle_data_response")
        self.register_event("DataUpdateEvent", "handle_data_update_event")
        self.register_event("DataUpdateResponseEvent", "handle_data_update_response")
        self.register_event("AgentDataByTypeEvent", "handle_agent_data_by_type_event")
        self.register_event("EndEvent", "handle_end_event")
        
        logger.info(f"ProxyEnv initialized for worker {node_id} connecting to {master_address}:{master_port} (batch_size={batch_size}, flush_interval={flush_interval}s)")
    
    def register_event(self, event_kind: str, method_name: str) -> None:
        """
        Register an event handler method for a specific event kind.
        
        Args:
            event_kind: The kind of event to register for
            method_name: The name of the method to call when this event is received
        """
        if event_kind not in self._event_schema:
            self._event_schema[event_kind] = []
        self._event_schema[event_kind].append(method_name)
    
    def add_event(self, event: Event) -> None:
        """
        Add an event to this environment's processing queue.
        
        Args:
            event: The event to process
        """
        self._queue.put_nowait(event)
    
    # Support both synchronous and asynchronous calls
    async def queue_event(self, event_data: Dict[str, Any]) -> None:
        """
        Queue an event for storage in the master's database.
        Handles being called from sync or async contexts.
        
        Args:
            event_data: The event data to store
        """
        try:
            # Check if we are inside an already running event loop
            loop = asyncio.get_running_loop()
            # If yes, schedule the task on the running loop
            await batch_processor.add_storage_event(event_data)
        except RuntimeError:
            # If no loop is running, we are in a synchronous context
            # Use run_async_safely to run the async function
            await batch_processor.add_storage_event(event_data)
        except Exception as e:
            # Catch other potential errors during task creation or execution
            logger.error(f"Error in queue_event: {e}")
    
    # Support both synchronous and asynchronous calls
    async def queue_decision(self, decision_data: Dict[str, Any]) -> None:
        """
        Queue a decision record for storage in the master's database.
        Handles being called from sync or async contexts.
        
        Args:
            decision_data: The decision data to store
        """
        try:
            # Check if we are inside an already running event loop
            loop = asyncio.get_running_loop()
            # If yes, schedule the task on the running loop
            await batch_processor.add_decision_record(decision_data)
        except RuntimeError:
            # If no loop is running, we are in a synchronous context
            # Use run_async_safely to run the async function
            await batch_processor.add_decision_record(decision_data)
        except Exception as e:
            # Catch other potential errors during task creation or execution
            logger.error(f"Error in queue_decision: {e}")
    
    async def handle_data_event(self, event: DataEvent) -> DataResponseEvent:
        """
        Handle a data access event.
        
        Args:
            event: The data access event
            
        Returns:
            The response event
        """
        try:
            # Forward the request to the master environment
            data_value = await grpc_impl.get_env_data_from_master(
                self.master_address,
                self.master_port,
                event.key,
                event.default
            )
            
            # Create response event
            response = DataResponseEvent(
                from_agent_id=self.env_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                data_value=data_value,
                success=True,
                parent_event_id=event.event_id
            )
            
            return response
        except Exception as e:
            logger.error(f"Error getting data from master: {e}")
            
            # Create error response
            error_response = DataResponseEvent(
                from_agent_id=self.env_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                data_value=None,
                success=False,
                error=str(e),
                parent_event_id=event.event_id
            )
            
            return error_response
    
    async def handle_data_response(self, event: DataResponseEvent) -> None:
        """
        Handle data response events.
        
        Args:
            event: The data response event
        """
        # Check if we're waiting for this response
        if hasattr(self, '_data_futures') and event.request_id in self._data_futures:
            future = self._data_futures.pop(event.request_id)
            
            if not future.done():
                if event.success:
                    future.set_result(event.data_value)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))
            
            # Signal that we've received a response
            self._sync_event.set()
    
    async def handle_data_update_event(self, event: DataUpdateEvent) -> DataUpdateResponseEvent:
        """
        Handle a data update event.
        
        Args:
            event: The data update event
            
        Returns:
            The response event
        """
        try:
            # Forward the request to the master environment
            success = await grpc_impl.update_env_data_on_master(
                self.master_address,
                self.master_port,
                event.key,
                event.value
            )
            
            # Create response event
            response = DataUpdateResponseEvent(
                from_agent_id=self.env_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=success,
                parent_event_id=event.event_id
            )
            
            return response
        except Exception as e:
            logger.error(f"Error updating data on master: {e}")
            
            # Create error response
            error_response = DataUpdateResponseEvent(
                from_agent_id=self.env_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=False,
                error=str(e),
                parent_event_id=event.event_id
            )
            
            return error_response
    
    async def handle_data_update_response(self, event: DataUpdateResponseEvent) -> None:
        """
        Handle data update response events.
        
        Args:
            event: The data update response event
        """
        # Check if we're waiting for this response
        if hasattr(self, '_data_update_futures') and event.request_id in self._data_update_futures:
            future = self._data_update_futures.pop(event.request_id)
            
            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))
            
            # Signal that we've received a response
            self._sync_event.set()
    
    async def handle_agent_data_by_type_event(self, event: Event) -> Event:
        """
        Handle a request for agent data by type.
        
        Args:
            event: The event containing the request
            
        Returns:
            The response event
        """
        try:
            # Forward the request to the master environment
            agent_type = event.agent_type
            key = event.key
            default = event.default
            
            values = await grpc_impl.get_agent_data_by_type_from_master(
                self.master_address,
                self.master_port,
                agent_type,
                key,
                default
            )
            
            # Create response event
            response = Event(
                from_agent_id=self.env_id,
                to_agent_id=event.from_agent_id,
                event_kind="DataResponseEvent",
                request_id=event.request_id,
                success=True,
                data_value=values,
                parent_event_id=event.event_id
            )
            
            return response
        except Exception as e:
            logger.error(f"Error getting agent data by type from master: {e}")
            
            # Create error response
            error_response = Event(
                from_agent_id=self.env_id,
                to_agent_id=event.from_agent_id,
                event_kind="DataResponseEvent",
                request_id=event.request_id,
                success=False,
                error=str(e),
                parent_event_id=event.event_id
            )
            
            return error_response
    
    async def handle_end_event(self, event: EndEvent) -> None:
        """
        Handle termination event.
        
        Args:
            event: The termination event
        """
        try:
            logger.info(f"Received termination event: {event.reason}")
            self._running = False
            
            # Cancel any running tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            await grpc_impl.request_simulation_stop(
                self.master_address,
                self.master_port,
                self.node_id
            )
        except Exception as e:
            logger.error(f"Error handling termination event: {e}")
    
    async def get_data(self, key: str, default: Any = None) -> Any:
        """
        Get data from the master environment.
        
        Args:
            key: The data key to retrieve
            default: Default value if data not found
            
        Returns:
            The requested data or default value
        """
        try:
            return await grpc_impl.get_env_data_from_master(
                self.master_address,
                self.master_port,
                key,
                default
            )
        except Exception as e:
            logger.error(f"Error getting data from master: {e}")
            return default
    
    async def update_data(self, key: str, value: Any) -> bool:
        """
        Update data in the master environment.
        
        Args:
            key: The data key to update
            value: The new value
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            return await grpc_impl.update_env_data_on_master(
                self.master_address,
                self.master_port,
                key,
                value
            )
        except Exception as e:
            logger.error(f"Error updating data on master: {e}")
            return False
    
    async def get_agent_data(self, agent_id: str, key: str, default: Optional[Any] = None) -> Any:
        """
        Get data from a specific agent through the master node.
        
        Args:
            agent_id: ID of the agent to query
            key: The data key to retrieve
            default: Default value if data not found
            
        Returns:
            The requested data or default value
        """
        try:
            # Forward the request to the master environment
            return await grpc_impl.get_agent_data_from_master(
                self.master_address,
                self.master_port,
                agent_id,
                key,
                default
            )
        except Exception as e:
            logger.error(f"Error getting agent data from master: {e}")
            return default
    
    async def get_agent_data_by_type(self, agent_type: str, key: str, default: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get data from agents of a specific type through the master node.
        
        Args:
            agent_type: Type of agents to query
            key: The data key to retrieve
            default: Default value if data not found
            
        Returns:
            Dictionary of agent IDs to their data values
        """
        try:
            # Forward the request to the master environment
            return await grpc_impl.get_agent_data_by_type_from_master(
                self.master_address,
                self.master_port,
                agent_type,
                key,
                default
            )
        except Exception as e:
            logger.error(f"Error getting agent data by type from master: {e}")
            return {}
    
    async def stop_simulation(self) -> None:
        """Request simulation termination on the master node."""
        try:
            self._running = False
            
            # Cancel any running tasks
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            # 请求停止模拟
            await grpc_impl.request_simulation_stop(
                self.master_address,
                self.master_port,
                self.node_id
            )
            
            # 执行资源清理
            await self.shutdown()
            
        except Exception as e:
            logger.error(f"Error requesting simulation stop: {e}")
    
    def set_trail_id(self, trail_id: str) -> None:
        """Set the trail ID for data storage."""
        self.trail_id = trail_id
        logger.info(f"Trail ID set to {trail_id}")
    
    async def handle_event(self, event: Event) -> List[Event]:
        """
        Handle an event received.
        
        Args:
            event: The event to handle
            
        Returns:
            List of response events
        """
        responses = []
        
        # Check if we have a handler for this event kind
        if event.event_kind in self._event_schema:
            for method_name in self._event_schema[event.event_kind]:
                method = getattr(self, method_name, None)
                if callable(method):
                    try:
                        response = await method(event)
                        if response is not None:
                            if isinstance(response, list):
                                responses.extend(response)
                            else:
                                responses.append(response)
                    except Exception as e:
                        logger.error(f"Error handling event {event.event_kind} with method {method_name}: {e}")
        
        return responses
    
    async def run_task(self):
        """Process events from the queue."""
        while self._running:
            try:
                event = await self._queue.get()
                
                # Process the event
                responses = await self.handle_event(event)
                
                # Send responses to the event bus
                event_bus = get_event_bus()
                for response in responses:
                    await event_bus.put(response)
                
                self._queue.task_done()
            except asyncio.CancelledError:
                logger.info("ProxyEnv task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in ProxyEnv run_task: {e}")
    
    async def run(self) -> List[asyncio.Task]:
        """
        Start the proxy environment. This method mimics the interface of BasicSimEnv.run().
        
        Returns:
            List of tasks that should be awaited by the caller
        """
        self._running = True
        
        # Create a task for processing events
        run_task = asyncio.create_task(self.run_task())
        self._tasks = [run_task]
        
        logger.info(f"ProxyEnv for worker {self.node_id} started")
        return self._tasks
    
    def set_batch_config(self, batch_size: Optional[int] = None, flush_interval: Optional[float] = None) -> None:
        """
        Update batch processing configuration.
        
        Args:
            batch_size: Maximum number of items to batch before sending, or None to keep current setting
            flush_interval: Seconds between forced batch flushes, or None to keep current setting
        """
        # 更新全局BatchProcessor的配置
        if batch_size is not None and batch_size > 0:
            batch_processor.batch_size = batch_size
        
        if flush_interval is not None and flush_interval > 0:
            batch_processor.max_wait_time = flush_interval
            
        logger.info(f"Updated batch config for worker {self.node_id}: batch_size={batch_processor.batch_size}, flush_interval={batch_processor.max_wait_time}s")
    
    async def shutdown(self):
        """Gracefully shutdown the proxy environment"""
        logger.info(f"Shutting down ProxyEnv for worker {self.node_id}")
        
        self._running = False
        
        # 取消所有任务
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling task: {e}")
        
        # BatchProcessor不需要在这里停止，它将由WorkerNode负责停止
        
        logger.info(f"ProxyEnv for worker {self.node_id} shutdown complete")
