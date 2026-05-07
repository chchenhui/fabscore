import asyncio
import threading
import copy
from .event import Event
from typing import Dict, List, Union, Optional, Any
from loguru import logger
from onesim.distribution.node import get_node, NodeRole
import time
import uuid
from collections import defaultdict
from onesim.utils.work_graph import WorkGraph
from onesim.distribution import grpc_impl

class EventBus:
    """事件总线，负责处理所有事件的分发"""
    
    def __init__(self):
        self.listeners = {}
        self.agent_registry = {}
        self.queue = asyncio.Queue()
        self._running = False
        self._paused = False
        self._pause_event = asyncio.Event()  # Initially not set (not paused)
        self._is_distributed = False
        self._master_node = None
        self._worker_node = None
        self._pending_tasks = []
        self._lock_registry = {}  # Registry to track active locks: {lock_id: {holder: node_id, expiry: timestamp}}
        self._lock_requests = {}  # Track pending lock requests: {lock_id: [waiting_nodes]}
        self._lock_mutex = asyncio.Lock()  # Async lock for lock operations
        
        # Event tracking for workflow visualization
        self._event_sequence = []  # List of (timestamp, event_kind, from_agent_id, to_agent_id, event_id) tuples
        self._event_flows = defaultdict(list)  # Tracks flows starting from StartEvents
        self._event_to_flow = {}  # Maps event_id to the flow_id it belongs to
        self._agent_types = {}  # Cache of agent types for logging: {agent_id: agent_type}
        self._event_mutex = asyncio.Lock()  # Mutex for event tracking operations
        
    def setup_distributed(self, node):
        """配置为分布式模式"""
        from onesim.distribution.node import NodeRole
        
        self._is_distributed = True
        if node.role == NodeRole.MASTER:
            self._master_node = node
        elif node.role == NodeRole.WORKER:
            self._worker_node = node
        
        logger.info(f"EventBus configured for distributed mode as {node.role.value}")
    
    async def dispatch_event(self, event: Event) -> None:
        """分发事件，如果是分布式模式，根据代理位置选择本地或远程分发"""
        # Track the event for workflow visualization
        
        
        # Special handling for lock-related events
        if event.event_kind in ["AcquireLockRequest", "ReleaseLockRequest", "AcquireLockResponse", "ReleaseLockResponse"]:
            await self._handle_lock_event(event)
            return
        # Special handling for global termination events
        if event.event_kind == "EndEvent" and event.to_agent_id == "all":
            # Put the event in the local queue first
            await self.queue.put(event)
            
            # In distributed mode, we need to forward to all other nodes
            if self._is_distributed:
                if self._master_node:
                    # Master needs to forward to all workers
                    for worker_id, worker in self._master_node.workers.items():
                        logger.info(f"Forwarding termination event to worker {worker_id}")
                        await self._master_node._grpc_module.send_event_to_worker(
                            worker.address, 
                            worker.port,
                            event
                        )
                elif self._worker_node:
                    # Worker needs to forward to master (who will distribute to other workers)
                    # logger.info("Forwarding termination event to master")
                    # MODIFIED: await the task
                    # await self._worker_node._grpc_module.send_event_to_master(
                    #     self._worker_node.master_address,
                    #     self._worker_node.master_port,
                    #     event
                    # )
                    # For global termination from worker, it should still primarily go to its local queue
                    # The master receiving it will then propagate.
                    # However, if a worker directly initiates a global stop, it should also notify master.
                    # This path is complex. Current logic: worker puts to local queue, master receives, master propagates.
                    # Let's assume worker's EndEvent to "all" is picked by its _process_event,
                    # which then sends to master if it's not a local agent.
                    # The current code for worker here seems to be for if event bus itself generates it.
                    # The logic for worker propagating "all" events to master is better handled in _process_event or a dedicated method.
                    # For now, if worker's dispatch_event gets an EndEvent to "all", it relies on its own queue processing.
                    await self.queue.put(event) # Worker still puts to its queue.
            return
            
        if not self._is_distributed:
            # 单机模式，直接放入队列
            await self.queue.put(event)
            return

        # 分布式模式下的处理
        if self._master_node:
            # Master节点上的处理
            to_agent_id = str(event.to_agent_id)
            
            # 确定目标代理的位置
            local_target = False
            remote_worker_id = None
            
            if to_agent_id == "ENV":  # 环境总是在Master上
                local_target = True
            else:
                # 查找代理位置
                worker_id = self._master_node.agent_locations.get(to_agent_id)
                if not worker_id:
                    # 如果找不到位置信息，假定代理在本地
                    local_target = True
                else:
                    # 远程代理
                    remote_worker_id = worker_id
            
            # 处理本地目标
            if local_target:
                await self.queue.put(event)
            
            # 处理远程目标
            elif remote_worker_id:
                worker = self._master_node.workers.get(remote_worker_id)
                if not worker:
                    logger.error(f"Worker {remote_worker_id} not found for event forwarding")
                    return
                    
                # MODIFIED: await the task
                await self._master_node._grpc_module.send_event_to_worker(
                    worker.address, 
                    worker.port,
                    event
                )
                
        elif self._worker_node:
            # Worker节点上的处理
            # 1. 检查事件是本地处理还是需要转发
            to_agent_id = str(event.to_agent_id)
            # 2. 如果目标不是本地代理，需要转发到Master
            if to_agent_id not in self.agent_registry:
                 # MODIFIED: await the task
                await self._worker_node._grpc_module.send_event_to_master(
                    self._worker_node.master_address,
                    self._worker_node.master_port,
                    event
                )
            else:
                await self.queue.put(event)
    
    async def _track_event(self, event: Event) -> None:
        """
        Track an event for workflow visualization without modifying the event
        
        Args:
            event (Event): The event to track
        """
        # Skip tracking for Data-related events
        if event.event_kind in ['DataEvent', 'DataResponseEvent', 'DataUpdateEvent', 'DataUpdateResponseEvent']:
            return
        
        async with self._event_mutex:
            # Use event's existing timestamp and ID instead of generating new ones
            current_time = getattr(event, 'timestamp', time.time())
            event_id = getattr(event, 'event_id', str(uuid.uuid4()))
            
            # Get agent types if available
            from_agent_type = self._get_agent_type(event.from_agent_id)
            to_agent_type = self._get_agent_type(event.to_agent_id)
            
            # Save event in sequence
            event_record = (current_time, event.event_kind, event.from_agent_id, event.to_agent_id, event_id, from_agent_type, to_agent_type)
            self._event_sequence.append(event_record)
            
            # Handle flow tracking
            if event.event_kind=="StartEvent":
                # This is a start event - create a new flow
                flow_id = f"flow_{event_id}"
                self._event_flows[flow_id].append(event_record)
                # Register this event as being in this flow
                self._event_to_flow[event_id] = flow_id
            elif hasattr(event, 'parent_event_id') and event.parent_event_id:
                # This event has a parent event, check if parent is part of a flow
                parent_flow_id = self._event_to_flow.get(event.parent_event_id)
                if parent_flow_id:
                    # Add to parent's flow
                    self._event_flows[parent_flow_id].append(event_record)
                    # Register this event as being in this flow
                    self._event_to_flow[event_id] = parent_flow_id
                else:
                    # Create new flow if parent not found in any flow
                    flow_id = f"flow_{event_id}"
                    self._event_flows[flow_id].append(event_record)
                    self._event_to_flow[event_id] = flow_id
            else:
                # No parent event, create a new flow
                flow_id = f"flow_{event_id}"
                self._event_flows[flow_id].append(event_record)
                self._event_to_flow[event_id] = flow_id
    
    def _get_agent_type(self, agent_id: str) -> str:
        """
        Get the agent type for an agent ID, caching results
        
        Args:
            agent_id (str): Agent ID to look up
            
        Returns:
            str: Agent type or "Unknown"
        """
        # Check cache first
        if agent_id in self._agent_types:
            return self._agent_types[agent_id]
            
        # Special cases
        if agent_id == "SimEnv" or agent_id == "ENV":
            self._agent_types[agent_id] = "ENV"
            return "ENV"
            
        # Try to find in registry
        for node_id, node_data in self.agent_registry.items():
            if node_id == agent_id and hasattr(node_data, "profile") and hasattr(node_data.profile, "agent_type"):
                agent_type = node_data.profile.agent_type
                self._agent_types[agent_id] = agent_type
                return agent_type
                
        # If we can't determine, return unknown
        self._agent_types[agent_id] = "Unknown"
        return "Unknown"
    
    async def log_event_flows(self) -> None:
        """
        Log all tracked event flows at the end of simulation
        """
        async with self._event_mutex:
            logger.info("======== Event Flow Visualization ========")
            
            if not self._event_flows:
                logger.info("No event flows were tracked during this simulation.")
                return
                
            flow_count = 0
            for flow_id, events in self._event_flows.items():
                flow_count += 1
                # Only log flows with more than one event
                # if len(events) <= 1:
                #     continue
                    
                # Find first event in flow
                initial_event = events[0]
                _, event_kind, from_agent_id, to_agent_id, _, from_agent_type, to_agent_type = initial_event
                
                if event_kind == "StartEvent":
                    from_agent_id="ENV"
                    from_agent_type="EnvAgent"
                logger.info(f"Flow #{flow_count}: Starting with {event_kind} from {from_agent_id}({from_agent_type}) to {to_agent_id}({to_agent_type})")
                
                # Build a sequential representation of the flow
                for i, event_data in enumerate(events):
                    _, event_kind, from_agent_id, to_agent_id, _, from_agent_type, to_agent_type = event_data
                    logger.info(f"  Step {i+1}: {event_kind}: {from_agent_id}({from_agent_type}) -> {to_agent_id}({to_agent_type})")
                
                logger.info("----------------------------------------")
            
            logger.info(f"Total event flows: {flow_count}")
            logger.info("======== End Event Flow Visualization ========")
    
    async def _handle_lock_event(self, event: Event) -> None:
        """
        Handle lock-related events (acquire/release requests and responses)
        
        Args:
            event (Event): The lock event to handle
        """
        if not self._is_distributed or not self._master_node:
            # Only master handles lock events in distributed mode
            logger.warning("Lock events can only be handled by the master node in distributed mode")
            return
            
        if event.event_kind == "AcquireLockRequest":
            # Handle lock acquisition
            lock_id = event.lock_id
            node_id = event.node_id
            timeout = event.timeout
            
            # Process lock request
            async with self._lock_mutex:
                current_time = time.time()
                
                # Check if lock exists and is still valid
                if lock_id in self._lock_registry:
                    lock_info = self._lock_registry[lock_id]
                    
                    # If lock has expired, clean it up
                    if current_time > lock_info['expiry']:
                        del self._lock_registry[lock_id]
                    else:
                        # Lock is already held by someone
                        if lock_info['holder'] == node_id:
                            # Renew the lock for this holder
                            lock_info['expiry'] = current_time + timeout
                            
                            # Send success response
                            response = {
                                'event_kind': 'AcquireLockResponse',
                                'lock_id': lock_id,
                                'node_id': node_id,
                                'success': True
                            }
                            await self._send_lock_response(response, event.from_agent_id)
                            return
                        else:
                            # Lock is held by someone else
                            # Add to waiting queue
                            if lock_id not in self._lock_requests:
                                self._lock_requests[lock_id] = []
                            self._lock_requests[lock_id].append({
                                'node_id': node_id,
                                'from_agent_id': event.from_agent_id,
                                'request_time': current_time
                            })
                            
                            # Send failure response
                            response = {
                                'event_kind': 'AcquireLockResponse',
                                'lock_id': lock_id,
                                'node_id': node_id,
                                'success': False
                            }
                            await self._send_lock_response(response, event.from_agent_id)
                            return
                
                # Lock is available, grant it
                self._lock_registry[lock_id] = {
                    'holder': node_id,
                    'expiry': current_time + timeout
                }
                
                # Send success response
                response = {
                    'event_kind': 'AcquireLockResponse',
                    'lock_id': lock_id,
                    'node_id': node_id,
                    'success': True
                }
                await self._send_lock_response(response, event.from_agent_id)
                
        elif event.event_kind == "ReleaseLockRequest":
            # Handle lock release
            lock_id = event.lock_id
            node_id = event.node_id
            
            async with self._lock_mutex:
                success = False
                
                # Check if lock exists and is held by the requesting node
                if lock_id in self._lock_registry:
                    lock_info = self._lock_registry[lock_id]
                    
                    if lock_info['holder'] == node_id:
                        # Release the lock
                        del self._lock_registry[lock_id]
                        success = True
                        
                        # Check if anyone is waiting for this lock
                        if lock_id in self._lock_requests and self._lock_requests[lock_id]:
                            # Get the next waiting node
                            next_requester = self._lock_requests[lock_id].pop(0)
                            next_node_id = next_requester['node_id']
                            next_agent_id = next_requester['from_agent_id']
                            
                            # Grant lock to next waiting node
                            current_time = time.time()
                            self._lock_registry[lock_id] = {
                                'holder': next_node_id,
                                'expiry': current_time + 30.0  # Default timeout
                            }
                            
                            # Send notification to the next node
                            notification = {
                                'event_kind': 'AcquireLockResponse',
                                'lock_id': lock_id,
                                'node_id': next_node_id,
                                'success': True
                            }
                            await self._send_lock_response(notification, next_agent_id)
                            
                            # Clean up empty request lists
                            if not self._lock_requests[lock_id]:
                                del self._lock_requests[lock_id]
                
                # Send release response
                response = {
                    'event_kind': 'ReleaseLockResponse',
                    'lock_id': lock_id,
                    'node_id': node_id,
                    'success': success
                }
                await self._send_lock_response(response, event.from_agent_id)
    
    async def _send_lock_response(self, response_data: Dict[str, Any], to_agent_id: str) -> None:
        """
        Send a lock response to the appropriate node
        
        Args:
            response_data: Lock response data
            to_agent_id: Target agent ID
        """
        from onesim.events import Event
        
        # Create response event
        response_event = Event(
            event_kind=response_data['event_kind'],
            from_agent_id='master',
            to_agent_id=to_agent_id,
            **response_data
        )
        
        # Find the target worker
        if self._master_node:
            worker_id = self._master_node.agent_locations.get(to_agent_id)
            if worker_id:
                worker = self._master_node.workers.get(worker_id)
                if worker:
                    # Send to worker
                    await self._master_node._grpc_module.send_event_to_worker(
                        worker.address,
                        worker.port,
                        response_event
                    )
                    return
            
            # If we couldn't find the worker or agent, queue locally (might be a local agent)
            await self.queue.put(response_event)
        
    def stop(self):
        self._running = False
    
    async def pause(self):
        """Pause the event bus processing"""
        # 如果已经暂停，直接返回成功
        if self._paused:
            logger.debug("Event bus is already paused")
            return True
            
        logger.info("Pausing event bus")
        self._paused = True
        self._pause_event.set()  # Set when paused (more intuitive)
        
        # Propagate pause to agents in distributed mode
        if self._is_distributed:
            # Create a pause event with the proper constructor
            pause_event = Event(
                from_agent_id="EventBus",
                to_agent_id="all",
                event_kind="PauseEvent"
            )
            await self.dispatch_event(pause_event)
        
        logger.info("Event bus paused")
        return True
        
    async def resume(self):
        """Resume the event bus processing"""
        # 如果没有暂停，直接返回成功
        if not self._paused:
            logger.debug("Event bus is not paused")
            return True
            
        logger.info("Resuming event bus")
        self._paused = False
        self._pause_event.clear()  # Clear when not paused (more intuitive)
        
        # Propagate resume to agents in distributed mode
        if self._is_distributed:
            # Create a resume event with the proper constructor
            resume_event = Event(
                from_agent_id="EventBus",
                to_agent_id="all",
                event_kind="ResumeEvent"
            )
            await self.dispatch_event(resume_event)
        
        logger.info("Event bus resumed")
        return True
    
    def is_paused(self) -> bool:
        """Check if the event bus is paused"""
        return self._paused
    
    def is_empty(self) -> bool:
        return self.queue.empty()

    def register_event(self, event_kind: str, agent: Any) -> None:
        if event_kind in self.listeners:
            self.listeners[event_kind].append(agent)
        else:
            self.listeners[event_kind] = [agent]

    def register_agent(self, agent_id: str, agent: Any) -> None:
        self.agent_registry[agent_id] = agent
        
        # If agent has a profile with agent_type, cache it for event tracking
        if hasattr(agent, "profile") and hasattr(agent.profile, "agent_type"):
            self._agent_types[agent_id] = agent.profile.agent_type

    def is_stopped(self):
        return not self._running

    async def run(self):
        """
        开始处理事件队列。该方法会一直运行，直到调用 stop 方法。
        在分布式环境中，Master 和 Worker 都需要运行自己的事件总线。
        """
        self._running = True
        logger.info("Event bus started")
        
        try:
            while self._running:
                try:
                    # Check if paused and wait if needed
                    if self._paused:
                        # 每隔一段较长时间才记录一次日志，避免刷屏
                        if not hasattr(self, "_last_pause_log_time") or time.time() - self._last_pause_log_time > 10.0:
                            logger.debug("Event bus is paused, waiting to resume")
                            self._last_pause_log_time = time.time()
                            
                        # Wait for pause to be cleared (resume called)
                        await asyncio.sleep(0.2)
                        continue
                    
                    # 从队列获取事件
                    # try:
                    #     # 使用 10 秒超时
                    #     queue_size = self.queue.qsize()
                    #     timeout = 1.0 if queue_size > 0 else 10.0
                    #     event = await asyncio.wait_for(self.queue.get(), timeout)
                    # except asyncio.TimeoutError:
                    #     logger.warning("EventBus: Timeout waiting for event, continuing...")
                    #     continue
                    event = await self.queue.get()
                    if event.event_kind == "EndEvent" and event.to_agent_id == "all":
                        logger.info("Received global termination event")
                        self._running = False
                        self.queue.task_done()
                        break
                    
                    # Special handling for pause/resume events
                    if event.event_kind == "PauseEvent" and event.to_agent_id == "all":
                        logger.info("Received global pause event")
                        self._paused = True
                        self._pause_event.set()  # Set when paused
                        self.queue.task_done()
                        continue
                        
                    if event.event_kind == "ResumeEvent" and event.to_agent_id == "all":
                        logger.info("Received global resume event")
                        self._paused = False
                        self._pause_event.clear()  # Clear when not paused
                        self.queue.task_done()
                        continue
                    
                    # 处理事件
                    await self._process_event(event)
                    self.queue.task_done()
                    
                except asyncio.CancelledError:
                    # 任务被取消
                    logger.info("Event processing cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error processing event: {e}")
                    logger.exception(e)
        finally:
            # Log event flows before shutting down
            await self.log_event_flows()
            logger.info("Event bus stopped")
            
    async def _process_event(self, event: Event):
        """处理单个事件"""
        # Check if paused - only process control events during pause
        if self._paused and event.event_kind not in ["PauseEvent", "ResumeEvent", "EndEvent"]:
            # Re-queue non-control events when paused but log less frequently
            if not hasattr(self, "_last_requeue_log_time") or time.time() - self._last_requeue_log_time > 10.0:
                logger.debug(f"EventBus paused, re-queuing events")
                self._last_requeue_log_time = time.time()
                
            await self.queue.put(event)
            return
            
        # Special handling for lock responses
        if event.event_kind in ["AcquireLockResponse", "ReleaseLockResponse"]:
            # Find the agent this is for
            agent_id = event.to_agent_id
            agent = self.agent_registry.get(agent_id)
            if agent:
                try:
                    agent.add_event(event)
                except Exception as e:
                    logger.error(f"Error handling lock response by agent {agent_id}: {e}")
            return
            
        # Special handling for termination events
        if event.event_kind == "EndEvent" and event.to_agent_id == "all":
            logger.info("Processing global termination event")
            
            # Send to all local agents
            for agent_id, agent in self.agent_registry.items():
                if agent_id != "ENV":  # Skip the simulation environment itself
                    try:
                        # Create a copy of the event specifically for this agent
                        agent_event = copy.deepcopy(event)
                        agent_event.to_agent_id = agent_id
                        agent.add_event(agent_event)
                        logger.debug(f"Sent termination event to agent {agent_id}")
                    except Exception as e:
                        logger.error(f"Error sending termination event to agent {agent_id}: {e}")
            
            # Set running flag to false
            self._running = False
            return
            
        await self._track_event(event)
        # 获取目标代理ID
        to_agent_id = str(event.to_agent_id)
        current_node = get_node()

        # Specific handling for Master node when an event is pulled from its local queue
        if self._is_distributed and current_node and current_node.role == NodeRole.MASTER:
            registered_entity = self.agent_registry.get(to_agent_id)

            if isinstance(registered_entity, dict):
                # This agent_id is registered as a dict (placeholder for a remote agent).
                # This means the event was incorrectly routed to the Master's local queue
                # instead of being reliably forwarded to the designated worker.
                logger.warning(
                    f"Master EventBus._process_event: Event for remote agent {to_agent_id} (ID: {event.event_id}, Kind: {event.event_kind}) "
                    f"was found in local queue (agent registered as dict). Attempting to forward to worker."
                )
                # self._master_node is the MasterNode instance associated with this EventBus
                if self._master_node and hasattr(self._master_node, 'forward_event'):
                    # MasterNode.forward_event will use agent_locations to find the worker
                    # and then use gRPC to send the event.
                    forward_success = await self._master_node.forward_event(event)
                    if not forward_success:
                        logger.error(
                            f"Master EventBus._process_event: Forwarding attempt for event to remote agent {to_agent_id} (ID: {event.event_id}) failed."
                        )
                else:
                    logger.error(
                        f"Master EventBus._process_event: Cannot forward event for remote agent {to_agent_id} (ID: {event.event_id}). "
                        f"MasterNode reference or forward_event method missing."
                    )
                return # Attempted to handle by forwarding; do not proceed to generic local dispatch logic.
            # If registered_entity is not a dict (e.g., it's a real local handler like SimEnv, or None),
            # it means this event is genuinely for a service on the Master, so fall through.
        
        # 分布式环境中的处理 (Worker Node specific logic from original code)
        # NOTE: This block's necessity is reduced if Worker's dispatch_event is corrected
        # to not queue events that it forwards to Master.
        if self._is_distributed and self._worker_node:
            # This EventBus instance is running on a WORKER node.
            agent_on_this_worker = to_agent_id in self.agent_registry

            if agent_on_this_worker:
                # Event is for a local agent on this worker.
                pass # Fall through to local processing logic below.
            elif to_agent_id == "ENV":
                # Event is for the Environment, send to Master.
                # logger.debug(f"EventBus (Worker): Routing event {event.event_id} ({event.event_kind}) for ENV to Master.")
                try:
                    proto_event = grpc_impl.event_to_proto(event)
                    # Use the existing send_event_to_master which uses connection_manager
                    await grpc_impl.send_event_to_master(
                        self._worker_node.master_address,
                        self._worker_node.master_port,
                        event # send_event_to_master expects the event object, not proto_event
                    )
                except Exception as e:
                    logger.error(f"EventBus (Worker): Error sending ENV event to Master: {e}")
                return # Event sent to Master, no further local processing.
            else:
                # Event is for a remote agent, try P2P.
                # logger.debug(f"EventBus (Worker): Event {event.event_id} ({event.event_kind}) to remote agent {to_agent_id}. Attempting P2P.")
                target_worker_location = await self._worker_node._get_agent_location(to_agent_id)

                if target_worker_location:
                    target_addr, target_port = target_worker_location
                    proto_event = grpc_impl.event_to_proto(event)

                    try:
                        await grpc_impl.connection_manager.with_stub(
                            target_addr, target_port,
                            grpc_impl.agent_pb2_grpc.AgentServiceStub,
                            'SendEvent',
                            proto_event
                        )
                        logger.debug(f"EventBus (Worker): Successfully sent event {event.event_id} P2P to {target_addr}:{target_port}")
                    except Exception as e:
                        logger.error(f"EventBus (Worker): Error sending event {event.event_id} P2P to {target_addr}:{target_port}: {e}")
                    return # Event sent (or attempted P2P), no further local processing.
                else:
                    logger.warning(f"EventBus (Worker): Could not find worker for agent {to_agent_id}. Event {event.event_id} ({event.event_kind}) will not be sent P2P.")
                    # Optional: Fallback to master. For now, just logs. If master fallback is needed, 
                    # it would be similar to the ENV block but might need careful thought to avoid loops.
                    return # Cannot find remote agent, do not process locally as if it were here.

        # Original logic for local dispatch (if not returned above)
        # This part will execute if:
        # 1. Not distributed mode.
        # 2. Is Master node AND event is for an actual local service (e.g., "ENV").
        # 3. Is Worker node AND event is for a local agent on this worker (assuming worker's dispatch_event is correct).
        agent_instance = self.agent_registry.get(to_agent_id)
        
        if agent_instance:
            # Safeguard: Ensure agent_instance is not a dictionary before calling add_event.
            # This primarily protects against misconfiguration or routing errors on Workers,
            # as Master should have handled dicts in the block above.
            if isinstance(agent_instance, dict):
                node_id_str = current_node.node_id if current_node else "UnknownNode"
                logger.critical(
                    f"CRITICAL FALLBACK: EventBus._process_event on node {node_id_str} trying to call add_event on a dict "
                    f"for agent {to_agent_id}. Event: {event.event_kind} (ID: {event.event_id}). This indicates a severe routing error."
                )
                return # Avoid AttributeError

            try:
                agent_instance.add_event(event)
            except AttributeError as ae:
                node_id_str = current_node.node_id if current_node else "UnknownNode"
                logger.error(
                    f"EventBus._process_event (node {node_id_str}): AttributeError when calling add_event for agent {to_agent_id} "
                    f"(target type: {type(agent_instance).__name__}). Event: {event.event_kind} (ID: {event.event_id}). Error: {ae}"
                )
                # logger.exception(ae) # Optionally log full stack for AttributeError
            except Exception as e:
                node_id_str = current_node.node_id if current_node else "UnknownNode"
                logger.error(
                    f"EventBus._process_event (node {node_id_str}): Exception calling add_event for agent {to_agent_id} "
                    f"(target type: {type(agent_instance).__name__}). Event: {event.event_kind} (ID: {event.event_id}). Error: {e}"
                )
                logger.exception(e) # Log full stack for other exceptions
        else:
            # Agent not found in local_registry
            node_id_str = current_node.node_id if current_node else "UnknownNode"
            logger.warning(
                f"EventBus._process_event (node {node_id_str}): Agent {to_agent_id} not found in agent_registry "
                f"for event {event.event_kind} (ID: {event.event_id})."
            )
            
    # Add a method to facilitate lock cleanup (useful for long-running systems)
    async def cleanup_expired_locks(self):
        """Clean up expired locks - should be called periodically"""
        if not self._is_distributed or not self._master_node:
            return
            
        async with self._lock_mutex:
            current_time = time.time()
            expired_locks = []
            
            # Find expired locks
            for lock_id, lock_info in self._lock_registry.items():
                if current_time > lock_info['expiry']:
                    expired_locks.append(lock_id)
            
            # Remove expired locks and notify waiters if needed
            for lock_id in expired_locks:
                logger.info(f"Cleaning up expired lock: {lock_id}")
                del self._lock_registry[lock_id]
                
                # Check if anyone is waiting for this lock
                if lock_id in self._lock_requests and self._lock_requests[lock_id]:
                    # Grant lock to next waiting node
                    next_requester = self._lock_requests[lock_id].pop(0)
                    next_node_id = next_requester['node_id']
                    next_agent_id = next_requester['from_agent_id']
                    
                    # Grant lock to next waiting node
                    self._lock_registry[lock_id] = {
                        'holder': next_node_id,
                        'expiry': current_time + 30.0  # Default timeout
                    }
                    
                    # Send notification
                    notification = {
                        'event_kind': 'AcquireLockResponse',
                        'lock_id': lock_id,
                        'node_id': next_node_id,
                        'success': True
                    }
                    await self._send_lock_response(notification, next_agent_id)
                    
                    # Clean up empty request lists
                    if not self._lock_requests[lock_id]:
                        del self._lock_requests[lock_id]

    
    async def export_event_flow_data(self, output_file: str = None) -> Dict[str, Any]:
        """
        Export event flow data to a file or return as a dictionary
        
        Args:
            output_file (str, optional): Path to output file
            
        Returns:
            Dict[str, Any]: Dictionary with flow data
        """
        async with self._event_mutex:
            # Prepare data for export
            export_data = {
                "flows": [],
                "timestamp": time.time()
            }
            
            for flow_id, events in self._event_flows.items():
                if len(events) <= 1:
                    continue
                    
                flow_data = {
                    "id": flow_id,
                    "steps": []
                }
                
                for event_data in events:
                    timestamp, event_kind, from_agent_id, to_agent_id, event_id, from_agent_type, to_agent_type = event_data
                    
                    # Get parent_event_id if it exists for this event
                    parent_event_id = None
                    for key, val in self._event_to_flow.items():
                        if val == flow_id and key != event_id:
                            # This could be a parent, check the timestamps to make sure
                            for e in events:
                                if e[4] == key and e[0] < timestamp:  # event_id is at index 4, timestamp at 0
                                    parent_event_id = key
                                    break
                    
                    step_data = {
                        "timestamp": timestamp,
                        "event_kind": event_kind,
                        "from_agent_id": from_agent_id,
                        "from_agent_type": from_agent_type,
                        "to_agent_id": to_agent_id,
                        "to_agent_type": to_agent_type,
                        "event_id": event_id
                    }
                    
                    if parent_event_id:
                        step_data["parent_event_id"] = parent_event_id
                    
                    flow_data["steps"].append(step_data)
                
                export_data["flows"].append(flow_data)
            
            # Save to file if requested
            if output_file:
                import json
                with open(output_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
                logger.info(f"Event flow data exported to {output_file}")
                
            return export_data

_event_bus_instance = None
_event_bus_lock = threading.Lock()

def get_event_bus() -> EventBus:
    """
    获取全局事件总线实例，确保整个应用程序中只有一个事件总线实例
    
    Returns:
        EventBus: 全局事件总线实例
    """
    global _event_bus_instance
    
    # 使用双重检查锁定模式确保线程安全
    if _event_bus_instance is None:
        with _event_bus_lock:
            if _event_bus_instance is None:
                _event_bus_instance = EventBus()
                logger.info("Created global EventBus instance")
    
    return _event_bus_instance

def reset_event_bus() -> None:
    """
    重置全局事件总线示例
    """
    global _event_bus_instance
    with _event_bus_lock:
        if _event_bus_instance and not _event_bus_instance.is_stopped():
            _event_bus_instance.stop()
        _event_bus_instance = None
        logger.info("Reset global EventBus instance")