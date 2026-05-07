import asyncio
from typing import Dict, List, Any, Optional, Tuple
import uuid
from loguru import logger
import importlib.util
import os
from tqdm import tqdm
from onesim.profile import AgentProfile, AgentSchema
from onesim.distribution.node import Node, NodeRole
from onesim.events import Event, get_event_bus, DataEvent, DataResponseEvent
from onesim.relationship import RelationshipManager
from onesim.memory import MemoryStrategy, AgentContext
from onesim.planning import PlanningBase
from onesim.models.core.model_manager import ModelManager
from onesim.distribution import grpc_impl # Assuming grpc_impl will be fixed
from onesim.distribution.node import get_node
from onesim.distribution.distributed_lock import  get_lock
import time

class WorkerNode(Node):
    """Worker node for hosting agents in a distributed setup"""

    def __init__(self, node_id: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(node_id, NodeRole.WORKER, config)
        self.master_address = config.get("master_address", "localhost")
        self.master_port = config.get("master_port", 10051)
        self.listen_port = config.get("listen_port", 10052)
        self.worker_address = config.get("worker_address", None) 
        self._server = None
        self._heartbeat_task = None
        self._registered = False
        self.model_config_name=ModelManager.get_instance().get_model_config_name(model_type="chat")
        self.agents={}
        self.env_path=None
        self.profile_id2agent={}
        self.agents_created = asyncio.Event()
        # 心跳配置
        self.heartbeat_interval = config.get("heartbeat_interval", 300)  # 减少心跳间隔为15秒
        self.heartbeat_max_retries = config.get("heartbeat_max_retries", 5)
        self.heartbeat_backoff_factor = config.get("heartbeat_backoff_factor", 2.0)
        self.heartbeat_max_interval = config.get("heartbeat_max_interval", 600)
        self.agent_location_cache: Dict[str, Tuple[str, int, float]] = {} # agent_id -> (worker_addr, worker_port, timestamp)
        self.agent_location_cache_ttl = config.get("agent_location_cache_ttl", 300) # 5 minutes TTL
        self.servicer_instance = None # Will hold the WorkerServicer instance
        # Stopped signal for graceful shutdown coordination
        self.stopped_event = asyncio.Event()

    async def initialize(self):
        """Initialize the worker node and connect to master"""
        await super().initialize()

        # 初始化批处理器
        from onesim.distribution.batch_processor import batch_processor
        batch_processor.start(
            master_address=self.master_address, 
            master_port=self.master_port
        )

        # Start gRPC server for receiving events
        if self._grpc_module:
            self._server = await self._grpc_module.create_worker_server(self, self.listen_port)
            logger.info(f"Worker server started on port {self.listen_port}")

            # Connect to master
            asyncio.create_task(self._connect_to_master())

            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._send_heartbeat())

    async def _connect_to_master(self):
        """Connect to the master node with retry mechanism"""
        if not self._grpc_module:
            logger.error("gRPC module not initialized")
            return

        retry_count = 0
        max_retries = 10  # 增加重试次数

        while retry_count < max_retries:
            try:
                # 在注册前先尝试确认连接性，使用单独通道
                # 这有助于避免注册时的连接问题
                logger.info(f"Attempting to connect to master at {self.master_address}:{self.master_port} (attempt {retry_count+1})")

                success = await self._grpc_module.register_with_master(
                    self.master_address,
                    self.master_port,
                    self.node_id,
                    self.listen_port,
                    self.worker_address
                )

                if success:
                    logger.info(f"Successfully registered node {self.node_id} {self.worker_address}:{self.listen_port} with master at {self.master_address}:{self.master_port}")
                    self._registered = True

                    # 注册成功后立即发送第一次心跳，确保连接保持活跃
                    await self._send_immediate_heartbeat()

                    # 启动定期心跳任务
                    if not self._heartbeat_task or self._heartbeat_task.done():
                        self._heartbeat_task = asyncio.create_task(self._send_heartbeat())

                    return
                else:
                    logger.error("Failed to register with master")

                # 增加指数退避重试延迟
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = min(2 ** retry_count, 60)  # 指数增长延迟，最多60秒
                    logger.info(f"Retrying connection in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

            except Exception as e:
                logger.error(f"Error connecting to master: {e}")
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = min(2 ** retry_count, 60)
                    logger.info(f"Retrying connection in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)

        logger.error(f"Failed to connect to master after {max_retries} attempts")

    async def _send_immediate_heartbeat(self):
        """立即发送一次心跳以确保连接保持活跃"""
        if not self._grpc_module:
            return

        try:
            # 不使用连接池发送初始心跳，确保有新的连接
            success = await self._grpc_module.send_heartbeat_to_master(
                self.master_address,
                self.master_port,
                self.node_id
            )

            if success:
                logger.info("Initial heartbeat sent successfully")
            else:
                logger.warning("Initial heartbeat failed")

        except Exception as e:
            logger.error(f"Error sending initial heartbeat: {e}")

    async def _send_heartbeat(self):
        """Send periodic heartbeat to master with exponential backoff on failure"""
        if not self._grpc_module:
            return

        retry_count = 0
        base_interval = self.heartbeat_interval

        while True:
            if self._registered:
                try:
                    success = await self._grpc_module.send_heartbeat_to_master(
                        self.master_address,
                        self.master_port,
                        self.node_id
                    )

                    if success:
                        # 重置重试计数
                        retry_count = 0
                        # 正常情况下使用较短心跳间隔，保持连接活跃
                        next_interval = base_interval
                        logger.debug(f"Heartbeat sent successfully, next in {next_interval}s")
                        await asyncio.sleep(next_interval)
                    else:
                        # 失败但没有异常，增加重试延迟
                        retry_count += 1
                        wait_time = min(base_interval * (self.heartbeat_backoff_factor ** retry_count), self.heartbeat_max_interval)
                        logger.warning(f"Heartbeat failed, retrying in {wait_time:.1f}s")
                        await asyncio.sleep(wait_time)

                except Exception as e:
                    retry_count += 1
                    wait_time = min(base_interval * (self.heartbeat_backoff_factor ** retry_count), self.heartbeat_max_interval)
                    logger.error(f"Error sending heartbeat: {e}, retrying in {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)

                    # 达到最大重试次数时重新尝试连接
                    if retry_count >= self.heartbeat_max_retries:
                        logger.warning("Heartbeat failed too many times, attempting to reconnect to master")
                        self._registered = False
                        # 强制清除连接池中可能损坏的连接
                        if hasattr(self._grpc_module, 'connection_manager'):
                            try:
                                await self._grpc_module.connection_manager._force_recreate_channel(
                                    self.master_address, 
                                    self.master_port
                                )
                            except Exception as e:
                                logger.error(f"Error recreating channel: {e}")

                        # 重新连接到master
                        asyncio.create_task(self._connect_to_master())
            else:
                # 未注册状态下仍定期检查
                await asyncio.sleep(5)

    async def _get_agent_location(self, agent_id: str) -> Optional[Tuple[str, int]]:
        """Get worker location for an agent, using cache or querying master."""
        # Check cache first
        if agent_id in self.agent_location_cache:
            addr, port, ts = self.agent_location_cache[agent_id]
            if time.time() - ts < self.agent_location_cache_ttl:
                logger.debug(f"Cache hit for agent {agent_id} location: {addr}:{port}")
                return addr, port
            else:
                logger.debug(f"Cache expired for agent {agent_id}")
                del self.agent_location_cache[agent_id]

        # If not in cache or expired, query master
        if self._registered and self._grpc_module: # Ensure grpc_module is loaded
            logger.debug(f"Querying master for agent {agent_id} location")
            location = await grpc_impl.locate_agent_on_master(
                self.master_address,
                self.master_port,
                agent_id
            )
            if location:
                self.agent_location_cache[agent_id] = (location[0], location[1], time.time())
                logger.debug(f"Cached agent {agent_id} location: {location[0]}:{location[1]}")
                return location
            else:
                logger.warning(f"Agent {agent_id} not found via master.")
                return None
        else:
            logger.warning("Cannot query agent location: Worker not registered or gRPC module not available.")
            return None

    async def handle_event(self, event: Event):
        """Handle an event received from the master or another worker."""
        logger.info(f"worker {self.node_id} handle_event: {event}")

        # If this event is a response to a P2P request originating from this worker,
        # it should be routed directly to the waiting future in GeneralAgent.
        # The GeneralAgent's handle_data_response (or similar) will be triggered by the local event bus.

        # If this event is an initial request that needs a P2P reply:
        # The reply info is already stored in WorkerServicer.p2p_reply_info_cache.
        # When GeneralAgent generates a response event, we'll need to check this cache.

        # Forward to local event bus for standard processing by the target agent
        event_bus = get_event_bus()
        await event_bus.dispatch_event(event)

    def find_worker_for_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Find worker information for an agent, only returning self if agent is local.
        
        Args:
            agent_id (str): ID of the agent to locate
            
        Returns:
            Optional[Dict[str, Any]]: Worker information or None
        """
        # Check if agent exists locally
        if agent_id in self.profile_id2agent:
            return {
                "worker_id": self.node_id,
                "address": "localhost",
                "port": self.listen_port
            }
        return None

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

    async def create_agents_batch(self, agent_configs: List[Dict[str, Any]]) -> List[str]:
        """批量创建多个本地Agent"""
        created_agent_ids = []
        created_agent_ids = self.create_local_agents(agent_configs)

        self.agents_created.set()
        return created_agent_ids

    async def handle_termination_signal(self, reason: str = "unknown") -> bool:
        """处理来自主节点的终止信号"""
        logger.info(f"Worker {self.node_id} received termination signal: {reason}")

        # 执行清理操作
        await self.shutdown()

        return True

    async def shutdown(self):
        """Clean shutdown of worker node resources"""
        logger.info(f"Shutting down worker node {self.node_id}")

        # 取消心跳任务
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()

        # 关闭批处理器
        from onesim.distribution.batch_processor import batch_processor
        batch_processor.stop()

        # 停止gRPC服务器
        if self._server:
            await self._server.stop(5)  # 5秒优雅停止

        # 调用父类shutdown方法
        await super().shutdown()

        # 通知外部等待方：已完成关闭
        if not self.stopped_event.is_set():
            self.stopped_event.set()

        logger.info(f"Worker node {self.node_id} shutdown complete")

    def load_memory(self, memory_config, model_config_name: str) -> MemoryStrategy:
        """Create memory strategy instance"""
        strategy_class_name = memory_config["strategy"]
        try:
            # Load memory module and class
            memory_module = importlib.import_module("onesim.memory")
            MemoryClass = getattr(memory_module, strategy_class_name)
            # Initialize memory instance
            memory_instance = MemoryClass(memory_config,model_config_name)
            return memory_instance
        except ImportError as e:
            logger.error(f"Failed to import module: {e}")
        except AttributeError as e:
            logger.error(f"Class {strategy_class_name} not found in onesim.memory: {e}")
        except Exception as e:
            logger.error(f"An error occurred during memory initialization: {e}")

        return None

    def load_agent_module_from_file(self, agent_type: str):
        """Load agent class from file"""
        module_path = os.path.join(self.env_path, "code", f"{agent_type}.py")
        logger.info(f"module_path: {module_path}")
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Agent module file not found: {module_path}")

        env_name = self.env_path.split(os.sep)[-1]
        package_name = f"envs.{env_name}.code"
        module_name = f"{package_name}.{agent_type}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, agent_type):
            raise AttributeError(f"Agent class '{agent_type}' not found in {module_path}")

        return getattr(module, agent_type)

    def load_planning(self, planning_config, model_config_name: str,sys_prompt: str) -> PlanningBase:
        """Create planning strategy instance"""
        if not planning_config:
            return None

        try:
            # Load memory module and class
            planning_module = importlib.import_module("onesim.planning")
            PlanningClass = getattr(planning_module, planning_config)

            # Initialize memory instance
            planning_instance = PlanningClass(model_config_name,sys_prompt)
            return planning_instance
        except ImportError as e:
            logger.error(f"Failed to import module: {e}")

    def create_local_agents(self, agent_configs: List[Dict]) -> None:
        """Create local agent instances with pre-configured relationships"""
        env_name= agent_configs[0]["env"].split(os.sep)[-1]
        self.env_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'..', "envs", env_name)
        # Group configurations by agent type
        event_bus = get_event_bus()
        created_agent_ids = []
        configs_by_type = {}
        for config in agent_configs:

            agent_type = config["type"]
            if agent_type not in configs_by_type:
                configs_by_type[agent_type] = []
            configs_by_type[agent_type].append(config)

        # Create agents for each type
        for agent_type, configs in configs_by_type.items():
            # Load the agent class
            AgentClass = self.load_agent_module_from_file(agent_type)
            # Create each agent instance
            for config in tqdm(configs):
                agent_id = config["id"]
                # Create profile instance from saved data
                profile = AgentProfile(
                    agent_type, 
                    AgentSchema(config["schema"]), 
                    profile_data=config["profile_data"]
                )
                profile.set_agent_profile_id(agent_id)
                # Create relationship manager and add relationships
                rm = RelationshipManager(profile_id=agent_id)
                # Add pre-loaded relationships if present
                if "relationships" in config:
                    for relationship in config["relationships"]:
                        rm.add_relationship(
                            target_id=relationship["target_id"],
                            description=relationship["description"],
                            target_info=relationship["target_info"]
                        )

                # Create memory instance
                memory_config = config.get("memory_config")
                memory_instance = self.load_memory(memory_config, self.model_config_name)
                planning_instance = self.load_planning(config["planning_config"], self.model_config_name,config["sys_prompt"])
                # Create agent instance
                agent = AgentClass(
                    #name=config["name"],
                    profile=profile,
                    sys_prompt=config["sys_prompt"],
                    model_config_name=self.model_config_name,
                    memory=memory_instance,
                    planning=planning_instance,
                    event_bus_queue=get_event_bus().queue,
                    relationship_manager=rm
                )

                # Store the agent in our dictionaries
                if agent_type not in self.agents:
                    self.agents[agent_type] = {}
                self.agents[agent_type][agent_id] = agent
                self.profile_id2agent[agent_id] = agent
                created_agent_ids.append(agent_id)
                event_bus.register_agent(agent_id, agent)
        return created_agent_ids

    async def route_event_to_destination(self, event: Event):
        """Intelligently routes an event: P2P, to master, or local."""
        to_agent_id = event.to_agent_id

        if to_agent_id == "ENV": # Send to Master/Environment
            logger.debug(f"Routing event {event.event_id} ({event.event_kind}) to ENV (Master)")
            if self._grpc_module:
                # Assuming event_to_proto is correctly in grpc_impl and handles all event types
                proto_event = grpc_impl.event_to_proto(event)
                await grpc_impl.connection_manager.with_stub(
                    self.master_address, self.master_port, 
                    grpc_impl.agent_pb2_grpc.AgentServiceStub, 
                    'SendEvent', proto_event
                )
            return

        if to_agent_id == self.node_id or to_agent_id in self.profile_id2agent: # Local agent
            logger.debug(f"Routing event {event.event_id} ({event.event_kind}) to local agent {to_agent_id}")
            await get_event_bus().dispatch_event(event) # Dispatch to local agent
            return

        # Try P2P for other agents
        target_worker_loc = await self._get_agent_location(to_agent_id)
        if target_worker_loc:
            worker_addr, worker_port = target_worker_loc
            logger.debug(f"Routing event {event.event_id} ({event.event_kind}) P2P to agent {to_agent_id} at {worker_addr}:{worker_port}")

            # Check if this is a response to a P2P request by looking up WorkerServicer's cache
            # This check is for *outgoing* responses. Incoming P2P requests are handled by WorkerServicer.SendEvent.
            is_p2p_response = False
            if self.servicer_instance and hasattr(event, 'request_id') and event.request_id in self.servicer_instance.p2p_reply_info_cache:
                # This event IS a response to a P2P request it received.
                # The WorkerServicer.p2p_reply_info_cache holds the address of the *original requester*.
                # So this `target_worker_loc` is where we send the reply.
                is_p2p_response = True
                # We don't need to add reply_to for a response.
                reply_addr, reply_port = self.servicer_instance.p2p_reply_info_cache.pop(event.request_id) # Get and remove reply info
                logger.debug(f"Event {event.event_id} is a P2P response. Original requester for {event.request_id} was at {reply_addr}:{reply_port}")
                # Ensure the event is now targeted to the original requester.
                # event.to_agent_id should already be the original requester for a response event.
                # The target_worker_loc for sending should be this reply_addr, reply_port
                worker_addr, worker_port = reply_addr, reply_port # Override target for P2P reply

            proto_event = grpc_impl.event_to_proto(event) 

            if not is_p2p_response and isinstance(event, DataEvent): # Example: Only add reply_to for initial DataEvents for P2P
                # This worker is initiating the P2P request to another worker (target_worker_loc is not self)
                if (worker_addr, worker_port) != (self.worker_address or grpc_impl.get_host_ip(), self.listen_port):
                    proto_event.reply_to_worker_address = self.worker_address or grpc_impl.get_host_ip()
                    proto_event.reply_to_worker_port = self.listen_port
                    logger.debug(f"Added P2P reply_to for initial event {event.event_id}: {proto_event.reply_to_worker_address}:{proto_event.reply_to_worker_port} to {worker_addr}:{worker_port}")

            if self._grpc_module:
                await grpc_impl.connection_manager.with_stub(
                    worker_addr, worker_port, 
                    grpc_impl.agent_pb2_grpc.AgentServiceStub, 
                    'SendEvent', proto_event
                )
        else:
            # Fallback: If agent not found via P2P, could send to master, or log error.
            # For now, log an error. A more robust solution might queue or retry.
            logger.error(f"Could not find worker for agent {to_agent_id}. Event {event.event_id} ({event.event_kind}) not sent P2P.")
            # Optionally, send to master as a fallback (less efficient)
            # if self._grpc_module:
            #     logger.warning(f"Fallback: Routing event {event.event_id} for {to_agent_id} via Master.")
            #     proto_event = grpc_impl.event_to_proto(event)
            #     await grpc_impl.connection_manager.with_stub(
            #         self.master_address, self.master_port,
            #         grpc_impl.agent_pb2_grpc.AgentServiceStub,
            #         'SendEvent', proto_event
            #     )

    def set_servicer_instance(self, servicer_instance):
        """Allows grpc_impl to set a reference to the WorkerServicer instance."""
        self.servicer_instance = servicer_instance

    async def collect_local_agent_data_batch(self, agent_type: str, data_key: str, default_value: Any) -> Dict[str, Any]:
        """Collects data from local agents of a specific type for batch processing."""
        collected_data = {}
        if agent_type in self.agents:
            agents_of_type = self.agents[agent_type]
            for agent_id, agent_instance in agents_of_type.items():
                try:
                    # Assuming agent_instance has a get_data method similar to GeneralAgent
                    if hasattr(agent_instance, 'get_data') and asyncio.iscoroutinefunction(agent_instance.get_data):
                        data = await agent_instance.get_data(data_key, default_value)
                    elif hasattr(agent_instance, 'get_data'): # Synchronous get_data
                        data = agent_instance.get_data(data_key, default_value)
                    else:
                        logger.warning(f"Agent {agent_id} of type {agent_type} does not have a get_data method.")
                        data = default_value
                    collected_data[agent_id] = data
                except Exception as e:
                    logger.error(f"Error collecting data from local agent {agent_id} (type {agent_type}) for key {data_key}: {e}")
                    collected_data[agent_id] = default_value
        else:
            logger.info(f"No agents of type {agent_type} found on worker {self.node_id} for batch data collection.")

        return collected_data
