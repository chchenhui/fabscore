import asyncio
from .base import AgentBase
from onesim.events import Event, DataEvent, DataResponseEvent, DataUpdateEvent, DataUpdateResponseEvent
from typing import Dict, List, Callable, Optional, Any
import json
import uuid
import time
from asyncio import Future
from loguru import logger
from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from onesim.profile import AgentProfile
from onesim.memory import *
from onesim.events import *
from onesim.relationship import RelationshipManager, Relationship
import inspect
from onesim.planning.base import PlanningBase
from onesim.distribution.node import get_node
from onesim.distribution import grpc_impl
from onesim.distribution.node import NodeRole
from onesim.distribution.distributed_lock import  get_lock
from onesim.utils.work_graph import WorkGraph
from datetime import datetime


class GeneralAgent(AgentBase):
    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name)
        
        self._queue = asyncio.Queue()
        '''
        The queue where the agent will store the events.
        '''

        self._event_schema: Dict[str, List[str]] = {}
        '''
        Key: event_kind, value: list of ability names.
        '''
        self._property_futures: Dict[str, Future] = {}
        self._event_bus_queue: asyncio.Queue = event_bus_queue
        self._sync_event = asyncio.Event()
        self.profile = profile
        self.memory=memory
        self.planning=planning
        #self.meory=memory
        if relationship_manager is None:
            self.relationship_manager = RelationshipManager(profile_id=self.profile.get_agent_profile_id())
        else:
            self.relationship_manager = relationship_manager
        self.relationships = self.relationship_manager.get_all_relationships()
        if self.memory:
            self.memory.set_agent_context(self.create_context())
        self.stopped=False
        self._hooks = {
            'before_event_handling': [],
            'after_event_handling': [],
            'before_action': [],
            'after_action': [],
            'before_reaction_generation': [],
            'after_reaction_generation': [],
            'before_memory_generation': [],
            'after_memory_generation': [],
        }
        # Register event handler for termination events
        self.register_event("EndEvent", "handle_end_event")
        
        # Register event handlers for data access
        self.register_event("DataEvent", "handle_data_event")
        self.register_event("DataResponseEvent", "handle_data_response")
        self.register_event("DataUpdateEvent", "handle_data_update_event")
        self.register_event("DataUpdateResponseEvent", "handle_data_update_response")
        
        # Data request futures dictionary
        self._data_futures: Dict[str, Future] = {}
        # Data update futures dictionary
        self._data_update_futures: Dict[str, Future] = {}

    def is_stopped(self) -> bool:
        return self.stopped


    def register_event(self, event_kind: str, ability_name: str) -> None:
        if event_kind not in self._event_schema:
            self._event_schema[event_kind] = []
        self._event_schema[event_kind].append(ability_name)

    def add_event(self, event: Event) -> None:
        self._queue.put_nowait(event)

    async def get_event(self) -> Event:
        pass

    
    def register_hook(self, hook_name: str, hook_function: Callable):
        if hook_name in self._hooks:
            self._hooks[hook_name].append(hook_function)
        else:
            raise ValueError(f"Unknown hook: {hook_name}")

    async def _execute_hooks(self, hook_name: str, **kwargs):
        for hook in self._hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self, **kwargs)
                else:
                    hook(self, **kwargs)
            except Exception as e:
                logger.error(f"Error in hook '{hook_name}': {e}")

    async def run_task(self, method: Callable, event: Event):
        # Record the incoming event
        await self.record_event(event)
        reses = await method(event)
        if not isinstance(reses, list):
            reses = [reses]
            
        for res in reses:
            if res is None:
                continue
            res.parent_event_id = event.event_id
            await self._event_bus_queue.put(res)

    async def run(self):
        while not self.is_stopped():
            try:
                event = await self._queue.get()
                await self._execute_hooks('before_event_handling', event=event)
                if event.event_kind not in self._event_schema:
                    # raise ValueError(
                    #      f"Event type {event.event_kind} not registered in {self.profile.agent_type}(ID:{self.profile_id}).")
                    logger.error(f"Event type {event.event_kind} not registered in {self.profile.agent_type}(ID:{self.profile_id}).")
                    self._queue.task_done()
                    continue

                for ability_name in self._event_schema[event.event_kind]:
                    # call self function with ability_name
                    method: Optional[Callable] = getattr(self, ability_name, None)
                    if not callable(method):
                        logger.error(f"Method {ability_name} not found in {self.__class__.__name__}.")
                        continue
                    await self._execute_hooks('before_action', ability_name=ability_name, event=event)
                    asyncio.create_task(self.run_task(method, event))
                    await self._execute_hooks('after_action', ability_name=ability_name, event=event)
                await self._execute_hooks('after_event_handling', event=event)
                self._queue.task_done()
            except asyncio.CancelledError:
                # logger.info(f"Agent {self.profile_id} task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in agent {self.profile.agent_type}(ID:{self.profile_id}) run loop: {e}")
                self._queue.task_done()
                
    async def handle_end_event(self, event: Event) -> None:
        """
        Handle termination event to gracefully stop the agent.
        
        Args:
            event (Event): The EndEvent containing termination details
        """

        
        # Perform any cleanup tasks
        try:
            
            # Set stopped flag to terminate the run loop
            self.stopped = True
            logger.info(f"Agent {self.profile.agent_type}(ID:{self.profile_id}) received termination event: {event.get('reason', 'unknown')}")
        except Exception as e:
            logger.error(f"Error during agent {self.profile.agent_type}(ID:{self.profile_id}) termination: {e}")
        
        # No response event needed for termination
        return None

    async def generate_memory(self, instruction: str, observation: str, reaction: dict) -> str:
        if not self.memory:
            return ""
        profile_str = self.profile.get_profile_str() if self.profile else "No profile information provided."
        # Build prompt for memory generation
        prompt_text = f"""
        ### Agent Profile:
        {profile_str}

        ### Relationship with other agents:
        {self.relationship_manager.get_all_relationships_str()}

        ### Event Details:
        Observation: {observation if observation else "No specific observation provided."}
        Instruction: {instruction}
        Reaction: {json.dumps(reaction, indent=2)}

        Based on the agent's profile and the complete event that occurred (the instruction received, what was observed, and how the agent reacted), generate a single sentence memory that captures this experience from the agent's perspective. The memory should be personal and reflect the complete interaction including what the agent was asked to do, what they perceived, and how they responded.

        Structure the response in JSON format with a single 'memory' field containing this sentence. Format the response in a json fenced code block as follows:
        ```json
        {{"memory": "Your memory sentence here"}}
        ```
        """
        ### the returned json format should comes from the memory manager
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt_text, role="user")
        )

        # Parse LLM JSON response
        try:
            response = await self.model.acall(prompt)
            parser = JsonBlockParser()
            res = parser.parse(response)
            memory=res.parsed['memory']
            #memory_msg=Message(self.name, memory, role="assistant")
            if self.memory:
                await self.memory.add(MemoryItem(self.agent_id,memory))
            return memory
        except (json.JSONDecodeError) as e:
            logger.error(f"LLM response is not valid JSON. {prompt}")
            raise ValueError("LLM response is not valid JSON.")

    async def generate_reaction(self, instruction: str, observation: str = None) -> json:
        # 获取Agent的Profile和Memory信息
        profile_str = self.profile.get_profile_str(include_private=True) if self.profile else "No profile information provided."
        if self.memory:     
            memory_msgs = (await self.memory.retrieve(observation))
            memory=""
            for msg in memory_msgs:
                memory+=msg.content
        else:
            memory=""

        caller_frame = inspect.currentframe().f_back
        # 获取调用者的函数名
        action_name=caller_frame.f_code.co_name
        work_graph=WorkGraph()
        successor_types=work_graph.get_successor_agent_types(f"{self.profile.agent_type}.{action_name}")
        relationships=self.relationship_manager.get_all_relationships()
        available_relations=[]
        for relation in relationships:
            if relation.target_info["agent_type"] in successor_types:
                available_relations.append(relation)
        if len(available_relations)==0:
            available_relations=relationships
        # Add guidance for empty relationships
        relationship_guidance = ""
        if not available_relations:
            if self.profile.agent_type in successor_types:
                relationship_guidance = f"Note: No available relationships found. You can send events to yourself (agent type: {self.profile.agent_type}, ID: {self.profile_id})."
            elif 'EnvAgent' in successor_types:
                relationship_guidance = "Note: No available relationships found. You can send events to the environment (agent type: EnvAgent, ID: ENV)."
            else:
                relationship_guidance = "Note: No available relationships found and no default targets available."
        
        #relations=self.relationship_manager.get_all_relationships()
        if self.planning:
            planning=await self.planning.plan(profile=profile_str,memory=memory,observation=observation,instruction=instruction,relationship=available_relations) 
            planning="### Planning:\n"+planning+"\n"
        else:
            planning=""
            
        # 构建Prompt，包含Profile、Memory、Observation和Instruction部分
        prompt_text = f"""
        ### Agent Profile:
        {profile_str}

        ### Memory:
        {memory}

        ### Relationship:
        {available_relations}

        ### Observation:
        {observation if observation else "No specific observation provided."}

        ### Instruction:
        {instruction}

        {planning}

        Please analyze the Agent's profile, memory, observation and planning. Based on these, generate a response that aligns with the Agent's identity and instruction. 
        When identifying or referring to other agents, you must ONLY use their **Target ID** from the Relationships data. Ignore any other IDs (such as Enterprise ID, Student ID) that might appear in the data.
        
        Important: You can always send events to yourself if needed. {relationship_guidance}
        
        Structure the response in JSON format, specifying a detailed action or reaction based on the instruction. You should respond a json object in a json fenced code block as follows:
        ```json
        Your JSON response here
        ```
        """
        start_time = time.time()
        prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt_text, role="user")
        )
        response = await self.model.acall(prompt)
        processing_time = time.time() - start_time
        
        try:
            parser = JsonBlockParser()
            res = parser.parse(response)
            reaction=res.parsed

            # Record decision for data storage - supports both local and distributed modes
            decision_data = {
                'agent_id': self.profile_id,
                'agent_type': self.profile.agent_type,
                'prompt': prompt_text,
                'output': response.text,
                'processing_time': processing_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'decision_id': str(uuid.uuid4()),
                'action': action_name,
                'context': {
                    'instruction': instruction,
                    'observation': observation,
                    'memory': memory
                },
                'feedback': None,
                'rating': None,
                'reason': None
            }
            await self._record_decision(decision_data)
            
            if self.memory:
                event_memory = await self.generate_memory(instruction, observation, reaction)
            else:
                event_memory = ""
                
            logger.info(f"{self.profile.agent_type}(ID:{self.profile_id}) Action: {action_name} - Event Log: "
            f"Observation: {observation or 'None'}, "
            f"Reaction: {json.dumps(reaction, indent=2)}, "
            f"Memory: {json.dumps(event_memory, indent=2)}")
            
            return reaction
            
        except json.JSONDecodeError:
            logger.error(f"{self.profile.agent_type}(ID:{self.profile_id}) - LLM response is not valid JSON.{prompt}")
            raise ValueError("LLM response is not valid JSON.")

    




    async def get_memory(self):
        if not self.memory:
            return []
        return await self.memory.get_all_memory()

    def add_relationship(self, target_id: str, description: str,target_info: Optional[Dict]=None):
        self.relationship_manager.add_relationship(target_id, description,target_info)

    def remove_relationship(self, target_id: str):
        self.relationship_manager.remove_relationship(target_id)

    def update_relationship(self, target_id: str, description: str):
        self.relationship_manager.update_relationship(target_id, description)

    def get_relationship(self, target_id: str) -> Optional[Relationship]:
        return self.relationship_manager.get_relationship(target_id)

    def get_all_relationships(self) -> List[Relationship]:
        return self.relationship_manager.get_all_relationships()

    def create_context(self):
        return AgentContext(self.profile_id,self.profile,self.relationship_manager)
    
    async def interact(self, message: str, chat_history: List[Dict[str, Any]] = None):
        """
        Process a user chat message and generate an in-character response from the agent.
        
        Args:
            message (str): The message from the user
            chat_history (List[Dict[str, Any]], optional): Previous chat history for context
            
        Returns:
            dict: The agent's in-character response
        """
        # Format chat history as context if available
        context = ""
        if chat_history and len(chat_history) > 0:
            context = "Previous conversation:\n"
            for entry in chat_history[-5:]:  # Only use the last 5 messages for context
                role = "User" if entry.get("role") == "user" else self.profile.get_data("name") or "Agent"
                context += f"{role}: {entry.get('message', '')}\n"
                
        # Construct a prompt that focuses on in-character role-playing
        instruction = f"""A user is talking directly to you. Respond as your character would in a conversation.
Stay completely in character based on your profile and respond naturally to the user's message.
Do not refer to yourself as an AI, a language model, or a simulation - you are the character described in your profile.
Respond in JSON format with a single 'message' field containing your response. Format the response in a json fenced code block as follows:
```json
{{
    "message": "Your response here"
}}
```
"""

        # Add context from chat history if available
        observation = f"{context}\nUser's message: {message}"
        
        reaction = await self.generate_reaction(instruction, observation)
        
        # Make sure we have a message field for the frontend
        if isinstance(reaction, dict) and "message" not in reaction:
            # If there's no message field but there is a response or content field, use that
            if "response" in reaction:
                reaction["message"] = reaction["response"]
            elif "content" in reaction:
                reaction["message"] = reaction["content"]
            elif "text" in reaction:
                reaction["message"] = reaction["text"]
            # Otherwise create a message from the reaction
            else:
                reaction["message"] = str(reaction)
                
        return reaction

    @property
    def profile_id(self):
        return self.profile.get_agent_profile_id() 

    def get_profile_str(self,include_private: bool = None):
        return self.profile.get_profile_str(include_private)
    
    def get_profile(self, include_private: bool = None):
        return self.profile.get_profile(include_private)

    def set_env(self, env):
        """Set the simulation environment associated with this agent."""
        self.env = env
        
    async def record_event(self, event: Event):
        """
        Record an event for data storage
        
        Args:
            event: Event object
        """
        
        if event.event_kind in ['DataEvent', 'DataUpdateEvent', 'DataResponseEvent', 'DataUpdateResponseEvent','PauseEvent','ResumeEvent', 'EndEvent']:
            return

        # Check if we have a local environment reference
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'queue_event'):
            # Local mode or using ProxyEnv - queue directly
            # Handle both sync and async implementations
            try:
                # If queue_event is synchronous
                await self.env.queue_event(event.to_dict())
            except Exception as e:
                logger.error(f"Error in synchronous queue_event: {e}")
        else:
            # Legacy distributed worker mode - send to master via gRPC
            node = get_node()
            if node.role == NodeRole.WORKER and node.master_address and node.master_port:
                # Create task to send data to master (don't await to avoid blocking)
                asyncio.create_task(
                    grpc_impl.send_storage_event_to_master(
                        node.master_address,
                        node.master_port,
                        event.to_dict()
                    )
                )
            else:
                logger.warning(f"Unable to record event: No environment reference and not in worker mode")
    
    async def _record_decision(self, decision_data: Dict[str, Any]):
        """
        Record a decision for data storage
        
        Args:
            decision_data: Decision data to record
        """
        # Check if we have a local environment reference
        if hasattr(self, 'env') and self.env is not None and hasattr(self.env, 'queue_decision'):
            # Local mode or using ProxyEnv - queue directly
            # Handle both sync and async implementations
            try:
                # If queue_decision is synchronous
                await self.env.queue_decision(decision_data)
            except Exception as e:
                logger.error(f"Error in synchronous queue_decision: {e}")
        else:
            # Legacy distributed worker mode - send to master via gRPC
            node = get_node()
            if node.role == NodeRole.WORKER and node.master_address and node.master_port:
                # Send data to master
                await grpc_impl.send_decision_record_to_master(
                    node.master_address,
                    node.master_port,
                    decision_data
                )
            else:
                logger.warning(f"Unable to record decision: No environment reference and not in worker mode")


    async def get_data(self, key: str,default: Optional[Any] = None):
        """Get data from agent profile"""
        if not key or not hasattr(self, 'profile') or not self.profile:
            return default
        if "profile" in key:
            key=key.replace("profile.","")
        return self.profile.get_data(key,default)

    async def handle_data_event(self, event: DataEvent) -> Optional[DataResponseEvent]:
        """
        Handle incoming data access requests
        
        Args:
            event (DataEvent): Data access event
            
        Returns:
            Optional[DataResponseEvent]: Response event or None
        """ 
        try:
            # Get data from profile
            data_value = await self.get_data(event.key)
            # Create response event
            response = DataResponseEvent(
                from_agent_id=self.profile_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                data_value=data_value,
                success=True,
                parent_event_id=event.event_id
            )
            return response
        except Exception as e:
            # Create error response
            error_response = DataResponseEvent(
                from_agent_id=self.profile_id,
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
        Handle incoming data response events
        
        Args:
            event (DataResponseEvent): Data response event
        """
        # Check if we're waiting for this response
        if event.request_id in self._data_futures:
            future = self._data_futures.pop(event.request_id)
            
            if not future.done():
                if event.success:
                    future.set_result(event.data_value)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))
            
            # If we have a sync event, set it
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                # Reset for next operation
                self._sync_event.clear()
    
    async def get_env_data(self, key: str, default: Optional[Any] = None, parent_event_id: Optional[str] = None) -> Any:
        """
        Get data from the environment
        
        Args:
            key (str): Data key to access
            default (Any, optional): Default value if key not found
            parent_event_id (str, optional): ID of parent event that triggered this request
            
        Returns:
            Any: The requested data or default value
        """
        # Create a unique request ID
        request_id = f"agent_env_req_{time.time()}_{id(self)}"
        
        # Create future for response
        future = Future()
        self._data_futures[request_id] = future
        
        # Create data request event
        data_event = DataEvent(
            from_agent_id=self.profile_id,
            to_agent_id="ENV",  # Special target for environment
            source_type="AGENT",
            target_type="ENV",
            key=key,
            default=default,
            request_id=request_id,
            parent_event_id=parent_event_id
        )
        
        # Send the request
        await self._event_bus_queue.put(data_event)
        
        # Wait for response with timeout
        try:
            if hasattr(self, '_sync_event'):
                # If we have a sync event, wait for it
                await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                return await future
            else:
                # Otherwise just wait for the future directly
                return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for environment data: {key}")
            self._data_futures.pop(request_id, None)
            return default
        except Exception as e:
            logger.error(f"Error getting environment data: {e}")
            self._data_futures.pop(request_id, None)
            return default
    
    async def get_agent_data(self, agent_id: str, key: str, default: Optional[Any] = None, parent_event_id: Optional[str] = None) -> Any:
        """
        Get data from another agent
        
        Args:
            agent_id (str): ID of agent to get data from
            key (str): Data key to access
            default (Any, optional): Default value if key not found
            parent_event_id (str, optional): ID of parent event that triggered this request
            
        Returns:
            Any: The requested data or default value
        """
        # Prevent self-queries
        if agent_id == self.profile_id:
            return await self.get_data(key)
            
        # Create a unique request ID
        request_id = f"agent_req_{time.time()}_{id(self)}"
        
        # Create future for response
        future = Future()
        self._data_futures[request_id] = future
        
        # Create data request event
        data_event = DataEvent(
            from_agent_id=self.profile_id,
            to_agent_id=agent_id,
            source_type="AGENT",
            target_type="AGENT",
            key=key,
            default=default,
            request_id=request_id,
            parent_event_id=parent_event_id
        )
        
        # Send the request
        await self._event_bus_queue.put(data_event)
        
        # Wait for response with timeout
        try:
            if hasattr(self, '_sync_event'):
                # If we have a sync event, wait for it
                await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                return await future
            else:
                # Otherwise just wait for the future directly
                return await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for data from agent {agent_id}")
            self._data_futures.pop(request_id, None)
            return default
        except Exception as e:
            logger.error(f"Error getting data from agent {agent_id}: {e}")
            self._data_futures.pop(request_id, None)
            return default

    async def update_data(self, key: str, value: Any) -> bool:
        """Update data in agent profile"""
        if not key or not hasattr(self, 'profile') or not self.profile:
            return False
        return self.profile.update_data(key, value)

    async def handle_data_update_event(self, event: DataUpdateEvent) -> Optional[DataUpdateResponseEvent]:
        """
        Handle incoming data update requests
        
        Args:
            event (DataUpdateEvent): Data update event
            
        Returns:
            Optional[DataUpdateResponseEvent]: Response event or None
        """
            
        try:
            # Update data in profile - Added await here
            success = await self.update_data(event.key, event.value)
            
            # Create response event
            response = DataUpdateResponseEvent(
                from_agent_id=self.profile_id,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=success,
                parent_event_id=event.event_id
            )
            
            return response
        except Exception as e:
            # Create error response
            error_response = DataUpdateResponseEvent(
                from_agent_id=self.profile_id,
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
        Handle incoming data update response events
        
        Args:
            event (DataUpdateResponseEvent): Data update response event
        """
        # Check if we're waiting for this response
        if event.request_id in self._data_update_futures:
            future = self._data_update_futures.pop(event.request_id)
            
            if not future.done():
                if event.success:
                    future.set_result(True)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))
            
            # If we have a sync event, set it
            if hasattr(self, '_sync_event'):
                self._sync_event.set()
                # Reset for next operation
                self._sync_event.clear()
    
    async def update_env_data(self, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        Update data in the environment with distributed locking
        
        Args:
            key (str): Data key to update
            value (Any): New value to set
            parent_event_id (str, optional): ID of parent event that triggered this request
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Create a unique request ID
        request_id = f"agent_env_update_req_{time.time()}_{id(self)}"
        
        # Create future for response
        future = Future()
        self._data_update_futures[request_id] = future
        
        # Create data update event
        data_update_event = DataUpdateEvent(
            from_agent_id=self.profile_id,
            to_agent_id="ENV",  # Special target for environment
            source_type="AGENT",
            target_type="ENV",
            key=key,
            value=value,
            request_id=request_id,
            parent_event_id=parent_event_id
        )
        
        # Get distributed lock for this key
        lock_id = f"env_data_lock_{key}"
        lock = await get_lock(lock_id)
        
        try:
            # Acquire lock before sending update
            async with lock:
                # Send the request
                await self._event_bus_queue.put(data_update_event)
                
                # Wait for response with timeout
                try:
                    if hasattr(self, '_sync_event'):
                        # If we have a sync event, wait for it
                        await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                        return await future
                    else:
                        # Otherwise just wait for the future directly
                        return await asyncio.wait_for(future, timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for environment data update: {key}")
                    self._data_update_futures.pop(request_id, None)
                    return False
                except Exception as e:
                    logger.error(f"Error updating environment data: {e}")
                    self._data_update_futures.pop(request_id, None)
                    return False
        except Exception as e:
            logger.error(f"Error acquiring lock for environment data update: {e}")
            return False
    
    async def update_agent_data(self, agent_id: str, key: str, value: Any, parent_event_id: Optional[str] = None) -> bool:
        """
        Update data in another agent with distributed locking
        
        Args:
            agent_id (str): ID of agent to update data in
            key (str): Data key to update
            value (Any): New value to set
            parent_event_id (str, optional): ID of parent event that triggered this request
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Prevent self-queries
        if agent_id == self.profile_id:
            return await self.update_data(key, value)
            
        # Create a unique request ID
        request_id = f"agent_update_req_{time.time()}_{id(self)}"
        
        # Create future for response
        future = Future()
        self._data_update_futures[request_id] = future
        
        # Create data update event
        data_update_event = DataUpdateEvent(
            from_agent_id=self.profile_id,
            to_agent_id=agent_id,
            source_type="AGENT",
            target_type="AGENT",
            key=key,
            value=value,
            request_id=request_id,
            parent_event_id=parent_event_id
        )
        
        # Get distributed lock for this agent and key
        lock_id = f"agent_data_lock_{agent_id}_{key}"
        lock = await get_lock(lock_id)
        
        try:
            # Acquire lock before sending update
            async with lock:
                # Send the request
                await self._event_bus_queue.put(data_update_event)
                
                # Wait for response with timeout
                try:
                    if hasattr(self, '_sync_event'):
                        # If we have a sync event, wait for it
                        await asyncio.wait_for(self._sync_event.wait(), timeout=30.0)
                        return await future
                    else:
                        # Otherwise just wait for the future directly
                        return await asyncio.wait_for(future, timeout=30.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for data update in agent {agent_id}")
                    self._data_update_futures.pop(request_id, None)
                    return False
                except Exception as e:
                    logger.error(f"Error updating data in agent {agent_id}: {e}")
                    self._data_update_futures.pop(request_id, None)
                    return False
        except Exception as e:
            logger.error(f"Error acquiring lock for agent data update: {e}")
            return False


    async def add_memory(self, memory: str):
        if not self.memory:
            return
        # Verify MemoryStrategy.add is truly async if it performs I/O.
        await self.memory.add(MemoryItem(self.agent_id,memory))