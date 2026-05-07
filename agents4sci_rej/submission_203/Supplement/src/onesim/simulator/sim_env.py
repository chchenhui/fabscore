from collections import defaultdict
from typing import Any, List, Dict, Optional, Union, Set
from enum import Enum
from dataclasses import dataclass, field
from onesim.events import Event, DataEvent, DataResponseEvent, DataUpdateEvent, DataUpdateResponseEvent
from onesim.events import Scheduler, EventBus
from onesim.agent import GeneralAgent
import asyncio
from loguru import logger
import threading
import os
import time
import json
import inspect
from onesim.data import (
    TrailManager, TrailStatus, 
    EnvironmentStateManager, 
    EventManager, 
    DecisionManager,
    AgentManager
)
from onesim.distribution.node import get_node, NodeRole
from onesim.distribution.distributed_lock import get_lock
from onesim.config import get_component_registry
from datetime import datetime
# Use aiofiles for asynchronous file operations
import aiofiles
import aiofiles.os

class SimulationMode(Enum):
    """Enumeration of simulation modes."""
    TIMED = "tick"
    ROUND = "round"

class SimulationState(Enum):
    """Enumeration of simulation states."""
    INITIALIZED = "initialized"  # Initialization complete but not started
    RUNNING = "running"          # Running
    PAUSED = "paused"            # Paused
    COMPLETED = "completed"      # Completed normally
    TERMINATED = "terminated"    # Terminated externally
    ERROR = "error"              # Error occurred

@dataclass
class SimulationConfig:
    """Configuration class for simulation environment."""
    mode: SimulationMode = SimulationMode.TIMED
    max_steps: int = 1
    interval: float = 60.0
    bus_idle_timeout: float = 120.0  # Time to wait before considering event bus as idle
    export_training_data: bool = False  # Whether to export training data
    export_event_data: bool = False
    export_event_flow: bool = False  # Whether to export event flow data
    additional_config: Dict[str, Any] = field(default_factory=dict)
    collection_interval: int = 30

class BasicSimEnv:
    """
    A flexible simulation environment supporting timed and round-based workflows.

    """

    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        data: Optional[Dict[str, Any]] = None,
        start_targets: Optional[Dict[str, Any]] = None,
        end_targets: Optional[Dict[str, Any]] = None,
        config: Optional[Union[Dict[str, Any], SimulationConfig]] = None,
        agents: Optional[Dict[str, GeneralAgent]] = None,
        env_path: Optional[str] = None,
        trail_id: Optional[str] = None,
        intervention_engine=None,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the simulation environment.

        Args:
            name (str): Name of the environment.
            event_bus (Any): Event bus for dispatching events.
            listeners (Optional[Dict[str, List]]): Event listeners.
            children (Optional[List[Env]]): Child environments.
            data (Optional[Dict[str, Any]]): Initial shared data.
            start_targets (Optional[List[int]]): Initial start targets.
            config (Optional[Union[Dict[str, Any], SimulationConfig]]): Simulation configuration.
            trail_id (Optional[str]): The ID of the trail to save data to.
        """

        # Config initialization
        if isinstance(config, dict):
            config['bus_idle_timeout'] = config.get('bus_idle_timeout', 10.0)
            self.config = SimulationConfig(
                mode=SimulationMode(config.get('mode', SimulationMode.TIMED.value)),
                max_steps=config.get('max_steps', 3),
                interval=config.get('interval', 60.0),
                bus_idle_timeout=config.get('bus_idle_timeout', 30.0),
                export_training_data=config.get('export_training_data', False),
                export_event_data=config.get('export_event_data', False),
                export_event_flow=config.get('export_event_flow', False),
                additional_config=config.get('additional_config', {}),
                collection_interval=config.get('collection_interval', 30),
            )
        elif config is None:
            self.config = SimulationConfig()

        self.name = name
        # Original initialization
        self.data = data or {}
        self.start_targets = start_targets or {}
        self.end_targets = end_targets or {}
        self.ended_agents = {}
        for agent_type, ids in self.end_targets.items():
            for id in ids:
                self.ended_agents[id] = 0

        self.event_bus = event_bus
        self.mode = self.config.mode
        self.max_steps = self.config.max_steps
        self.agents = agents
        self.env_path = env_path
        # Store output directory for simulation results
        # If output_dir is not provided, it will be created by the caller
        self.output_dir = output_dir
        self.tot_time = 0.0
        self.current_step = 1 # Unified counter for rounds/triggers

        # Event handling
        self._queue = asyncio.Queue()
        self._event_schema = {}
        self._lock = asyncio.Lock()

        # Bus monitoring
        self._last_event_time = time.time()
        self._bus_idle_start = None

        # Unified state management
        self._state = SimulationState.INITIALIZED
        self._state_change_time = time.time()
        self._pause_cumulative_time = 0.0

        # Event control - use more intuitive names and logic
        self._pause_signal = asyncio.Event()  # Pause signal: set = paused, clear = running
        # Initially not paused, so clear the signal
        self._termination_signal = asyncio.Event()  # Termination signal
        # Signal when stop_simulation fully completed
        self.stopped_event = asyncio.Event()

        # Data storage managers
        self.trail_id = trail_id
        self._trail_manager = TrailManager() if trail_id else None
        self._env_state_manager = EnvironmentStateManager() if trail_id else None
        self._event_manager = EventManager() if trail_id else None
        self._decision_manager = DecisionManager() if trail_id else None
        self._agent_manager = AgentManager() if trail_id else None
        # Temporary storage for events and decisions to be saved at the end of each step
        self._pending_events = []
        self._pending_decisions = []

        # Set to track agents that have made decisions
        self._agent_decisions: Dict[int, Dict[str, int]] = {}  # step_num -> set of agent_ids

        # Runtime intervention engine
        self.intervention_engine = intervention_engine

        # Register data event handlers
        self.register_event("DataEvent", "handle_data_event")
        self.register_event("DataUpdateEvent", "handle_data_update_event")

        # Initialize futures dictionaries for event-based data access
        self._data_futures = {}
        self._data_update_futures = {}

        # Initialize dictionaries to store sync events for each request
        self._data_sync_events = {}
        self._data_update_sync_events = {}

        # Register data-related event handlers
        self.register_event("DataResponseEvent", "handle_data_response")
        self.register_event("DataUpdateResponseEvent", "handle_data_update_response")

        if self.mode == SimulationMode.ROUND:
            self._init_round_mode()
        else:
            self._init_timed_mode()
        # Initialize scheduler for timed mode
        # self.scheduler = Scheduler(self.event_bus) if self.mode == SimulationMode.TIMED else None

        # Create metrics directory path (directory creation moved to async methods)
        self.metrics_save_dir = self.get_metrics_directory()

        # Run async initialization steps
        asyncio.create_task(self.initialize())

    async def initialize(self):
        """Perform asynchronous initialization steps."""
        self.data["simulation_start_time"] = time.time()
        await self.load_initial_data()
        self.register_event("PauseEvent", "handle_pause_event")
        self.register_event("ResumeEvent", "handle_resume_event")

    def _init_round_mode(self):
        """Initialize round mode specific attributes."""
        self.current_step = 1
        self.ended_agents = {id: 0 for ids in self.end_targets.values() for id in ids}
        self.tot_time = 0.0
        self.scheduler = None

    def _init_timed_mode(self):
        """Initialize timed mode specific attributes."""
        self.current_step = 1
        self.agent_triggers = defaultdict(list)  # Track which steps each agent participated in
        self.scheduler = Scheduler(self.event_bus)

    def register_event(self, event_kind: str, method_name: str) -> None:
        """Register an event type with its handling method."""
        if event_kind not in self._event_schema:
            self._event_schema[event_kind] = []
        self._event_schema[event_kind].append(method_name)

    def add_event(self, event: Event) -> None:
        """Add an event to the environment's event queue."""
        self._queue.put_nowait(event)

    async def run(self) -> List[asyncio.Task]:
        """
        Initialize and return all environment tasks that need to be run.
        
        Returns:
            List[asyncio.Task]: List of tasks that need to be executed
        """
        try:
            # Save initial state (step 0) before starting
            if self.trail_id:
                await self._save_initial_state()
                # Update trail status to RUNNING
                await self._trail_manager.update_trail_status(self.trail_id, TrailStatus.RUNNING)
                logger.info(f"Trail {self.trail_id} status updated to RUNNING")

            tasks = []

            # Create event processing task
            event_processing = asyncio.create_task(
                self.process_events(),
                name=f"{self.name}_event_processing"
            )
            tasks.append(event_processing)

            # Create periodic metrics collection task if enabled and env path exists
            # Ensure metrics task is only created if directory can be used later
            if self.metrics_save_dir:
                self._metrics_collection_task = asyncio.create_task(
                    self._periodic_metrics_collection(),
                    name=f"{self.name}_metrics_collection"
                )
                tasks.append(self._metrics_collection_task)
            else:
                self._metrics_collection_task = None
                logger.warning(
                    "Metrics collection disabled: metrics directory not available."
                )

            if self.mode == SimulationMode.TIMED:
                # Initialize timed mode scheduling
                assert self.scheduler is not None

                # Create scheduler task
                # scheduler_task = asyncio.create_task(
                #     self.scheduler.run(),
                #     name=f"{self.name}_scheduler"
                # )
                # tasks.append(scheduler_task)
                await asyncio.sleep(0)  # Ensure scheduler has started
                await self._schedule_start_events_timed()
            else:
                # For round mode, start the first round
                await self.start()

            return tasks

        except Exception as e:
            logger.error(f"Error initializing environment tasks in {self.name}: {e}")
            raise

    async def _save_initial_state(self):
        """Save the initial environment state (step 0)"""
        if not self.trail_id or not self._env_state_manager:
            return

        try:
            # Create a copy of environment data to save
            state_data = await self.get_data(None)

            # Save as step 0
            await self._env_state_manager.save_state(
                trail_id=self.trail_id,
                step=0,
                state=state_data
            )

            # Save initial state for each agent
            if self._agent_manager and self.agents:
                for agent_type, agents in self.agents.items():
                    for agent_id, agent in agents.items():
                        try:
                            try:
                                # Check if agent exists by attempting to get it
                                existing_agent = await self._agent_manager.get_agent(
                                    trail_id=self.trail_id,
                                    agent_id=agent_id,
                                    universe_id="main"
                                )
                                # If agent doesn't exist, register it
                                if not existing_agent:
                                    # Extract essential info for registration
                                    agent_type = getattr(agent, 'agent_type', 'general')
                                    name = getattr(agent, 'name', agent_id)
                                    initial_profile = agent.get_profile() if hasattr(agent, 'get_profile') else {}
                                    system_prompt = getattr(agent, 'system_prompt', None)
                                    model_config = getattr(agent, 'model_config', None)
                                    memory_config = getattr(agent, 'memory_config', None)
                                    planning_config = getattr(agent, 'planning_config', None)

                                    # Register the agent
                                    success = await self._agent_manager.register_agent(
                                        trail_id=self.trail_id,
                                        agent_id=agent_id,
                                        agent_type=agent_type,
                                        name=name,
                                        initial_profile=initial_profile,
                                        universe_id="main",
                                        system_prompt=system_prompt,
                                        model_config_name=model_config,
                                        memory_config=memory_config,
                                        planning_config=planning_config
                                    )

                                    if not success:
                                        logger.warning(f"Failed to register agent {agent_id} for trail {self.trail_id}")
                                        continue
                                    else:
                                        logger.info(f"Registered agent {agent_id} for trail {self.trail_id}")
                            except Exception as e:
                                logger.error(f"Error checking/registering agent {agent_id}: {e}")
                                continue

                            # Extract agent data from its current state
                            profile = agent.get_profile() if hasattr(agent, 'get_profile') else None
                            memory = await agent.get_memory() if hasattr(agent, 'get_memory') else None
                            relationships = agent.get_all_relationships() if hasattr(agent, 'get_relationships') else None

                            # Additional state can contain any agent-specific data not covered by standard fields
                            additional_state = {
                                "current_step": 0, # Will change to current_step later if needed
                            
                            }
                            await self._agent_manager.save_agent_state(
                                trail_id=self.trail_id,
                                agent_id=agent_id,
                                step=0,
                                profile=profile,
                                memory=memory, 
                                relationships=relationships,
                                additional_state=additional_state,
                                universe_id="main"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save initial state for agent {agent_id}: {e}")

            logger.info(f"Saved initial environment state (step 0) for trail {self.trail_id}")
        except Exception as e:
            logger.error(f"Error saving initial environment state: {e}")

    async def _save_step_data(self, step_num: int):
        """Save environment state, events, and decisions for a completed step"""
        try:
            await self.collect_metrics()

            # 1. Save environment state
            if self._env_state_manager:
                state_data = await self.get_data(None)
                await self._env_state_manager.save_state(
                    trail_id=self.trail_id,
                    step=step_num,
                    state=state_data
                )

                # Save state for each agent at this step
                # Use self.agents.values() directly assuming it's a dict {type: {id: agent}}
                # or adjust if the structure is different {id: agent}
                agents_to_save = []
                if self._agent_manager and self.agents:
                    for agent_type_dict in self.agents.values():
                        agents_to_save.extend(agent_type_dict.items())

                    for agent_id, agent in agents_to_save:
                        try:
                            # We don't need to register agents here since they should have been
                            # registered in _save_initial_state or elsewhere before steps begin

                            # Extract agent data asynchronously if methods are async
                            profile = agent.get_profile() if hasattr(agent, 'get_profile') else None
                            memory = await agent.get_memory() if hasattr(agent, 'get_memory') else None
                            relationships = agent.get_all_relationships() if hasattr(agent, 'get_relationships') else None

                            additional_state = {
                                "current_step": step_num, 
                            }

                            await self._agent_manager.save_agent_state(
                                trail_id=self.trail_id,
                                agent_id=agent_id,
                                step=step_num,
                                profile=profile,
                                memory=memory,
                                relationships=relationships,
                                additional_state=additional_state,
                                universe_id="main"
                            )
                        except Exception as e:
                            logger.warning(f"Failed to save state for agent {agent_id} at step {step_num}: {e}")

            # 2. Save pending events
            if self._event_manager and self._pending_events:
                for event_data in self._pending_events:
                    event_data['payload']=event_data.pop('data',{})
                    await self._event_manager.create_event(
                        trail_id=self.trail_id,
                        **event_data
                    )
                logger.info(f"Saved {len(self._pending_events)} events for step {step_num}")
                self._pending_events = []  # Clear after saving

            # 3. Save pending decisions
            if self._decision_manager and self._pending_decisions:
                for decision_data in self._pending_decisions:
                    await self._decision_manager.record_decision(
                        trail_id=self.trail_id,
                        **decision_data
                    )
                logger.info(f"Saved {len(self._pending_decisions)} decisions for step {step_num}")

            # Update step count in trail
            if self._trail_manager:
                await self._trail_manager.increment_step(self.trail_id)
                # Queue metrics for DB storage if trail_id is available

            if self.trail_id and self._env_state_manager:
                metrics = {
                    'step_id': self.current_step, 
                    'duration': self.data['step_data'][self.current_step]['duration'],
                }

                # Update trail metadata with these metrics
                if self._trail_manager:
                    step_key = f'step_{self.current_step}' 
                    await self._trail_manager.update_trail_metadata(
                        self.trail_id, 
                        {step_key: metrics},
                        merge=True
                    )

            # Export training data if enabled in config
            if self.config.export_training_data:
                dataset_dir = self.get_datasets_directory()

                # Create dataset directory if it doesn't exist
                await aiofiles.os.makedirs(dataset_dir, exist_ok=True)

                # Use human-readable timestamp for filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = os.path.join(dataset_dir, f"decisions_{timestamp}.json")

                # Make export directory creation async
                try:
                    await aiofiles.os.makedirs(dataset_dir, exist_ok=True)
                except Exception as e:
                    logger.error(f"Failed to create dataset directory {dataset_dir}: {e}")
                    # Decide how to handle this - maybe skip export?
                    # For now, log and continue, export might fail

                try:
                    # Try to export from database if trail_id exists
                    if self.trail_id:
                        try:
                            from onesim.data import DecisionManager
                            decision_mgr = DecisionManager()
                            data = await decision_mgr.export_training_data(trail_id=self.trail_id)

                            # Use aiofiles for writing
                            async with aiofiles.open(export_path, "w") as f:
                                await f.write(data) # Assuming data is already a JSON string
                            logger.info(f"Exported training data to {export_path} from database")
                        except Exception as db_error:
                            logger.warning(f"Failed to export data from database: {db_error}. Falling back to direct export.")
                            # Make the fallback async
                            await self._export_data_from_pending_decisions(export_path)
                    else:
                        # Direct export from pending decisions when not using database (make it async)
                        await self._export_data_from_pending_decisions(export_path)
                except Exception as e:
                    logger.error(f"Error exporting training data: {e}")

            # Export metrics as images if save directory is available
            monitor_dir = self.get_metrics_directory()
            if monitor_dir:
                try:
                    # Create step-specific directory asynchronously
                    step_dir = os.path.join(monitor_dir, f'step_{self.current_step}')
                    await aiofiles.os.makedirs(step_dir, exist_ok=True)

                    profiles_dir=os.path.join(step_dir,"profiles")
                    await aiofiles.os.makedirs(profiles_dir, exist_ok=True) # Make async
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    profiles_path=os.path.join(profiles_dir,f"profiles_{timestamp}.json")
                    profiles=[]
                    # Iterate through agents correctly
                    agents_to_profile = []
                    if self.agents:
                        for agent_type_dict in self.agents.values():
                            agents_to_profile.extend(agent_type_dict.values())

                    for agent in agents_to_profile:
                        profile = agent.get_profile() if hasattr(agent, 'get_profile') else None
                        if profile:
                            profiles.append(profile)

                    # Write profiles asynchronously
                    try:
                        async with aiofiles.open(profiles_path, "w") as f:
                            # json.dumps is synchronous, consider run_in_executor for very large profiles list
                            await f.write(json.dumps(profiles))
                    except Exception as e:
                        logger.error(f"Error saving agent profiles to {profiles_path}: {e}")

                    # Get the monitor manager from registry
                    registry = get_component_registry()
                    monitor_manager = registry.get_instance("monitor")

                    if monitor_manager:
                        monitor_manager.export_metrics_as_images(step_dir, self.current_step)
                        logger.info(f"Metrics plots for step {self.current_step} saved to {step_dir}")
                    else:
                        logger.warning("No monitor manager found in registry for metrics")

                except Exception as e:
                    logger.error(f"Error saving metrics plots: {e}")

        except Exception as e:
            logger.error(f"Error saving step {step_num} data: {e}")
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Error saving step {step_num} data: {e}\nTraceback: {error_traceback}")

    async def _execute_interventions(self, step_num: int):
        """Execute runtime interventions at the end of each step"""
        if not self.intervention_engine:
            return

        try:
            # Get scheduled interventions for this step
            scheduled_interventions = (
                self.intervention_engine.get_scheduled_interventions(step_num)
            )
            for intervention_item in scheduled_interventions:
                await self.intervention_engine.execute_intervention(
                    intervention_item['spec'], self
                )

            # Check conditional interventions
            simulation_context = {
                'current_step': step_num,
                'environment': self,
                'agents': self.agents,
            }
            conditional_interventions = (
                self.intervention_engine.get_conditional_interventions(
                    simulation_context, step_num
                )
            )
            for intervention_item in conditional_interventions:
                await self.intervention_engine.execute_intervention(
                    intervention_item['spec'], self
                )

        except Exception as e:
            logger.error(f"Error executing interventions at step {step_num}: {e}")

    # Methods to queue events and decisions for later saving
    async def queue_event(self, event_data: Dict[str, Any]):
        """Queue an event to be saved at the end of the step and broadcast it"""
        if event_data['event_type'] in [
            'DataEvent',
            'DataUpdateEvent',
            'DataResponseEvent',
            'DataUpdateResponseEvent',
            'PauseEvent',
            'ResumeEvent',
        ]:
            return
        self._pending_events.append(event_data)

        # Get environment name from path
        env_name = os.path.basename(self.env_path)
        # Add step number to event data if available
        if hasattr(self, 'current_step'):
            event_data['step'] = self.current_step

        import sys
        if 'backend' not in sys.modules:
            return
        from backend.utils.websocket import connection_manager

        # Check if there are active WebSocket connections before broadcasting
        if connection_manager.has_active_connections(env_name):
            # Send any buffered events first (only once)
            await self._send_buffered_events(env_name, connection_manager)
            # Then broadcast current event
            asyncio.create_task(
                connection_manager.broadcast_event(env_name, event_data)
            )
        else:
            # If no connections, buffer all events
            if not hasattr(self, '_buffered_events'):
                self._buffered_events = []
            self._buffered_events.append(event_data)

    async def _send_buffered_events(self, env_name: str, connection_manager):
        """Send all buffered events to frontend and clear the buffer"""
        if hasattr(self, '_buffered_events') and self._buffered_events:
            logger.info(
                f"Sending {len(self._buffered_events)} buffered events to {env_name}"
            )
            for buffered_event in self._buffered_events:
                await connection_manager.broadcast_event(env_name, buffered_event)
            # Clear the buffer after sending (ensure events are sent only once)
            self._buffered_events = []

    async def queue_decision(self, decision_data: Dict[str, Any]):
        """Queue a decision to be saved at the end of the step"""
        if 'step' not in decision_data or decision_data['step'] == None:
            decision_data['step'] = self.current_step
        self._pending_decisions.append(decision_data)

        # Track which agents have made decisions this step
        agent_type = decision_data.get('agent_type')
        if agent_type:
            current_step_val = self.current_step # Use the unified counter
            if current_step_val not in self._agent_decisions:
                self._agent_decisions[current_step_val] = {}
            if agent_type not in self._agent_decisions[current_step_val]:
                self._agent_decisions[current_step_val][agent_type]=0
            self._agent_decisions[current_step_val][agent_type]+=1

    async def process_events(self) -> None:
        """Process events from the event queue."""
        logger.info(f"Starting event processing in {self.name}")

        while True:
            try:
                # Check if terminated
                if self.is_terminated():
                    logger.info(f"Termination requested, stopping event processing in {self.name}")
                    break

                # Check pause signal, wait if set
                if self._pause_signal.is_set():
                    logger.debug(f"Simulation {self.name} paused, waiting for resume")
                    # Wait directly for the pause signal to be cleared (resume)
                    while self._pause_signal.is_set() and not self.is_terminated():
                        await asyncio.sleep(0.2)  # Use a fixed sleep time instead of waiting

                    # Check if terminated
                    if self.is_terminated():
                        logger.info(f"Termination requested during pause, stopping event processing in {self.name}")
                        break

                    logger.debug(f"Simulation {self.name} resuming event processing")

                try:
                    # Use a shorter timeout during active processing to be more responsive
                    # to pause/resume/stop signals
                    event = await asyncio.wait_for(self._queue.get(), timeout=30.0)
                    current_time = time.time()
                    self._last_event_time = current_time

                    await self.handle_event(event)
                    self._queue.task_done()

                except asyncio.TimeoutError:
                    # Skip completion checks if paused
                    if self._pause_signal.is_set():
                        continue

                    # Check completion for both modes
                    if self.mode == SimulationMode.ROUND:
                        await self._check_round_completion()
                    else:
                        await self._check_timed_completion()
                    continue

            except asyncio.CancelledError:
                break
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                logger.error(f"Error in event processing: {e}\n{error_detail}")
                continue

    async def _periodic_metrics_collection(self):
        """Periodically collect metrics from the simulation environment"""
        logger.info(f"Starting periodic metrics collection task for {self.name}")
        while True:
            try:
                # Exit if simulation is terminated
                if self.is_terminated():
                    logger.info(f"Metrics collection task stopping - simulation terminated")
                    break

                # Skip collection if paused
                if not self._pause_signal.is_set():
                    await self.collect_metrics()

                # Wait for next collection interval
                await asyncio.sleep(self.config.collection_interval)

            except asyncio.CancelledError:
                logger.info(f"Metrics collection task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.config.collection_interval)  # Wait before trying again

    async def _check_round_completion(self):
        """Check for round completion in round mode with dynamic timeout."""
        logger.warning("Checking step completion for ROUND mode")
        current_time = time.time()

        # If paused or terminated, skip the check
        if self._pause_signal.is_set() or self.is_terminated():
            logger.debug(f"Skipping step completion check - paused: {self._pause_signal.is_set()}, terminated: {self.is_terminated()}")
            return

        # Prevent duplicate completion checks for the same step
        step_start_time_key = f'step_{self.current_step}_time_start'
        step_end_time_key = f'step_{self.current_step}_time_end'

        # If this step has already been completed, skip
        if self.data.get(step_end_time_key, 0) > 0:
            logger.debug(
                f"Step {self.current_step} already completed, skipping duplicate check"
            )
            return

        # Calculate dynamic timeout
        idle_timeout = self.config.bus_idle_timeout # Default timeout

        # Round can complete if either:
        # 1. All agents have terminated
        # 2. No events for dynamic time
        all_terminated = all(count >= self.current_step for count in self.ended_agents.values())
        long_idle = (
            (current_time - self._last_event_time > idle_timeout) and
            self.event_bus.is_empty() and
            self._queue.empty()
        )

        # Save round data before moving to next round

        if all_terminated or long_idle:
            if long_idle:
                logger.warning(f"Step {self.current_step} (Round Mode) completed due to idle timeout (dynamic timeout: {idle_timeout:.1f}s)")

            step_duration = current_time - self.data.get(step_start_time_key, current_time)

            # Store round duration in round_data before saving
            if 'step_data' not in self.data: self.data['step_data'] = {}
            if self.current_step not in self.data['step_data']: self.data['step_data'][self.current_step] = {}
            self.data['step_data'][self.current_step]['duration'] = step_duration
            self.tot_time += step_duration

            logger.info(f"Step {self.current_step} (Round Mode) Time: {step_duration:.2f} seconds")
            logger.info(f"Total Time: {self.tot_time:.2f} seconds")
            # Mark this step as completed to prevent duplicate processing
            self.data[step_end_time_key] = current_time

            # Save round data *before* potentially stopping
            await self._save_step_data(self.current_step)

            if long_idle:
                # Log which agents didn't complete
                incomplete = [agent_id for agent_id, count in self.ended_agents.items()
                            if count < self.current_step]
                # logger.warning(f"{len(incomplete)} agents didn't complete: {incomplete}")

            # Reset pause time accumulator for the next round
            self._pause_cumulative_time = 0.0
            self._last_event_time = time.time()
            if self.current_step < self.max_steps:
                self.current_step += 1
                await self.start()
            else:
                logger.info("All steps (rounds) completed")
                # Update trail status before stopping
                if self.trail_id and self._trail_manager:
                    await self._trail_manager.update_trail_status(self.trail_id, TrailStatus.COMPLETED)
                    logger.info(f"Trail {self.trail_id} status updated to COMPLETED")
                # Call stop_simulation to properly terminate in both single and distributed modes
                await self.stop_simulation()

    async def _check_timed_completion(self):
        """Check for simulation completion in timed mode."""
        current_time = time.time()

        # If paused or terminated, skip check
        if self._pause_signal.is_set() or self.is_terminated():
            logger.debug(f"Skipping timed completion check - paused: {self._pause_signal.is_set()}, terminated: {self.is_terminated()}")
            return

        # Check for long idle
        long_idle = (current_time - self._last_event_time > self.config.bus_idle_timeout and
                    self.event_bus.is_empty() and
                    self._queue.empty())

        if long_idle and self.scheduler.is_done():
            # Log completion statistics
            stats = self.get_statistics()
            logger.info("Simulation completed:")
            logger.info(f"Total triggers executed: {stats['total_triggers']}")
            logger.info(f"Agent completion distribution: {stats['completion_distribution']}")

            # Add pause time to statistics if there was any
            if self._pause_cumulative_time > 0:
                logger.info(f"Total time paused: {self._pause_cumulative_time:.2f} seconds")
                stats['total_pause_time'] = self._pause_cumulative_time

            await self._save_step_data(self.current_step)
            # Update trail status to COMPLETED
            if self._trail_manager:
                await self._trail_manager.update_trail_status(self.trail_id, TrailStatus.COMPLETED)
                logger.info(f"Trail {self.trail_id} status updated to COMPLETED")

            # Call stop_simulation to properly terminate in both single and distributed modes
            await self.stop_simulation()

    async def handle_event(self, event: Event) -> None:
        """
        Handle a single event.
        
        Args:
            event (Event): The event to handle
        """
        # Record the event for broadcasting
        await self.queue_event(event.to_dict())

        # Special handling for Pause/Resume events
        if event.event_kind == "PauseEvent":
            if not self._pause_signal.is_set() and event.to_agent_id in ["ENV", "all"]:
                await self.pause_simulation()
                return
        elif event.event_kind == "ResumeEvent":
            if self._pause_signal.is_set() and event.to_agent_id in ["ENV", "all"]:
                await self.resume_simulation()
                return

        # If paused, skip handling non-system events
        if self._pause_signal.is_set() and event.event_kind not in ["EndEvent", "PauseEvent", "ResumeEvent"]:
            logger.debug(f"Skipping event processing during pause: {event.event_kind}")
            return

        if event.event_kind not in self._event_schema:
            logger.warning(f"Event type {event.event_kind} not registered in environment {self.name}")
            return

        for method_name in self._event_schema[event.event_kind]:
            method = getattr(self, method_name, None)
            if not callable(method):
                logger.error(f"Method {method_name} not found in {self.__class__.__name__}")
                continue

            try:
                if inspect.iscoroutinefunction(method):
                    await method(event)
                else:
                    method(event)
            except Exception as e:
                logger.error(f"Error handling event {event.event_kind} with method {method_name}: {e}")

    async def load_initial_data(self):
        """Load initial environment data asynchronously."""
        default_env_file = os.path.join(self.env_path, 'env_data.json') if self.env_path else None
        env_file = self.config.additional_config.get('env_data_path', default_env_file)

        if env_file and await aiofiles.os.path.exists(env_file):
            try:
                async with aiofiles.open(env_file, 'r') as f:
                    data = json.loads(await f.read())
                # Update data using the async lock
                async with self._lock:
                    self.data.update(data)
                logger.info(f"Loaded initial environment data from {env_file}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from environment file: {env_file}")
            except Exception as e:
                logger.error(f"Error loading environment data from {env_file}: {e}")
        elif env_file:
            logger.warning(f"Initial environment data file not found: {env_file}")
        else:
            logger.info("No initial environment data file specified.")

    async def handle_pause_event(self, event: Event) -> None:
        """Handle a pause event."""
        if not self._pause_signal.is_set() and (event.to_agent_id == "ENV" or event.to_agent_id == "all"):
            await self.pause_simulation()

    async def handle_resume_event(self, event: Event) -> None:
        """Handle a resume event."""
        if self._pause_signal.is_set() and (event.to_agent_id == "ENV" or event.to_agent_id == "all"):
            await self.resume_simulation()

    async def collect_metrics(self):
        """Collect metrics for the current round and store them."""
        if 'step_data' not in self.data:
            self.data['step_data'] = {}

        if self.current_step not in self.data['step_data']:
            self.data['step_data'][self.current_step] = {}

        self.data['step_data'][self.current_step]['step_id'] = self.current_step

        # Add more metrics as needed
        current_time = time.time()
        step_start_time_key = f'step_{self.current_step}_time_start'
        self.data['step_data'][self.current_step]['duration'] = current_time - self.data.get(step_start_time_key, current_time)

        # Count events in this round
        event_count = len(self._pending_events) if hasattr(self, '_pending_events') else 0
        self.data['step_data'][self.current_step]['event_count'] = event_count
        logger.info(f"Event count for step {self.current_step}: {event_count} events")

        # Get token usage statistics from all nodes in distributed mode
        try:
            # Check if we're in distributed mode with a master node
            node = get_node()
            is_distributed = node and node.role == NodeRole.MASTER

            if is_distributed:
                # Get master node token usage stats
                from onesim.models.utils.token_usage import get_token_usage_stats
                master_stats = get_token_usage_stats()

                # Initialize merged stats with master's stats
                merged_stats = {
                    "total_prompt_tokens": master_stats.get("total_prompt_tokens", 0),
                    "total_completion_tokens": master_stats.get("total_completion_tokens", 0),
                    "total_tokens": master_stats.get("total_tokens", 0),
                    "request_count": master_stats.get("request_count", 0),
                    "model_usage": master_stats.get("model_usage", {}),
                    "worker_stats": {"master": master_stats}
                }

                # Collect from all worker nodes using grpc client functions
                master_node = node
                for worker_id, worker_info in master_node.workers.items():
                    try:
                        # Import the function to get token usage from a worker
                        from onesim.distribution.grpc_impl import get_token_usage_from_worker
                        worker_stats = await get_token_usage_from_worker(
                            worker_info.address,
                            worker_info.port,
                            worker_id
                        )

                        if worker_stats:
                            # Record worker's statistics
                            merged_stats["worker_stats"][worker_id] = worker_stats

                            # Add to total counts
                            merged_stats["total_prompt_tokens"] += worker_stats.get("total_prompt_tokens", 0)
                            merged_stats["total_completion_tokens"] += worker_stats.get("total_completion_tokens", 0)
                            merged_stats["total_tokens"] += worker_stats.get("total_tokens", 0)
                            merged_stats["request_count"] += worker_stats.get("request_count", 0)

                            # Merge model usage
                            for model, usage in worker_stats.get("model_usage", {}).items():
                                if model not in merged_stats["model_usage"]:
                                    merged_stats["model_usage"][model] = {
                                        "prompt_tokens": 0,
                                        "completion_tokens": 0,
                                        "total_tokens": 0,
                                        "request_count": 0
                                    }

                                merged_stats["model_usage"][model]["prompt_tokens"] += usage.get("prompt_tokens", 0)
                                merged_stats["model_usage"][model]["completion_tokens"] += usage.get("completion_tokens", 0)
                                merged_stats["model_usage"][model]["total_tokens"] += usage.get("total_tokens", 0)
                                merged_stats["model_usage"][model]["request_count"] += usage.get("request_count", 0)
                    except Exception as e:
                        logger.error(f"Error collecting token usage from worker {worker_id}: {e}")

                # Use the merged stats
                token_stats = merged_stats
                logger.info(f"Collected token usage from {len(merged_stats.get('worker_stats', {}))} workers")
            else:
                # Standard non-distributed mode
                from onesim.models.utils.token_usage import get_token_usage_stats
                token_stats = get_token_usage_stats()

            # Add token usage to round data
            self.data['step_data'][self.current_step]['token_usage'] = {
                'total_tokens': token_stats.get('total_tokens', 0),
                'total_prompt_tokens': token_stats.get('total_prompt_tokens', 0),
                'total_completion_tokens': token_stats.get('total_completion_tokens', 0),
                'request_count': token_stats.get('request_count', 0),
                'model_usage': token_stats.get('model_usage', {})
            }

            # If distributed, also store worker-specific stats
            if is_distributed and 'worker_stats' in token_stats:
                self.data['step_data'][self.current_step]['token_usage']['worker_stats'] = token_stats.get('worker_stats', {})

            logger.info(f"Token usage for step {self.current_step}: {token_stats.get('total_tokens', 0)} tokens")

            # Get the monitor manager from registry
            registry = get_component_registry()
            monitor_manager = registry.get_instance("monitor")

            if monitor_manager:
                logger.info(f"Exporting metrics for step {self.current_step}")
                monitor_dir = self.get_metrics_directory()
                if monitor_dir:
                    step_dir = os.path.join(monitor_dir, f'step_{self.current_step}')
                    await aiofiles.os.makedirs(step_dir, exist_ok=True)
                    # 使用新的统一导出接口
                    monitor_manager.export_metrics_as_images(
                        step_dir, self.current_step
                    )
                    logger.info(f"Metrics plots saved to {step_dir}")
                else:
                    logger.warning(
                        "Monitor directory not available - skipping metrics export"
                    )
            else:
                logger.warning("No monitor manager found in registry for metrics")
        except ImportError:
            logger.warning("Token usage module not available, skipping token statistics")
        except Exception as e:
            logger.error(f"Error collecting token statistics: {e}")

    def describe(self, **kwargs: Any) -> str:
        """
        Provide a descriptive string of the current environment state.

        Returns:
            str: Description of the environment's current state.
        """
        return (
            f"SimEnv {self.name} "
            f"current data: {self.data}, "
            f"mode: {self.mode.value}, "
            f"step: {self.current_step}"
        )

    def get_env_id(self) -> str:
        """
        Retrieve the environment's unique identifier.

        Returns:
            int: Environment ID (default implementation returns 0).
        """
        return "ENV"

    def get_metrics_directory(self) -> Optional[str]:
        """
        Get the directory path for metrics and monitor output.

        Returns:
            Optional[str]: Path to metrics directory, or None if output_dir is not set
        """
        if not self.output_dir:
            return None

        # Check for custom metrics or monitor directory in additional_config
        # Support both metrics_output_dir and monitor_output_dir for backward compatibility
        custom_dir = self.config.additional_config.get('metrics_output_dir')
        if custom_dir:
            # If it's an absolute path, use it directly
            if os.path.isabs(custom_dir):
                return custom_dir
            # If it's a relative path, make it relative to output_dir
            return os.path.join(self.output_dir, custom_dir)

        # Default to metrics_plots subdirectory in output_dir
        return os.path.join(self.output_dir, 'metrics_plots')

    def get_datasets_directory(self) -> Optional[str]:
        """
        Get the directory path for training data export.

        Returns:
            Optional[str]: Path to datasets directory, or None if output_dir is not set
        """
        if not self.output_dir:
            return None

        # Check for custom datasets directory in additional_config
        custom_dir = self.config.additional_config.get('datasets_output_dir')
        if custom_dir:
            # If it's an absolute path, use it directly
            if os.path.isabs(custom_dir):
                return custom_dir
            # If it's a relative path, make it relative to output_dir
            return os.path.join(self.output_dir, custom_dir)

        # Default to datasets subdirectory in output_dir
        return os.path.join(self.output_dir, 'datasets')

    def get_events_directory(self) -> Optional[str]:
        """
        Get the directory path for event data export.

        Returns:
            Optional[str]: Path to events directory, or None if output_dir is not set
        """
        if not self.output_dir:
            return None

        # Check for custom events directory in additional_config
        custom_dir = self.config.additional_config.get('events_output_dir')
        if custom_dir:
            # If it's an absolute path, use it directly
            if os.path.isabs(custom_dir):
                return custom_dir
            # If it's a relative path, make it relative to output_dir
            return os.path.join(self.output_dir, custom_dir)

        # Default to events subdirectory in output_dir
        return os.path.join(self.output_dir, 'events')

    async def _schedule_start_events_timed(self) -> None:
        """Schedule start events for each target in timed mode."""
        if not self.start_targets:
            logger.warning("No start targets defined for timed mode simulation")
            return

        for agent_type in self.start_targets.keys():
            for target_id in self.start_targets[agent_type]:
                start_event = await self._create_start_event(target_id)
                assert self.scheduler is not None
                await self.queue_event(start_event.to_dict())
                self.scheduler.schedule_task(
                    self.config.interval, 
                    start_event,
                    max_count=self.config.max_steps
                )

    async def _create_start_event(self, target_id: int) -> Event:
        """
        Create a start event for a specific target.

        Args:
            target_id (int): Target ID to create start event for.

        Returns:
            Event: Newly created start event.

        Raises:
            NotImplementedError: Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses must implement _create_start_event method")

    async def start(self, **kwargs: Any) -> None:
        """Trigger start event to begin or continue the workflow."""
        # Execute runtime interventions if enabled
        await self._execute_interventions(self.current_step)

        if self.mode == SimulationMode.ROUND:
            if self.current_step > self.max_steps:
                logger.info("Maximum steps (rounds) reached. No more steps will be started.")
                return

            logger.info(f"Starting step {self.current_step} (Round Mode)")
            self.data[f'step_{self.current_step}_time_start'] = time.time()

            # Dispatch start events for all targets
            for agent_type in self.start_targets.keys():
                for target_id in self.start_targets[agent_type]:
                    start_event = await self._create_start_event(target_id)
                    await self.queue_event(start_event.to_dict())
                    await self.event_bus.dispatch_event(start_event)

    def terminate(self, event: Event, **kwargs: Any) -> None:
        """Handle agent termination with bus state awareness. This should be async due to lock usage"""
        # This method acquires self._lock implicitly via _check_round_completion
        # Let's make it explicitly async to handle the lock correctly if needed directly.
        # However, the current logic calls _check_round_completion which is async.
        # We only need to ensure the lock usage within this function or called functions is async.

        # Skip if we're in paused state - we'll process this after resume
        if self._pause_signal.is_set():
            logger.info(f"Received termination event during pause, will process after resume: {event}")
            return

        if self.mode == SimulationMode.ROUND:
            logger.info(f"Step Workflow (Round Mode) End Info: {event}")
            agent_id = event.from_agent_id
            # Use lock correctly if modifying shared state directly here
            # Current logic updates ended_agents and calls async check function
            if agent_id in self.ended_agents:
                # Ensure update is atomic if needed, though check_round_completion handles the logic
                self.ended_agents[agent_id] = min(self.ended_agents[agent_id]+1,self.current_step)
            else:
                # Agent not in end_targets? Log warning.
                logger.warning(f"Agent {agent_id} terminated but was not in end_targets.")
                # Optionally add it if needed: self.ended_agents[agent_id] = self.current_step

            self._last_event_time = time.time() # Update last event time
            # Schedule the async check instead of awaiting it here to avoid blocking terminate
            asyncio.create_task(self._check_round_completion())
        else:
            # For timed mode, increment agent's trigger count
            # This access to agent_triggers might need locking if modified elsewhere concurrently
            # Assuming it's only updated here based on termination events simplifies things.
            self.agent_triggers[event.from_agent_id].append(self.current_step) # Store current_step when agent finished
            self._last_event_time = time.time()
            self.current_step += 1 # Increment total steps processed for timed mode (this might be an issue, see below)

    async def get_data(self, key: str=None, default: Optional[Any] = None) -> Any:
        """Get value from shared data, supporting multi-level dot notation. (Async due to lock)"""
        # Use asyncio.Lock
        async with self._lock:
            if key is None:
                # Return a deep copy if nested structures need isolation, shallow copy otherwise
                # return copy.deepcopy(self.data) # Requires import copy
                return self.data.copy() # Return a shallow copy

            parts = key.split('.')
            value = self.data

            for part in parts:
                if isinstance(value, dict):
                    if part in value:
                        value = value[part]
                    else:
                        return default # Key not found in dictionary
                elif isinstance(value, list):
                    try:
                        idx = int(part)
                        if 0 <= idx < len(value):
                            value = value[idx]
                        else:
                            return default # Index out of bounds
                    except (ValueError, TypeError):
                        return default # Part is not a valid integer index
                else:
                    # Cannot traverse further if it's not a dict or list
                    return default 

            return value

    async def get(self, key: str=None, default: Optional[Any] = None) -> Any:
        """Alias for get_data for dictionary-like access."""
        return await self.get_data(key, default)

    async def update_data(self, key: str, data: Any) -> Any:
        """Update shared data. (Async due to lock)"""
        # Use asyncio.Lock
        async with self._lock:
            # Handle nested updates if needed, e.g., key = "a.b.c"
            # This simple implementation only updates top-level keys
            self.data[key] = data
            return data

    def get_statistics(self) -> Dict[str, Any]:
        """Get the simulation statistics."""
        # Calculate basic statistics
        stats = {
            'total_steps': self.current_step, 
            'mode': self.mode.value,
            'total_agents': len(self.end_targets),
            'paused': self._pause_signal.is_set(),
            'total_pause_time': self._pause_cumulative_time
        }

        if self.mode == SimulationMode.ROUND:
            stats.update({
                'total_steps': self.current_step, 
                'total_time': self.tot_time,
                'step_times': {i: self.data.get(f'step_{i}_time')  
                               for i in range(1, self.current_step + 1)},
                'completed_agents': sum(1 for count in self.ended_agents.values() 
                                     if count >= self.current_step)
            })
        else: # TIMED mode
            stats.update({'completion_distribution': {}})

            trigger_counts_per_agent = [len(triggers) for triggers in self.agent_triggers.values()]

            # Initialize the distribution dictionary
            # Counts how many agents completed N triggers
            completion_counts = defaultdict(int)
            for agent_id, completed_triggers_list in self.agent_triggers.items():
                completion_counts[len(completed_triggers_list)] += 1

            # Transform the distribution into the desired format (cumulative distribution)
            # How many agents completed *at least* N triggers
            cumulative_distribution = {}

            # max_completed_triggers could be self.config.max_steps or max(completion_counts.keys())
            # Let's use self.config.max_steps as the upper limit for reporting
            sorted_unique_completions = sorted(completion_counts.keys())

            # Calculate agent completion distribution: for each number of triggers completed (X), how many agents completed X triggers.
            # Then, calculate cumulative: how many agents completed *at least* X triggers.
            agent_completion_counts = defaultdict(int)
            for agent_id, triggers_participated in self.agent_triggers.items():
                agent_completion_counts[len(triggers_participated)] += 1

            cumulative_distribution_map = {}
            num_total_relevant_agents = len(self.agent_triggers) # Agents that participated in at least one trigger

            # Iterate from max_steps down to 1 for "at least" count
            for i in range(self.config.max_steps, 0, -1):
                count_at_least_i = 0
                for completed_count, num_agents in agent_completion_counts.items():
                    if completed_count >= i:
                        count_at_least_i += num_agents
                cumulative_distribution_map[i] = count_at_least_i

            stats['completion_distribution'] = cumulative_distribution_map

        return stats

    async def stop_simulation(self):
        """
        Stop the simulation gracefully in both single and distributed modes.
        Sends termination signal to all registered agents.
        """
        logger.info("Stopping simulation...")

        # Set state to terminated
        await self.set_simulation_state(SimulationState.TERMINATED, reason="user_requested")

        # Stop external MonitorManager tasks early to avoid post-termination RPCs
        try:
            registry = get_component_registry()
            monitor_manager = registry.get_instance("monitor")
            if monitor_manager and hasattr(monitor_manager, 'stop_all_metrics'):
                await monitor_manager.stop_all_metrics()
                logger.info("Stopped MonitorManager metric tasks")
        except Exception as e:
            logger.warning(f"Failed to stop MonitorManager tasks: {e}")

        # Cancel the metrics collection task if it exists
        if hasattr(self, '_metrics_collection_task') and self._metrics_collection_task is not None:
            if not self._metrics_collection_task.done():
                self._metrics_collection_task.cancel()
                try:
                    await asyncio.wait_for(self._metrics_collection_task, timeout=10.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    # Expected when cancelling
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling metrics collection task: {e}")
            logger.info("Metrics collection task cancelled")

        # Save any remaining data if not already saved
        await self._save_step_data(self.current_step)

        # Update trail status if not already updated
        if self._trail_manager:
            await self._trail_manager.update_trail_status(self.trail_id, TrailStatus.COMPLETED)

        # Create a global termination event using EndEvent class
        from onesim.events import EndEvent
        # 保持一致逻辑：先向所有Agent广播EndEvent（包括分布式场景）
        termination_event = EndEvent(
            from_agent_id="ENV", to_agent_id="all", reason="simulation_completed"
        )

        # Export event flow data if enabled in config
        if hasattr(self, 'config') and self.config.export_event_flow:
            output_file = self.config.additional_config.get('event_flow_output_file')
            logger.info("Exporting event flow visualization data...")
            try:
                flow_data = await self.event_bus.export_event_flow_data(output_file)
                logger.info(
                    f"Event flow data exported: {len(flow_data['flows'])} flows"
                )
            except Exception as e:
                logger.error(f"Error exporting event flow data: {e}")

        # Send termination event to event bus
        await self.event_bus.dispatch_event(termination_event)

        # For distributed mode, we need to make sure the event is propagated
        # to all nodes. The event bus will handle this based on the "all" target.
        logger.info("Termination event dispatched to all agents")

        # Stop the scheduler if it exists
        if self.scheduler:
            self.scheduler.stop()
            logger.info("Scheduler stopped")

        # Record final state (use async lock if modifying shared data)
        # This modifies self.data, use the lock
        async with self._lock:
            self.data["simulation_complete"] = True
            end_time = time.time()
            self.data["simulation_end_time"] = end_time
            self.data["total_simulation_time"] = end_time - self.data.get("simulation_start_time", end_time)

        # Final export of training data at simulation end if enabled
        if self.config.export_training_data:
            # This involves file I/O, should be async

            dataset_dir = self.get_datasets_directory()

            # Create dataset directory if it doesn't exist (async)
            await aiofiles.os.makedirs(dataset_dir, exist_ok=True)

            # Use human-readable timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(dataset_dir, f"decisions_{timestamp}.json")

            try:
                # Try to export from database if trail_id exists
                if self.trail_id:
                    try:
                        from onesim.data import DecisionManager
                        decision_mgr = DecisionManager()
                        data = await decision_mgr.export_training_data(trail_id=self.trail_id)

                        # Write async
                        async with aiofiles.open(export_path, "w") as f:
                            await f.write(data) # Assuming data is already JSON string
                        logger.info(f"Exported training data to {export_path} from database")
                    except Exception as db_error:
                        logger.warning(f"Failed to export data from database: {db_error}. Falling back to direct export.")
                        # Use async export
                        await self._export_data_from_pending_decisions(export_path)
                else:
                    # Direct export from pending decisions when not using database (async)
                    await self._export_data_from_pending_decisions(export_path)
            except Exception as e:
                logger.error(f"Error exporting training data: {e}")

        if self.config.export_event_data and hasattr(self, '_pending_events') and self._pending_events:
            # This involves file I/O, should be async
            env_name = os.path.basename(self.env_path) if self.env_path else "unknown_env"
            dataset_dir = self.get_events_directory() or "events"
            await aiofiles.os.makedirs(dataset_dir, exist_ok=True)

            # Use trail_id or timestamp for the filename
            filename = f"{self.trail_id or f'simulation_{int(time.time())}'}.json"
            export_path = os.path.join(dataset_dir, filename)
            await self._export_data_from_pending_events(export_path)

        logger.info(f"Simulation stopped. Total time: {self.data['total_simulation_time']:.2f} seconds")

        # Ensure worker nodes can see the termination signal
        await asyncio.sleep(0.5)  # Allow some time for the termination signal to propagate to all nodes

        # Mark environment fully stopped
        if not self.stopped_event.is_set():
            self.stopped_event.set()

    async def pause_simulation(self):
        """
        Pause the simulation, suspending event processing and agent activities.
        When paused, the simulation will:
        1. Stop processing events from the queue
        2. Pause the event bus if supported
        3. Pause the scheduler if in timed mode
        4. Record pause timestamp for timing calculations
        
        This is a non-blocking method that returns immediately after setting
        the pause state. Event processing will pause at the next check point.
        """
        if self._pause_signal.is_set():
            logger.info(f"Simulation {self.name} is already paused")
            return

        logger.info(f"Pausing simulation {self.name}")

        # Set state to paused
        await self.set_simulation_state(SimulationState.PAUSED, reason="user_requested")

        # Pause the scheduler if in timed mode
        if self.mode == SimulationMode.TIMED and self.scheduler:
            if hasattr(self.scheduler, 'pause'):
                self.scheduler.pause()

        logger.info(f"Simulation {self.name} paused at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")

    async def resume_simulation(self):
        """
        Resume a paused simulation, continuing event processing and agent activities.
        When resumed, the simulation will:
        1. Adjust timing calculations to account for time spent paused
        2. Continue processing events from the queue
        3. Resume the scheduler if in timed mode
        4. Record the time spent paused for accurate timing calculations
        
        This is a non-blocking method that returns immediately after setting
        the resume state. Event processing will resume at the next check point.
        """
        if not self._pause_signal.is_set():
            logger.info(f"Simulation {self.name} is not paused")
            return

        logger.info(f"Resuming simulation {self.name}")

        # Set state to running
        await self.set_simulation_state(SimulationState.RUNNING, reason="user_requested")

        # Resume the scheduler if in timed mode
        if self.mode == SimulationMode.TIMED and self.scheduler:
            if hasattr(self.scheduler, 'resume'):
                self.scheduler.resume()

        logger.info(f"Simulation {self.name} resumed after {time.time() - self._state_change_time:.2f} seconds")

    def is_running(self) -> bool:
        """Check if the simulation is in a running state."""
        return self._state == SimulationState.RUNNING

    def is_paused(self) -> bool:
        """Check if the simulation is in a paused state."""
        return self._state == SimulationState.PAUSED

    def is_terminated(self) -> bool:
        """Check if the simulation has been terminated (including normal completion and external termination)."""
        return self._state in [SimulationState.TERMINATED, SimulationState.COMPLETED]

    async def set_simulation_state(self, new_state: SimulationState, reason: str = None) -> bool:
        """
        Update the simulation state and perform corresponding actions.
        
        Args:
            new_state: The new simulation state
            reason: Reason for the state change (optional)
            
        Returns:
            bool: Whether the state update was successful
        """
        if self._state == new_state:
            logger.debug(f"Simulation already in state {new_state}")
            return False

        # Record previous state
        previous_state = self._state
        timestamp = time.time()

        # Update state
        self._state = new_state

        # Broadcast state change
        # await self._broadcast_state_change(previous_state, new_state, reason)

        # Handle logic for specific states
        if new_state == SimulationState.PAUSED:
            # Set pause signal - set when paused
            self._pause_signal.set()
            self._state_change_time = timestamp

            # Pause the event bus if available
            if hasattr(self.event_bus, 'pause'):
                try:
                    await self.event_bus.pause()
                except Exception as e:
                    logger.warning(f"Failed to pause event bus: {e}")

        elif new_state == SimulationState.RUNNING:
            # If resuming from paused state
            if previous_state == SimulationState.PAUSED:
                pause_duration = timestamp - self._state_change_time
                self._pause_cumulative_time += pause_duration
                # Update last event time to prevent false idle detection
                self._last_event_time = timestamp

                # Resume the event bus if available
                if hasattr(self.event_bus, 'resume'):
                    try:
                        await self.event_bus.resume()
                    except Exception as e:
                        logger.warning(f"Failed to resume event bus: {e}")

            # Clear pause signal - clear when running
            self._pause_signal.clear()

        elif new_state == SimulationState.TERMINATED or new_state == SimulationState.COMPLETED:
            # Set termination signal
            self._termination_signal.set()
            # If terminating from paused state, also update pause time
            if previous_state == SimulationState.PAUSED:
                pause_duration = timestamp - self._state_change_time
                self._pause_cumulative_time += pause_duration
                # Clear pause signal to prevent tasks from getting stuck permanently
                self._pause_signal.clear()

        # Log the state change
        logger.info(f"Simulation state changed from {previous_state.value} to {new_state.value}" + 
                   (f" - Reason: {reason}" if reason else ""))

        return True

    async def handle_data_event(self, event: DataEvent) -> None:
        """
        Handle data access events coming from agents
        
        Args:
            event (DataEvent): The data access event
        """
        if event.target_type != "ENV":
            # Only handle events targeted at the environment
            return

        try:
            # Access the requested data
            data_value = await self.get_data(event.key, event.default)

            # Create and send response event
            response_event = DataResponseEvent(
                from_agent_id=self.name,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                data_value=data_value,
                success=True
            )

            # Dispatch the response via event bus
            await self.event_bus.dispatch_event(response_event)

        except Exception as e:
            # Send error response
            error_response = DataResponseEvent(
                from_agent_id=self.name,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                data_value=None,
                success=False,
                error=str(e)
            )

            await self.event_bus.dispatch_event(error_response)

    async def get_agent_data(self, agent_id: str, key: str, default: Optional[Any] = None) -> Any:
        """
        Get data from a specific agent, handling distributed case if needed.
        
        Args:
            agent_id (str): ID of the agent to get data from
            key (str): Data key to access
            default (Any, optional): Default value if key not found
            
        Returns:
            Any: The requested data or default value
        """
        # Check if agent is in local environment
        local_agent = None
        if self.agents:
            for agent_type, agents in self.agents.items():
                if agent_id in agents:
                    local_agent = agents[agent_id]
                    break

        if isinstance(local_agent, GeneralAgent):
            # Local access - properly handle async get_data method
            try:
                # GeneralAgent's get_data is async, so we need to await it
                return await local_agent.get_data(key, default)
            except Exception as e:
                logger.error(f"Error getting data from local agent {agent_id}: {e}")
                return default
        else:
            # Distributed access - create future for response
            response_future = asyncio.Future()

            # Create a unique ID for tracking this request
            request_id = f"env_data_req_{time.time()}_{id(response_future)}"

            # Store future for later resolution when response comes back
            if not hasattr(self, '_data_futures'):
                self._data_futures = {}
            self._data_futures[request_id] = response_future

            # Create a specific sync event for this request
            sync_event = asyncio.Event()
            self._data_sync_events[request_id] = sync_event

            # Create data event
            data_event = DataEvent(
                from_agent_id="ENV",
                to_agent_id=agent_id,
                source_type="ENV",
                target_type="AGENT",
                key=key,
                default=default,
                request_id=request_id
            )
            try:
                # Send the request
                await self.event_bus.dispatch_event(data_event)

                # Wait for the response with timeout
                try:
                    # Wait for this request's specific sync event
                    await asyncio.wait_for(sync_event.wait(), timeout=60.0)
                    return await response_future
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for data from agent {agent_id}")
                    self._data_futures.pop(request_id, None)
                    self._data_sync_events.pop(request_id, None)
                    return default
            except Exception as e:
                logger.error(f"Error getting data from agent {agent_id}: {e}")
                self._data_futures.pop(request_id, None)
                self._data_sync_events.pop(request_id, None)
                return default

    async def update_agent_data(self, agent_id: str, key: str, value: Any) -> bool:
        """
        Update data in a specific agent with distributed locking
        
        Args:
            agent_id (str): ID of the agent to update data in
            key (str): Data key to update
            value (Any): New value to set
            
        Returns:
            bool: Success status of the update
        """
        # Check if agent is in local environment
        local_agent = None
        if self.agents:
            for agent_type, agents in self.agents.items():
                if agent_id in agents:
                    local_agent = agents[agent_id]
                    break

        if local_agent:
            # Local update with distributed locking
            try:
                # Get distributed lock for this agent and key
                lock_id = f"agent_data_lock_{agent_id}_{key}"
                lock = await get_lock(lock_id)

                # Acquire lock before updating
                async with lock:
                    # GeneralAgent's update_data is async, so we need to await it
                    return await local_agent.update_data(key, value)
            except Exception as e:
                logger.error(f"Error updating data in local agent {agent_id}: {e}")
                return False
        else:
            # Distributed update - create future for response
            response_future = asyncio.Future()

            # Create a unique ID for tracking this request
            request_id = f"env_data_update_req_{time.time()}_{id(response_future)}"

            # Store future for later resolution when response comes back
            if not hasattr(self, '_data_update_futures'):
                self._data_update_futures = {}
            self._data_update_futures[request_id] = response_future

            # Create a specific sync event for this request
            sync_event = asyncio.Event()
            self._data_update_sync_events[request_id] = sync_event

            # Create data update event
            data_update_event = DataUpdateEvent(
                from_agent_id="ENV",
                to_agent_id=agent_id,
                source_type="ENV",
                target_type="AGENT",
                key=key,
                value=value,
                request_id=request_id
            )

            try:
                # Send the request
                await self.event_bus.dispatch_event(data_update_event)

                # Wait for the response with timeout
                try:
                    # Wait for this request's specific sync event
                    await asyncio.wait_for(sync_event.wait(), timeout=60.0)
                    return await response_future
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for data update in agent {agent_id}")
                    self._data_update_futures.pop(request_id, None)
                    self._data_update_sync_events.pop(request_id, None)
                    return False
            except Exception as e:
                logger.error(f"Error updating data in agent {agent_id}: {e}")
                self._data_update_futures.pop(request_id, None)
                self._data_update_sync_events.pop(request_id, None)
                return False

    async def _export_data_from_pending_decisions(self, export_path: str) -> None:
        """Export training data from pending decisions directly to a file.
        
        Used when database is not available or as a fallback.
        
        Args:
            export_path (str): Path to save the exported data
        """
        try:
            # Check if we have pending decisions or stored decisions
            if hasattr(self, '_pending_decisions') and self._pending_decisions:
                # Export just the decisions array to match the format in decision.py and test.json
                with open(export_path, "w", encoding='utf-8') as f:
                    json.dump(self._pending_decisions, f, indent=4)

                logger.info(f"Exported {len(self._pending_decisions)} decisions directly to {export_path}")
            else:
                logger.warning("No decisions to export - file not created")
        except Exception as e:
            logger.error(f"Error in direct export of decisions: {e}")

    async def _export_data_from_pending_events(self, export_path: str) -> None:
        """Export training data from pending events directly to a file.
        
        Used when database is not available or as a fallback.
        
        Args:
            export_path (str): Path to save the exported data
        """
        try:
            # Check if we have pending events
            if hasattr(self, '_pending_events') and self._pending_events:
                # Export just the events array to match the format in event.py and test.json
                with open(export_path, "w", encoding='utf-8') as f:
                    json.dump(self._pending_events, f, indent=4)

                logger.info(f"Exported {len(self._pending_events)} events directly to {export_path}")
            else:
                logger.warning("No events to export - file not created")
        except Exception as e:
            logger.error(f"Error in direct export of events: {e}")

    async def get_agent_data_by_type(self, agent_type: str, key: str, default: Optional[Any] = None) -> Any:
        """
        Get data from all agents of a specific type
        
        Args:
            agent_type (str): Type of agents to get data from
            key (str): Data key to access
            default (Any, optional): Default value if key not found
            
        Returns:
            Any: The requested data or default value
        """
        # Check if agent type exists in agents dictionary
        if agent_type not in self.agents:
            logger.warning(f"Agent type {agent_type} not found in agents dictionary")
            return default

        node = get_node()
        is_distributed_master = node and node.role == NodeRole.MASTER

        result = {}
        if is_distributed_master:
            # Distributed mode: Collect data from workers in batch
            master_node = node # Should be an instance of MasterNode
            tasks = []
            # Import the client function for batch collection
            from onesim.distribution.grpc_impl import collect_data_batch_from_worker

            worker_infos = list(master_node.workers.values()) # Get a stable list of workers

            for worker_info in worker_infos:
                tasks.append(
                    collect_data_batch_from_worker(
                        worker_address=worker_info.address,
                        worker_port=worker_info.port,
                        agent_type=agent_type,
                        data_key=key,
                        default_value=default
                    )
                )

            worker_results = await asyncio.gather(*tasks, return_exceptions=True)

            for res in worker_results:
                if isinstance(res, Exception):
                    logger.error(f"Error collecting batch data from a worker: {res}")
                elif res is not None: # res is a dict {agent_id: data_value}
                    result.update(res)
                # If res is None, it means an error occurred that was logged by collect_data_batch_from_worker

        else: # Non-distributed mode or Worker node
            # Original logic for non-distributed or if this SimEnv instance is on a worker
            local_agents_of_type = self.agents.get(agent_type, {})
            for agent_id, agent_instance in local_agents_of_type.items():
                try:
                    result[agent_id] = await agent_instance.get_data(key, default)
                except Exception as e:
                    logger.error(f"Error getting data from agent {agent_id} (type {agent_type}): {e}")
                    result[agent_id] = default
        return result

    async def handle_data_response(self, event: DataResponseEvent) -> None:
        """
        Handle data response events from agents
        
        Args:
            event (DataResponseEvent): The data response event
        """
        # Only handle responses for requests we sent
        if hasattr(self, '_data_futures') and event.request_id in self._data_futures:
            future = self._data_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(event.data_value)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))

            # Signal that we've received a response
            if event.request_id in self._data_sync_events:
                self._data_sync_events.pop(event.request_id).set()

    async def handle_data_update_response(self, event: DataUpdateResponseEvent) -> None:
        """
        Handle data update response events from agents
        
        Args:
            event (DataUpdateResponseEvent): The data update response event
        """
        # Only handle responses for requests we sent
        if hasattr(self, '_data_update_futures') and event.request_id in self._data_update_futures:
            future = self._data_update_futures.pop(event.request_id)

            if not future.done():
                if event.success:
                    future.set_result(event.success)
                else:
                    future.set_exception(ValueError(event.error or "Unknown error"))

            # Signal that we've received a response
            if event.request_id in self._data_update_sync_events:
                self._data_update_sync_events.pop(event.request_id).set()

    async def handle_data_update_event(self, event: DataUpdateEvent) -> None:
        """
        Handle data update events coming from agents with distributed locking
        
        Args:
            event (DataUpdateEvent): The data update event
        """
        if event.target_type != "ENV":
            # Only handle events targeted at the environment
            return

        try:
            # Get distributed lock for this key
            lock_id = f"env_data_lock_{event.key}"
            lock = await get_lock(lock_id)

            # Acquire lock before updating data
            async with lock:
                # Update the requested data
                success = await self.update_data(event.key, event.value)

                # Create and send response event
                response_event = DataUpdateResponseEvent(
                    from_agent_id=self.name,
                    to_agent_id=event.from_agent_id,
                    request_id=event.request_id,
                    key=event.key,
                    success=success
                )

                # Dispatch the response via event bus
                await self.event_bus.dispatch_event(response_event)

        except Exception as e:
            # Send error response
            error_response = DataUpdateResponseEvent(
                from_agent_id=self.name,
                to_agent_id=event.from_agent_id,
                request_id=event.request_id,
                key=event.key,
                success=False,
                error=str(e)
            )

            await self.event_bus.dispatch_event(error_response)
