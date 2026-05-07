import argparse
import asyncio
import importlib.util
import sys
import os
import time
import uuid
from typing import List, Dict, Type, Optional
import json
from datetime import datetime

from loguru import logger
import onesim
from onesim.config import (
    get_config, 
    get_component_registry,
    parse_json, 
    OneSimConfig
)
from onesim.simulator import AgentFactory, BasicSimEnv
from onesim.models import ModelManager
from onesim.events import get_event_bus, EventBus, Scheduler
from onesim.agent import GeneralAgent
from onesim.utils.work_graph import WorkGraph
from onesim.distribution.node import get_node, NodeRole
from onesim.utils.common import setup_logging
# Component identifier constants from onesim module
from onesim import (
    COMPONENT_MODEL,
    COMPONENT_DATABASE,
    COMPONENT_DISTRIBUTION,
    COMPONENT_MONITOR
)
from onesim.distribution.proxy_env import ProxyEnv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Multi-Agent Simulator.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    parser.add_argument("--model_config", type=str,
                        required=False, help="Model configuration file.")
    parser.add_argument("--model_config_name", type=str,
                        required=False, help="Model configuration name to use.")
    parser.add_argument("--env", type=str, required=False,
                        help="Simulation environment.")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="Output directory for simulation results. Defaults to src/envs/{env} if not specified.")
    parser.add_argument("--log_dir", type=str, required=False,
                        help="Log directory for simulation results. Defaults to src/envs/{env} if not specified.")
    
    # Add distributed mode arguments
    parser.add_argument("--mode", type=str, choices=["single", "master", "worker"],
                       default="single", help="Operating mode: single, master, or worker")
    parser.add_argument("--master_address", type=str, default="localhost",
                       help="Master node address (for worker mode)")
    parser.add_argument("--master_port", type=int, default=50051,
                       help="Master node port")
    parser.add_argument('--worker_address', type=str, 
                  help='Worker node address (for master mode)')
    parser.add_argument("--worker_port", type=int, default=0,
                       help="Worker node port (0 for auto-assign)")
    parser.add_argument("--node_id", type=str, default=None,
                       help="Node identifier (generated if not provided)")
    
    parser.add_argument("--expected_workers", type=int, default=1,
                       help="Number of worker nodes to wait for (for master mode)")
                       
    # Component selection arguments
    parser.add_argument("--enable_db", action="store_true",
                       help="Enable database component")
    parser.add_argument("--enable_observation", action="store_true",
                       help="Enable observation system")
                       
    args = parser.parse_args()
    return args

def load_sim_env_from_file(env_path, env_name) -> Type[BasicSimEnv]:
    # Make sure env_path is in sys.path so env_name can be treated as a top-level package
    if env_path not in sys.path:
        sys.path.append(env_path)

    # Load SimEnv module based on package name
    module_name = f"{env_name}.code.SimEnv"
    sim_env_module = importlib.import_module(module_name)

    # Get the SimEnv class
    if not hasattr(sim_env_module, "SimEnv"):
        raise AttributeError(
            f"The module {module_name} does not contain a class named 'SimEnv'")

    SimEnv = getattr(sim_env_module, "SimEnv")
    return SimEnv

async def run_agents(agents_dict: Dict[str, List[GeneralAgent]]) -> List[asyncio.Task]:
    """Create tasks for all agents from the agent dictionary."""
    tasks = []
    for agent_type in agents_dict:
        for agent in agents_dict[agent_type]:
            tasks.append(agent.run())
    return tasks

def build_graph(conf: OneSimConfig, agent_factory):
    events_path = os.path.join(conf.env_path, "events.json")
    actions_path = os.path.join(conf.env_path, "actions.json")
    events = parse_json(events_path)
    actions = parse_json(actions_path)

    # Create work graph
    simulator_config = conf.simulator_config
    work_graph = WorkGraph()
    work_graph.load_workflow_data(actions, events)
    start_agent_types = work_graph.get_start_agent_types()
    end_agent_types = work_graph.get_end_agent_types()
    start_agent_ids = agent_factory.get_agent_profile_ids(start_agent_types)
    end_agent_ids = agent_factory.get_agent_profile_ids(end_agent_types)
    for agent_type, ids in end_agent_ids.items():
        for id in ids:
            agent_factory.add_env_relationship(id)
    return work_graph, start_agent_ids, end_agent_ids

async def init_agents(conf: OneSimConfig, model_config_name: str = None):
    # Get first LLM config name for compatibility with legacy code

    if model_config_name is None:
        model_config_name = "chat_load_balancer"
    
    # Create agent factory with both simulator_config and agent_config
    agent_factory = AgentFactory(
        simulator_config=conf.simulator_config,
        model_config_name=model_config_name,
        env_path=conf.env_path,
        agent_config=conf.agent_config
    )
    agents = await agent_factory.create_agents()
    return agents, agent_factory

async def initialize_distributed_environment(config: OneSimConfig, args):
    """Initialize environment in distributed mode"""
    registry = get_component_registry()
    node = registry.get_instance(COMPONENT_DISTRIBUTION)
    
    if not node:
        logger.error("Distribution component not initialized")
        raise RuntimeError("Distribution component must be initialized first")
    
    event_bus = get_event_bus()
    event_bus.setup_distributed(node)
    
    if node.role == NodeRole.MASTER:
        # Master node: wait for all workers to connect, then create environment
        logger.info(f"Master node waiting for {node.expected_worker_count} workers to connect")
        await node.initialized.wait()  # Wait for all workers to connect
        logger.info("All workers connected, initializing environment...")
        
        # Create environment (only on master)
        env = await initialize_environment(config, args)
        
        return env
    elif node.role == NodeRole.WORKER:
        events_path = os.path.join(config.env_path, "events.json")
        actions_path = os.path.join(config.env_path, "actions.json")
        events = parse_json(events_path)
        actions = parse_json(actions_path)
        work_graph = WorkGraph()
        work_graph.load_workflow_data(actions, events)
        
        # Create a proxy environment for worker node
        logger.info(f"Creating ProxyEnv for worker node {node.node_id}")
        proxy_env = ProxyEnv(
            node_id=node.node_id,
            master_address=node.master_address,
            master_port=node.master_port
        )
        
        # Register proxy environment with event bus
        proxy_env_id = f"{node.node_id}_ENV"
        event_bus.register_agent(proxy_env_id, proxy_env)
        
        # Store proxy_env in node for agent access
        node.proxy_env = proxy_env
        
        # Worker uses proxy environment instead of real environment
        return proxy_env
    else:
        raise ValueError(f"Invalid node role: {node.role}")

async def initialize_environment(config: OneSimConfig, args) -> Optional[BasicSimEnv]:
    """Initialize simulation environment"""
    registry = get_component_registry()

    # Get node if we're in distributed mode
    node = registry.get_instance(COMPONENT_DISTRIBUTION)
    node_role = node.role if node else NodeRole.SINGLE

    # Create or get event bus
    event_bus = get_event_bus()

    # Create Agent Factory and Agents
    logger.info("Creating agents for environment")
    agents, agent_factory = await init_agents(config, args.model_config_name)

    # Build workflow graph and get start/end nodes
    logger.info("Building workflow graph")
    work_graph, start_agent_ids, end_agent_ids = build_graph(config, agent_factory)

    # Load and create environment
    logger.info(f"Loading environment class from {config.env_path}")
    SimEnv = load_sim_env_from_file(os.path.join(config.base_dir, "envs"), config.env_name)

    # Initialize trail for data storage (only on master or single node)
    trail_id = None
    if node_role != NodeRole.WORKER and registry.is_initialized(COMPONENT_DATABASE):
        from onesim.data import ScenarioManager, TrailManager
        try:
            # Create scenario if it doesn't exist
            scenario_mgr = ScenarioManager()
            env_config = config.simulator_config.environment
            env_name = env_config.get('name', config.env_name)

            # Create or get scenario ID
            try:
                # Try to find existing scenario
                scenarios = await scenario_mgr.get_scenario_by_name(name=env_name, exact_match=True)
                scenario_id = None
                if scenarios and len(scenarios) > 0:
                    for scenario in scenarios:
                        if scenario['name'] == env_name:
                            scenario_id = scenario['scenario_id']
                            logger.info(f"Using existing scenario {scenario_id} for {env_name}")
                            break
                if scenario_id is None:
                    # Create new scenario
                    scenario_id = await scenario_mgr.create_scenario(
                        name=env_name,
                        folder_path=f"envs/{env_name}",
                        description=env_config.get('description', f"Simulation scenario for {env_name}"),
                        tags={
                            "domain": env_config.get('domain', ''), 
                            "version": env_config.get('version', '1.0')
                        }
                    )
                    logger.info(f"Created new scenario {scenario_id} for {env_name}")
            except Exception as e:
                logger.warning(f"Error finding/creating scenario: {e}, generating temporary ID")
                scenario_id = str(uuid.uuid4())

            # Create trail
            trail_mgr = TrailManager()
            trail_name = f"{env_name}_run_{time.strftime('%Y%m%d_%H%M%S')}"
            trail_id = await trail_mgr.create_trail(
                scenario_id=scenario_id,
                name=trail_name,
                description=f"Simulation run for {env_name}",
                config=config.simulator_config.to_dict()
            )
            logger.info(f"Created trail {trail_id} for data storage")
        except Exception as e:
            logger.error(f"Error initializing data storage: {e}, continuing without storage")
    
    # Create environment instance
    logger.info("Creating environment instance")
    # Use the environment settings from simulator_config
    env_settings = config.simulator_config.environment

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = create_timestamped_output_dir(config.env_path)

    sim_env = SimEnv(
        config.env_name,
        event_bus,
        {},  # initial data
        start_agent_ids,
        end_agent_ids,
        env_settings,
        agents,
        config.env_path,
        trail_id,  # Pass trail_id to environment
        None,  # intervention_engine
        output_dir,  # Pass the timestamped output directory
    )

    # Register termination events
    end_events = work_graph.get_end_events()
    for event_name in end_events:
        logger.info(f"Registering termination event: {event_name}")
        sim_env.register_event(event_name, 'terminate')

    # Register environment with event bus
    if node_role == NodeRole.MASTER or node_role == NodeRole.SINGLE:
        event_bus.register_agent("ENV", sim_env)

        # Set environment reference in master node for data forwarding from workers
        if node_role == NodeRole.MASTER and node:
            # Provide sim_env to master node for data forwarding from workers
            if hasattr(node, 'set_sim_env'):
                node.set_sim_env(sim_env)
                logger.info("Set simulation environment reference in master node")
        else:
            for agent_type in agents:
                for agent_id, agent in agents[agent_type].items():
                    agent.set_env(sim_env)

        for agent_type in agents:
            for agent_id, agent in agents[agent_type].items():
                event_bus.register_agent(agent_id, agent)

    logger.info(f"Environment '{config.env_name}' initialized successfully")

    return sim_env

async def run_simulation(
    sim_env: Optional[BasicSimEnv],
    config: OneSimConfig, 
    args
) -> None:
    """Coordinate running all simulation components."""
    try:
        registry = get_component_registry()

        # Get node if in distributed mode
        node = registry.get_instance(COMPONENT_DISTRIBUTION)
        node_role = node.role if node else NodeRole.SINGLE

        # Get event bus
        event_bus = get_event_bus()

        # Create a termination event
        termination_event = asyncio.Event()

        # Register signal handlers for graceful shutdown
        import signal

        def signal_handler():
            logger.info("Received interrupt signal, initiating graceful shutdown")
            termination_event.set()

        try:
            if sys.platform.startswith('win'):
                # Windows: use traditional signal handling
                signal.signal(signal.SIGINT, signal_handler)
            else:
                # Unix/Linux/macOS: try asyncio first, fallback to traditional
                try:
                    loop = asyncio.get_running_loop()
                    loop.add_signal_handler(signal.SIGINT, signal_handler)
                    loop.add_signal_handler(signal.SIGTERM, signal_handler)
                except (NotImplementedError, RuntimeError):
                    signal.signal(signal.SIGINT, signal_handler)
                    signal.signal(signal.SIGTERM, signal_handler)
        except Exception as e:
            logger.warning(f"Signal handler setup failed: {e}")

        if node_role == NodeRole.MASTER or node_role == NodeRole.SINGLE:
            if not sim_env:
                logger.error("Simulation environment is required for Master node")
                return

            logger.info(f"{node_role.name} node starting simulation with environment")

            if config.monitor_config.enabled:
                from onesim.monitor import MonitorManager
                await MonitorManager.setup_metrics(
                    env=sim_env
                )

            # Get environment tasks
            env_tasks = await sim_env.run()

            # If single mode, create agent tasks
            agent_tasks = []
            if node_role == NodeRole.SINGLE:
                agents = sim_env.agents
                # Create agent tasks
                for agent_type in agents:
                    for agent_id, agent in agents[agent_type].items():
                        agent_tasks.append(agent.run())

                # Run all tasks with termination handling
                event_bus_task = asyncio.create_task(event_bus.run())
                all_tasks = [event_bus_task] + agent_tasks + env_tasks

                # Wait for either tasks to complete or termination signal
                done, pending = await asyncio.wait(
                    [asyncio.create_task(termination_event.wait())] + all_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Check if termination was requested
                if any(t for t in done if termination_event.is_set()):
                    logger.info("Manual termination requested")
                    await sim_env.stop_simulation()

                # Ensure cleanup of remaining tasks
                for task in pending:
                    if not task.done():
                        task.cancel()
            else:
                # Master mode - run event bus and environment tasks separately
                event_bus_task = asyncio.create_task(event_bus.run())

                try:
                    # Wait for either env tasks to complete or termination signal
                    done, pending = await asyncio.wait(
                        [asyncio.create_task(termination_event.wait())] + env_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # Check if termination was requested
                    if any(t for t in done if termination_event.is_set()):
                        logger.info("Manual termination requested")
                        await sim_env.stop_simulation()
                finally:
                    # Ensure event bus task can be cancelled
                    if not event_bus_task.done():
                        event_bus_task.cancel()
                        try:
                            await event_bus_task
                        except asyncio.CancelledError:
                            pass

                    # Ensure environment fully stopped before node shutdown
                    try:
                        if hasattr(sim_env, 'stopped_event'):
                            await sim_env.stopped_event.wait()
                    except Exception as e:
                        logger.warning(f"Waiting env stopped_event failed: {e}")

        elif node_role == NodeRole.WORKER:
            logger.info("Worker node starting event processing")
            agents = node.agents if node else {}
            agent_tasks = []

            # Set the ProxyEnv reference for all agents on this worker
            if hasattr(node, 'proxy_env') and node.proxy_env and agents:
                logger.info(f"Setting proxy environment for {sum(len(agents[t]) for t in agents)} agents")
                for agent_type in agents:
                    for agent_id, agent in agents[agent_type].items():
                        agent.set_env(node.proxy_env)

            # Create agent tasks
            for agent_type in agents:
                for agent_id, agent in agents[agent_type].items():
                    agent_tasks.append(agent.run())

            # Get proxy environment tasks
            env_tasks = []
            if hasattr(node, 'proxy_env') and node.proxy_env:
                env_tasks = await node.proxy_env.run()

            # Worker needs to run event bus, proxy env, and agent tasks
            if agent_tasks or env_tasks:
                event_bus_task = asyncio.create_task(event_bus.run())
                all_tasks = [event_bus_task] + agent_tasks + env_tasks

                # Wait for either tasks to complete or termination signal
                done, pending = await asyncio.wait(
                    [asyncio.create_task(termination_event.wait())] + all_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )

                # If termination was requested or event bus stopped, cancel remaining tasks
                if termination_event.is_set():
                    logger.info("Worker node termination requested")

                    # Stop simulation through proxy environment if available
                    if hasattr(node, 'proxy_env') and node.proxy_env:
                        await node.proxy_env.stop_simulation()

                for task in pending:
                    if not task.done():
                        task.cancel()
            else:
                logger.warning("No tasks to run, only running event bus")
                await event_bus.run()
        else:
            logger.error(f"Unknown node role: {node_role}")

    except asyncio.CancelledError:
        logger.info("Simulation cancelled")
        # Ensure simulation is stopped if it was running
        if sim_env:
            await sim_env.stop_simulation()
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
        raise

async def async_main():
    """Asynchronous main entry function"""
    # Parse command line arguments
    args = parse_args()

    # Determine which components to initialize
    components_to_init = [COMPONENT_MODEL]
    components_to_init.append(COMPONENT_MONITOR)

    # Initialize database if requested or if enabled in config
    if args.enable_db or ("--config" in sys.argv and is_database_enabled(args.config)):
        components_to_init.append(COMPONENT_DATABASE)

    # # Initialize observation if requested
    # if args.enable_observation:
    #     components_to_init.append(COMPONENT_MONITOR)

    # Initialize distribution component if not in single mode or if enabled in config
    if args.mode != "single" or ("--config" in sys.argv and is_distribution_enabled(args.config)):
        components_to_init.append(COMPONENT_DISTRIBUTION)

    # Build initialization configuration
    init_config = {
        "distribution": {
            "enabled": args.mode != "single",
            "mode": args.mode,
            "node_id": args.node_id,
            "master_address": args.master_address,
            "master_port": args.master_port,
            "worker_address": args.worker_address,
            "worker_port": args.worker_port,
            "expected_workers": args.expected_workers
        }
    }

    # Initialize OneSim with only the requested components
    await onesim.init(
        config_path=args.config,
        model_config_path=args.model_config,
        components=components_to_init,
        **init_config
    )

    # Load simulation configuration
    sim_config = onesim.load_simulation_config(
        args.config, 
        args.model_config, 
        args.env
    )
    setup_logging(sim_config.env_name, args.log_dir)
    # Determine if we're in distributed mode
    registry = get_component_registry()
    is_distributed = registry.is_initialized(COMPONENT_DISTRIBUTION)

    # Initialize environment based on mode
    if not is_distributed:
        # Single mode
        env = await initialize_environment(sim_config, args)
        await run_simulation(env, sim_config, args)
    else:
        # Distributed mode (master/worker)
        node = registry.get_instance(COMPONENT_DISTRIBUTION)

        # Initialize distributed environment
        env = await initialize_distributed_environment(sim_config, args)

        if node.role == NodeRole.WORKER:
            # For worker, wait for agent creation
            logger.info(f"Worker node {node.node_id} waiting for agent creation...")
            await node.agents_created.wait()
            logger.info(f"Worker node {node.node_id} agents created: {sum(len(node.agents[t]) for t in node.agents)} agents")

        await run_simulation(env, sim_config, args)

        # Keep node alive until interrupted
        if node.role == NodeRole.WORKER:
            # 由 WorkerNode 的 stopped_event 退出，不使用固定超时
            try:
                await node.stopped_event.wait()
            except KeyboardInterrupt:
                logger.info("Worker stopping due to keyboard interrupt")
                # 键盘中断时执行一次清理
                try:
                    await node.shutdown()
                except Exception as e:
                    logger.error(f"Error during worker shutdown: {e}")
            return
        else:
            # Master节点在模拟完成后也应该退出
            logger.info("Master node simulation completed, shutting down")
            # 等待环境完全关闭，再优雅关闭Master分布式资源
            try:
                if hasattr(env, 'stopped_event'):
                    await env.stopped_event.wait()
            except Exception as e:
                logger.warning(f"Waiting env stopped_event failed: {e}")
            try:
                await node.shutdown()
            except Exception as e:
                logger.error(f"Error during master shutdown: {e}")
            return

def is_database_enabled(config_path):
    """Check if database is enabled in config file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Check if database section is directly in config (older format)
        if "database" in config:
            return config.get("database", {}).get("enabled", False)

        return False
    except Exception:
        return False

def is_distribution_enabled(config_path):
    """Check if distribution is enabled in config file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get("distribution", {}).get("enabled", False)
    except Exception:
        return False


def create_timestamped_output_dir(env_path: str) -> str:
    """
    Create a timestamped output directory under env_path/runs.

    Args:
        env_path (str): Path to the environment directory

    Returns:
        str: Path to the created timestamped output directory
    """
    if not env_path:
        raise ValueError("env_path cannot be None or empty")

    # Create timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create runs directory under env_path
    runs_dir = os.path.join(env_path, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # Create timestamped directory
    output_dir = os.path.join(runs_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def cli_entry_point():
    """Synchronous entry point for console script."""
    asyncio.run(async_main())

if __name__ == "__main__":

    cli_entry_point()
