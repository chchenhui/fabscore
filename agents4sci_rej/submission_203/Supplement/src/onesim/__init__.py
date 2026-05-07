"""
OneSim - A reactive multi-agent simulation framework
"""

import os
import sys
import uuid
import asyncio
from typing import Optional, Dict, Any, List, Set, Union
from loguru import logger
import re
import textwrap
import importlib.util
import json

# Version information
__version__ = "1.0.0"

# Import configuration system
from onesim.config import (
    get_config, 
    get_component_registry,
    ObservationConfig, 
    OneSimConfig,
    ModelConfig,
    DatabaseConfig,
    MonitorConfig,
    DistributionConfig,
    AgentConfig,
    SimulatorConfig,
    AgentMemoryConfig
)

# Component identifier constants
COMPONENT_MODEL = "model"
COMPONENT_DATABASE = "database"
COMPONENT_DISTRIBUTION = "distribution"
COMPONENT_OBSERVATION = "observation"
COMPONENT_MONITOR = "monitor"

async def init(
    config_path: Optional[str] = None, 
    model_config_path: Optional[str] = None,
    components: Optional[List[str]] = None,
    config_dict: Optional[Dict[str, Any]] = None,
    model_config_dict: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Initialize the OneSim framework with only requested components.
    
    Args:
        config_path: Path to configuration file (YAML or JSON)
        model_config_path: Path to model configuration file
        components: List of component names to initialize. If None, initializes based on config.
            Available components: "model", "database", "distribution", "observation", "monitor"
        config_dict: Dictionary containing configuration (alternative to config_path)
        model_config_dict: Dictionary containing model configuration (alternative to model_config_path)
        **kwargs: Configuration overrides
    
    Returns:
        Configured OneSimConfig instance
    """
    config = get_config()
    registry = get_component_registry()
    
    # Load from file if provided
    if config_path:
        config.load_from_file(config_path)
    
    # Load from dictionary if provided (higher priority than file)
    if config_dict:
        config.load_from_dict(config_dict)
    
    if model_config_path:
        config.model_config.load_from_file(model_config_path)
    
    # Load model config from dictionary if provided
    if model_config_dict:
        config.model_config.load_from_dict(model_config_dict)
    
    # Apply overrides
    if kwargs:
        config.update(**kwargs)
    # Determine which components to initialize
    components_to_init = set()
    
    if components is not None:
        # If specific components were requested, use those
        components_to_init = set(components)
    else:
        # Otherwise, initialize components that are enabled in config
        if config.model_config.enabled:
            components_to_init.add(COMPONENT_MODEL)
        if config.database_config.enabled:
            components_to_init.add(COMPONENT_DATABASE)
        if config.distribution_config.enabled:
            components_to_init.add(COMPONENT_DISTRIBUTION)
        if config.observation_config.enabled:
            components_to_init.add(COMPONENT_OBSERVATION)
        if config.monitor_config.enabled:
            components_to_init.add(COMPONENT_MONITOR)
    
    # Initialize components
    init_tasks = []
    
    # Model component initialization
    if COMPONENT_MODEL in components_to_init and not registry.is_initialized(COMPONENT_MODEL):
        init_tasks.append(init_model_component(config.model_config))
    
    # Database component initialization
    if COMPONENT_DATABASE in components_to_init and not registry.is_initialized(COMPONENT_DATABASE):
        init_tasks.append(init_database_component(config.database_config))
    
    # Observation component initialization
    if COMPONENT_OBSERVATION in components_to_init and not registry.is_initialized(COMPONENT_OBSERVATION):
        init_tasks.append(init_observation_component(config.observation_config))
    
    # Monitor component initialization
    if COMPONENT_MONITOR in components_to_init and not registry.is_initialized(COMPONENT_MONITOR):
        init_tasks.append(init_monitor_component(config.monitor_config))
    
    # Distribution component initialization (must be last due to potential dependencies)
    if COMPONENT_DISTRIBUTION in components_to_init and not registry.is_initialized(COMPONENT_DISTRIBUTION):
        init_tasks.append(init_distribution_component(dist_config=config.distribution_config))
    
    # Wait for all component initializations to complete
    if init_tasks:
        await asyncio.gather(*init_tasks)
    
    # Log initialized components
    initialized = [comp for comp in components_to_init if registry.is_initialized(comp)]
    if initialized:
        logger.info(f"OneSim {__version__} initialized with components: {', '.join(initialized)}")
    else:
        logger.info(f"OneSim {__version__} initialized with no components")
    
    return config

async def init_model_component(model_config: ModelConfig):
    """Initialize model component"""
    registry = get_component_registry()
    
    try:
        from onesim.models import get_model_manager
        
        # Initialize ModelManager with the model configurations
        model_manager = get_model_manager()
        
        # Create combined config with proper structure
        model_configs = {
            "chat": model_config.chat_configs,
            "embedding": model_config.embedding_configs
        }
        
        # Load model configurations
        if model_config.chat_configs or model_config.embedding_configs:
            model_manager.load_model_configs(model_configs)
            
            # Configure load balancers
            if model_config.chat_configs:
                model_manager.configure_load_balancer(
                    model_configs=None,  # Auto-detect all chat models
                    strategy="round_robin",
                    config_name="chat_load_balancer",
                    model_type="chat"
                )
            
            if model_config.embedding_configs:
                model_manager.configure_load_balancer(
                    model_configs=None,  # Auto-detect all embedding models
                    strategy="round_robin",
                    config_name="embedding_load_balancer",
                    model_type="embedding"
                )
                
            logger.info(f"Initialized ModelManager with {len(model_config.chat_configs)} chat and "
                        f"{len(model_config.embedding_configs)} embedding models")
        else:
            logger.info("Initialized ModelManager with no models configured")
        
        # Register the component as initialized
        registry.register(COMPONENT_MODEL, model_manager)
        
    except ImportError as e:
        logger.error(f"Failed to initialize model component: {e}")
        raise

async def init_monitor_component(monitor_config: MonitorConfig):
    """Initialize monitor component"""
    registry = get_component_registry()
    
    if not monitor_config.enabled:
        logger.info("Monitor component disabled")
        return
    
    try:
        from onesim.monitor import MonitorManager, MetricDefinition, VariableSpec
        from json import load as json_load
        
        # Create monitor manager
        monitor_manager = MonitorManager()
        
        # 尝试从环境目录加载 scene_info.json
        env_path = os.path.dirname(monitor_config.metrics_path) if monitor_config.metrics_path else None
        metrics_module = None
        
        if env_path and os.path.exists(env_path):
            # 加载 scene_info.json 获取监控配置和指标定义
            scene_info_path = os.path.join(env_path, "scene_info.json")
            if os.path.exists(scene_info_path):
                try:
                    with open(scene_info_path, 'r', encoding='utf-8') as f:
                        scene_info = json.load(f)
                    
                    # 从 scene_info.json 加载指标定义
                    if "metrics" in scene_info and isinstance(scene_info["metrics"], list):
                        # 首先尝试加载指标计算函数模块
                        if monitor_config.metrics_path and os.path.exists(monitor_config.metrics_path):
                            try:
                                if os.path.isdir(monitor_config.metrics_path):
                                    # 如果是目录，查找metrics.py
                                    module_path = os.path.join(monitor_config.metrics_path, "metrics.py")
                                else:
                                    module_path = monitor_config.metrics_path
                                
                                if os.path.exists(module_path):
                                    # 使用importlib动态导入模块
                                    spec = importlib.util.spec_from_file_location("metrics_module", module_path)
                                    metrics_module = importlib.util.module_from_spec(spec)
                                    spec.loader.exec_module(metrics_module)
                                    logger.info(f"成功导入指标计算模块: {module_path}")
                            except Exception as e:
                                logger.error(f"导入指标计算模块失败: {e}")
                        
                        # 加载并注册指标
                        metrics_loaded = 0
                        for metric_def in scene_info["metrics"]:
                            try:
                                # 创建变量规格
                                variables = []
                                for var in metric_def.get("variables", []):
                                    variables.append(VariableSpec(
                                        name=var["name"],
                                        source_type=var["source_type"],
                                        path=var["path"],
                                        agent_type=var.get("agent_type"),
                                        required=var.get("required", True)
                                    ))
                                
                                # 获取函数名
                                function_name = metric_def.get("function_name") or metric_def.get("id")
                                if not function_name:
                                    function_name = re.sub(r'[^\w\-_]', '_', metric_def["name"])
                                
                                # 创建指标定义
                                metric_definition = MetricDefinition(
                                    id=metric_def.get("id", function_name),
                                    name=metric_def["name"],
                                    description=metric_def["description"],
                                    variables=variables,
                                    visualization_type=metric_def["visualization_type"],
                                    update_interval=metric_def.get("update_interval", 60)
                                )
                                
                                # 尝试从metrics模块查找计算函数
                                calculation_function = None
                                if metrics_module:
                                    # 首先尝试通过get_metric_function获取
                                    if hasattr(metrics_module, 'get_metric_function'):
                                        calculation_function = metrics_module.get_metric_function(function_name)
                                    
                                    # 如果没有找到，直接尝试获取同名函数
                                    if calculation_function is None and hasattr(metrics_module, function_name):
                                        calculation_function = getattr(metrics_module, function_name)
                                
                                if calculation_function:
                                    # 注册指标
                                    monitor_manager.register_metric(
                                        metric_definition, 
                                        calculation_function=calculation_function
                                    )
                                    metrics_loaded += 1
                                else:
                                    logger.warning(f"未找到指标 '{metric_def['name']}' 的计算函数 '{function_name}'")
                            except Exception as e:
                                logger.error(f"加载指标 '{metric_def.get('name', 'unknown')}' 失败: {e}")
                        
                        logger.info(f"从scene_info.json中加载了 {metrics_loaded} 个指标")
                except Exception as e:
                    logger.error(f"从scene_info.json加载数据失败: {e}")
        
        # 注册监控管理器
        registry.register(COMPONENT_MONITOR, monitor_manager)
        logger.info("监控系统初始化完成")
        
        # 如果enabled为true，尝试获取环境并启动监控
        active_env = None
        
        # 1. 尝试从simulator获取环境 (如果已注册)
        try:
            simulator = registry.get_instance("simulator")
            if simulator and hasattr(simulator, "env"):
                active_env = simulator.env
        except:
            pass
            
        # 2. 尝试从已知模块获取环境
        if not active_env:
            try:
                from onesim.simulator import get_current_environment
                active_env = get_current_environment()
            except:
                pass
                
        # 如果找到环境，配置监控并启动
        
        if active_env:
            monitor_manager.setup(active_env)
            monitor_manager.start_all_metrics()
            logger.info("已自动启动所有指标的监控")
        else:
            logger.info("未找到活跃环境，监控将在环境可用时启动")
        
    except ImportError as e:
        logger.error(f"初始化监控组件失败: {e}")
        raise

async def init_database_component(db_config: DatabaseConfig):
    """Initialize database component"""
    registry = get_component_registry()
    
    if not db_config.enabled:
        logger.info("Database component disabled")
        return
    
    try:
        from onesim import data
        
        # Initialize with database configuration
        data_manager = data.DatabaseManager.get_instance({
            "enabled": True,
            "host": db_config.host,
            "port": db_config.port,
            "dbname": db_config.dbname,
            "user": db_config.user,
            "password": db_config.password
        })
        
        # Initialize database schema
        data_manager.initialize_schema()
        logger.info("Data storage module initialized")
        
        # Register the component as initialized
        registry.register(COMPONENT_DATABASE, data_manager)
        
    except ImportError:
        logger.warning("Database dependencies not installed. Data storage will be disabled.")
        
        # Initialize with disabled flag to ensure managers can still be instantiated
        from onesim import data
        data_manager = data.DatabaseManager.get_instance({"enabled": False})
        logger.info("Data storage module disabled due to missing dependencies")

async def init_observation_component(obs_config: ObservationConfig):
    """Initialize observation component"""
    registry = get_component_registry()
    
    if not obs_config.enabled:
        logger.info("Observation component disabled")
        return
    
    try:
        from onesim.observation import setup_loki_logger
        
        if obs_config.loki_url:
            setup_loki_logger(obs_config.loki_url)
            logger.info(f"Loki logging set up with URL: {obs_config.loki_url}")
        
        # Register component as initialized
        registry.register(COMPONENT_OBSERVATION)
        
    except ImportError:
        logger.warning("Observation dependencies not installed. Observation will be disabled.")

async def init_distribution_component(dist_config: DistributionConfig):
    """Initialize distribution component"""
    registry = get_component_registry()
    logger.info(f"dist_config: {dist_config}")
    if not dist_config.enabled:
        logger.info("Distribution component disabled")
        return
    
    try:
        import grpc
        from onesim.distribution.node import initialize_node, NodeRole
        
        # Generate node ID if not provided
        node_id = dist_config.node_id
        if not node_id:
            node_id = f"node-{uuid.uuid4().hex[:8]}"
        
        # Initialize node based on mode
        mode = dist_config.mode
        if mode in ["master", "worker", "single"]:
            if mode == "master":
                node_config = {
                    "listen_port": dist_config.master_port,
                    "expected_workers": dist_config.expected_workers
                }
                node = await initialize_node(node_id, "master", node_config)
                logger.info(f"Initialized master node {node_id} on port {node_config['listen_port']}")
                
            elif mode == "worker":
                node_config = {
                    "master_address": dist_config.master_address,
                    "master_port": dist_config.master_port,
                    "listen_port": dist_config.worker_port
                }
                node = await initialize_node(node_id, "worker", node_config)
                logger.info(f"Initialized worker node {node_id} {dist_config.worker_address}:{dist_config.worker_port} connecting to {node_config['master_address']}:{node_config['master_port']}")
                
            else:  # single mode
                node = await initialize_node(node_id, "single")
                logger.info(f"Initialized single node {node_id}")
                
            # Register the component as initialized
            registry.register(COMPONENT_DISTRIBUTION, node)
        else:
            logger.warning(f"Unknown mode: {mode}. Using 'single' mode.")
            node = await initialize_node(node_id, "single")
            registry.register(COMPONENT_DISTRIBUTION, node)
            
    except ImportError:
        logger.error("gRPC dependencies not found. Please install with: pip install grpcio grpcio-tools")
        raise

def load_simulation_config(config_path: str, model_config_path: str = None, env_name: str = None) -> OneSimConfig:
    """
    Load simulation configuration from files.
    
    Args:
        config_path: Path to main configuration file
        model_config_path: Path to model configuration file (optional)
        env_name: Optional environment name override
        
    Returns:
        OneSimConfig object
    """
    config = get_config()
    return config.load_simulation_config(config_path, model_config_path, env_name)

async def init_models_only(config_path: str = None, model_config_path: str = None, **kwargs):
    """
    Initialize only the model component for lightweight usage.
    
    Args:
        config_path: Path to configuration file
        model_config_path: Path to model configuration file
        **kwargs: Model configuration overrides
        
    Returns:
        Model manager instance
    """
    config = get_config()
    
    # Load configuration if path provided
    if config_path:
        config.load_from_file(config_path)
    
    # Load model config if provided
    if model_config_path:
        config.model_config.load_from_file(model_config_path)
    
    # Apply any overrides
    if kwargs:
        model_conf = {"model": kwargs}
        config.update(**model_conf)
    
    # Ensure models are enabled
    config.model_config.enabled = True
    
    # Initialize only model component
    await init(components=[COMPONENT_MODEL])
    
    # Return model manager for immediate use
    registry = get_component_registry()
    return registry.get_instance(COMPONENT_MODEL)