from typing import Dict, Optional, List, Union, Any
from enum import Enum
import asyncio
import importlib
from loguru import logger

class NodeRole(Enum):
    SINGLE = "single"
    MASTER = "master"
    WORKER = "worker"

class Node:
    """Base class for distributed node functionality"""
    
    def __init__(self, 
                 node_id: str,
                 role: NodeRole = NodeRole.SINGLE,
                 config: Optional[Dict[str, Any]] = None):
        self.node_id = node_id
        self.role = role
        self.config = config or {}
        self._initialized = False
        self._grpc_module = None
        
    async def initialize(self):
        """Initialize the node based on its role"""
        if self._initialized:
            return
            
        if self.role != NodeRole.SINGLE:
            # Lazy import gRPC dependencies only when needed
            try:
                self._grpc_module = importlib.import_module('onesim.distribution.grpc_impl')
                # 初始化连接管理器
                from onesim.distribution.connection_manager import connection_manager
                # 确保连接管理器正确初始化（异步方式）
                await connection_manager.initialize()
                # 确保批处理器也被导入
                from onesim.distribution.batch_processor import batch_processor
                logger.info(f"Initialized {self.role.value} node with gRPC support")
            except ImportError as e:
                print(e)
                logger.error("Failed to import gRPC dependencies. Please install them with: pip install grpcio grpcio-tools")
                raise
        
        self._initialized = True
        
    @property
    def is_distributed(self) -> bool:
        """Check if this node is part of a distributed setup"""
        return self.role != NodeRole.SINGLE
    
    async def shutdown(self):
        """Base shutdown method for proper resource cleanup"""
        logger.info(f"Base node {self.node_id} shutting down")
        
        if self.role != NodeRole.SINGLE and self._initialized:
            try:
                # 清理连接管理器资源
                from onesim.distribution.connection_manager import connection_manager
                await connection_manager.stop()
                
                # 停止批处理器
                if self._grpc_module:
                    from onesim.distribution.batch_processor import batch_processor
                    batch_processor.stop()
                    
                logger.info(f"Node {self.node_id} connection resources cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up node resources: {e}")
        
        # Will be overridden by subclasses

# Global node instance
_node_instance = None

def get_node() -> Node:
    """Get the global node instance"""
    global _node_instance
    if _node_instance is None:
        _node_instance = Node("default", NodeRole.SINGLE)
    return _node_instance

async def initialize_node(node_id: str, role: str, config: Optional[Dict[str, Any]] = None) -> Node:
    """Initialize the global node with specified settings"""
    global _node_instance
    
    # Convert role string to enum
    try:
        if role.lower() == "single":
            role_enum = NodeRole.SINGLE
            _node_instance = Node(node_id, role_enum, config)
        elif role.lower() == "master":
            role_enum = NodeRole.MASTER
            # Dynamic import to avoid circular imports
            from onesim.distribution.master import MasterNode
            _node_instance = MasterNode(node_id, config)
        elif role.lower() == "worker":
            role_enum = NodeRole.WORKER
            # Dynamic import to avoid circular imports
            from onesim.distribution.worker import WorkerNode
            _node_instance = WorkerNode(node_id, config)
        else:
            logger.error(f"Invalid role: {role}. Using SINGLE.")
            role_enum = NodeRole.SINGLE
            _node_instance = Node(node_id, role_enum, config)
    except ValueError:
        logger.error(f"Invalid role: {role}. Using SINGLE.")
        role_enum = NodeRole.SINGLE
        _node_instance = Node(node_id, role_enum, config)
    
    # Initialize the node
    await _node_instance.initialize()
    return _node_instance