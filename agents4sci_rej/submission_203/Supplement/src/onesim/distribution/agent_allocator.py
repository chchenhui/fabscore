import asyncio
from typing import Dict, List, Any, Optional, Set
from loguru import logger
import json

from onesim.distribution.node import get_node, NodeRole
from onesim.distribution.master import MasterNode

class AgentAllocator:
    """负责在分布式环境中为Agent分配Worker节点"""
    
    def __init__(self, master_node: MasterNode):
        self.master_node = master_node
        self.allocation_cache = {}  # 记录已分配的agent_id -> worker_id映射
        

    async def allocate_agents_batch(self, agent_configs: List[Dict[str, Any]]) -> Dict[str, str]:
        """批量分配一组Agent到Worker节点，尽量将有交互关系的不同类型Agent放在同一Worker
        
        Args:
            agent_configs: 包含多个Agent配置的列表，每个配置必须包含id和type字段
            
        Returns:
            Dict[str, str]: agent_id到worker_id的映射
        """
        if not self.master_node.workers:
            logger.error("No workers available for agent allocation")
            return {}
        
        worker_count = len(self.master_node.workers)
        worker_ids = list(self.master_node.workers.keys())
        if not worker_ids:
            return {}
        
        # 创建ID到配置的映射
        id_to_config = {str(config.get("id")): config for config in agent_configs}
        
        # 收集关系信息 - 构建关系图
        relationship_graph = {}  # agent_id -> list of (target_id, weight)
        
        for config in agent_configs:
            agent_id = str(config.get("id"))
            agent_type = config.get("type", "unknown")
            if not agent_id:
                continue
                
            # 初始化图节点
            if agent_id not in relationship_graph:
                relationship_graph[agent_id] = []
                
            # 添加关系边
            relationships = config.get("relationships", [])
            for rel in relationships:
                target_id = rel.get("target_id")
                if not target_id or target_id not in id_to_config:
                    continue
                
                # 检查目标Agent的类型
                target_type = id_to_config[target_id].get("type", "unknown")
                
                # 不同类型之间的关系权重更高（更倾向于将它们放在一起）
                weight = 3 if agent_type != target_type else 1
                    
                # 添加带权重的边
                relationship_graph[agent_id].append((target_id, weight))
                
                # 确保目标节点存在于图中
                if target_id not in relationship_graph:
                    relationship_graph[target_id] = []
                relationship_graph[target_id].append((agent_id, weight))
        
        # 查找子图（交互组）
        def find_communities():
            """使用简化的社区检测算法找出交互紧密的Agent组"""
            visited = set()
            communities = []
            
            # 按连接数排序，从连接最多的开始处理
            sorted_nodes = sorted(
                relationship_graph.keys(), 
                key=lambda x: sum(w for _, w in relationship_graph.get(x, []))
            )
            
            for node in sorted_nodes:
                if node in visited:
                    continue
                    
                # 使用带权重的BFS查找交互组
                community = []
                queue = [(node, 0)]  # (node_id, distance from start)
                community_visited = set()
                
                while queue:
                    current, dist = queue.pop(0)
                    if current in community_visited:
                        continue
                        
                    community_visited.add(current)
                    visited.add(current)
                    community.append(current)
                    
                    # 遍历相邻节点，按权重排序（优先处理不同类型交互）
                    neighbors = sorted(
                        relationship_graph.get(current, []),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for neighbor, weight in neighbors:
                        if neighbor not in community_visited:
                            # 距离越远，优先级越低
                            queue.append((neighbor, dist + 1))
                
                if community:
                    communities.append(community)
            
            # 添加没有关系的单独Agent
            for agent_id in id_to_config:
                if agent_id not in visited:
                    communities.append([agent_id])
                    visited.add(agent_id)
            
            return communities
        
        communities = find_communities()
        
        # 计算各worker的理想负载和最大允许负载
        agent_count = len(agent_configs)
        avg_load = max(1, agent_count // worker_count)
        max_load = avg_load + min(avg_load, 5)  # 最大允许负载
        
        # 处理过大的组
        final_communities = []
        for community in communities:
            if len(community) > max_load:
                # 使用贪婪算法来拆分大组，尽量保持不同类型之间的交互关系
                subgroups = []
                current_group = []
                agent_types_in_group = set()
                
                # 按类型多样性对社区中的agent排序
                def type_diversity(agent_id):
                    agent_type = id_to_config[agent_id].get("type", "unknown")
                    connected_types = set()
                    for neighbor, _ in relationship_graph.get(agent_id, []):
                        neighbor_type = id_to_config[neighbor].get("type", "unknown")
                        connected_types.add(neighbor_type)
                    return len(connected_types)
                
                sorted_community = sorted(community, key=type_diversity, reverse=True)
                
                for agent_id in sorted_community:
                    agent_type = id_to_config[agent_id].get("type", "unknown")
                    
                    # 如果当前组已达最大负载或添加会导致同类型过多，则开始新组
                    if len(current_group) >= max_load:
                        if current_group:
                            subgroups.append(current_group)
                        current_group = [agent_id]
                        agent_types_in_group = {agent_type}
                    else:
                        current_group.append(agent_id)
                        agent_types_in_group.add(agent_type)
                
                if current_group:  # 添加最后一个子组
                    subgroups.append(current_group)
                
                # 将所有子组添加到最终列表
                final_communities.extend(subgroups)
            else:
                final_communities.append(community)
        
        # 按社区大小和类型多样性排序
        def community_priority(community):
            # 计算社区中不同类型的数量
            types = set(id_to_config[aid].get("type", "unknown") for aid in community)
            # 返回(类型数量, 社区大小)作为优先级
            return (len(types), len(community))
        
        # 大社区和类型多样性高的社区优先分配
        final_communities.sort(key=community_priority, reverse=True)
        
        # 初始化worker负载跟踪
        worker_loads = {wid: 0 for wid in worker_ids}
        worker_type_counts = {wid: {} for wid in worker_ids}  # 跟踪每个worker上各类型的数量
        allocations = {}  # agent_id -> worker_id
        
        # 分配社区到worker
        for community in final_communities:
            # 找到最适合的worker
            # 1. 首先考虑负载
            # 2. 然后考虑类型的互补性（偏好已有互补类型的worker）
            
            best_worker = None
            best_score = float('-inf')
            
            for worker_id in worker_ids:
                # 如果worker负载已达到最大，跳过
                if worker_loads[worker_id] + len(community) > max_load * 1.2:
                    continue
                    
                # 计算与此worker已有类型的兼容性
                compatibility_score = 0
                community_types = {}
                
                # 统计社区中各类型的数量
                for agent_id in community:
                    agent_type = id_to_config[agent_id].get("type", "unknown")
                    community_types[agent_type] = community_types.get(agent_type, 0) + 1
                
                # 检查与worker上现有类型的兼容性
                for comm_type, comm_count in community_types.items():
                    # 如果worker上已有此类型的Agent，得分降低（避免同类型过多）
                    worker_type_count = worker_type_counts[worker_id].get(comm_type, 0)
                    if worker_type_count > 0:
                        compatibility_score -= comm_count * worker_type_count
                    else:
                        # 没有此类型，增加多样性得分
                        compatibility_score += comm_count
                
                # 负载得分（负载越低越好）
                load_score = -worker_loads[worker_id]
                
                # 总得分
                total_score = load_score * 2 + compatibility_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_worker = worker_id
            
            # 如果没找到合适的worker，选择负载最低的
            if best_worker is None:
                best_worker = min(worker_loads.items(), key=lambda x: x[1])[0]
            
            # 分配社区到选定的worker
            for agent_id in community:
                agent_type = id_to_config[agent_id].get("type", "unknown")
                allocations[agent_id] = best_worker
                worker_loads[best_worker] += 1
                
                # 更新此worker的类型计数
                if agent_type not in worker_type_counts[best_worker]:
                    worker_type_counts[best_worker][agent_type] = 0
                worker_type_counts[best_worker][agent_type] += 1
        
        # 更新master节点的位置映射和worker信息
        for agent_id, worker_id in allocations.items():
            self.master_node.agent_locations[agent_id] = worker_id
            
            worker = self.master_node.workers.get(worker_id)
            if worker:
                if not hasattr(worker, 'agent_count'):
                    worker.agent_count = 0
                if not hasattr(worker, 'agent_ids'):
                    worker.agent_ids = []
                
                if agent_id not in worker.agent_ids:
                    worker.agent_count += 1
                    worker.agent_ids.append(agent_id)
        
        # 验证每个worker的负载均衡性
        min_load = min(worker_loads.values()) if worker_loads else 0
        max_load = max(worker_loads.values()) if worker_loads else 0
        load_variance = max_load - min_load
        
        logger.info(f"Agent分配完成: 总Agent {len(allocations)}个, Worker {len(worker_ids)}个")
        logger.info(f"负载情况: 最小 {min_load}, 最大 {max_load}, 差异 {load_variance}")
        
        self.allocation_cache.update(allocations)
        return allocations
    
    async def create_agents_on_workers(self, allocations: Dict[str, str], agent_configs: List[Dict[str, Any]]) -> bool:
        """在Worker上批量创建已分配的Agent
        
        Args:
            allocations: agent_id到worker_id的映射
            agent_configs: Agent配置列表
            
        Returns:
            bool: 是否所有Agent都创建成功
        """
        # 按worker分组，减少通信次数
        configs_by_worker = {}
        for config in agent_configs:
            agent_id = config.get("id")
            if not agent_id or agent_id not in allocations:
                continue
                
            worker_id = allocations[agent_id]
            if worker_id not in configs_by_worker:
                configs_by_worker[worker_id] = []
            configs_by_worker[worker_id].append(config)
        
        # 并行向各个Worker发送创建请求
        tasks = []
        for worker_id, configs in configs_by_worker.items():
            worker = self.master_node.workers.get(worker_id)
            if not worker:
                logger.error(f"Worker {worker_id} not found")
                continue
            
            task = asyncio.create_task(
                self._create_agents_batch(
                    worker.address,
                    worker.port,
                    configs
                )
            )
            tasks.append(task)
        
        if not tasks:
            return False
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return all(r is True for r in results if not isinstance(r, Exception))
    
    async def _create_agents_batch(self, worker_address: str, worker_port: int, agent_configs: List[Dict[str, Any]]) -> bool:
        """在指定Worker上批量创建Agent"""
        if not self.master_node._grpc_module:
            logger.error("gRPC module not initialized")
            return False
        agent_ids = [config.get("id") for config in agent_configs]
        try:
            return await self.master_node._grpc_module.create_agents_batch_on_worker(
                worker_address,
                worker_port,
                agent_configs
            )
        except Exception as e:
            logger.error(f"Error creating agents on worker {worker_address}:{worker_port}: {e}")
            return False 