# -*- coding: utf-8 -*-
"""
文化传播模型的监控指标计算模块 - 适配新的Schema结构
"""

from typing import Dict, Any, List, Optional, Union, Callable
import math
from collections import Counter, defaultdict
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)


def get_agent_dimension_data(data: Dict[str, Any], dimensions: List[str]) -> tuple:
    """
    获取agent维度数据，支持列表和字典两种格式
    
    Returns:
        (all_agent_ids: set, dimension_agent_values: dict)
    """
    all_agent_ids = set()
    dimension_agent_values = {}
    
    for dimension in dimensions:
        dimension_data = safe_get(data, dimension, [])
        agent_values = {}
        
        if isinstance(dimension_data, list):
            # 对于列表格式，使用索引作为agent_id
            for i, value in enumerate(dimension_data):
                if value:  # 只考虑有效值
                    agent_values[str(i)] = value
                    all_agent_ids.add(str(i))
        elif isinstance(dimension_data, dict):
            # 如果是字典格式，直接使用
            for agent_id, value in dimension_data.items():
                if value:
                    agent_values[agent_id] = value
                    all_agent_ids.add(agent_id)
        
        dimension_agent_values[dimension] = agent_values
    
    return all_agent_ids, dimension_agent_values


def build_grid_adjacency(agent_ids: List[str], grid_size: int = 10) -> Dict[str, List[str]]:
    """
    构建10x10网格的邻接关系
    
    Args:
        agent_ids: agent ID列表（从"0"开始的字符串）
        grid_size: 网格大小，默认10
        
    Returns:
        邻接表字典，键为agent_id，值为相邻agent_id列表
    """
    adjacency = defaultdict(list)
    agent_positions = {}
    
    # Agent IDs是从"0"开始的字符串，直接转换为整数索引
    for agent_id in agent_ids:
        try:
            # 将agent_id转换为整数索引
            idx = int(agent_id)
            x, y = idx % grid_size, idx // grid_size
            agent_positions[agent_id] = (x, y)
        except ValueError:
            # 如果转换失败，使用列表中的位置作为索引
            idx = list(agent_ids).index(agent_id)
            x, y = idx % grid_size, idx // grid_size
            agent_positions[agent_id] = (x, y)
    
    # 构建邻接关系
    for agent_id, (x, y) in agent_positions.items():
        neighbors = []
        
        # 检查四个方向的邻居：上下左右
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            
            # 检查边界
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                # 查找对应位置的agent
                for neighbor_id, (neighbor_x, neighbor_y) in agent_positions.items():
                    if neighbor_x == nx and neighbor_y == ny:
                        neighbors.append(neighbor_id)
                        break
        
        adjacency[agent_id] = neighbors
    
    return dict(adjacency)


def calculate_cultural_distribution(data: Dict[str, Any]) -> Any:
    """
    计算指标: cultural_distribution
    描述: 追踪随时间变化的各种文化特征的分布，可视化文化传播模式
    可视化类型: bar  # 修改为bar图以支持多维度显示
    更新频率: 5 秒
    
    返回所有维度文化特征的合并分布，适合柱状图显示
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("cultural_distribution", ValueError("无效的数据输入"), {"data": data})
            return {}

        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 合并所有维度的文化特征分布
        combined_distribution = {}
        
        # 对每个文化维度计算分布
        for dimension in dimensions:
            # 获取所有agent的该维度偏好
            dimension_values = safe_list(safe_get(data, dimension, []))
            
            # 过滤掉None值
            valid_values = [value for value in dimension_values if value]
            
            # 如果没有有效数据，跳过该维度
            if not valid_values:
                continue
            
            # 计算每种值的数量
            value_counts = Counter(valid_values)
            
            # 将该维度的分布加入到合并结果中，使用dimension前缀区分
            for trait_value, count in value_counts.items():
                combined_key = f"{dimension}_{trait_value}"
                combined_distribution[combined_key] = count
        
        return combined_distribution
    
    except Exception as e:
        log_metric_error("cultural_distribution", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}


def calculate_cultural_homogeneity(data: Dict[str, Any]) -> Any:
    """
    计算指标: cultural_homogeneity
    描述: 测量整个网络的文化同质性，范围从0(完全多样化)到1(完全同质化)
    可视化类型: line
    更新频率: 5 秒
    
    对5个维度分别计算同质性，然后取平均值
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("cultural_homogeneity", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 计算每个维度的同质性指数
        homogeneity_indices = []
        
        for dimension in dimensions:
            # 获取所有agent的该维度偏好
            dimension_values = safe_list(safe_get(data, dimension, []))
            
            # 过滤掉None值和空值
            valid_values = [value for value in dimension_values if value]
            
            # 如果没有有效数据，跳过该维度
            if not valid_values:
                continue
            
            # 计算每种值的数量
            value_counts = Counter(valid_values)
            
            # 计算最常见值的使用比例
            total_agents = len(valid_values)
            most_common_value, most_common_count = value_counts.most_common(1)[0]
            dimension_homogeneity = most_common_count / total_agents
            
            homogeneity_indices.append(dimension_homogeneity)
        
        # 如果没有收集到任何维度的同质性指数，返回0
        if not homogeneity_indices:
            return 0
        
        # 返回所有维度同质性指数的平均值
        return sum(homogeneity_indices) / len(homogeneity_indices)
    
    except Exception as e:
        log_metric_error("cultural_homogeneity", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_adoption_rate(data: Dict[str, Any]) -> Any:
    """
    计算指标: adoption_rate
    描述: 每轮推荐导致文化特征采纳的百分比
    可视化类型: line
    更新频率: 5 秒
    
    返回一个0到1之间的数值，表示当前轮次的采纳率
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("adoption_rate", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 获取当前轮次
        current_round = safe_number(safe_get(data, "round_number", 0))
        
        # 获取所有agent的采纳历史
        adoption_histories = safe_list(safe_get(data, "adoption_history", []))
        
        # 如果没有采纳历史，返回0
        if not adoption_histories:
            return 0
        
        # 统计当前轮次的采纳情况
        total_recommendations = 0
        successful_adoptions = 0
        
        for agent_history in adoption_histories:
            if not isinstance(agent_history, list):
                continue
                
            for entry in agent_history:
                if not isinstance(entry, dict):
                    continue
                    
                entry_round = safe_number(entry.get("round", -1))
                
                # 只统计当前轮次的数据
                if entry_round == current_round:
                    total_recommendations += 1
                    if entry.get("adopted", False):
                        successful_adoptions += 1
        
        # 计算采纳率
        if total_recommendations == 0:
            return 0
            
        adoption_rate = successful_adoptions / total_recommendations
        return adoption_rate
    
    except Exception as e:
        log_metric_error("adoption_rate", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_cultural_regions(data: Dict[str, Any]) -> Any:
    """
    计算指标: cultural_regions
    描述: 共享相同文化特征的相连agent群体的数量
    可视化类型: line  # 修改为line图显示单个总数值
    更新频率: 5 秒
    
    返回一个数值，表示基于完整文化向量的文化区域总数
    将所有文化维度作为整体考虑，只有完全相同的文化向量才算相同
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("cultural_regions", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 使用helper函数获取所有agent数据
        all_agent_ids, _ = get_agent_dimension_data(data, dimensions)
        
        if not all_agent_ids:
            return 0
        
        # 构建10x10网格邻接关系
        adjacency = build_grid_adjacency(list(all_agent_ids))
        
        # 为每个agent构建完整的文化向量
        agent_cultures = {}
        for agent_id in all_agent_ids:
            culture_vector = []
            for dimension in dimensions:
                dimension_data = safe_get(data, dimension, {})
                if isinstance(dimension_data, dict):
                    value = dimension_data.get(agent_id, "")
                elif isinstance(dimension_data, list):
                    try:
                        idx = int(agent_id)
                        value = dimension_data[idx] if idx < len(dimension_data) else ""
                    except (ValueError, IndexError):
                        value = ""
                else:
                    value = ""
                culture_vector.append(value)
            
            # 将文化向量转换为元组作为唯一标识
            agent_cultures[agent_id] = tuple(culture_vector)
        
        # 使用BFS搜索标识文化区域
        visited = set()
        total_regions = 0
        
        for agent_id in agent_cultures:
            if agent_id in visited:
                continue
                
            # 发现新区域
            culture = agent_cultures[agent_id]
            total_regions += 1
            
            # BFS标记该区域的所有agent
            queue = [agent_id]
            visited.add(agent_id)
            
            while queue:
                current = queue.pop(0)
                
                # 使用网格邻接关系
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited and agent_cultures.get(neighbor) == culture:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        return total_regions
    
    except Exception as e:
        log_metric_error("cultural_regions", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_influence_distribution(data: Dict[str, Any]) -> Any:
    """
    计算指标: influence_distribution
    描述: 可视化哪些agent成功地影响其他人采纳了他们的文化特征
    可视化类型: bar
    更新频率: 5 秒
    
    返回一个字典，其中键是agent ID，值是该agent成功影响他人的次数
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("influence_distribution", ValueError("无效的数据输入"), {"data": data})
            return {}
        
        # 获取所有agent的采纳历史
        adoption_histories = safe_list(safe_get(data, "adoption_history", []))
        
        # 统计每个agent的影响力
        influence_counts = Counter()
        
        for agent_history in adoption_histories:
            if not isinstance(agent_history, list):
                continue
                
            for entry in agent_history:
                if not isinstance(entry, dict):
                    continue
                    
                # 只统计成功采纳的情况
                if entry.get("adopted", False):
                    recommender = entry.get("recommender")
                    if recommender:
                        influence_counts[recommender] += 1
        
        # 返回影响力分布
        return dict(influence_counts)
    
    except Exception as e:
        log_metric_error("influence_distribution", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}


def calculate_dimension_influence(data: Dict[str, Any]) -> Any:
    """
    计算指标: dimension_influence
    描述: 分析哪些文化维度更容易被传播和采纳
    可视化类型: bar
    更新频率: 5 秒
    
    返回一个字典，其中键是文化维度，值是该维度被采纳的次数
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("dimension_influence", ValueError("无效的数据输入"), {"data": data})
            return {}
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 获取所有agent的采纳历史
        adoption_histories = safe_list(safe_get(data, "adoption_history", []))
        
        # 统计每个维度的采纳次数
        dimension_counts = Counter()
        
        for agent_history in adoption_histories:
            if not isinstance(agent_history, list):
                continue
                
            for entry in agent_history:
                if not isinstance(entry, dict):
                    continue
                    
                # 只统计成功采纳的情况
                if entry.get("adopted", False):
                    dimension = entry.get("dimension")
                    if dimension in dimensions:
                        dimension_counts[dimension] += 1
        
        # 返回维度影响力分布
        return dict(dimension_counts)
    
    except Exception as e:
        log_metric_error("dimension_influence", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}


def calculate_num_cultural_regions(data: Dict[str, Any]) -> Any:
    """
    计算指标: num_cultural_regions
    描述: Number of distinct contiguous cultural regions at simulation end
    可视化类型: line
    更新频率: 5 秒
    
    返回一个数值，表示总的文化区域数量，现在直接调用calculate_cultural_regions
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("num_cultural_regions", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 直接调用calculate_cultural_regions，因为它现在返回总数
        total_regions = calculate_cultural_regions(data)
        
        return total_regions
    
    except Exception as e:
        log_metric_error("num_cultural_regions", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_region_size_distribution(data: Dict[str, Any]) -> Any:
    """
    计算指标: region_size_distribution  
    描述: Distribution of sizes (number of agents) of all cultural regions at simulation end
    可视化类型: bar
    更新频率: 5 秒
    
    返回一个字典，其中键是区域大小，值是该大小区域的数量
    将所有文化维度作为整体考虑，只有完全相同的文化向量才算相同
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("region_size_distribution", ValueError("无效的数据输入"), {"data": data})
            return {}
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 使用helper函数获取所有agent数据
        all_agent_ids, _ = get_agent_dimension_data(data, dimensions)
        
        if not all_agent_ids:
            return {}
        
        # 构建10x10网格邻接关系
        adjacency = build_grid_adjacency(list(all_agent_ids))
        
        # 为每个agent构建完整的文化向量
        agent_cultures = {}
        for agent_id in all_agent_ids:
            culture_vector = []
            for dimension in dimensions:
                dimension_data = safe_get(data, dimension, {})
                if isinstance(dimension_data, dict):
                    value = dimension_data.get(agent_id, "")
                elif isinstance(dimension_data, list):
                    try:
                        idx = int(agent_id)
                        value = dimension_data[idx] if idx < len(dimension_data) else ""
                    except (ValueError, IndexError):
                        value = ""
                else:
                    value = ""
                culture_vector.append(value)
            
            # 将文化向量转换为元组作为唯一标识
            agent_cultures[agent_id] = tuple(culture_vector)
        
        # 使用BFS找到所有区域的大小
        visited = set()
        all_region_sizes = []
        
        for agent_id in agent_cultures:
            if agent_id in visited:
                continue
                
            # 发现新区域，计算大小
            culture = agent_cultures[agent_id]
            region_size = 0
            
            # BFS遍历区域
            queue = [agent_id]
            visited.add(agent_id)
            
            while queue:
                current = queue.pop(0)
                region_size += 1
                
                # 使用网格邻接关系
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited and agent_cultures.get(neighbor) == culture:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if region_size > 0:
                all_region_sizes.append(region_size)
        
        # 统计区域大小分布
        size_distribution = Counter(all_region_sizes)
        
        # 转换为字符串键以便可视化
        return {f"Size {size}": count for size, count in size_distribution.items()}
    
    except Exception as e:
        log_metric_error("region_size_distribution", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return {}


def calculate_average_trait_adoption_rate(data: Dict[str, Any]) -> Any:
    """
    计算指标: average_trait_adoption_rate
    描述: Average number of trait adoptions per agent per step
    可视化类型: line
    更新频率: 5 秒
    
    返回一个数值，表示平均每个agent每步的特征采纳率
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("average_trait_adoption_rate", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 获取所有agent的采纳历史
        adoption_histories = safe_list(safe_get(data, "adoption_history", []))
        
        if not adoption_histories:
            return 0
        
        # 统计所有agent的采纳次数和总步数
        total_adoptions = 0
        total_agents = len(adoption_histories)
        
        # 获取当前轮数（假设所有agent都经历了相同的轮数）
        max_rounds = 0
        
        for agent_history in adoption_histories:
            if not isinstance(agent_history, list):
                continue
                
            agent_adoptions = 0
            agent_max_round = 0
            
            for entry in agent_history:
                if not isinstance(entry, dict):
                    continue
                    
                # 统计成功采纳
                if entry.get("adopted", False):
                    agent_adoptions += 1
                
                # 跟踪最大轮数
                entry_round = safe_number(entry.get("round", 0))
                if entry_round > agent_max_round:
                    agent_max_round = entry_round
            
            total_adoptions += agent_adoptions
            if agent_max_round > max_rounds:
                max_rounds = agent_max_round
        
        # 计算平均每个agent每步的采纳率
        if total_agents == 0 or max_rounds == 0:
            return 0
        
        # 注意：max_rounds可能从0开始，所以需要+1来获得实际步数
        total_steps = max_rounds + 1 if max_rounds > 0 else 1
        average_rate = total_adoptions / (total_agents * total_steps)
        
        return average_rate
    
    except Exception as e:
        log_metric_error("average_trait_adoption_rate", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_final_trait_diversity(data: Dict[str, Any]) -> Any:
    """
    计算指标: final_trait_diversity
    描述: Shannon entropy of trait vectors across all agents at simulation end
    可视化类型: line
    更新频率: 5 秒
    
    返回一个数值，表示基于Shannon熵的特征向量多样性指数
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("final_trait_diversity", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 收集所有agent的完整特征向量
        trait_vectors = []
        
        # 使用helper函数获取所有agent数据
        agent_ids, _ = get_agent_dimension_data(data, dimensions)
        
        # 为每个agent构建特征向量
        for agent_id in agent_ids:
            trait_vector = []
            
            for dimension in dimensions:
                dimension_data = safe_get(data, dimension, {})
                if isinstance(dimension_data, dict):
                    trait_value = dimension_data.get(agent_id, "")
                    trait_vector.append(trait_value)
                else:
                    trait_vector.append("")
            
            # 将特征向量转换为元组以便计数
            trait_vectors.append(tuple(trait_vector))
        
        if not trait_vectors:
            return 0
        
        # 计算特征向量的分布
        vector_counts = Counter(trait_vectors)
        total_agents = len(trait_vectors)
        
        # 计算Shannon熵
        shannon_entropy = 0
        for count in vector_counts.values():
            if count > 0:
                probability = count / total_agents
                shannon_entropy -= probability * math.log2(probability)
        
        return shannon_entropy
    
    except Exception as e:
        log_metric_error("final_trait_diversity", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_local_convergence_index(data: Dict[str, Any]) -> Any:
    """
    计算指标: local_convergence_index
    描述: Measures local cultural convergence by calculating the average size of cultural regions normalized by total population
    可视化类型: line
    更新频率: 5 秒
    
    基于参考代码的local convergence计算逻辑，衡量局部文化趋同程度
    将所有文化维度作为整体考虑，只有完全相同的文化向量才算相同
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("local_convergence_index", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 使用helper函数获取所有agent数据
        all_agent_ids, _ = get_agent_dimension_data(data, dimensions)
        if not all_agent_ids:
            return 0
        
        # 构建10x10网格邻接关系（根据现有的网格大小调整）
        adjacency = build_grid_adjacency(list(all_agent_ids))
        total_agents = len(all_agent_ids)
        
        # 为每个agent构建完整的文化向量
        agent_cultures = {}
        for agent_id in all_agent_ids:
            culture_vector = []
            for dimension in dimensions:
                dimension_data = safe_get(data, dimension, {})
                if isinstance(dimension_data, dict):
                    value = dimension_data.get(agent_id, "")
                elif isinstance(dimension_data, list):
                    try:
                        idx = int(agent_id)
                        value = dimension_data[idx] if idx < len(dimension_data) else ""
                    except (ValueError, IndexError):
                        value = ""
                else:
                    value = ""
                culture_vector.append(value)
            
            # 将文化向量转换为元组作为唯一标识
            agent_cultures[agent_id] = tuple(culture_vector)
        
        # 使用BFS找到所有文化区域及其大小
        visited = set()
        all_region_sizes = []
        
        for agent_id in agent_cultures:
            if agent_id in visited:
                continue
                
            # 发现新区域，计算大小
            culture = agent_cultures[agent_id]
            region_size = 0
            
            # BFS遍历区域
            queue = [agent_id]
            visited.add(agent_id)
            
            while queue:
                current = queue.pop(0)
                region_size += 1
                
                # 使用网格邻接关系
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited and agent_cultures.get(neighbor) == culture:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if region_size > 0:
                all_region_sizes.append(region_size)
        
        # 计算local convergence index = 平均区域大小 / 总人口
        if not all_region_sizes:
            return 0
        
        average_region_size = sum(all_region_sizes) / len(all_region_sizes)
        local_convergence_index = average_region_size / total_agents
        
        return local_convergence_index
    
    except Exception as e:
        log_metric_error("local_convergence_index", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


def calculate_global_polarization_index(data: Dict[str, Any]) -> Any:
    """
    计算指标: global_polarization_index
    描述: Measures global cultural polarization by calculating the number of distinct cultural regions normalized by total population
    可视化类型: line
    更新频率: 5 秒
    
    基于参考代码的global polarization计算逻辑，衡量全局文化分化程度
    将所有文化维度作为整体考虑，只有完全相同的文化向量才算相同
    """
    try:
        # 验证输入数据
        if not data or not isinstance(data, dict):
            log_metric_error("global_polarization_index", ValueError("无效的数据输入"), {"data": data})
            return 0
        
        # 文化维度列表
        dimensions = [
            "music_preference",
            "culinary_preference", 
            "fashion_style", 
            "political_orientation", 
            "leisure_activity"
        ]
        
        # 使用helper函数获取所有agent数据
        all_agent_ids, _ = get_agent_dimension_data(data, dimensions)
        
        if not all_agent_ids:
            return 0
        
        # 构建10x10网格邻接关系
        adjacency = build_grid_adjacency(list(all_agent_ids))
        total_agents = len(all_agent_ids)
        
        # 为每个agent构建完整的文化向量
        agent_cultures = {}
        for agent_id in all_agent_ids:
            culture_vector = []
            for dimension in dimensions:
                dimension_data = safe_get(data, dimension, {})
                if isinstance(dimension_data, dict):
                    value = dimension_data.get(agent_id, "")
                elif isinstance(dimension_data, list):
                    try:
                        idx = int(agent_id)
                        value = dimension_data[idx] if idx < len(dimension_data) else ""
                    except (ValueError, IndexError):
                        value = ""
                else:
                    value = ""
                culture_vector.append(value)
            
            # 将文化向量转换为元组作为唯一标识
            agent_cultures[agent_id] = tuple(culture_vector)
        
        # 使用BFS找到所有文化区域
        visited = set()
        total_regions = 0
        
        for agent_id in agent_cultures:
            if agent_id in visited:
                continue
                
            # 发现新区域
            culture = agent_cultures[agent_id]
            total_regions += 1
            
            # BFS标记该区域的所有agent
            queue = [agent_id]
            visited.add(agent_id)
            
            while queue:
                current = queue.pop(0)
                
                # 使用网格邻接关系
                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited and agent_cultures.get(neighbor) == culture:
                        visited.add(neighbor)
                        queue.append(neighbor)
        
        # 计算global polarization index = 区域总数 / 总人口
        # 值越高表示文化分化程度越高
        if total_agents == 0:
            return 0
            
        global_polarization_index = total_regions / total_agents
        
        return global_polarization_index
    
    except Exception as e:
        log_metric_error("global_polarization_index", e, {"data_keys": list(data.keys()) if isinstance(data, dict) else None})
        return 0


# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
    'calculate_cultural_distribution': calculate_cultural_distribution,
    'calculate_cultural_homogeneity': calculate_cultural_homogeneity,
    'calculate_adoption_rate': calculate_adoption_rate,
    'calculate_cultural_regions': calculate_cultural_regions,
    'calculate_influence_distribution': calculate_influence_distribution,
    'calculate_dimension_influence': calculate_dimension_influence,
    'calculate_num_cultural_regions': calculate_num_cultural_regions,
    'calculate_region_size_distribution': calculate_region_size_distribution,
    'calculate_average_trait_adoption_rate': calculate_average_trait_adoption_rate,
    'calculate_final_trait_diversity': calculate_final_trait_diversity,
    'calculate_local_convergence_index': calculate_local_convergence_index,
    'calculate_global_polarization_index': calculate_global_polarization_index,
}


def get_metric_function(function_name: str) -> Optional[Callable]:
    """
    根据函数名获取对应的指标计算函数
    
    Args:
        function_name: 函数名
        
    Returns:
        指标计算函数或None
    """
    return METRIC_FUNCTIONS.get(function_name)
