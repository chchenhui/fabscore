from typing import Dict, List, Any, Callable, Optional
import logging
from collections import defaultdict
import os
import asyncio
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import time
import importlib.util
import re

# Import get_config to access global configuration
from onesim.config import get_config

from .metric import (
    MetricDefinition, 
    VariableSpec,
    MetricResult, 
    TimeSeriesMetricData,
    CategoryMetricData
)

from loguru import logger

from .utils import create_line_chart_option, create_pie_chart_option, create_bar_chart_option, create_time_series_chart_option


class DataCollector:
    """数据收集器，负责从环境和Agent收集所需数据"""
    
    async def collect_env_data(self, env: Any, variables: List[VariableSpec]) -> Dict:
        """
        从环境中收集指定变量 (using var.path and env.get_data)
        
        Args:
            env: 环境对象 (assuming env has a get_data method)
            variables: 变量规范列表
            
        Returns:
            变量名到值的映射
        """
        result = {}
        if not hasattr(env, 'get_data') or not callable(env.get_data):
            logger.error("Environment object is missing a callable 'get_data' method.")
            # Provide default values (e.g., None) for all requested env vars
            for var in variables:
                if var.source_type == "env":
                    result[var.name] = None
            return result
            
        for var in variables:
            if var.source_type != "env":
                continue
            
            # Use env.get_data which supports dot notation for nested access
            value = await env.get_data(var.path) # Using var.path directly
            result[var.name] = value # Store result using the metric's variable name
            
        return result
        
    async def collect_agent_data(self, env: Any, agent_type: str, variables: List[VariableSpec]) -> Dict:
        """
        从特定类型的所有Agent收集数据 (using env.get_agent_data_by_type)
        
        Args:
            env: 环境对象 (must have get_agent_data_by_type method)
            agent_type: Agent类型
            variables: 变量规范列表
            
        Returns:
            变量名到值的映射，对于Agent变量，值是列表
        """
        result = defaultdict(list)
        
        # Filter variables relevant to this agent type
        agent_vars_for_type = [var for var in variables if var.source_type == "agent" and var.agent_type == agent_type]
        if not agent_vars_for_type:
            return {} # No relevant variables for this type

        # Check if environment has the required method
        if not hasattr(env, 'get_agent_data_by_type') or not callable(env.get_agent_data_by_type):
            logger.error(f"Environment is missing callable 'get_agent_data_by_type' method.")
            # Return empty lists for all variables of this type
            for var in agent_vars_for_type:
                result[var.name] = []
            return result

        # Iterate through each relevant variable definition
        for var in agent_vars_for_type:
            try:
                # Use the environment's method to get data for this variable from all agents of the type
                # This method should handle dot notation in var.path and potential distribution
                agent_data_dict = await env.get_agent_data_by_type(agent_type, var.path)
                
                # The expected return format for collect_agent_data is a list of values.
                # Extract values from the dictionary returned by get_agent_data_by_type.
                # The order might not be guaranteed, but for aggregation it often doesn't matter.
                if isinstance(agent_data_dict, dict):
                    collected_values = list(agent_data_dict.values())
                elif agent_data_dict is None:
                    # If the method returns None (e.g., error), provide an empty list
                    collected_values = []
                    logger.warning(f"Received None from get_agent_data_by_type for {agent_type}.{var.path}")
                else:
                    # Handle unexpected return types
                    logger.warning(f"Unexpected return type {type(agent_data_dict)} from get_agent_data_by_type for {agent_type}.{var.path}")
                    collected_values = []
                    
            except Exception as e:
                logger.error(f"Error calling env.get_agent_data_by_type for '{agent_type}', path '{var.path}': {e}")
                collected_values = [] # Provide empty list on error
                
            # Store the list of collected values under the metric variable name
            result[var.name] = collected_values
            
        return result
    
    async def collect_for_metric(self, env: Any, metric_def: MetricDefinition) -> Dict:
        """
        收集特定指标所需的所有数据
        
        Args:
            env: 环境对象
            metric_def: 指标定义
            
        Returns:
            变量名到值的映射
        """
        result = {}
        
        # 按变量来源分组
        env_vars = []
        agent_vars_by_type = defaultdict(list)
        
        for var in metric_def.variables:
            if var.source_type == "env":
                env_vars.append(var)
            elif var.source_type == "agent" and var.agent_type:
                agent_vars_by_type[var.agent_type].append(var)
        
        # 收集环境变量
        if env_vars:
            env_data = await self.collect_env_data(env, env_vars)
            result.update(env_data)
        
        # 收集每种类型的Agent变量
        for agent_type, vars_list in agent_vars_by_type.items():
            agent_data = await self.collect_agent_data(env, agent_type, vars_list)
            result.update(agent_data)
        
        return result


class MetricProcessor:
    """指标处理器，执行计算并格式化结果"""
    
    def calculate(self, metric_def: MetricDefinition, data: Dict) -> Any:
        """
        执行指标计算
        
        Args:
            metric_def: 指标定义
            data: 所需的数据
            
        Returns:
            计算结果
        """
        try:
            # 检查是否所有必需的变量都有值
            missing_vars = []
            for var in metric_def.variables:
                # Also check for None if required, as None might break calculations
                if var.required and (var.name not in data or data[var.name] is None):
                    missing_vars.append(var.name)
                    
            if missing_vars:
                logger.warning(f"指标 {metric_def.name} 缺少必要变量或其值为None: {', '.join(missing_vars)}. Returning None.")
                return None # Calculation function should handle None input if possible
                
            # 执行计算函数
            result = metric_def.calculation_function(data)
            return result
        except Exception as e:
            logger.error(f"计算指标 {metric_def.name} 时发生错误: {str(e)}", exc_info=True)
            return None # Return None on calculation error
            
    def format_for_visualization(self, raw_result: Any, metric_def: MetricDefinition, ts_data: Optional[TimeSeriesMetricData] = None) -> Dict:
        """
        将原始结果转换为可视化格式
        
        Args:
            raw_result: 原始计算结果
            metric_def: 指标定义
            ts_data: 时间序列数据 (用于折线图)
            
        Returns:
            适用于ECharts的数据结构
        """
        if raw_result is None:
             logger.debug(f"Raw result for {metric_def.name} is None, returning empty viz data.")
             return {} # Return empty dict if calculation failed or returned None
            
        viz_type = metric_def.visualization_type
        
        if viz_type == "line":
            # 折线图使用时间序列数据 (ts_data handles formatting)
            if ts_data is None:
                logger.warning(f"TimeSeries data not provided for line chart: {metric_def.name}")
                return {"xAxis": [], "series": []} # Default empty structure
            try:
                return ts_data.get_echarts_data() # Delegate formatting
            except Exception as e:
                logger.error(f"Error getting ECharts data from TimeSeriesMetricData for {metric_def.name}: {e}")
                return {"xAxis": [], "series": []}
            
        elif viz_type == "bar":
            # 柱状图处理 - Expects dict {category: value} or list [(cat, val), ...]
            try:
                if isinstance(raw_result, dict):
                    categories = list(raw_result.keys())
                    values = list(raw_result.values())
                elif isinstance(raw_result, (list, tuple)) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw_result):
                    # 处理 [(key, value), ...] 结构
                    categories = [item[0] for item in raw_result]
                    values = [item[1] for item in raw_result]
                elif isinstance(raw_result, (int, float)): # Handle single value case for bar? Maybe should be pie?
                     logger.warning(f"Single value {raw_result} received for bar chart {metric_def.name}. Formatting as single bar.")
                     categories = [metric_def.name] # Use metric name as category
                     values = [raw_result]
                else:
                    logger.error(f"无法将结果格式化为柱状图 (不支持的类型 {type(raw_result)}): {metric_def.name}")
                    return {"xAxis": [], "series": []}
                    
                return {"xAxis": categories, "series": values}
            except Exception as e:
                 logger.error(f"Error formatting bar chart data for {metric_def.name}: {e}")
                 return {"xAxis": [], "series": []}
            
        elif viz_type == "pie":
            # 饼图处理 - Expects dict {category: value} or list [(cat, val), ...]
            try:
                series_data = []
                if isinstance(raw_result, dict):
                    series_data = [{"name": k, "value": v} for k, v in raw_result.items() if isinstance(v, (int, float)) and v >= 0]
                elif isinstance(raw_result, (list, tuple)) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw_result):
                    # 处理 [(key, value), ...] 结构
                    series_data = [{"name": item[0], "value": item[1]} for item in raw_result if isinstance(item[1], (int, float)) and item[1] >= 0]
                else:
                    logger.error(f"无法将结果格式化为饼图 (不支持的类型 {type(raw_result)}): {metric_def.name}")
                    return {"series": []}
                
                # Filter out zero/negative values as they don't make sense in pie charts
                series_data = [item for item in series_data if item['value'] > 0]
                
                return {"series": series_data}
            except Exception as e:
                 logger.error(f"Error formatting pie chart data for {metric_def.name}: {e}")
                 return {"series": []}
            
        # Fallback for unknown viz_type or if logic missed a case
        logger.warning(f"Unknown visualization type '{viz_type}' or unhandled format for metric {metric_def.name}. Returning raw result.")
        return {"raw": raw_result} # Or return empty dict {}?
 

class MonitorScheduler:
    """指标更新调度器"""
    
    def __init__(self):
        self.tasks = {}  # 存储每个指标的调度任务 {metric_name: task}
        self.lock = asyncio.Lock()  # 使用asyncio.Lock而不是threading.Lock
        
    async def schedule_metric(self, metric_name: str, env: Any, monitor_manager: 'MonitorManager', frequency: int):
        """异步调度指标定期更新"""
        # 如果该指标已在调度中，先停止
        if metric_name in self.tasks:
            await self.pause_metric(metric_name)
        
        # 创建异步任务
        task = asyncio.create_task(
            self._update_loop(metric_name, env, monitor_manager, frequency)
        )
        
        # 保存任务
        async with self.lock:
            self.tasks[metric_name] = task
        
        logger.debug(f"指标 {metric_name} 调度已启动，更新频率: {frequency}秒")
    
    async def _update_loop(self, metric_name: str, env: Any, monitor_manager: 'MonitorManager', frequency: int):
        """异步指标更新循环"""
        try:
            while True:
                # 执行更新
                await monitor_manager.update_metric(metric_name, env)
                
                # 异步等待
                await asyncio.sleep(frequency)
        except asyncio.CancelledError:
            logger.debug(f"指标 {metric_name} 更新任务已取消")
        except Exception as e:
            logger.error(f"指标 {metric_name} 更新循环出错: {e}")
    
    async def pause_metric(self, metric_name: str):
        """
        暂停指标更新
        
        Args:
            metric_name: 指标名称
        """
        async with self.lock:
            if metric_name in self.tasks:
                task = self.tasks[metric_name]
                task.cancel()
                # Wait for the task to actually finish cancellation
                try:
                    await task 
                except asyncio.CancelledError:
                    pass # Expected
                self.tasks.pop(metric_name, None)
                logger.debug(f"指标 {metric_name} 调度已暂停")
    
    async def update_interval(self, metric_name: str, env: Any, monitor_manager: 'MonitorManager', new_frequency: int):
        """
        更新指标的更新频率
        
        Args:
            metric_name: 指标名称
            env: 环境对象
            monitor_manager: 监控管理器
            frequency: 更新频率(秒)，如果不提供则使用指标定义中的频率或全局配置
        """
        # Fetch the global update interval from MonitorConfig
        global_update_interval = get_config().monitor_config.update_interval
        
        metric_def = monitor_manager.metrics.get(metric_name)
        if not metric_def:
             logger.error(f"Cannot resume metric {metric_name}: definition not found.")
             return

        # Determine the frequency to use
        effective_frequency = global_update_interval if global_update_interval is not None else metric_def.update_interval
        
        # Use the provided frequency if it was explicitly passed (overrides default logic)
        if new_frequency is not None:
            effective_frequency = new_frequency

        await self.schedule_metric(metric_name, env, monitor_manager, effective_frequency)
        logger.debug(f"指标 {metric_name} 调度已恢复，更新频率: {effective_frequency}秒")
    
    async def resume_metric(self, metric_name: str, env: Any, monitor_manager: 'MonitorManager', frequency: int = None):
        """
        恢复指标更新
        
        Args:
            metric_name: 指标名称
            env: 环境对象
            monitor_manager: 监控管理器
            frequency: 更新频率(秒)，如果不提供则使用指标定义中的频率
        """
        if frequency is None:
            frequency = monitor_manager.metrics[metric_name].update_interval
        await self.schedule_metric(metric_name, env, monitor_manager, frequency)
        logger.debug(f"指标 {metric_name} 调度已恢复")
    
    async def stop_all(self):
        """停止所有指标更新"""
        for metric_name in list(self.tasks.keys()):
            await self.pause_metric(metric_name)


class MonitorManager:
    """监控系统总控制器"""
    
    def __init__(self):
        # 存储所有注册的指标定义
        self.metrics: Dict[str, MetricDefinition] = {}
        
        # 存储指标计算结果
        self.results: Dict[str, MetricResult] = {}
        
        # 存储时间序列数据(用于折线图)
        self.time_series_data: Dict[str, TimeSeriesMetricData] = {}
        
        # 存储类别数据(用于柱状图和饼图)
        self.category_data: Dict[str, CategoryMetricData] = {}
        
        # 数据收集器和处理器
        self.collector = DataCollector()
        self.processor = MetricProcessor()
        
        # 调度器
        self.scheduler = MonitorScheduler()
        
        # 线程安全锁
        self.lock = asyncio.Lock()  # 使用asyncio.Lock而不是threading.Lock
        
        # 环境对象引用
        self.env = None
        
        # 监控状态
        self.is_monitoring = False
        self.metric_index = 0
        
    def setup(self, env: Any):
        """
        设置监控系统，关联环境对象
        
        Args:
            env: 环境对象
        """
        self.env = env
        logger.info(f"监控系统已关联环境对象")
        return self
    
    @staticmethod
    async def setup_metrics(env: Any):
        """
        在环境中设置和启动监控系统
        
        Args:
            env: 环境对象

        Returns:
            MonitorManager实例
        """
        from onesim.config import get_component_registry
        
        # 尝试从注册表获取监控管理器
        registry = get_component_registry()
        monitor_manager = registry.get_instance("monitor")
        
        # 如果监控管理器不存在，创建一个新的
        if not monitor_manager:
            logger.warning("监控组件未初始化，正在创建新的监控管理器")
            monitor_manager = MonitorManager()
            registry.register("monitor", monitor_manager)
        
        # 设置环境
        monitor_manager.setup(env)

      
            
        env_path = env.env_path
        try:
            # 导入指标计算模块
            metrics_path = os.path.join(env_path, "code", "metrics")
            metrics_module = None
            if os.path.exists(metrics_path):
                if os.path.isdir(metrics_path):
                    module_path = os.path.join(metrics_path, "metrics.py")
                else:
                    module_path = metrics_path
                
                if os.path.exists(module_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("metrics_module", module_path)
                    metrics_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(metrics_module)
                    logger.info(f"成功导入指标计算模块: {module_path}")
            
            # 加载scene_info.json中的指标定义
            scene_info_path = os.path.join(env_path, "scene_info.json")
            if os.path.exists(scene_info_path):
                import json
                import re
                from .metric import VariableSpec, MetricDefinition
                
                with open(scene_info_path, 'r', encoding='utf-8') as f:
                    scene_info = json.load(f)
                
                if "metrics" in scene_info and isinstance(scene_info["metrics"], list):
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
                                name=metric_def["name"],
                                description=metric_def["description"],
                                variables=variables,
                                visualization_type=metric_def["visualization_type"],
                                update_interval=metric_def.get("update_interval", 60),
                                calculation_function=function_name
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
            logger.error(f"加载指标失败: {e}")
        
        # 启动监控
        await monitor_manager.start_all_metrics()
        
        return monitor_manager
        
    def register_metric(self, metric_def: MetricDefinition, calculation_function: Callable = None):
        """
        注册新指标
        
        Args:
            metric_def: 指标定义
            calculation_function: 计算函数，如果提供则覆盖metric_def中的函数
        """
        # 检查指标名是否已存在
        if metric_def.name in self.metrics:
            logger.warning(f"指标 {metric_def.name} 已存在，将被覆盖")
        
        # 如果提供了计算函数，覆盖指标定义中的函数
        if calculation_function:
            metric_def.calculation_function = calculation_function
            
        # 存储指标定义
        self.metrics[metric_def.name] = metric_def
        
        # 初始化数据存储
        if metric_def.visualization_type == "line":
            self.time_series_data[metric_def.name] = TimeSeriesMetricData()
        else:  # "bar" or "pie"
            self.category_data[metric_def.name] = CategoryMetricData()
            
        logger.info(f"指标 {metric_def.name} 已注册")
    
    async def set_update_interval(self, metric_name: str, update_interval: int):
        """
        设置指标的更新频率
        
        Args:
            metric_name: 指标名称
            update_interval: 更新频率(秒)
        """
        async with self.lock:
            if metric_name not in self.metrics:
                logger.error(f"无法设置更新频率：指标 {metric_name} 未定义")
                return False
            
            # 更新指标定义中的更新频率
            self.metrics[metric_name].update_interval = update_interval
            
            # 如果指标正在监控中，更新调度频率
            if self.is_monitoring and hasattr(self, 'env') and self.env:
                await self.scheduler.update_interval(metric_name, self.env, self, update_interval)
                logger.info(f"指标 {metric_name} 更新频率已设置为 {update_interval}秒")
            return True
        
    async def unregister_metric(self, metric_name: str):
        """
        注销指标
        
        Args:
            metric_name: 指标名称
        """
        async with self.lock:
            # 先停止调度
            await self.scheduler.pause_metric(metric_name)
            
            # 移除指标定义和相关数据
            self.metrics.pop(metric_name, None)
            self.results.pop(metric_name, None)
            self.time_series_data.pop(metric_name, None)
            self.category_data.pop(metric_name, None)
            
            logger.info(f"指标 {metric_name} 已注销")
            
    def get_metric_definition(self, metric_name: str) -> Optional[MetricDefinition]:
        """
        获取指标定义
        
        Args:
            metric_name: 指标名称
            
        Returns:
            指标定义，如果不存在则返回None
        """
        return self.metrics.get(metric_name)
            
    async def start_all_metrics(self, env: Any = None):
        """
        启动所有指标的监控
        
        Args:
            env: 环境对象，如果不提供则使用已设置的环境
        """
        if env:
            self.env = env
            
        if not self.env:
            logger.error("无法启动监控：未设置环境对象")
            return
        
        # Fetch the global update interval from MonitorConfig
        global_update_interval = get_config().monitor_config.update_interval
        if global_update_interval is not None:
            logger.info(f"使用全局监控更新间隔: {global_update_interval}秒")
        
        async with self.lock:
            for metric_name, metric_def in self.metrics.items():
                # Determine the frequency for this specific metric
                frequency = global_update_interval if global_update_interval is not None else metric_def.update_interval
                
                await self.scheduler.schedule_metric(
                    metric_name, 
                    self.env, 
                    self, 
                    frequency # Use the determined frequency
                )
            self.is_monitoring = True
            logger.info(f"已启动 {len(self.metrics)} 个指标的监控")
                
    async def stop_all_metrics(self):
        """停止所有指标的监控"""
        await self.scheduler.stop_all()
        self.is_monitoring = False
        logger.info("已停止所有指标的监控")
            
    async def update_metric(self, metric_name: str, env: Any = None):
        """
        异步更新指定指标的值
        
        Args:
            metric_name: 指标名称
            env: 环境对象，如果不提供则使用已设置的环境
        """
        if not env and not self.env:
            logger.error(f"无法更新指标 {metric_name}：未提供环境对象")
            return
            
        env = env or self.env
        logger.info(f"更新指标 {metric_name}")
        async with self.lock:
            # 获取指标定义
            metric_def = self.metrics.get(metric_name)
            if not metric_def:
                logger.error(f"无法更新指标 {metric_name}：指标未定义")
                return
                
            # 收集数据
            data = await self.collector.collect_for_metric(env, metric_def)
            # 计算指标值
            raw_result = self.processor.calculate(metric_def, data)
            logger.info(raw_result)
            if raw_result is None:
                return
            
            # 根据可视化类型处理数据
            if metric_def.visualization_type == "line":
                # 规范化折线图数据
                normalized_result = self._normalize_line_data(raw_result)
                # 时间序列数据(折线图)
                ts_data = self.time_series_data[metric_name]
                ts_data.add_point(normalized_result)
                viz_data = ts_data.get_echarts_data()
            else:
                # 类别数据(柱状图或饼图)
                viz_data = self.processor.format_for_visualization(raw_result, metric_def)
                
                # 更新类别数据存储
                cat_data = self.category_data[metric_name]
                if metric_def.visualization_type == "bar":
                    cat_data.update_data(viz_data["xAxis"], viz_data["series"])
                elif metric_def.visualization_type == "pie" and "series" in viz_data:
                    categories = [item["name"] for item in viz_data["series"]]
                    values = [item["value"] for item in viz_data["series"]]
                    cat_data.update_data(categories, values)
            
            # 保存结果
            result = MetricResult(
                metric_name=metric_name,
                raw_data=raw_result,
                visualization_data=viz_data
            )
            self.results[metric_name] = result
            
            logger.debug(f"指标 {metric_name} 已更新")
            
    def get_result(self, metric_name: str) -> Optional[MetricResult]:
        """
        获取指标结果
        
        Args:
            metric_name: 指标名称
            
        Returns:
            指标结果，如果不存在则返回None
        """
        return self.results.get(metric_name)
        
    def get_all_results(self) -> Dict[str, MetricResult]:
        """
        获取所有指标结果
        
        Returns:
            指标名称到结果的映射
        """
        return self.results.copy()
        
    def get_results_by_type(self, visualization_type: str) -> Dict[str, MetricResult]:
        """
        获取特定可视化类型的所有指标结果
        
        Args:
            visualization_type: 可视化类型 ("bar", "pie", "line")
            
        Returns:
            指标名称到结果的映射
        """
        results = {}
        for name, metric_def in self.metrics.items():
            if metric_def.visualization_type == visualization_type:
                result = self.results.get(name)
                if result:
                    results[name] = result
        return results
        
    def get_time_series_data(self, metric_name: str, last_n: Optional[int] = None) -> Dict:
        """
        获取指标的时间序列数据
        
        Args:
            metric_name: 指标名称
            last_n: 如果指定，获取最近n个数据点
            
        Returns:
            格式化的时间序列数据
        """
        ts_data = self.time_series_data.get(metric_name)
        if not ts_data:
            return {"xAxis": [], "series": []}
            
        if last_n:
            return ts_data.get_last_n_points(last_n)
        return ts_data.get_echarts_data()
    
    def get_metric_data(self, metric_name: str, format: str = "echarts") -> Dict:
        """
        获取指标数据，支持多种输出格式
        
        Args:
            metric_name: 指标名称
            format: 输出格式，可选 "echarts" 或 "matplotlib"
        
        Returns:
            适合指定格式的数据结构
        """
        metric_def = self.metrics.get(metric_name)
        if not metric_def:
            return {}
            
        viz_type = metric_def.visualization_type
        
        if viz_type == "line":
            ts_data = self.time_series_data.get(metric_name)
            if not ts_data:
                return {"xAxis": [], "series": []}
                
            if format == "matplotlib":
                return ts_data.get_matplotlib_data()
            else:
                return ts_data.get_echarts_data()
        else:  # "bar" or "pie"
            cat_data = self.category_data.get(metric_name)
            if not cat_data:
                return {}
                
            return cat_data.get_data(format=format, viz_type=viz_type)
    
    def get_metrics_for_api(self) -> Dict[str, Any]:
        """返回适合API传输的所有指标数据, data字段为完整的ECharts Option"""
        metrics_data = {}
        for metric_name, metric_def in self.metrics.items():
            result = self.get_result(metric_name)
            if result and result.visualization_data is not None:
                viz_type = metric_def.visualization_type
                raw_viz_data = result.visualization_data # This holds the basic data, e.g. {"xAxis": [...], "series": [...]} or {"series": [...]} for pie
                echarts_option = {}

                try:
                    if viz_type == "line":
                        # Line charts use TimeSeriesMetricData which formats series data internally for type: time.
                        # raw_viz_data is expected to be {"series": [{name:..., type:'line', data:[[ts, val],...]}, ...]}}
                        series_list = raw_viz_data.get("series", []) # This contains the correctly formatted data
                        
                        # Use the dedicated time-series helper
                        echarts_option = create_time_series_chart_option(
                            title=metric_def.description or metric_name,
                            series_list=series_list
                        )
                        
                        # Removed old manual construction:
                        # x_axis = raw_viz_data.get("xAxis", []) 
                        # legend_data = [s.get('name', f'Series {i+1}') for i, s in enumerate(series_list)]
                        # echarts_option = {
                        #     "title": {"text": metric_def.description or metric_name, "left": "center"},
                        #     "tooltip": {"trigger": "axis"},
                        #     "legend": {"data": legend_data, "bottom": 10, "type": "scroll"}, 
                        #     "grid": {"left": '3%', "right": '4%', "bottom": '10%', "containLabel": True},
                        #     "xAxis": {"type": "category", "boundaryGap": False, "data": x_axis},
                        #     "yAxis": {"type": "value"},
                        #     "series": series_list
                        # }
                        
                    elif viz_type == "pie":
                        # Assuming raw_viz_data format: {"series": List[Dict[str, Any]]} with format {"name": ..., "value": ...}
                        series_data = raw_viz_data.get("series", [])
                        # Ensure it's in the correct format [{"name": ..., "value": ...}, ...]
                        if series_data and not isinstance(series_data[0], dict):
                             logger.warning(f"Pie chart data for {metric_name} has unexpected format. Attempting conversion.")
                             # Attempt conversion if it's just a list of values or simple list of lists/tuples
                             if isinstance(series_data[0], (int, float)): 
                                series_data = [{'name': f'Category {i}', 'value': v} for i, v in enumerate(series_data)]
                             else: # Give up if format is too weird
                                series_data = [] 
                                
                        echarts_option = create_pie_chart_option(
                            title=metric_def.description or metric_name,
                            series_data=series_data,
                            series_name=metric_name # Use metric name as series name for pie
                        )
                    elif viz_type == "bar":
                        # Assuming raw_viz_data format from CategoryMetricData.get_bar_chart_data: {"xAxis": List[str], "series": List[Any]}
                        x_axis = raw_viz_data.get("xAxis", [])
                        series_values = raw_viz_data.get("series", []) # Should be a list of values for a single series
                        
                        # The helper function create_bar_chart_option handles converting this to the ECharts series list format
                        echarts_option = create_bar_chart_option(
                             title=metric_def.description or metric_name,
                             x_axis_data=x_axis,
                             series_data=series_values, # Pass the list of values
                             series_name=metric_name # Default series name if only one
                        )
                        # Old fallback:
                        # logger.warning(f"Bar chart option generation not fully implemented for {metric_name}. Returning raw data structure.")
                        # echarts_option = self._format_for_api_display(raw_viz_data, viz_type)
                    else:
                        # For unknown types, return raw visualization data
                        echarts_option = raw_viz_data

                except Exception as e:
                    logger.error(f"Error generating ECharts option for {metric_name} ({viz_type}): {e}")
                    echarts_option = {"error": f"Failed to generate chart option: {e}"} 

                metrics_data[metric_name] = {
                    "name": metric_name,
                    "description": metric_def.description,
                    "visualization_type": viz_type,
                    "data": echarts_option, # Assign the generated ECharts option
                    "raw_data": result.raw_data, 
                    "timestamp": result.timestamp,
                    "formatted_time": result.formatted_time
                }
            elif metric_name in self.metrics: # Metric exists but no result yet
                 # Provide a default structure based on viz_type so frontend doesn't break
                 metric_def = self.metrics[metric_name]
                 viz_type = metric_def.visualization_type
                 default_option = {}
                 if viz_type == "line":
                     # Use the time series helper for the default option as well
                     default_option = create_time_series_chart_option(title=metric_def.description or metric_name, series_list=[])
                 elif viz_type == "pie":
                     default_option = create_pie_chart_option(title=metric_def.description or metric_name, series_data=[])
                 elif viz_type == "bar":
                     default_option = create_bar_chart_option(title=metric_def.description or metric_name, x_axis_data=[], series_data=[])
                 else:
                     default_option = {"message": "No data available yet."}
                     
                 metrics_data[metric_name] = {
                     "name": metric_name,
                     "description": metric_def.description,
                     "visualization_type": viz_type,
                     "data": default_option,
                     "raw_data": None,
                     "timestamp": int(time.time()),
                     "formatted_time": time.strftime('%Y-%m-%d %H:%M:%S')
                 }
                 
        return metrics_data
        
    def _format_for_api_display(self, viz_data: Dict, viz_type: str) -> Dict:
        """
        将可视化数据格式化为适合API传输的格式
        
        Args:
            viz_data: 可视化数据
            viz_type: 可视化类型
            
        Returns:
            格式化后的数据
        """
        if not viz_data:
            if viz_type == "line":
                return {"xAxis": [], "series": []}
            elif viz_type == "bar":
                return {"xAxis": [], "series": []}
            elif viz_type == "pie":
                return {"series": []}
            return {}
            
        # 对于折线图，确保series是数组格式
        if viz_type == "line" and "series" in viz_data:
            # 已经是数组格式，直接返回
            if isinstance(viz_data["series"], list):
                return viz_data
                
            # 将字典格式转换为数组格式
            if isinstance(viz_data["series"], dict):
                series_array = []
                for name, values in viz_data["series"].items():
                    series_array.append({
                        "name": name,
                        "type": "line",
                        "data": values
                    })
                return {
                    "xAxis": viz_data.get("xAxis", []),
                    "series": series_array
                }
        
        # 对于条形图，标准化格式
        elif viz_type == "bar" and "xAxis" in viz_data and "series" in viz_data:
            # 如果series是字典格式，转换为适合前端的格式
            if isinstance(viz_data["series"], dict):
                series_array = []
                for name, values in viz_data["series"].items():
                    series_array.append({
                        "name": name,
                        "type": "bar",
                        "data": values
                    })
                return {
                    "xAxis": viz_data["xAxis"],
                    "series": series_array
                }
            # 如果series是数组但不是对象数组，转换为对象数组
            elif isinstance(viz_data["series"], list) and (not viz_data["series"] or not isinstance(viz_data["series"][0], dict)):
                return {
                    "xAxis": viz_data["xAxis"],
                    "series": [{
                        "name": "Value",
                        "type": "bar",
                        "data": viz_data["series"]
                    }]
                }
                
        # 对于饼图，标准化格式
        elif viz_type == "pie" and "series" in viz_data:
            # 如果series已经是正确格式，直接返回
            if isinstance(viz_data["series"], list) and (not viz_data["series"] or isinstance(viz_data["series"][0], dict)):
                return viz_data
                
        # 默认返回原始数据
        return viz_data

    def plot_metrics(self, metrics_data: Dict[str, Any], save_dir: str, round_num: Optional[int] = None) -> None:
        """
        Plot metrics data and save as images.
        
        Args:
            metrics_data (Dict[str, Any]): Dictionary containing metrics data
            save_dir (str): Directory to save the plots
            round_num (Optional[int]): Current step number if applicable
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        #plt.style.use('seaborn')
        
        # Plot step duration
        if 'round_duration' in metrics_data:
            plt.figure(figsize=(10, 6))
            plt.plot(metrics_data['round_duration'], marker='o')
            plt.title('Step Duration Over Time')
            plt.xlabel('Step')
            plt.ylabel('Duration (seconds)')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'round_duration.png'))
            plt.close()
        
        if 'total_tokens' in metrics_data:
            fig = plt.figure(figsize=(10, 6))
            plt.plot(metrics_data['total_tokens'], marker='o')
            plt.title('Total Tokens Over Time')
            plt.xlabel('Step')
            plt.ylabel('Total Tokens')
            plt.savefig(os.path.join(save_dir, 'total_tokens.png'))
        
            plt.close(fig)
        
        # Save metrics data as JSON for reference
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_file = os.path.join(save_dir, f'metrics_{timestamp}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)

    def collect_metrics(self, env_data: Dict[str, Any], round_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Collect metrics from environment data.
        
        Args:
            env_data (Dict[str, Any]): Environment data dictionary
            round_num (Optional[int]): Current step number if applicable
            
        Returns:
            Dict[str, Any]: Dictionary containing collected metrics
        """
        metrics = {}
        
        # Extract completion rate
        if 'step_data' in env_data:
            step_data = env_data['step_data']

            metrics['round_duration'] = [
                step_data[r]['duration'] 
                for r in sorted(step_data.keys())
            ]

            
            metrics['total_tokens'] = [
                step_data[r]['token_usage']['total_tokens']
                for r in sorted(step_data.keys())
            ]
            metrics['total_prompt_tokens']=[
                step_data[r]['token_usage']['total_prompt_tokens']
                for r in sorted(step_data.keys())
            ]
            metrics['total_completion_tokens']=[
                step_data[r]['token_usage']['total_completion_tokens']
                for r in sorted(step_data.keys())
            ]
            metrics['request_count']=[
                step_data[r]['token_usage']['request_count']
                for r in sorted(step_data.keys())
            ]
        
        
            metrics['event_count'] = [
                step_data[r]['event_count']
                for r in sorted(step_data.keys())
            ]
        
        # # Extract agent participation
        # if 'agent_decisions' in env_data:
        #     metrics['agent_participation'] = env_data['agent_decisions']
        
        # # Extract decision distribution
        # if 'decision_distribution' in env_data:
        #     metrics['decision_distribution'] = env_data['decision_distribution']
        
        return metrics

    def plot_registered_metrics(self, save_dir: str, round_num: Optional[int] = None) -> None:
        """
        Plot registered (scene-specific) metrics data and save them as images.
        
        Args:
            save_dir (str): Directory to save the plots
            round_num (Optional[int]): Current step number if applicable
        """
        if not self.results:
            logger.warning("No registered metrics data available to plot")
            return
            
        # Create scene-specific metrics directory
        scene_metrics_dir = os.path.join(save_dir, 'scene_metrics')
        os.makedirs(scene_metrics_dir, exist_ok=True)
        
        # Plot each registered metric
        for metric_name, result in self.results.items():
            fig = None
            try:
                metric_def = self.metrics.get(metric_name)
                if not metric_def:
                    continue
                    
                viz_type = metric_def.visualization_type
                fig = plt.figure(figsize=(12, 7))
                
                # Create metric-specific directory
                metric_dir = os.path.join(scene_metrics_dir, metric_name)
                os.makedirs(metric_dir, exist_ok=True)
                
                # 使用统一的数据获取接口，指定matplotlib格式
                data = self.get_metric_data(metric_name, format="matplotlib")
                
                if viz_type == "line":
                    # 处理折线图
                    if data and "xAxis" in data and "series" in data:
                        # 获取X轴数据 (时间戳) 和 Y轴数据
                        timestamps = data["xAxis"]
                        num_points = len(timestamps)
                        
                        # 绘制 Series
                        if isinstance(data["series"], dict):
                            for series_name, series_values in data["series"].items():
                                plt.plot(range(num_points), series_values, marker='o', linestyle='-', markersize=4, label=series_name)
                            plt.legend(loc='best', frameon=True, fancybox=True, framealpha=0.7)
                        else:
                            plt.plot(range(num_points), data["series"], marker='o', linestyle='-', markersize=4)

                        # 优化 X 轴刻度显示
                        max_ticks = 10  # 最多显示 10 个刻度
                        if num_points > 1:
                            step = max(1, num_points // max_ticks)
                            tick_indices = range(0, num_points, step)
                            tick_labels = [timestamps[i].strftime('%H:%M:%S') if isinstance(timestamps[i], datetime) else timestamps[i] for i in tick_indices] # Format time
                            plt.xticks(tick_indices, tick_labels, rotation=30, ha='right') # 设置刻度位置和标签
                        elif num_points == 1:
                             tick_labels = [timestamps[0].strftime('%H:%M:%S') if isinstance(timestamps[0], datetime) else timestamps[0]]
                             plt.xticks([0], tick_labels)


                        plt.title(f'{metric_name} Over Time', fontsize=14, fontweight='bold')
                        # 使用 "Step" 或 "Index" 作为 X 轴标签更通用
                        plt.xlabel('Step', fontsize=12) 
                        plt.ylabel(metric_name, fontsize=12)
                        plt.grid(True, linestyle='--', alpha=0.6)
                        plt.tight_layout()
                            
                elif viz_type == "bar":
                    # 处理柱状图
                    if data and "xAxis" in data and "series" in data:
                        categories = data["xAxis"]
                        
                        if isinstance(data["series"], dict):
                            # 多series条形图
                            series_count = len(data["series"])
                            width = 0.8 / series_count  # 每个条的宽度
                            
                            for i, (series_name, values) in enumerate(data["series"].items()):
                                positions = [j + (i - series_count/2 + 0.5) * width for j in range(len(categories))]
                                plt.bar(positions, values, width, label=series_name)
                            
                            plt.xticks(range(len(categories)), categories, rotation=45)
                            plt.legend(loc='best')
                        else:
                            # 兼容旧格式单series
                            plt.bar(categories, data["series"])
                        
                        plt.title(f'{metric_name} Distribution', fontsize=14, fontweight='bold')
                        plt.xlabel('Category', fontsize=12)
                        plt.ylabel('Value', fontsize=12)
                        plt.tight_layout()
                        
                elif viz_type == "pie":
                    # 处理饼图
                    if data:
                        if "categories" in data and "values" in data:
                            # 处理单系列饼图
                            plt.pie(data["values"], labels=data["categories"], autopct='%1.1f%%', 
                                  shadow=True, startangle=90)
                        elif "series" in data and isinstance(data["series"], list):
                            # 处理新格式饼图数据
                            values = [item["value"] for item in data["series"]]
                            labels = [item["name"] for item in data["series"]]
                            plt.pie(values, labels=labels, autopct='%1.1f%%', 
                                  shadow=True, startangle=90)
                        
                        plt.title(f'{metric_name} Distribution', fontsize=14, fontweight='bold')
                        plt.axis('equal')
                
                # 保存图表，包含时间戳和回合数
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                round_info = f"_round{round_num}" if round_num is not None else ""
                filename = f"{metric_name}{round_info}_{timestamp}.png"
                plt.savefig(os.path.join(metric_dir, filename), dpi=300, bbox_inches='tight')
                
                # 保存原始数据为JSON
                data_filename = f"{metric_name}{round_info}_{timestamp}.json"
                with open(os.path.join(metric_dir, data_filename), 'w') as f:
                    # Save the data used for plotting instead of the raw data
                    json.dump(data, f, indent=4)
                    
            except Exception as e:
                logger.error(f"Error plotting metric {metric_name}: {e}")
            finally:
                # Always close the figure, even if an exception occurs
                if fig is not None:
                    plt.close(fig)
                
    def export_metrics_as_images(self, save_dir: str, round_num: Optional[int] = None) -> None:
        """
        将所有指标保存为本地图片文件（综合接口）
        
        Args:
            save_dir (str): 图片保存目录
            round_num (Optional[int]): 当前回合数（可选）
        """
        try:
            # 1. 创建保存目录
            os.makedirs(save_dir, exist_ok=True)
            
            # 2. 导出常规指标图表（使用现有方法）
            # 收集常规指标数据
            general_metrics = self.collect_metrics(self.env.data if hasattr(self, 'env') and self.env else {}, round_num)
            general_dir = os.path.join(save_dir, 'general')
            self.plot_metrics(general_metrics, general_dir, round_num)
            logger.info(f"Saved general metrics plots to {general_dir}")
            
            # 3. 导出注册的特定指标图表
            self.plot_registered_metrics(save_dir, round_num)
            logger.info(f"Saved registered metrics plots to {save_dir}")
        finally:
            # Make sure to close any remaining figures
            plt.close('all')

    def _normalize_line_data(self, raw_result: Any) -> Any:
        """
        规范化折线图数据格式，确保存储的是一致格式
        
        Args:
            raw_result: 原始数据
            
        Returns:
            规范化后的数据
        """
        # 如果已经是字典形式，适合多series，直接返回
        if isinstance(raw_result, dict):
            # 检查字典值是否全为数值类型
            for key, value in raw_result.items():
                if not isinstance(value, (int, float, bool)) and value is not None:
                    # 如果包含非数值，可能是嵌套结构，尝试扁平化
                    try:
                        return self._flatten_nested_dict(raw_result)
                    except:
                        # 无法扁平化，返回原字典
                        pass
            return raw_result
            
        # 如果是简单类型，转换为默认series
        if isinstance(raw_result, (int, float, bool, str)) or raw_result is None:
            return {"default": raw_result}
            
        # 如果是列表或其他复杂类型，尝试转换为字典
        try:
            if isinstance(raw_result, (list, tuple)) and len(raw_result) > 0:
                # 如果是键值对列表，转换为字典
                if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in raw_result):
                    return {str(item[0]): item[1] for item in raw_result}
                    
                # 如果是单值列表，使用索引作为键
                return {f"series_{i}": val for i, val in enumerate(raw_result) if val is not None}
        except Exception:
            pass
            
        # 如果无法规范化，转换为默认series
        return {"default": raw_result}
        
    def _flatten_nested_dict(self, nested_dict: Dict, prefix: str = "") -> Dict:
        """
        将嵌套字典扁平化为一级字典
        
        Args:
            nested_dict: 嵌套字典
            prefix: 键前缀
            
        Returns:
            扁平化后的字典
        """
        result = {}
        for key, value in nested_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                # 递归处理嵌套字典
                result.update(self._flatten_nested_dict(value, new_key))
            elif isinstance(value, (int, float, bool)) or value is None:
                # 只保留数值类型
                result[new_key] = value
        return result