from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

@dataclass
class VariableSpec:
    """描述指标计算所需的变量来源"""
    
    name: str  # 变量名称
    source_type: str  # 来源类型: "env" 或 "agent"
    path: str  # 变量在data/profile中的路径(支持点表示法,如"economy.gdp")
    required: bool = True  # 是否必需
    agent_type: Optional[str] = None  # 若source_type为"agent"，指定agent类型

@dataclass
class MetricDefinition:
    """指标定义类，描述一个监控指标的所有信息"""
    
    name: str  # 指标唯一名称
    description: str  # 指标描述
    visualization_type: str  # 可视化类型: "bar", "pie", "line"
    variables: List[VariableSpec]  # 所需变量列表
    calculation_function: str  # 计算函数名
    update_interval: int = 60  # 更新频率(秒)
    visualization_config: Dict = field(default_factory=dict)  # ECharts可视化配置
    
    def __post_init__(self):
        # 验证可视化类型
        valid_types = ["bar", "pie", "line"]
        if self.visualization_type not in valid_types:
            raise ValueError(f"可视化类型必须是 {', '.join(valid_types)} 之一")

@dataclass
class MetricResult:
    """指标计算结果结构"""
    
    metric_name: str  # 对应指标名称
    raw_data: Any  # 原始计算结果
    visualization_data: Dict  # 适配ECharts的数据格式
    timestamp: float = field(default_factory=time.time)  # 计算时间戳
    metadata: Dict = field(default_factory=dict)  # 附加元数据
    
    @property
    def formatted_time(self) -> str:
        """返回格式化的时间字符串"""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')


class TimeSeriesMetricData:
    """处理时间序列指标数据，用于折线图"""
    
    def __init__(self, max_points: int = 1000):
        """
        初始化时间序列数据存储
        
        Args:
            max_points: 最大保存的历史数据点数量
        """
        self.timestamps: List[float] = []
        self.series_data: Dict[str, List[Any]] = {}  # 存储每个series的数据
        self.max_points = max_points
    
    def add_point(self, value: Any, timestamp: Optional[float] = None):
        """
        添加一个数据点
        
        Args:
            value: 数据值，可以是单个值或字典（多series）
            timestamp: 时间戳，默认为当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        
        # 处理多series数据
        if isinstance(value, dict):
            for series_name, series_value in value.items():
                if series_name not in self.series_data:
                    self.series_data[series_name] = []
                    
                # 填充历史缺失值
                while len(self.series_data[series_name]) < len(self.timestamps) - 1:
                    self.series_data[series_name].append(None)
                    
                self.series_data[series_name].append(series_value)
        else:
            # 单个值的情况，使用"default"作为默认series名
            if "default" not in self.series_data:
                self.series_data["default"] = []
                
            # 填充历史缺失值
            while len(self.series_data["default"]) < len(self.timestamps) - 1:
                self.series_data["default"].append(None)
                
            self.series_data["default"].append(value)
        
        # 超出最大点数时，移除最早的点
        if len(self.timestamps) > self.max_points:
            self.timestamps.pop(0)
            for series in self.series_data.values():
                if series:
                    series.pop(0)
    
    def has_multiple_series(self) -> bool:
        """检查是否包含多个series"""
        return len(self.series_data) > 1 or (len(self.series_data) == 1 and "default" not in self.series_data)
    
    def get_series_names(self) -> List[str]:
        """获取所有series名称"""
        return list(self.series_data.keys())
    
    def get_echarts_data(self) -> Dict:
        """获取适合ECharts时间序列折线图的数据格式 (type: 'time')"""
        # formatted_times = [datetime.fromtimestamp(ts).strftime('%H:%M:%S') 
        #                    for ts in self.timestamps] # No longer needed for type: time
        
        # 如果没有数据，返回空结构
        if not self.timestamps:
            # Return structure expected by the new helper (just series list)
            return {"series": []} 
        
        series = []
        for series_name, values in self.series_data.items():
            # 确保values长度与timestamps匹配
            adjusted_values = values.copy()
            while len(adjusted_values) < len(self.timestamps):
                adjusted_values.append(None)
            
            # Format data as [[timestamp_ms, value], ...]
            # Multiply timestamp by 1000 for milliseconds
            time_value_pairs = []
            for i, ts in enumerate(self.timestamps):
                 val = adjusted_values[i]
                 # Echarts time axis can handle nulls for gaps
                 time_value_pairs.append([datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), val])
                
            series.append({
                "name": series_name,
                "type": "line", # Type might be adjusted in the helper
                "data": time_value_pairs # Use the time-value pair format
            })
        
        # Return only the series list, as xAxis is implicit in the data for type: time
        return {
            # "xAxis": formatted_times, # Removed
            "series": series
        }
    
    def get_matplotlib_data(self) -> Dict:
        """获取适合matplotlib折线图的数据格式"""
        formatted_times = [datetime.fromtimestamp(ts).strftime('%H:%M:%S') 
                           for ts in self.timestamps]
        
        # 如果没有数据，返回空结构
        if not self.timestamps:
            return {"xAxis": [], "series": {}}
        
        # 确保所有series长度一致
        series_data = {}
        for series_name, values in self.series_data.items():
            adjusted_values = values.copy()
            while len(adjusted_values) < len(self.timestamps):
                adjusted_values.append(None)
            series_data[series_name] = adjusted_values
        
        return {
            "xAxis": formatted_times,
            "series": series_data
        }
    
    def get_last_n_points(self, n: int, format: str = "echarts") -> Dict:
        """
        获取最近n个数据点
        
        Args:
            n: 获取的点数量
            format: 数据格式，"echarts"或"matplotlib"
            
        Returns:
            格式化的数据
        """
        if n >= len(self.timestamps):
            return self.get_echarts_data() if format == "echarts" else self.get_matplotlib_data()
        
        sliced_times = self.timestamps[-n:]
        sliced_series = {}
        for series_name, values in self.series_data.items():
            # 处理series长度小于n的情况
            if len(values) < n:
                # 填充None值
                padding = [None] * (n - len(values))
                sliced_series[series_name] = padding + values
            else:
                sliced_series[series_name] = values[-n:]
        
        formatted_times = [datetime.fromtimestamp(ts).strftime('%H:%M:%S') 
                           for ts in sliced_times]
        
        if format == "echarts":
            series = []
            for series_name, values in sliced_series.items():
                series.append({
                    "name": series_name,
                    "type": "line",
                    "data": values
                })
            return {
                "xAxis": formatted_times,
                "series": series
            }
        else:
            return {
                "xAxis": formatted_times,
                "series": sliced_series
            }
            
    def clear(self):
        """清空所有数据"""
        self.timestamps = []
        self.series_data = {}
    
    def merge(self, other: 'TimeSeriesMetricData'):
        """
        合并另一个时间序列数据
        
        Args:
            other: 另一个TimeSeriesMetricData实例
        """
        if not other.timestamps:
            return
            
        # 合并时间戳和数据
        for ts_idx, ts in enumerate(other.timestamps):
            if ts not in self.timestamps:
                self.timestamps.append(ts)
                
                # 为现有series添加None值
                for series in self.series_data.values():
                    series.append(None)
            
            ts_position = self.timestamps.index(ts)
            
            # 更新各个series数据
            for series_name, values in other.series_data.items():
                if series_name not in self.series_data:
                    self.series_data[series_name] = [None] * len(self.timestamps)
                
                if ts_idx < len(values):
                    self.series_data[series_name][ts_position] = values[ts_idx]
        
        # 排序时间戳和对应的数据
        sorted_data = sorted(zip(self.timestamps, range(len(self.timestamps))))
        self.timestamps = [item[0] for item in sorted_data]
        sort_indices = [item[1] for item in sorted_data]
        
        for series_name in self.series_data:
            self.series_data[series_name] = [self.series_data[series_name][i] for i in sort_indices]
            
        # 裁剪超出max_points的部分
        while len(self.timestamps) > self.max_points:
            self.timestamps.pop(0)
            for series in self.series_data.values():
                series.pop(0)


class CategoryMetricData:
    """处理类别型指标数据，用于柱状图和饼图"""
    
    def __init__(self):
        """初始化类别数据存储"""
        self.categories: List[str] = []
        self.values: List[float] = []
        self.timestamp: float = time.time()
    
    def update_data(self, categories: List[str], values: List[float], timestamp: Optional[float] = None):
        """
        更新类别数据
        
        Args:
            categories: 类别列表
            values: 对应的值列表
            timestamp: 时间戳，默认为当前时间
        """
        if len(categories) != len(values):
            raise ValueError("类别列表和值列表长度必须相同")
        
        self.categories = categories
        self.values = values
        self.timestamp = timestamp or time.time()
    
    def get_data(self, format: str = "echarts", viz_type: str = "bar") -> Dict:
        """
        获取数据，支持多种格式和可视化类型
        
        Args:
            format: 数据格式，"echarts"或"matplotlib"
            viz_type: 可视化类型，"bar"或"pie"
            
        Returns:
            格式化的数据
        """
        if format == "matplotlib":
            return self.get_matplotlib_data(viz_type)
        else:
            if viz_type == "pie":
                return self.get_pie_chart_data()
            else:
                return self.get_bar_chart_data()
    
    def get_bar_chart_data(self) -> Dict:
        """获取适合ECharts柱状图的数据格式"""
        return {
            "xAxis": self.categories,
            "series": self.values
        }
    
    def get_pie_chart_data(self) -> Dict:
        """获取适合ECharts饼图的数据格式"""
        series_data = [{"name": cat, "value": val} 
                      for cat, val in zip(self.categories, self.values)]
        
        return {
            "series": series_data
        }
        
    def get_matplotlib_data(self, viz_type: str = "bar") -> Dict:
        """
        获取适合matplotlib的数据格式
        
        Args:
            viz_type: 可视化类型，"bar"或"pie"
            
        Returns:
            适合matplotlib的数据格式
        """
        if viz_type == "pie":
            return {
                "categories": self.categories,
                "values": self.values
            }
        else:  # bar
            return {
                "xAxis": self.categories,
                "series": self.values
            } 