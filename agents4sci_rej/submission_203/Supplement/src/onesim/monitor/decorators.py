from typing import List, Dict, Any, Callable, Optional
from functools import wraps
from .metric import MetricDefinition, VariableSpec

def metric(
    name: str,
    description: str,
    variables: List[Dict],
    visualization_type: str = "line",
    update_interval: int = 60,
    visualization_config: Optional[Dict] = None
) -> Callable:
    """
    指标定义装饰器，简化用户创建指标的过程
    
    Args:
        name: 指标名称
        description: 指标描述
        variables: 简化的变量声明列表，每项格式为:
                  {
                    "name": "变量名",
                    "source_type": "env|agent",
                    "path": "数据路径",
                    "agent_type": "AgentType" (仅当source_type为agent时必需),
                    "required": True|False (可选，默认True)
                  }
        visualization_type: 可视化类型 ("bar", "pie", "line")
        update_interval: 更新频率(秒)
        visualization_config: 可视化配置 (可选)

    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        # 转换变量规范
        variable_specs = []
        for var in variables:
            variable_specs.append(VariableSpec(
                name=var["name"],
                source_type=var["source_type"],
                path=var["path"],
                agent_type=var.get("agent_type"),
                required=var.get("required", True)
            ))
            
        # 创建指标定义
        metric_def = MetricDefinition(
            name=name,
            description=description,
            visualization_type=visualization_type,
            variables=variable_specs,
            calculation_function=func,
            update_interval=update_interval,
            visualization_config=visualization_config or {}
        )
        
        # 将指标定义附加到函数
        func.metric_definition = metric_def
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
            
        wrapper.metric_definition = metric_def
        return wrapper
        
    return decorator


def register_custom_metric(
    name: str,
    description: str,
    variables: List[VariableSpec],
    calculation_function: Callable,
    visualization_type: str = "line",
    update_interval: int = 60,
    visualization_config: Optional[Dict] = None
) -> MetricDefinition:
    """
    注册自定义指标
    
    Args:
        name: 指标名称
        description: 指标描述
        variables: 变量规范列表
        calculation_function: 计算函数
        visualization_type: 可视化类型 ("bar", "pie", "line")
        update_interval: 更新频率(秒)
        visualization_config: 可视化配置 (可选)
        
    Returns:
        创建的指标定义
    """
    return MetricDefinition(
        name=name,
        description=description,
        visualization_type=visualization_type,
        variables=variables,
        calculation_function=calculation_function,
        update_interval=update_interval,
        visualization_config=visualization_config or {}
    ) 