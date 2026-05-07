from typing import Dict, List, Any, Optional, Union, Callable
import re
import os
from loguru import logger
import textwrap
import json
import random
from collections import defaultdict

from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from onesim.models.parsers import CodeBlockParser
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)
from .base import AgentBase

class MetricAgent(AgentBase):
    """基于场景生成监控指标的Agent，优化版本具有更强的数据验证和错误处理能力"""
    
    def __init__(
        self,
        model_config_name: str,
        sys_prompt: str = '',
    ):
        """
        初始化指标生成Agent
        
        Args:
            model_config_name: 模型配置名称
            sys_prompt: 系统提示，默认为空
        """
        super().__init__(
            sys_prompt=sys_prompt or (
                "你是一个专门负责分析多智能体系统性能指标的AI助手。"
                "你的任务是基于系统描述和数据模型生成有意义的监控指标，并为每个指标生成计算函数。"
                "生成的函数必须具有强大的错误处理能力，能够处理None值、空列表和类型错误等各种异常情况。"
            ),
            model_config_name=model_config_name,
        )
        self.parser = JsonBlockParser()
        self.code_parser = CodeBlockParser(language="python")
        self.visualization_types = ["line", "bar", "pie"]
        self.source_types = ["env", "agent"]

    def generate_metrics(self, scenario_description: str, agent_types: List[str], system_data_model: Dict = None, num_metrics: int = 3) -> List[Dict]:
        """
        分析场景，生成适用的指标列表
        
        Args:
            scenario_description: 场景描述
            agent_types: Agent类型列表
            system_data_model: 系统数据模型，包含环境变量和各类Agent变量
            num_metrics: 生成指标的数量
            
        Returns:
            指标定义列表
        """
        if not scenario_description:
            logger.error("场景描述不能为空")
            return []
            
        if not agent_types:
            logger.error("代理类型列表不能为空")
            return []
            
        prompt = self._create_generation_prompt(scenario_description, agent_types, system_data_model, num_metrics)
        
        # 使用模型获取响应
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt + self.parser.format_instruction, role="user")
        )
        
        response = self.model(prompt_message)
        
        # 解析响应，提取指标定义
        try:
            result = self.parser.parse(response)
            metrics = result.parsed.get("metrics", [])
            logger.info(f"为场景生成了 {len(metrics)} 个指标")
            
            # 验证指标
            if system_data_model:
                metrics = self.validate_metrics(metrics, system_data_model)
                
            return metrics
        except Exception as e:
            logger.error(f"解析指标生成响应时出错: {str(e)}")
            # 尝试使用正则表达式提取JSON块
            try:
                import re
                json_pattern = r'```json\s*([\s\S]*?)\s*```'
                matches = re.findall(json_pattern, response)
                if matches:
                    import json
                    metrics_data = json.loads(matches[0])
                    metrics = metrics_data.get("metrics", [])
                    logger.info(f"通过备用方法提取到 {len(metrics)} 个指标")
                    
                    # 验证指标
                    if system_data_model:
                        metrics = self.validate_metrics(metrics, system_data_model)
                        
                    return metrics
            except Exception as backup_error:
                logger.error(f"备用提取方法也失败: {str(backup_error)}")
            
            return []
    
    def validate_metrics(self, metrics: List[Dict], system_data_model: Dict = None) -> List[Dict]:
        """
        验证指标定义的有效性，确保所有引用的变量都存在
        
        Args:
            metrics: 指标定义列表
            system_data_model: 系统数据模型
            
        Returns:
            验证后的指标列表，移除无效指标
        """
        if system_data_model is None:
            logger.warning("无法验证指标，system_data_model未提供")
            return metrics
        
        available_variables = set()
        
        # 收集所有可用变量
        if "environment" in system_data_model and "variables" in system_data_model["environment"]:
            for var in system_data_model["environment"]["variables"]:
                available_variables.add(var["name"])
        
        if "agents" in system_data_model:
            for agent_type, agent_data in system_data_model["agents"].items():
                if "variables" in agent_data:
                    for var in agent_data["variables"]:
                        available_variables.add(var["name"])
        
        valid_metrics = []
        for metric in metrics:
            is_valid = True
            invalid_vars = []
            
            # 检查可视化类型是否有效
            if "visualization_type" not in metric or metric["visualization_type"] not in self.visualization_types:
                logger.warning(f"指标 '{metric.get('name', 'unknown')}' 使用了无效的可视化类型: {metric.get('visualization_type', 'missing')}，已设为默认值'line'")
                metric["visualization_type"] = "line"
            
            # 检查指标中使用的变量是否存在
            for var in metric.get("variables", []):
                if var.get("name") not in available_variables:
                    is_valid = False
                    invalid_vars.append(var.get("name", "unknown"))
                
                # 检查source_type是否有效
                if "source_type" not in var or var["source_type"] not in self.source_types:
                    logger.warning(f"指标 '{metric.get('name', 'unknown')}' 的变量 '{var.get('name', 'unknown')}' 使用了无效的source_type: {var.get('source_type', 'missing')}，已设为默认值'env'")
                    var["source_type"] = "env"
            
            if not is_valid:
                logger.warning(f"指标 '{metric.get('name', 'unknown')}' 使用了不存在的变量: {', '.join(invalid_vars)}，将跳过")
            else:
                valid_metrics.append(metric)
        
        logger.info(f"验证完成: {len(valid_metrics)}/{len(metrics)} 个指标有效")
        return valid_metrics
    
    def _create_generation_prompt(self, scenario_description: str, agent_types: List[str], system_data_model: Dict = None, num_metrics: int = 3) -> str:
        """
        创建用于生成指标的提示
        
        Args:
            scenario_description: 场景描述
            agent_types: Agent类型列表
            system_data_model: 系统数据模型，包含环境变量和各类Agent变量
            num_metrics: 生成指标的数量
            
        Returns:
            提示字符串
        """
        system_data_model_str = ""
        available_variables = []
        
        if system_data_model:
            # 格式化系统数据模型为字符串
            system_data_model_str = json.dumps(system_data_model, indent=2)
            
            # 提取环境变量
            if "environment" in system_data_model and "variables" in system_data_model["environment"]:
                for var in system_data_model["environment"]["variables"]:
                    available_variables.append({
                        "name": var["name"],
                        "type": var["type"],
                        "source_type": "env",
                        "path": var["name"]  # Path can be simple or nested e.g., "stats.value"
                    })
            
            # 提取智能体变量
            if "agents" in system_data_model:
                for agent_type, agent_data in system_data_model["agents"].items():
                    if "variables" in agent_data:
                        for var in agent_data["variables"]:
                            available_variables.append({
                                "name": var["name"],
                                "type": var["type"],
                                "source_type": "agent",
                                "agent_type": agent_type,
                                "path": var["name"],  # Path can be simple or nested e.g., "group.name"
                                "is_list": True  # 标记为列表类型
                            })
        
        # 格式化可用变量列表
        available_variables_str = json.dumps(available_variables, indent=4)

        return f"""
Metric Generation Task

Scenario Description:
```
{scenario_description}
```

Agent Types:
```
{", ".join(agent_types)}
```

System Data Model:
```json
{system_data_model_str}
```

Available Variables:
```json
{available_variables_str}
```

Task: Generate monitoring metrics that would be valuable for analyzing this multi-agent system.

Requirements:
1. Consider key performance indicators that would provide insights into:
   - System-level outcomes
   - Agent-specific behaviors
   - Resource utilization
   - Interaction patterns
   - Emergent phenomena

2. For each metric, specify:
   - Descriptive name
   - Clear explanation of what it measures
   - Variables needed from environment or agents
   - Calculation logic
   - Appropriate visualization type

3. Use only available data sources from the system data model:
   - Environment variables (via "env" source_type) - these are single values accessed directly from data dictionary
   - Agent variables (via "agent" source_type) - these are lists of values, one for each agent of that type

4. Support these visualization types:
   - "line": For time-series data plotting changes over time (can have multiple lines)
   - "bar": For comparing values across categories
   - "pie": For showing proportions of a whole

5. IMPORTANT: Remember that agent data comes as lists of values (one per agent of that type), not as single values.
   For agent variables, you will need to include appropriate aggregation methods (sum, average, max, min, etc.).

6. CRITICAL: Be aware that data might have None values, empty lists, or unexpected types. Your calculation logic
   must describe how to handle these edge cases safely (using default values, skipping calculations, etc.)

7. MULTI-SERIES SUPPORT: For appropriate metrics, consider returning multiple series of data:
   - For "line" charts: Return a dictionary where each key is a series name and value is the data point
   - For "bar" charts: Return a dictionary where each key is a category name and value is the bar height
   - For "pie" charts: Return a dictionary where each key is a slice name and value is the proportion

Output Format:
```json
{{
  "metrics": [
    {{
      "name": "metric_name",
      "description": "What this metric measures",
      "visualization_type": "line|bar|pie",
      "update_interval": 5,
      "variables": [
        {{
          "name": "variable_name",
          "source_type": "env|agent",
          "agent_type": "AgentType",  // Only when source_type is "agent"
          "path": "variable_name_or_nested.path",    // Variable name or dot-separated path
          "required": true,
          "is_list": true  // Set to true for agent variables which are lists
        }}
      ],
      "calculation_logic": "Explanation of how this metric is calculated from the variables, including how list data is aggregated, how edge cases are handled, and how multiple series are formed (if applicable)"
    }}
  ]
}}
```

Generate {num_metrics} metrics that would be most valuable for monitoring this scenario, making sure to use ONLY the variables defined in the system data model.
"""
    
    def generate_calculation_function(self, metric_def: Dict, system_data_model: Dict = None) -> str:
        """
        为指标生成计算函数代码
        
        Args:
            metric_def: 指标定义字典
            system_data_model: 系统数据模型，包含环境变量和各类Agent变量
            
        Returns:
            函数代码字符串
        """
        # 验证输入
        if not metric_def or "name" not in metric_def:
            logger.error("指标定义无效，无法生成计算函数")
            return "def invalid_metric(data: Dict[str, Any]) -> Any:\n    return 0"
        
        function_name = re.sub(r'[^\w\-_]', '_', metric_def["name"])
        prompt = self._create_function_prompt(metric_def, system_data_model, function_name)
        
        # 使用模型获取响应
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user")
        )
        response = self.model(prompt_message)
        
        # 提取响应中的Python代码
        try:
            # 先尝试使用code_parser解析
            code_block = self.code_parser.parse(response)
            function_code = code_block.parsed
            
            # 确保代码中包含正确的函数名
            expected_def = f"def {function_name}"
            if expected_def not in function_code:
                # 如果函数名不匹配，尝试替换
                function_code = re.sub(r'def\s+([a-zA-Z0-9_]+)', f'def {function_name}', function_code)
                
                # 如果还是没有正确的函数定义，使用我们自己的模板
                if expected_def not in function_code:
                    logger.warning(f"无法找到正确的函数定义: {expected_def}，将创建一个基本函数")
                    return self._create_default_function_code(function_name, metric_def)
            
            # 确保函数有文档字符串
            if '"""' not in function_code:
                # 在函数定义后添加文档字符串
                docstring = self._generate_function_docstring(metric_def)
                function_code = re.sub(
                    f'def {function_name}\\([^)]*\\)[^:]*:',
                    f'def {function_name}(data: Dict[str, Any]) -> Any:\n    """{docstring}"""',
                    function_code
                )
            
            # 检查是否使用了安全工具函数
            safe_functions = ["safe_get", "safe_list", "safe_avg", "safe_sum", "safe_max", "safe_min", "safe_count"]
            has_safe_functions = any(func in function_code for func in safe_functions)
            
            # 如果没有使用安全函数，可能需要添加额外的错误处理
            if not has_safe_functions and "try:" not in function_code:
                logger.warning(f"生成的函数 {function_name} 缺少适当的错误处理")
                # 考虑在这里增强错误处理，但这需要解析函数结构，较为复杂
            
            return function_code
                
        except Exception as e:
            logger.error(f"解析计算函数代码时出错: {str(e)}")
            return self._create_default_function_code(function_name, metric_def)
    
    def _generate_function_docstring(self, metric_def: Dict) -> str:
        """
        为指标函数生成文档字符串
        
        Args:
            metric_def: 指标定义字典
            
        Returns:
            文档字符串
        """
        return f"""
    计算指标: {metric_def.get('name', 'unknown')}
    描述: {metric_def.get('description', '无描述')}
    可视化类型: {metric_def.get('visualization_type', 'line')}
    更新频率: {metric_def.get('update_interval', 5)} 秒
    
    Args:
        data: 包含所有变量的数据字典，注意agent变量是列表形式
        
    Returns:
        根据可视化类型返回不同格式的结果:
        - line: 返回单个数值
        - bar/pie: 返回字典，键为分类，值为对应数值
        
    注意:
        此函数处理各种异常情况，包括None值、空列表和类型错误等
    """
    
    def _create_default_function_code(self, function_name: str, metric_def: Dict) -> str:
        """
        创建默认的函数代码
        
        Args:
            function_name: 函数名
            metric_def: 指标定义字典
            
        Returns:
            函数代码字符串
        """
        docstring = self._generate_function_docstring(metric_def)
        
        # 检查必需变量
        required_vars = [v["name"] for v in metric_def.get("variables", []) if v.get("required", True)]
        required_vars_check = "\n        ".join([
            f"# 检查必需变量 {var} 是否存在",
            f"if '{var}' not in data:",
            f"    log_metric_error('{metric_def.get('name', 'unknown')}', ValueError(f'缺少必需变量: {var}'))",
            f"    return 0 if '{metric_def.get('visualization_type', 'line')}' == 'line' else {{}}"
        ] for var in required_vars)
        
        # 变量分类
        env_vars = [v for v in metric_def.get("variables", []) if v.get("source_type") == "env"]
        agent_vars = [v for v in metric_def.get("variables", []) if v.get("source_type") == "agent"]
        
        # 生成变量访问代码
        env_vars_code = "\n        ".join([
            f"# 访问环境变量 {v['name']}",
            f"{v['name']}_value = safe_get(data, '{v['name']}')"
        ] for v in env_vars)
        
        agent_vars_code = "\n        ".join([
            f"# 安全处理代理变量 {v['name']} (列表形式)",
            f"{v['name']}_data = safe_list(safe_get(data, '{v['name']}'))",
            f"# 对列表数据进行安全聚合",
            f"{v['name']}_avg = safe_avg({v['name']}_data)",
            f"{v['name']}_sum = safe_sum({v['name']}_data)",
            f"{v['name']}_max = safe_max({v['name']}_data)",
            f"{v['name']}_min = safe_min({v['name']}_data)",
            f"{v['name']}_count = safe_count({v['name']}_data)"
        ] for v in agent_vars)
        
        # 根据可视化类型提供适当的返回值
        vis_type = metric_def.get("visualization_type", "line")
        if vis_type == "line":
            result_code = "result = 0  # 默认值，根据实际逻辑修改"
        elif vis_type in ["bar", "pie"]:
            result_code = "result = {'类别1': 0, '类别2': 0}  # 默认示例，根据实际逻辑修改"
        else:
            result_code = "result = 0  # 默认值，根据实际逻辑修改"
        
        return f"""def {function_name}(data: Dict[str, Any]) -> Any:
    \"\"\"{docstring}\"\"\"
    try:
        # 检查输入数据有效性
        if not data or not isinstance(data, dict):
            log_metric_error('{metric_def.get('name', 'unknown')}', ValueError('无效的数据输入'), {{'data': data}})
            return 0 if '{vis_type}' == 'line' else {{}} 
        
        {required_vars_check}
        
        # 处理环境变量 (单值)
        {env_vars_code}
        
        # 处理代理变量 (列表形式)
        {agent_vars_code}
        
        # 计算指标结果
        # TODO: 实现具体的计算逻辑，基于指标定义中的 calculation_logic
        {result_code}
        
        return result
    except Exception as e:
        # 使用日志工具记录错误，而不是简单地打印
        log_metric_error('{metric_def.get('name', 'unknown')}', e, {{'data_keys': list(data.keys()) if isinstance(data, dict) else None}})
        # 根据可视化类型返回适当的默认值
        return 0 if '{vis_type}' == 'line' else {{}}
"""
    
    def _create_function_prompt(self, metric_def: Dict, system_data_model: Dict = None, function_name: str = None) -> str:
        """
        创建用于生成计算函数的提示
        
        Args:
            metric_def: 指标定义字典
            system_data_model: 系统数据模型，包含环境变量和各类Agent变量
            function_name: 函数名称，默认根据指标名生成
            
        Returns:
            提示字符串
        """
        if function_name is None:
            function_name = re.sub(r'[^\w\-_]', '_', metric_def["name"])
            
        variables_str = "\n".join([
            f"- {v.get('name', 'unknown')}: from {'environment' if v.get('source_type') == 'env' else v.get('agent_type', 'unknown')}" +
            (f" (optional)" if not v.get('required', True) else "") +
            (f" (LIST OF VALUES)" if v.get('source_type') == 'agent' or v.get('is_list', False) else "") +
            (f" (Path: {v.get('path', v.get('name', 'unknown'))})" if v.get('path') != v.get('name') else "") # Show path if different from name
            for v in metric_def.get("variables", [])
        ])
        
        system_data_model_str = ""
        if system_data_model:
            system_data_model_str = json.dumps(system_data_model, indent=2)
        
        return f"""
Metric Calculation Function Generation Task

Metric Definition:
- Name: {metric_def.get('name', 'unknown')}
- Description: {metric_def.get('description', '无描述')}
- Visualization Type: {metric_def.get('visualization_type', 'line')}

Available Variables:
{variables_str}

System Data Model:
```json
{system_data_model_str}
```

Calculation Logic:
{metric_def.get('calculation_logic', '未提供计算逻辑')}

Task: Write a Python function named "{function_name}" to calculate this metric.

Requirements:
1. Function name MUST be "{function_name}" (not "calculate")
2. Takes a single parameter: data (dict containing all variables collected by the monitor)
3. Return appropriate data structure for the visualization type:
   - For "line": Return a dict with series names as keys and values as data points
   - For "bar": Return a dict where keys are categories and values are measurements
   - For "pie": Return a dict where keys are categories and values are proportions

4. DATA VALIDATION AND ERROR HANDLING IS CRITICAL:
   - The 'data' dictionary contains the collected values using the 'name' specified in the metric definition's 'variables' list.
   - Check if required variables exist in the data dictionary using their 'name'.
   - Handle None values, empty lists, and invalid data types for the *values* within the 'data' dict.
   - Handle division by zero scenarios.
   - Use the utility functions imported from onesim.monitor.utils module.
   - Log errors with context using log_metric_error function.

5. Available Utility Functions:
   - safe_get(data, key, default=None): Safely gets a value from a dict. IMPORTANT: This function DOES NOT support dot notation directly on the input 'data' dictionary. Use it to get top-level variables by their 'name'.
   - safe_number(value, default=0): Safely converts a value to a number
   - safe_list(value): Ensures a value is a list
   - safe_sum(values, default=0): Safely sums a list of values
   - safe_avg(values, default=0): Safely calculates the average of a list
   - safe_max(values, default=0): Safely finds the maximum value in a list
   - safe_min(values, default=0): Safely finds the minimum value in a list
   - safe_count(values, predicate=None): Safely counts elements in a list
   - log_metric_error(metric_name, error, context=None): Logs metric calculation errors

6. IMPORTANT: Agent data is provided as LISTS of values, one value per agent of that type. The path definition was handled during data collection. The function receives a dictionary where keys are the variable 'name' and values are either single environment values or lists of agent values.
   - You MUST handle agent variables (which are lists) using appropriate aggregation (sum, average, max, min) using the safe_* helper functions.
   - Environment variables are single values.
   - ALWAYS check if list is empty before operations like sum() or calculating averages.
   - Use the safe_* helper functions available in the module.

7. MULTI-SERIES SUPPORT:
   - For "line" charts: Return a dictionary where each key represents a different line
     Example: {{'series1': value1, 'series2': value2}}
   - For "bar" charts with multiple series: Return a nested dictionary or appropriate format
     that represents multiple data series for grouped bar charts
   - For "pie" charts: Return a dictionary where each key represents a slice

8. EXTREMELY IMPORTANT: Make your function robust to all of these edge cases:
   - Variable 'name' might be missing from the data dictionary
   - Values associated with a 'name' might be None
   - Agent variables (lists) might be empty
   - Lists might contain None values
   - Values might be of unexpected types
   - Division by zero scenarios
   - Other potential errors or exceptions

9. VARIABLE ACCESS PATTERN:
   - Access variables from the input 'data' dictionary using the 'name' defined in the metric's 'variables' list.
   - Example: `value = safe_get(data, 'variable_name_from_metric_def')`
   - The monitor system handles data collection based on the variable's 'path':
     * For most variables, 'path' is the same as 'name' (simple key access)
     * For nested data structures, 'path' uses dot notation (e.g., "stats.value")
   - IMPORTANT: Regardless of how complex the original 'path' is, inside this function
     you ALWAYS access data using ONLY the variable 'name' with safe_get(data, 'name')
   - The system has already traversed any complex paths during data collection and placed
     the values in the 'data' dictionary under the corresponding 'name' keys.

Return the function in a Python code block:

```python
from typing import Dict, Any

def {function_name}(data: Dict[str, Any]) -> Any:
    \"\"\"
    计算指标: {metric_def.get('name', 'unknown')}
    描述: {metric_def.get('description', '无描述')}
    可视化类型: {metric_def.get('visualization_type', 'line')}
    更新频率: {metric_def.get('update_interval', 5)} 秒
    
    Args:
        data: 包含所有变量的数据字典，注意agent变量是列表形式
        
    Returns:
        根据可视化类型返回不同格式的结果:
        - line: 返回字典，键为系列名称，值为对应数值
        - bar/pie: 返回字典，键为分类，值为对应数值
        
    注意:
        此函数处理各种异常情况，包括None值、空列表和类型错误等
    \"\"\"
    try:
        # Check if required variables exist and validate input data
        if not data or not isinstance(data, dict):
            log_metric_error("{metric_def.get('name', 'unknown')}", ValueError("Invalid data input"), {{"data": data}})
            return {{}} if "{metric_def.get('visualization_type', 'line')}" != "line" else {{"default": 0}}

        # Example: Accessing a required environment variable named 'env_var_name'
        env_value = safe_get(data, 'env_var_name')
        if env_value is None: # Check if required value is missing or None
            log_metric_error("{metric_def.get('name', 'unknown')}", ValueError("Missing required variable: env_var_name"))
            return {{}} if "{metric_def.get('visualization_type', 'line')}" != "line" else {{"default": 0}}

        # Example: Accessing an agent variable list named 'agent_var_name'
        agent_data_list = safe_list(safe_get(data, 'agent_var_name', [])) # Use default [] if not present

        # Safe aggregation using helper functions
        agent_avg = safe_avg(agent_data_list)
        agent_sum = safe_sum(agent_data_list)
        
        # Implementation
        # [Your calculation logic here using safe_get(data, var_name) to access values]
        
        # Return result in appropriate format
        result = {{}} # Placeholder
        return result
    except Exception as e:
        log_metric_error("{metric_def.get('name', 'unknown')}", e, {{"data_keys": list(data.keys()) if isinstance(data, dict) else None}})
        return {{}} if "{metric_def.get('visualization_type', 'line')}" != "line" else {{"default": 0}}
```
"""

    def format_metrics_for_export(self, metrics: List[Dict]) -> List[Dict]:
        """
        将生成的指标格式化为导出格式，使其更适合存储在scene_info.json中
        
        Args:
            metrics: 原始指标定义列表
            
        Returns:
            格式化后的指标列表
        """
        formatted_metrics = []
        
        for metric in metrics:
            # 创建安全的函数名
            function_name = re.sub(r'[^\w\-_]', '_', metric.get("name", "unknown_metric"))
            
            # 格式化指标
            formatted_metric = {
                "id": function_name,
                "name": metric.get("name", "未命名指标"),
                "description": metric.get("description", "无描述"),
                "visualization_type": metric.get("visualization_type", "line"),
                "update_interval": metric.get("update_interval", 60),
                "variables": metric.get("variables", []),
                "calculation_logic": metric.get("calculation_logic", "无计算逻辑"),
                "function_name": function_name
            }
            
            formatted_metrics.append(formatted_metric)
            
        return formatted_metrics

    def generate_metrics_code_file(self, metrics: List[Dict], output_dir: str) -> Dict[str, str]:
        """
        为指标生成计算函数代码文件
        
        Args:
            metrics: 指标定义列表
            output_dir: 输出目录
            
        Returns:
            指标名称到文件路径的映射
        """
        os.makedirs(output_dir, exist_ok=True)
        file_paths = {}
        
        for metric in metrics:
            metric_name = metric.get('name', 'unknown_metric')
            function_code = self.generate_calculation_function(metric)
            
            # 清理文件名，替换非法字符
            safe_name = re.sub(r'[^\w\-_]', '_', metric_name)
            file_path = os.path.join(output_dir, f"metric_{safe_name}.py")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"""# -*- coding: utf-8 -*-
\"\"\"
指标: {metric.get('name', 'unknown_metric')}
描述: {metric.get('description', '无描述')}
可视化类型: {metric.get('visualization_type', 'line')}
更新频率: {metric.get('update_interval', 60)} 秒
\"\"\"

from typing import Dict, Any, List, Optional, Union
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)

{function_code}
""")
            file_paths[metric_name] = file_path
            logger.info(f"已生成指标计算代码: {file_path}")
            
        return file_paths

    def generate_metrics_module(self, metrics: List[Dict], output_dir: str, system_data_model: Dict = None) -> str:
        """
        为所有指标生成单个计算模块文件
        
        Args:
            metrics: 指标定义列表
            output_dir: 输出目录
            system_data_model: 系统数据模型，包含环境变量和各类Agent变量
            
        Returns:
            模块文件路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建metrics.py文件
        module_path = os.path.join(output_dir, "metrics.py")
        with open(module_path, 'w', encoding='utf-8') as f:
            # 添加文件头和工具函数导入
            f.write("""# -*- coding: utf-8 -*-
\"\"\"
自动生成的监控指标计算模块
\"\"\"

from typing import Dict, Any, List, Optional, Union, Callable
import math
from loguru import logger
from onesim.monitor.utils import (
    safe_get, safe_number, safe_list, safe_sum, 
    safe_avg, safe_max, safe_min, safe_count, log_metric_error
)

""")
            
            # 为每个指标生成计算函数
            for metric in metrics:
                metric_name = metric.get('name', 'unknown_metric')
                # 生成安全的函数名
                function_name = re.sub(r'[^\w\-_]', '_', metric_name)
                # 生成函数代码
                function_code = self.generate_calculation_function(metric, system_data_model)
                
                # 添加函数注释
                f.write(f"""
{function_code}
""")
            
            # 写入辅助函数
            f.write("""
# 指标函数字典，用于查找
METRIC_FUNCTIONS = {
""")
            for metric in metrics:
                function_name = re.sub(r'[^\w\-_]', '_', metric.get("name", "unknown_metric"))
                f.write(f"    '{function_name}': {function_name},\n")
            f.write("}\n\n")
            
            # 添加获取函数的工具方法
            f.write('''
def get_metric_function(function_name: str) -> Optional[Callable]:
    """
    根据函数名获取对应的指标计算函数
    
    Args:
        function_name: 函数名
        
    Returns:
        指标计算函数或None
    """
    return METRIC_FUNCTIONS.get(function_name)
''')
        
        logger.info(f"已生成指标计算模块: {module_path}")
        return module_path