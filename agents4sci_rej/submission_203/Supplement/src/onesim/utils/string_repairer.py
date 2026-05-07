# -*- coding: utf-8 -*-
"""
String Replacement Repairer
Handles string-based repair operations with intelligent correction.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from loguru import logger
from onesim.models.core.message import Message
from onesim.models import JsonBlockParser

from .debugging_data_structures import (
    RepairParams, RepairItem, RepairResult, RepairContext
)
from .repair_corrector import IntelligentCorrector


class StringReplacementRepairer:
    """字符串替换修复器 - 实现灵活的修复粒度"""
    
    def __init__(self, model, debugging_agent=None):
        self.model = model
        self.debugging_agent = debugging_agent
        self.corrector = IntelligentCorrector(model)
        self.json_parser = JsonBlockParser()
    
    async def repair_with_replacement(self, error,
                                    context: RepairContext) -> RepairResult:
        """使用字符串替换进行修复"""
        
        # 1. 读取多个相关文件（基于selected_context）
        files_content = {}
        
        # 首先读取错误文件（如果存在）
        if error.file_path:
            error_file_content = self.read_file(error.file_path)
            if error_file_content:
                files_content[error.file_path] = error_file_content
        
        # 读取selected_context中指定的相关文件
        files_to_read = getattr(context.selected_context, 'files_to_read', [])
        if not files_to_read:
            # 如果没有指定files_to_read，使用selected_related_files作为fallback
            files_to_read = context.selected_context.selected_related_files
        
        for file_path in files_to_read:
            if file_path and file_path not in files_content:
                file_content = self.read_file(file_path)
                if file_content:
                    files_content[file_path] = file_content
        
        # 确保至少有一个文件被读取
        if not files_content:
            return RepairResult(
                success=False,
                error_message="No files could be read for repair"
            )
        
        # 2. LLM生成修复参数（传入所有文件内容）
        repair_params = await self.generate_repair_params(error, context, files_content)
        if not repair_params or not repair_params.repairs:
            return RepairResult(
                success=False,
                error_message="Failed to generate repair parameters"
            )
        
        # 3. 执行多文件修复
        repair_result = await self.execute_multi_file_replacement(repair_params, files_content)
        
        return repair_result
    
    def read_file(self, file_path: str, line_number: Optional[int] = None) -> str:
        """读取文件内容，复用上下文读取逻辑"""
        # if self.debugging_agent and hasattr(self.debugging_agent, 'context_manager') and self.debugging_agent.context_manager:
        #     return self.debugging_agent.context_manager.llm_selector.read_file_with_context(file_path, line_number)
        
        try:
            actual_file_path = file_path
            if self.debugging_agent:
                actual_file_path = self.debugging_agent.container_path_to_host_path(file_path)
            
            with open(actual_file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return ""
    
    def write_file(self, file_path: str, content: str) -> bool:
        """写入文件内容（带安全检查）"""
        try:
            # Convert container path to host path if necessary
            actual_file_path = file_path
            if self.debugging_agent:
                actual_file_path = self.debugging_agent.container_path_to_host_path(file_path)
                
                # Safety check: only allow modifications to environment files
                current_env = getattr(self.debugging_agent, 'current_env_name', None)
                if current_env:
                    is_allowed, reason = self.debugging_agent.is_file_modification_allowed(file_path, current_env)
                    if not is_allowed:
                        logger.error(f"File modification blocked: {reason}")
                        return False
                    else:
                        logger.info(f"File modification allowed: {reason}")
            
            # Create backup before writing
            if self.debugging_agent and Path(actual_file_path).exists():
                backup_path = f"{actual_file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                try:
                    shutil.copy2(actual_file_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
                except Exception as backup_error:
                    logger.warning(f"Failed to create backup: {backup_error}")
            
            with open(actual_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write file {file_path}: {e}")
            return False
    
    async def generate_repair_params(self, error,
                                   context: RepairContext,
                                   files_content: Dict[str, str]) -> Optional[RepairParams]:
        """生成修复参数（支持多文件）"""
        
        repair_prompt = self.build_repair_prompt(error, context, files_content)
        
        try:
            formatted_prompt = self.model.format(
                Message("user", repair_prompt, role="user")
            )
            
            # Log LLM call if debugging agent is available
            if self.debugging_agent:
                self.debugging_agent.log_llm_call(
                    "repair_generation",
                    repair_prompt,
                    "",  # Will be filled after response
                    {
                        "error_type": str(error.error_type),
                        "file_path": error.file_path,
                        "files_count": len(files_content)
                    }
                )
            
            llm_response = await self.model.acall(formatted_prompt)
            
            # Log response
            if self.debugging_agent:
                self.debugging_agent.log_llm_call(
                    "repair_generation",
                    repair_prompt,
                    str(llm_response.text),
                    {
                        "error_type": str(error.error_type),
                        "file_path": error.file_path,
                        "files_count": len(files_content)
                    }
                )
            
            repair_params = self.parse_repair_params(llm_response)
            return repair_params
        except Exception as e:
            logger.error(f"Failed to generate repair params: {e}")
            return None
    
    def build_repair_prompt(self, error,
                          context: RepairContext,
                          files_content: Dict[str, str]) -> str:
        """构建修复提示（支持多文件）"""
        
        # Check if this is a metric-related error
        is_metric_error = (
            (error.file_path and 'metrics.py' in error.file_path) or
            'metric' in error.error_message.lower() or
            'monitor' in error.error_message.lower() or
            any(keyword in error.error_message.lower() for keyword in [
                'safe_get', 'safe_list', 'safe_sum', 'safe_avg', 'safe_count',
            ])
        )
        
        # Build files content section
        files_content_section = ""
        for file_path, file_content in files_content.items():
            files_content_section += f"\n### 文件: {file_path}\n```python\n{file_content}\n```\n"
        
        base_prompt = f"""
你是OneSim平台的代码修复专家。请分析错误并生成精确的修复参数。

## 错误信息
错误类型: {error.error_type}
错误消息: {error.error_message}
文件路径: {error.file_path}
函数名: {error.function_name}
行号: {error.line_number}

## 相关上下文
{context.get_formatted_context()}

## 相关文件内容
{files_content_section}

## 历史修复经验
{context.get_historical_fixes_summary()}"""


        code_template="""
## 代码模板

### 核心理解：项目代码是根据固定模板生成的，具有以下特征：

### Handler模板模式:
```python
async def handler_name(self, event: Event) -> List[Event]:
    # 1. 从event中获取数据
    field_value = event.field_name
    
    # 2. 构建observation和instruction
    observation = f"Context: {field_value}"
    instruction = '''请根据上下文返回JSON格式:
    {
        "key": "value",
        "target_ids": ["agent_id"] 或 "single_id"
    }'''
    
    # 3. 调用generate_reaction
    result = await self.generate_reaction(instruction, observation)
    
    # 4. 处理target_ids (关键错误点)
    target_ids = result.get('target_ids', [])
    if not isinstance(target_ids, list):
        target_ids = [target_ids]
    
    # 5. 创建并返回events
    events = []
    for target_id in target_ids:
        event = NextEventClass(self.profile_id, target_id, data)
        events.append(event)
    return events
```

### Event类模板模式:
```python
class EventName(Event):
    def __init__(self, from_agent_id: str, to_agent_id: str, field: type = default, **kwargs):
        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)
        self.field = field
```

### 常见错误模式与精确修复策略:

#### 1. target_ids类型错误 (最常见)
**问题**: `target_ids`可能是字符串或列表，未统一处理
**识别**: 错误信息包含"list", "iteration", "target_ids"
**修复模式**:
```python
# 错误代码:
target_ids = result.get('target_ids', [])
for target_id in target_ids:  # 如果target_ids是字符串会报错

# 修复代码:
target_ids = result.get('target_ids', [])
if not isinstance(target_ids, list):
    target_ids = [target_ids]
for target_id in target_ids:
```

#### 2. Event构造参数错误
**问题**: Event类构造函数参数顺序或类型不匹配
**识别**: 错误信息包含Event类名，"TypeError", "argument"
**修复策略**: 检查对应Event类的__init__参数定义，确保参数匹配

#### 3. generate_reaction格式错误
**问题**: instruction未明确要求JSON格式或target_ids
**识别**: 错误信息包含"generate_reaction", "JSON", "format"
**修复策略**: 在instruction中明确要求返回JSON格式并包含target_ids字段

#### 4. Event导入错误
**问题**: 未正确导入生成的Event类
**识别**: 错误信息包含"NameError", Event类名
**修复策略**: 添加 `from .events import EventClassName`

#### 5. 框架方法调用错误
**问题**: self.profile_id, self.generate_reaction等框架方法使用错误
**识别**: 错误信息包含"AttributeError", "profile_id", "generate_reaction"
**修复策略**: 确保正确继承GeneralAgent并使用框架提供的方法

### 修复优先级指导:
1. 优先检查target_ids处理 (80%的错误)
2. 其次检查Event构造函数参数
3. 最后检查generate_reaction的instruction格式

### Note:
- The `profile_id` is a property in base class GeneralAgent and is already defined. The use of 'self.profile_id' is correct.
- The `env` is a property in base class GeneralAgent and is already defined. The use of 'self.env' is correct. 
- Instructions for generate_reaction must explicitly request target_ids (can be single ID or a list)
- target_ids should be derived from the generate_reaction() response
- Code must handle both single ID and list responses (convert single IDs to a list with one element)
- Observation context included when needed
"""
        # Add metric-specific guidance if this is a metric error
        metric_guidance = ""
        if is_metric_error:
            metric_guidance = f"""

## 🔧 指标计算专项修复指导

### 指标系统架构理解
1. **scene_info.json**: 包含指标定义、变量来源、计算逻辑描述
   - variables: 定义数据来源 (source_type: "env"/"agent", path, agent_type等)
   - calculation_logic: 描述计算逻辑
   - function_name: 对应的计算函数名

2. **metrics/metrics.py**: 实际的指标计算函数实现
   - 函数名必须与scene_info.json中的function_name完全一致
   - 使用onesim.monitor.utils中的安全函数处理数据

3. **profile/schema/**: Agent schema文件，仅作参考，不需修改

### 常见指标错误类型与修复策略

**A. 数据访问错误**
- 问题: KeyError, AttributeError访问不存在的数据字段
- 修复: 使用safe_get(data, 'field_name', default_value)替换直接访问
- 示例: `data['field']` → `safe_get(data, 'field')`

**B. 数据类型错误**
- 问题: TypeError处理None值或错误数据类型
- 修复: 使用safe_list(), safe_number()等函数
- 示例: `len(data['list'])` → `safe_count(safe_list(data.get('list', [])))`

**C. 计算错误**
- 问题: ZeroDivisionError, ValueError在数学计算中
- 修复: 添加边界检查，使用safe_sum(), safe_avg()等
- 示例: 在除法前检查分母是否为0

**D. 函数名不匹配**
- 问题: 函数名与scene_info.json中的function_name不一致
- 修复: 确保函数名完全匹配，包括大小写

### 必须遵循的指标计算模式
```python
def Metric_Function_Name(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # 1. 输入验证
        if not data or not isinstance(data, dict):
            log_metric_error("Metric Name", ValueError("Invalid data"), {{"data": data}})
            return {{"default_key": default_value}}
        
        # 2. 安全数据提取
        field_value = safe_get(data, 'field_name', default_value)
        list_value = safe_list(safe_get(data, 'list_field', []))
        
        # 3. 数据处理与计算
        processed_data = [item for item in list_value if item is not None]
        result = safe_sum(processed_data) / safe_count(processed_data) if processed_data else 0
        
        # 4. 返回正确格式
        return {{"metric_key": result}}
        
    except Exception as e:
        log_metric_error("Metric Name", e, {{"data_keys": list(data.keys()) if isinstance(data, dict) else None}})
        return {{"metric_key": default_value}}
```

### 根据scene_info.json定义修复
- 检查variables定义的source_type和path是否正确使用
- 确保返回值格式符合visualization_type要求:
  - line: {{"series_name": [values]}}
  - bar: {{"category": value}}
  - pie: {{"category1": proportion1, "category2": proportion2}}
"""

        current_env = getattr(self.debugging_agent, 'current_env_name', 'current_env') if self.debugging_agent else 'current_env'
        
        prompt = base_prompt + code_template + metric_guidance + f"""

## 修复要求

请提供精确的字符串替换参数来修复这个错误。**注意：每个修复操作都必须指定目标文件路径**

```json
{{
    "repair_strategy": "string_replacement",
    "repairs": [
        {{
            "file_path": "需要修复的具体文件路径",
            "old_string": "要替换的原始代码片段(包含足够上下文)",
            "new_string": "修复后的代码片段", 
            "expected_replacements": 1,
            "reasoning": "修复原因说明"
        }}
    ],
    "backup_strategy": "如果字符串匹配失败的备选方案"
}}
```

## 关键修复原则
1. **文件路径明确**: 每个repair都必须指定file_path，可以修复多个文件
2. **精确匹配**: old_string必须是文件中的EXACT字符串，包含正确的缩进和空白字符
3. **上下文充足**: 包含错误行前后至少2-3行代码确保唯一匹配
4. **缩进一致**: new_string必须保持与原代码相同的缩进级别
5. **避免重复**: 绝对不要在new_string中包含行号标记或重复内容
6. **完整性**: 确保替换后的代码语法正确且逻辑完整
7. **OneSim兼容**: 考虑异步函数、事件系统等OneSim特性
8. **安全限制**: 只修改{str(Path('src') / 'envs' / current_env)}/目录下的文件，不得修改框架代码
9. **指标一致性**: 对于指标文件，确保修复后的代码符合指标计算规范和数据安全处理要求

## 常见错误避免
- 不要在new_string中包含"6: "这样的行号标记
- 不要重复现有代码或创建缩进混乱
- 绝对不要修改Agent类的初始化参数，添加新的初始化参数，只能修改Agent类的action函数
- 确保old_string完全匹配文件中的内容（包括所有空格和换行）
- 绝对不要尝试修改src/onesim/或其他框架目录下的文件
- 对于指标错误，始终使用安全函数处理数据，避免直接访问可能不存在的字段
- 可以在一次修复中处理多个文件，每个文件的修复都要单独指定file_path
"""
        return prompt
    
    def extract_error_context(self, file_content: str, error, context_lines: int = 10) -> str:
        """提取错误相关的代码上下文"""
        if not error.line_number or not file_content:
            return file_content
        
        lines = file_content.split('\n')
        error_line_idx = error.line_number - 1  # Convert to 0-based index
        
        # Calculate context range
        start_line = max(0, error_line_idx - context_lines)
        end_line = min(len(lines), error_line_idx + context_lines + 1)
        
        # Extract context lines with line numbers
        context_lines_with_numbers = []
        for i in range(start_line, end_line):
            line_num = i + 1
            marker = ">>> " if i == error_line_idx else "    "
            context_lines_with_numbers.append(f"{marker}{line_num:3d}: {lines[i]}")
        
        return "\n".join(context_lines_with_numbers)
    
    def parse_repair_params(self, response) -> Optional[RepairParams]:
        """解析修复参数"""
        try:
            # Create a ModelResponse object if we have a string
            if isinstance(response, str):
                from onesim.models.core.model_response import ModelResponse
                model_response = ModelResponse(text=response)
            else:
                model_response = response
                
            parsed = self.json_parser.parse(model_response)
            if not parsed.parsed:
                return None
                
            data = parsed.parsed
            repairs = []
            
            for repair_data in data.get('repairs', []):
                repair = RepairItem(
                    file_path=repair_data.get('file_path', ''),
                    old_string=repair_data.get('old_string', ''),
                    new_string=repair_data.get('new_string', ''),
                    expected_replacements=repair_data.get('expected_replacements', 1),
                    reasoning=repair_data.get('reasoning', '')
                )
                repairs.append(repair)
            
            return RepairParams(
                repair_strategy=data.get('repair_strategy', 'string_replacement'),
                repairs=repairs,
                backup_strategy=data.get('backup_strategy', '')
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse repair params: {e}")
            return None
    
    async def execute_multi_file_replacement(self, repair_params: RepairParams,
                                            files_content: Dict[str, str]) -> RepairResult:
        """执行多文件修复"""
        
        total_changes_made = 0
        modified_files = []
        repair_details = []
        
        # Group repairs by file path
        repairs_by_file = {}
        for repair in repair_params.repairs:
            file_path = repair.file_path
            if file_path not in repairs_by_file:
                repairs_by_file[file_path] = []
            repairs_by_file[file_path].append(repair)
        
        # Process each file
        for file_path, file_repairs in repairs_by_file.items():
            if file_path not in files_content:
                repair_details.append(f"File {file_path} not found in available files")
                continue
            
            actual_file_path = file_path
            if self.debugging_agent:
                actual_file_path = self.debugging_agent.container_path_to_host_path(file_path)
            
            with open(actual_file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            modified_content = original_content
            file_changes = 0
            
            # Apply all repairs for this file
            for repair in file_repairs:
                # Check occurrences
                occurrences = modified_content.count(repair.old_string)
                logger.info(f"repair: {repair}")
                logger.info(f"occurrences: {occurrences}")
                if occurrences == repair.expected_replacements:
                    # Perfect match - apply replacement
                    modified_content = modified_content.replace(
                        repair.old_string, repair.new_string, repair.expected_replacements
                    )
                    file_changes += 1
                    repair_details.append(f"✓ {file_path}: {repair.reasoning}")
                    
                elif occurrences > repair.expected_replacements:
                    # Too many matches - need more specific context
                    repair_details.append(f"✗ {file_path}: Too many matches ({occurrences}), need more specific context")
                    continue
                    
                else:
                    # No match - try intelligent correction
                    corrected_repair = await self.corrector.correct_mismatch(repair, modified_content)
                    
                    if corrected_repair:
                        modified_content = modified_content.replace(
                            corrected_repair.old_string, corrected_repair.new_string
                        )
                        file_changes += 1
                        repair_details.append(f"✓ {file_path}: {repair.reasoning} (corrected)")
                    else:
                        repair_details.append(f"✗ {file_path}: No matching code found for repair")
            
            # Write modified file if changes were made
            if file_changes > 0 and modified_content != original_content:
                if self.write_file(file_path, modified_content):
                    modified_files.append(file_path)
                    total_changes_made += file_changes
                    repair_details.append(f"📝 {file_path}: File updated with {file_changes} changes")
                else:
                    repair_details.append(f"✗ {file_path}: Failed to write file")
        
        # Determine overall success
        success = total_changes_made > 0
        
        return RepairResult(
            success=success,
            repaired_file=", ".join(modified_files) if modified_files else "",
            changes_made=total_changes_made,
            error_message="" if success else "No successful repairs applied",
            repair_details={"files_modified": modified_files, "repair_details": repair_details}
        )