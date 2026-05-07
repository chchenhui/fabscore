# -*- coding: utf-8 -*-
"""
ReAct Debugging Agent
Implements Reasoning and Acting approach for complex debugging tasks.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from loguru import logger

from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from onesim.utils.debugging_data_structures import (
    ReActAction, ReActState, ThinkingResult, RepairContext, RepairItem, RepairParams
)
from onesim.utils.context_selector import ContextualFileReader
from onesim.utils.error_parser import ParsedError

class ReActDebuggingAgent:
    """
    ReAct (Reasoning and Acting) debugging agent
    Use iterative approach to read files, analyze errors, and generate fixes in each repair attempt
    """
    
    def __init__(self, model, debugging_agent):
        self.model = model
        self.debugging_agent = debugging_agent
        self.json_parser = JsonBlockParser()
        self.contextual_reader = ContextualFileReader(model, debugging_agent)
    
    async def debug_with_react(self, error, 
                              repair_context: RepairContext) -> ReActState:
        """Debug using improved ReAct approach with proper Think-Act-Observe cycle"""
        
        state = ReActState(current_error=error)
        state.observations.append("Initial error analysis needed")
        max_iterations = self.debugging_agent.config.max_react_iterations
        
        for iteration in range(max_iterations):
            logger.info(f"ReAct iteration {iteration + 1}/{max_iterations}")
            

            thinking_result = await self.think_and_analyze(state, repair_context)
            
            if not thinking_result:
                logger.warning("No thinking result, stopping ReAct loop")
                state.termination_reason = "Failed to generate thinking result"
                break
            
            action = await self.create_and_execute_action(thinking_result, state, repair_context)
            
            if not action:
                logger.warning("No action generated, stopping ReAct loop")
                state.termination_reason = "Failed to generate action"
                break
            
            observation = action.result
            
            state.update_with_observation(action, observation, thinking_result.reasoning)
            
            # Check termination conditions
            should_terminate, reason = self._should_terminate_early(state)
            if should_terminate:
                state.termination_reason = reason
                if "successfully" in reason.lower():
                    state.completed = True
                    state.success = True
                break
        
        # Final state evaluation
        if not state.completed and not state.termination_reason:
            state.termination_reason = f"Reached maximum iterations ({max_iterations})"
            
        return state
    
    def _adapt_strategy_from_failures(self, state: ReActState):
        """从失败中学习并调整策略"""
        failed_actions = [action for action in state.actions_taken if not action.success]
        
        if failed_actions:
            # 记录失败模式
            failure_patterns = {}
            for action in failed_actions:
                pattern_key = f"{action.action_type}_{action.target}"
                if pattern_key not in failure_patterns:
                    failure_patterns[pattern_key] = []
                failure_patterns[pattern_key].append(action.result)
            
            # 在state中记录失败模式，供后续决策参考
            if not hasattr(state, 'failure_patterns'):
                state.failure_patterns = {}
            state.failure_patterns.update(failure_patterns)
            
            logger.info(f"Learned from {len(failed_actions)} failures, identified {len(failure_patterns)} failure patterns")
    
    async def think_and_analyze(self, state: ReActState, repair_context: RepairContext) -> Optional[ThinkingResult]:
        """整合错误分析和推理的Think阶段"""
        
        # Add null check for current_error
        if state.current_error is None:
            logger.error("No current error in state")
            return None
        
        # Get environment name for permission guidance
        env_name = getattr(self.debugging_agent, 'current_env_name', 'current_env') if self.debugging_agent else 'current_env'
        
        # 构建包含错误分析的思考提示
        thinking_prompt = f"""
# ReAct Debugging - Think阶段

## 当前错误
错误类型: {state.current_error.error_type.value}
错误信息: {state.current_error.error_message}
文件: {state.current_error.file_path or "Unknown"}
行号: {state.current_error.line_number or "Unknown"}

## 已有信息
已尝试行动: {[f"{a.action_type}({a.target})" for a in state.actions_taken[-3:]] if state.actions_taken else "None"}

## 已读取文件的具体内容
{self._format_file_contents(state)}

## 可用修复上下文
{repair_context.get_formatted_context()}

## 代码模板和错误模式知识
{self._get_generation_template_knowledge()}

## Experience Patterns (Learning from Previous Repairs)
{self._get_experience_patterns_summary(state, repair_context)}

## 代码设计上下文 (用于理解设计意图)
{self._get_design_context_summary(state, repair_context)}

## 环境文件结构 (src/envs/{env_name}/)
{self._get_environment_file_structure(env_name)}

## 可用上下文文件 (来自Context Selector)
{self._get_available_context_files(repair_context)}

## 文件访问权限说明
### 读取权限 (read_file):
- 可以读取项目中的任意文件
- 包括框架文件 (src/onesim/) 和环境文件 (src/envs/{env_name}/)
- 可以读取配置文件、依赖库源码等任何需要的文件来理解问题

### 修改权限 (generate_and_apply_fix):
- **只能修改环境文件**: src/envs/{env_name}/ 目录下的文件
- 允许修改的文件类型: *.py, *.json, *.yaml, *.yml
- **禁止修改框架文件**: src/onesim/ 目录下的所有文件
- **禁止修改系统文件**: config/, requirements.txt, Dockerfile等

### 修复策略建议:
- 如果错误源于框架文件，需要通过修改环境文件来适配或规避
- 所有修复操作必须在 src/envs/{env_name}/ 范围内进行
- 如果需要理解框架代码，先用 read_file 读取，再在环境文件中实现修复
- 对于metric/monitor错误，优先考虑修改 src/envs/{env_name}/code/metrics/metrics.py

## 思考任务
请进行系统性分析：

1. **错误根因分析**: 
   - 这个错误的根本原因是什么？
   - 是否需要更多信息来确定根因？

2. **信息需求评估**: 
   - 还需要读取哪些文件来理解问题？(可以读取任意文件)
   - 已有的文件内容是否足够？

3. **下一步行动规划**: 
   - 基于当前分析，最合适的下一步行动是什么？
   - read_file: 需要读取特定文件 (可以是任意文件，请从上面的环境文件结构中选择具体路径)
   - generate_and_apply_fix: 已有足够信息，可以生成并应用修复 (仅限环境文件)

4. **置信度评估**: 
   - 对当前分析的置信度 (0.0-1.0)
   - 如果置信度低，说明原因

**重要提示**: 
- action_target必须是具体的文件路径，从上面的环境文件结构或可用上下文文件中选择
- 对于read_file: 
  - 可以是单个文件路径 (如 "src/envs/{env_name}/code/events.py")
  - 也可以是文件路径列表 (如 ["src/envs/{env_name}/code/events.py", "src/envs/{env_name}/code/Agent.py"])
  - 一次读取多个文件能更高效地收集相关信息
- 对于generate_and_apply_fix: 只能选择单个需要修复的文件路径 (如 src/envs/{env_name}/code/Agent.py)

请返回JSON格式结果：
```json
{{
    "error_analysis": "详细的错误分析",
    "information_needs": "还需要什么信息",
    "recommended_action": "read_file|generate_and_apply_fix",
    "action_target": "单个文件路径字符串或文件路径列表,必须从上面的文件列表中选择",
    "reasoning": "选择这个行动的详细推理过程，包括为什么选择这些具体文件",
    "confidence": 0.0-1.0
}}
```
"""
        
        try:
            prompt = self.model.format(
                Message("user", thinking_prompt, role="user")
            )
            response = await self.model.acall(prompt)
            
          
            # Parse the JSON response
            parsed = self.json_parser.parse(response)
              # Log the thinking call
            if self.debugging_agent:
                self.debugging_agent.log_llm_call(
                    "react_thinking", 
                    thinking_prompt, 
                    str(parsed.parsed),
                    {"iteration": len(state.actions_taken) + 1}
                )
            
            if parsed.parsed:
                result_data = parsed.parsed
                # Handle action_target as either string or list
                action_target = result_data.get('action_target', '')
                if isinstance(action_target, list):
                    # Convert list to JSON string for storage in action.target
                    import json
                    action_target = json.dumps(action_target)
                
                return ThinkingResult(
                    analysis=result_data.get('error_analysis', ''),
                    information_needs=result_data.get('information_needs', ''),
                    action_target=action_target,
                    recommended_action=result_data.get('recommended_action', 'read_file'),
                    confidence=float(result_data.get('confidence', 0.5)),
                    reasoning=result_data.get('reasoning', '')
                )
            else:
                logger.warning("Failed to parse thinking result JSON")
                return None
                
        except Exception as e:
            logger.error(f"Failed to generate thinking result: {e}")
            return None
    
    def _format_file_contents(self, state: ReActState) -> str:
        """格式化已读取文件的具体内容"""
        if not state.files_read:
            return "无已读取文件内容"
        
        formatted_content = []
        for i, (file_path, content) in enumerate(state.files_read.items(), 1):
           
            
            formatted_content.append(f"""
### 已读取文件 {i}: {file_path}
```
{content}
```
""")
        
        # Add summary if multiple files
        if len(state.files_read) > 1:
            file_list = list(state.files_read.keys())
            summary = f"\n**已读取文件总结**: 共{len(file_list)}个文件 - {', '.join([f.split('/')[-1] for f in file_list])}"
            formatted_content.insert(0, summary)
        
        return "\n".join(formatted_content)
    
    def _should_terminate_early(self, state: ReActState) -> Tuple[bool, str]:
        """智能早期终止判断"""
        
        # Pattern 1: 成功生成并应用修复
        if state.actions_taken and state.actions_taken[-1].action_type == 'generate_and_apply_fix' and state.actions_taken[-1].success:
            return True, "Successfully generated and applied fix"
        
        # Pattern 2: 重复的相同失败
        if len(state.actions_taken) >= 3:
            recent_failures = state.actions_taken[-3:]
            if all(not action.success for action in recent_failures):
                # 检查是否是相同的错误
                error_messages = [action.result for action in recent_failures]
                if len(set(error_messages)) <= 1:  # 相同的错误消息
                    return True, "Repeated identical failures detected"
        
        # Pattern 3: 置信度持续下降
        if len(state.confidence_progression) >= 2:
            recent_confidence = state.confidence_progression[-2:]
            if all(conf < 0.3 for conf in recent_confidence):
                return True, "Low confidence in recent actions"
        
        # Pattern 4: 无法读取关键文件
        recent_read_failures = [a for a in state.actions_taken[-3:] 
                               if a.action_type == 'read_file' and not a.success 
                               and 'cannot read' in a.result.lower()]
        if len(recent_read_failures) >= 2:
            return True, "Persistent file access failures"
            
        return False, ""
    
    async def create_and_execute_action(self, thinking_result: ThinkingResult, state: ReActState, 
                                       repair_context: RepairContext) -> Optional[ReActAction]:
        """基于thinking结果创建并执行动作"""
        
        try:
            # 从thinking结果中提取action_target
            action_target = ""
            if thinking_result.action_target:
                action_target = thinking_result.action_target
            elif hasattr(thinking_result, 'reasoning') and thinking_result.reasoning:
                # 尝试从reasoning中提取target
                for line in thinking_result.reasoning.split('\n'):
                    if 'action_target' in line.lower():
                        parts = line.split(':')
                        if len(parts) > 1:
                            action_target = parts[1].strip().strip('"\'')
                            break
            
            # 如果没有找到target，使用默认逻辑
            if not action_target:
                if thinking_result.recommended_action == 'read_file':
                    action_target = self._determine_file_to_read(state, repair_context)
                elif thinking_result.recommended_action == 'generate_and_apply_fix':
                    action_target = state.current_error.file_path or "error_file"
                else:
                    action_target = "unknown"
            
            # 验证action_target的有效性
            if thinking_result.recommended_action == 'read_file':
                # For read_file, action_target can be a string or JSON list
                if not action_target or action_target in ["unknown", "error_file"]:
                    action_target = self._determine_file_to_read(state, repair_context)
            elif thinking_result.recommended_action == 'generate_and_apply_fix':
                # For generate_and_apply_fix, only single file is supported
                import json
                try:
                    if action_target.startswith('[') and action_target.endswith(']'):
                        file_list = json.loads(action_target)
                        if isinstance(file_list, list) and file_list:
                            action_target = file_list[0]  # Take first file for fix operations
                            logger.info(f"generate_and_apply_fix only supports single file, using: {action_target}")
                except (json.JSONDecodeError, AttributeError):
                    pass  # action_target is already a string, continue
            
            # 创建ReActAction
            action = ReActAction(
                action_type=thinking_result.recommended_action,
                target=action_target,
                reasoning=thinking_result.reasoning,
                confidence=thinking_result.confidence,
                parameters={}
            )
            
            # 执行动作
            await self.execute_action(action, state, repair_context)
            
            return action
            
        except Exception as e:
            logger.error(f"Failed to create and execute action: {e}")
            return None
    
    def _determine_file_to_read(self, state: ReActState, repair_context: RepairContext) -> str:
        """确定需要读取的文件"""
        
        # 优先读取错误文件
        if state.current_error and state.current_error.file_path:
            error_file = state.current_error.file_path
            if error_file not in state.files_read:
                return error_file
        
        # 然后读取相关上下文文件
        if repair_context and repair_context.selected_context.files_to_read:
            for file_path in repair_context.selected_context.files_to_read:
                if file_path not in state.files_read:
                    return file_path
        
        # 默认返回错误文件路径
        return state.current_error.file_path if state.current_error else "unknown_file"
    
    
    async def execute_action(self, action: ReActAction, state: ReActState, 
                           repair_context: RepairContext):
        """执行动作"""
        
        try:
            if action.action_type == 'read_file':
                await self.execute_read_file(action, state)
            elif action.action_type == 'generate_and_apply_fix':
                await self.execute_generate_and_apply_fix(action, state, repair_context)
            else:
                action.success = False
                action.result = f"Unknown action type: {action.action_type}"
        except Exception as e:
            action.success = False
            action.result = f"Action execution failed: {e}"
            logger.error(f"Action execution failed: {e}")
    
    async def execute_read_file(self, action: ReActAction, state: ReActState):
        """执行读取文件动作 - 支持读取单个文件或多个文件"""
        
        # Parse action target - could be a single file or a list of files
        file_targets = action.target
        env_name = getattr(self.debugging_agent, 'current_env_name', '')
        
        # Check if action.target is a JSON string (list of files)
        import json
        try:
            if file_targets.startswith('[') and file_targets.endswith(']'):
                file_paths = json.loads(file_targets)
                if not isinstance(file_paths, list):
                    file_paths = [file_targets]
            else:
                file_paths = [file_targets]
        except (json.JSONDecodeError, AttributeError):
            file_paths = [file_targets]
        
        # 解析模板路径：将 {env_name} 替换为实际环境名
        resolved_file_paths = []
        for file_path in file_paths:
            if '{env_name}' in file_path and env_name:
                resolved_file_paths.append(file_path.replace('{env_name}', env_name))
            else:
                resolved_file_paths.append(file_path)
        
        successfully_read_files = []
        failed_files = []
        total_lines_read = 0
        
        try:
            for file_path in resolved_file_paths:
                try:
                    # 使用ContextualFileReader进行智能读取
                    error_line = None
                    if state.current_error and state.current_error.line_number:
                        error_line = state.current_error.line_number
                    
                    # 传统方式：使用上下文窗口读取
                    content = await self.contextual_reader.read_file_with_context(
                        file_path=file_path,
                        error_line=error_line,
                        context_window=50
                    )
                    
                    # 新方式：LLM引导的整文件内容提取
                    purpose = f"The user wants to debug an error. The reasoning for reading this file is: '{action.reasoning}'."

                    extracted_content = await self.contextual_reader.extract_relevant_content_with_llm(
                        file_path=file_path,
                        purpose=purpose,
                        max_ranges=5
                    )
                    
                    # 组合传统内容和LLM提取内容
                    if extracted_content.content_sections:
                        combined_content = f"# Summary: {extracted_content.summary}\n\n"
                        combined_content += "\n\n".join(extracted_content.content_sections)
                        
                        state.files_read[file_path] = combined_content
                        successfully_read_files.append(f"{file_path} (LLM extracted {len(extracted_content.extracted_ranges)} sections)")
                    else:
                        # Fallback 到传统方式
                        state.files_read[file_path] = content
                        successfully_read_files.append(f"{file_path} (contextual)")
                    
                    # Count lines for summary
                    lines_count = len(content.split('\n'))
                    total_lines_read += lines_count
                    
                except Exception as e:
                    failed_files.append(f"{file_path}: {str(e)}")
                    logger.error(f"Failed to read file {file_path}: {e}")
            
            # 构建结果消息
            if successfully_read_files:
                action.success = True
                if len(successfully_read_files) == 1:
                    action.result = f"Successfully read file: {successfully_read_files[0]}"
                else:
                    action.result = f"Successfully read {len(successfully_read_files)} files: {', '.join(successfully_read_files)}"
                
                if total_lines_read > 0:
                    action.result += f" (total {total_lines_read} lines)"
                
                if failed_files:
                    action.result += f". Failed files: {', '.join(failed_files)}"
            else:
                action.success = False
                action.result = f"Failed to read any files. Errors: {', '.join(failed_files)}"
            
        except Exception as e:
            action.success = False
            action.result = f"Failed to execute read_file action: {e}"
            logger.error(f"Failed to execute read_file action: {e}")
    
    async def execute_generate_and_apply_fix(self, action: ReActAction, state: ReActState, 
                                           repair_context):
        """生成修复并立即应用 - 合并原generate_fix和modify_file功能"""
        
        target_file = action.target
        env_name = getattr(self.debugging_agent, 'current_env_name', '')
        
        # 解析模板路径
        if '{env_name}' in target_file and env_name:
            target_file = target_file.replace('{env_name}', env_name)
        
        # 安全检查：只修改环境文件
        is_allowed, reason = self.debugging_agent.is_file_modification_allowed(target_file, env_name)
        if not is_allowed:
            action.success = False
            action.result = f"File {target_file} modification not allowed: {reason}"
            return

        # 1. 首先生成修复方案
        fix_prompt = self.build_fix_generation_prompt(state, repair_context, action)
        
        try:
            # 生成修复
            formatted_prompt = self.model.format(Message("user", fix_prompt, role="user"))
            response = await self.model.acall(formatted_prompt)
            
            # Log LLM call
            if self.debugging_agent:
                self.debugging_agent.log_llm_call(
                    "react_generate_and_apply_fix", 
                    fix_prompt, 
                    response.text if hasattr(response, 'text') else str(response),
                    {"target": action.target, "iteration": len(state.actions_taken)}
                )
            
            # 解析修复方案
            parsed = self.json_parser.parse(response)
            if not parsed.parsed:
                action.success = False
                action.result = "Failed to parse fix generation response"
                return
                
            fix_data = parsed.parsed
            state.fix_attempts.append(fix_data)
            
  
            
            # 检查修复方案是否包含修复操作
            if 'repairs' not in fix_data or not fix_data['repairs']:
                action.success = False
                action.result = f"No repair operations found in generated fix for {target_file}"
                return
            
            # # 应用修复
            # repair_result = await self._apply_fix_to_file(target_file, fix_data, env_name)
            
            # if repair_result.success:
            #     action.success = True
            #     action.result = f"Successfully generated and applied fix to {target_file}: {repair_result.message}"
            # else:
            #     action.success = False
            #     action.result = f"Fix generation succeeded but application failed for {target_file}: {repair_result.error_message}"
                
        except Exception as e:
            action.success = False
            action.result = f"Generate and apply fix failed for {target_file}: {e}"
            logger.error(f"Generate and apply fix failed for {target_file}: {e}")
    
    async def _apply_fix_to_file(self, file_path: str, fix_data, env_name: str):
        """应用修复到文件的辅助方法"""
        from onesim.utils.debugging_data_structures import RepairResult
        
        try:
            # 读取当前文件内容
            actual_path = self.debugging_agent.container_path_to_host_path(file_path)
            with open(actual_path, 'r', encoding='utf-8') as f:
                current_content = f.read()
            
            # 应用所有修复操作
            modified_content = current_content
            applied_repairs = 0
            
            for repair in fix_data['repairs']:
                if repair.get('file_path') == file_path:
                    old_string = repair.get('old_string', '')
                    new_string = repair.get('new_string', '')
                    
                    if old_string in modified_content:
                        modified_content = modified_content.replace(old_string, new_string, 1)
                        applied_repairs += 1
                    else:
                        logger.warning(f"Old string not found in {file_path}: {old_string}")
            
            if applied_repairs == 0:
                return RepairResult(
                    success=False,
                    error_message=f"No repair operations could be applied to {file_path}"
                )
            
            # 写入修改后的内容
            with open(actual_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            return RepairResult(
                success=True,
                message=f"Applied {applied_repairs} repair operations to {file_path}"
            )
            
        except Exception as e:
            return RepairResult(
                success=False,
                error_message=f"Failed to apply fix to {file_path}: {e}"
            )
    
    def build_fix_generation_prompt(self, state: ReActState, 
                                   repair_context: RepairContext,
                                   action: ReActAction) -> str:
        """构建修复生成提示, 包含ReAct的推理和上下文信息"""
        
        if state.current_error is None:
            return "ERROR: No current error information available for fix generation"

        target_file_path = action.target
        if target_file_path in state.files_read:
            target_file_content = state.files_read[target_file_path]
        else:
            with open(target_file_path, 'r', encoding='utf-8') as f:
                target_file_content = f.read()

        # 添加CodeAgent生成模板知识
        generation_template_knowledge = self._get_generation_template_knowledge()

        prompt = f"""
# 代码修复生成任务

作为一名专家程序员，您的任务是修复一个错误。根据所提供的信息，您必须生成一个精确的代码替换方案来解决该错误。

{generation_template_knowledge}

## 1. 原始错误信息
**错误回溯:**
```
{state.current_error.error_message}
```
**错误文件:** {state.current_error.file_path or '未知'}
**行号:** {state.current_error.line_number or '未知'}

## 2. 修复推理
上一步的分析为生成修复方案提供了以下推理：
```
{action.reasoning}
```

## 3. 待修复的目标文件
修复的目标文件是: `{target_file_path}`

## 4. 目标文件 (`{target_file_path}`) 的内容
```
{target_file_content}
```

## 5. 📚 Experience Patterns (Learning from Previous Repairs)
{self._get_experience_patterns_for_fix_generation(state, repair_context)}
## 常见错误避免
- 不要在new_string中包含"6: "这样的行号标记
- 不要重复现有代码或创建缩进混乱
- 不要修改Agent类的初始化参数，只能修改handler函数
- 确保old_string完全匹配文件中的内容（包括所有空格和换行）
- 绝对不要尝试修改src/onesim/或其他框架目录下的文件

## 6. 修复指令
根据以上所有信息，**特别是CodeAgent生成模板知识**，生成一个包含所需代码修改的JSON对象。

- **`old_string`**: 必须是源文件内容中精确、可能包含多行的子字符串。
- **`new_string`**: 将用于替换 `old_string` 的新代码。
- **`reasoning`**: 简要解释为什么此更改可以修复错误。

**JSON 输出格式:**
```json
{{
    "repair_strategy": "string_replacement",
    "repairs": [
        {{
            "file_path": "{target_file_path}",
            "old_string": "需要被替换的、精确的、可能包含多行的原始代码块。",
            "new_string": "将替换旧代码的新代码块。",
            "expected_replacements": 1,
            "reasoning": "对本次修复的清晰简洁的解释。"
        }}
    ]
}}
```
"""
        return prompt
    
    def _get_generation_template_knowledge(self) -> str:
        """获取CodeAgent生成模板知识"""
        return """
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
    
    def _get_experience_patterns_summary(self, state: ReActState, repair_context: RepairContext) -> str:
        """Get experience patterns summary for thinking prompt"""
        if not state.current_error or not self.debugging_agent.experience_manager:
            return "No experience patterns available."
        
        try:
            # Get pattern guidance from experience manager
            pattern_guidance = self.debugging_agent.experience_manager.get_pattern_guidance(state.current_error)
            if pattern_guidance:
                return pattern_guidance
            
            # Fallback to formatted patterns
            formatted_patterns = self.debugging_agent.experience_manager.format_patterns_for_llm(state.current_error, max_patterns=2)
            return formatted_patterns if formatted_patterns else "No relevant experience patterns found."
        except Exception as e:
            return f"Error retrieving experience patterns: {e}"
    
    def _get_experience_patterns_for_fix_generation(self, state: ReActState, repair_context: RepairContext) -> str:
        """Get experience patterns for fix generation prompt"""
        if not state.current_error or not self.debugging_agent.experience_manager:
            return "No experience patterns available for guidance."
        
        try:
            # Get both local and global patterns
            local_patterns, global_patterns = self.debugging_agent.experience_manager.find_combined_patterns(state.current_error)
            
            if not local_patterns and not global_patterns:
                return "No relevant experience patterns found for this error type."
            
            pattern_summaries = []
            
            # Add local patterns
            for pattern in local_patterns[:1]:  # Only show top 1 for fix generation
                pattern_summaries.append(f"""
**Local Pattern**: {pattern.pattern_name}
- **Solution**: {pattern.solution_template}
- **Key Insights**: {'; '.join(pattern.key_insights[:2])}
- **Success Rate**: {pattern.success_rate:.1%}
""")
            
            # Add global patterns
            for pattern in global_patterns[:1]:  # Only show top 1 for fix generation
                pattern_summaries.append(f"""
**Global Pattern**: {pattern.pattern_name}
- **Solution**: {pattern.solution_template}
- **Key Insights**: {'; '.join(pattern.key_insights[:2])}
- **Success Rate**: {pattern.success_rate:.1%}
""")
            
            return "\n".join(pattern_summaries)
        except Exception as e:
            return f"Error retrieving experience patterns: {e}"
    
    def _get_design_context_summary(self, state: ReActState, repair_context: RepairContext) -> str:
        """获取代码设计上下文总结"""
        if not state.current_error:
            return "No current error to analyze design context."
        
        env_name = getattr(self.debugging_agent, 'current_env_name', '')
        if not env_name:
            return "Environment name not available."
        
        try:
            # 获取代码结构信息
            code_structure = self.debugging_agent.get_code_structure(env_name)
            
            context_parts = []
            
            # 原始需求描述
            if "odd_protocol" in code_structure:
                context_parts.append(f"**系统需求**: {code_structure['odd_protocol']['overview']['system_goal']}")
            
            # 相关的action定义
            if "agents" in code_structure and state.current_error.file_path:
                related_actions = self._find_related_actions(state.current_error, code_structure["agents"])
                if related_actions:
                    context_parts.append("**相关Action定义**:")
                    for action in related_actions[:2]:  # 最多显示2个
                        context_parts.append(f"- {action.get('name', 'unknown')}: {action.get('description', 'No description')}")
            
            # 相关的事件流
            if "events" in code_structure:
                related_events = self._find_related_events(state.current_error, code_structure["events"])
                if related_events:
                    context_parts.append("**相关Event流**:")
                    for event in related_events[:2]:  # 最多显示2个
                        context_parts.append(f"- {event.get('event_name', 'unknown')}: {event.get('from_agent_type', 'unknown')} → {event.get('to_agent_type', 'unknown')}")
            
            # 错误位置的代码特征提示
            if state.current_error.file_path:
                code_hints = self._get_code_location_hints(state.current_error)
                if code_hints:
                    context_parts.append(f"**代码位置特征**: {code_hints}")
            
            return "\n".join(context_parts) if context_parts else "设计上下文信息不完整。"
            
        except Exception as e:
            return f"获取设计上下文失败: {e}"
    
    def _find_related_actions(self, error: ParsedError, agents: Dict[str, List[Dict]]) -> List[Dict]:
        """查找与错误相关的action定义"""
        related_actions = []
        
        if not error.file_path:
            return related_actions
            
        # 从文件路径推断agent类型
        file_name = error.file_path.split('/')[-1].replace('.py', '')
        
        # 查找匹配的agent和其actions
        for agent_type, actions in agents.items():
            if agent_type.lower() in file_name.lower() or file_name.lower() in agent_type.lower():
                # 如果有函数名，优先匹配对应的action
                if error.function_name:
                    for action in actions:
                        if action.get('name', '').lower() == error.function_name.lower():
                            related_actions.append(action)
                            break
                    else:
                        # 如果没找到完全匹配的，添加该agent的所有actions
                        related_actions.extend(actions)
                else:
                    related_actions.extend(actions)
                break
        
        return related_actions
    
    def _find_related_events(self, error: ParsedError, events: Dict[int, Dict]) -> List[Dict]:
        """查找与错误相关的事件定义"""
        related_events = []
        
        if not error.file_path:
            return related_events
        
        # 从错误信息中提取可能的事件名
        error_text = (error.error_message + " " + (error.code_context or "")).lower()
        
        for event in events.values():
            event_name = event.get('event_name', '').lower()
            if event_name and event_name in error_text:
                related_events.append(event)
            elif error.function_name and event.get('to_action_name', '').lower() == error.function_name.lower():
                related_events.append(event)
        
        return related_events  # 最多返回3个相关事件
    
    def _get_code_location_hints(self, error: ParsedError) -> str:
        """根据错误位置提供代码特征提示"""
        hints = []
        
        if error.file_path:
            if "events.py" in error.file_path:
                hints.append("Event类定义文件，注意构造函数参数和继承")
            elif error.file_path.endswith(".py") and "events.py" not in error.file_path and "SimEnv.py" not in error.file_path:
                hints.append("Agent handler文件，注意async函数、generate_reaction调用、target_ids处理")
        
        if error.function_name:
            if error.function_name.startswith("__init__"):
                hints.append("构造函数错误，检查参数匹配")
            elif "async" in str(error.code_context or "").lower():
                hints.append("异步handler函数，检查事件处理流程")
        
        return "; ".join(hints)
    
    def _get_environment_file_structure(self, env_name: str) -> str:
        """获取环境文件结构信息，提供实际发现的文件路径"""
        if not self.debugging_agent:
            return "Environment file structure not available."
        
        try:
            # 使用context selector的discover_environment_files方法获取实际文件
            from onesim.utils.context_selector import LLMContextSelector
            context_selector = LLMContextSelector(self.model, self.debugging_agent)
            env_files, config_files = context_selector.discover_environment_files()
            
            if not env_files and not config_files:
                return f"No files discovered in environment {env_name}"
            
            # 按类别组织文件
            code_files = []
            profile_files = []
            root_config_files = []
            
            # 处理环境文件(.py文件)
            for file_path in env_files:
                full_path = f"src/envs/{env_name}/{file_path}"
                if file_path.startswith('code/'):
                    if 'metrics/metrics.py' in file_path:
                        code_files.append(f"{full_path} - Monitor/metric calculations during simulation")
                    elif file_path.endswith('Agent.py'):
                        agent_name = file_path.split('/')[-1].replace('.py', '')
                        code_files.append(f"{full_path} - {agent_name} class and action handlers")
                    elif 'events.py' in file_path:
                        code_files.append(f"{full_path} - All event class definitions")
                    elif 'SimEnv.py' in file_path:
                        code_files.append(f"{full_path} - Simulation environment definition")
                    else:
                        code_files.append(f"{full_path} - Code implementation")
            
            # 处理配置文件(.json/.yaml文件)
            for file_path in config_files:
                full_path = f"src/envs/{env_name}/{file_path}"
                if 'scene_info.json' in file_path:
                    root_config_files.append(f"{full_path} - Simulation scenario and metric definitions")
                elif 'actions.json' in file_path:
                    root_config_files.append(f"{full_path} - Action definitions for workflow")
                elif 'events.json' in file_path:
                    root_config_files.append(f"{full_path} - Event definitions for workflow")
                elif 'system_data_model.json' in file_path:
                    root_config_files.append(f"{full_path} - Data models used in simulation")
                elif 'env_data.json' in file_path:
                    root_config_files.append(f"{full_path} - Environment data loaded before simulation")
                elif 'schema' in file_path:
                    root_config_files.append(f"{full_path} - Agent profile schema definition")
                else:
                    root_config_files.append(f"{full_path} - Configuration file")
            
            # 构建结构化输出
            structure_parts = []
            
            if code_files:
                structure_parts.append("### Core Implementation (/code/)")
                structure_parts.extend(code_files)
            
            if profile_files:
                structure_parts.append("\n### Agent Configuration (/profile/)")
                structure_parts.extend(profile_files)
            
            if root_config_files:
                structure_parts.append("\n### System Configuration (root level)")
                structure_parts.extend(root_config_files)
            
            # # 添加文件统计信息
            # structure_parts.append(f"\n### File Summary")
            # structure_parts.append(f"Total files discovered: {len(env_files) + len(config_files)}")
            # structure_parts.append(f"- Python files: {len(env_files)}")
            # structure_parts.append(f"- Configuration files: {len(config_files)}")
            
            return "\n".join(structure_parts) if structure_parts else f"Environment {env_name} discovered but no categorizable files found"
            
        except Exception as e:
            return f"Failed to get environment file structure: {e}"
    
    def _get_available_context_files(self, repair_context: RepairContext) -> str:
        """获取可用的上下文文件列表，便于ReAct决策时选择具体文件"""
        if not repair_context or not repair_context.selected_context:
            return "No context files available from context selector."
        
        context_parts = []
        
        # 来自context selector的推荐文件
        if repair_context.selected_context.files_to_read:
            context_parts.append("### Context Selector推荐读取的文件:")
            for file_path in repair_context.selected_context.files_to_read:
                context_parts.append(f"- {file_path}")
        
        # 相关文件列表 
        if repair_context.selected_context.selected_related_files:
            context_parts.append("\n### 相关文件:")
            for file_path in repair_context.selected_context.selected_related_files:
                context_parts.append(f"- {file_path}")
        
        # 相关的agent和event信息
        context_info = []
        if repair_context.selected_context.selected_agents:
            context_info.append(f"相关Agents: {', '.join(repair_context.selected_context.selected_agents)}")
        
        if repair_context.selected_context.selected_events:
            context_info.append(f"相关Events: {', '.join(repair_context.selected_context.selected_events)}")
        
        if context_info:
            context_parts.append(f"\n### 上下文分析结果:")
            context_parts.extend([f"- {info}" for info in context_info])
        
        # Context selector的推理说明
        if repair_context.selected_context.reasoning:
            context_parts.append(f"\n### Context Selector推理:")
            context_parts.append(repair_context.selected_context.reasoning)
        
        return "\n".join(context_parts) if context_parts else "Context selector没有提供具体的文件建议."