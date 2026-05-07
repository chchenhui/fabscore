# -*- coding: utf-8 -*-
"""
Intelligent Debugging Agent
Main debugging agent class using modular architecture.
Refactored from a monolithic implementation.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import uuid
import fnmatch

from loguru import logger

from .base import AgentBase
from .context_manager import ContextManager
from .react_debugging_agent import ReActDebuggingAgent
from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from onesim.utils.docker_runner import DockerRunner
from onesim.utils.error_parser import ErrorParser, ParsedError
from onesim.utils.experience import ExperienceManager, RepairRecord
from onesim.utils.debugging_config import DebuggingConfig
from onesim.utils.debugging_data_structures import (
    DebuggingResult, SimulationResult, ErrorClassification, ReActState,
    RepairResult, RepairParams, RepairItem
)
from onesim.utils.string_repairer import StringReplacementRepairer


class DebuggingAgent(AgentBase):
    """Intelligent Debugging Agent - focused on automatic repair"""
    
    def __init__(self, model_config_name: str, config: Optional[DebuggingConfig] = None, **kwargs):
        super().__init__(
            sys_prompt="""You are an intelligent debugging Agent for the OneSim platform, focused on automatic code error repair.
Your core tasks are:
1. Analyze runtime errors
2. Select the most relevant context information
3. Generate precise code fixes
4. Ensure repaired code complies with OneSim framework specifications""",
            model_config_name=model_config_name
        )
        
        # Use the provided configuration or the default one
        self.config = config or DebuggingConfig()
        
        # Initialize ReAct debugging agent
        if self.config.enable_react_debugging:
            self.react_agent = ReActDebuggingAgent(model=self.model, debugging_agent=self)
            logger.info("ReAct debugging agent initialized")
        else:
            self.react_agent = None
            logger.info("ReAct debugging agent disabled")
        
        self.repairer = StringReplacementRepairer(model=self.model, debugging_agent=self)
        
        # Docker runner for execution
        self.docker_runner = DockerRunner(
            container_name=self.config.container_name,
            host_project_path=self.config.host_project_path,
            container_workdir=self.config.container_project_path,
            image_name=self.config.docker_image,
            default_timeout=self.config.timeout_per_iteration
        )
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.debug_logger = None  # Will be set in auto_debug
        self.current_env_name = None  # Track current environment for safety checks
    
    def container_path_to_host_path(self, container_path: str) -> str:
        """Convert container path to host path with proper path resolution"""
        if container_path.startswith(self.config.container_project_path):
            # Remove container workdir prefix and add host project path
            relative_path = container_path[len(self.config.container_project_path):].lstrip('/')
            host_path = Path(self.config.host_project_path) / relative_path
            return str(host_path)
        
        # Handle environment-relative paths
        if not container_path.startswith(('/', 'src/')):
            # Assume it's relative to current environment
            if hasattr(self, 'current_env_name') and self.current_env_name:
                host_path = Path(self.config.host_project_path) / "src" / "envs" / self.current_env_name / container_path
                return str(host_path)
        
        return container_path
    
    def is_file_modification_allowed(self, file_path: str, env_name: str) -> Tuple[bool, str]:
        """
        Check if a file modification is allowed - only environment files are permitted
        
        Args:
            file_path: Path to the file to be modified
            env_name: Current environment name
            
        Returns:
            Tuple[bool, str]: (is_allowed, reason)
        """
        if not file_path:
            return False, "File path is empty or None"
        
        # Convert container path to host path for checking
        host_file_path = self.container_path_to_host_path(file_path)
        
        # Convert to relative path from project root
        try:
            project_root = Path(self.config.host_project_path)
            file_path_obj = Path(host_file_path)
            
            # Get relative path
            if file_path_obj.is_absolute():
                try:
                    relative_path = file_path_obj.relative_to(project_root)
                except ValueError:
                    return False, f"File {file_path} is outside project directory"
            else:
                relative_path = file_path_obj
            
            # Only allow modifications within environment directory
            env_path_prefix = Path("src") / "envs" / env_name
            env_prefix = str(env_path_prefix) + os.sep
            if str(relative_path).startswith(env_prefix) or str(relative_path).startswith(env_prefix.replace(os.sep, "/")):
                # Additional pattern checks for file types
                allowed_patterns = [
                    str(env_path_prefix / "**" / "*.py"),
                    str(env_path_prefix / "**" / "*.json"), 
                    str(env_path_prefix / "**" / "*.yaml"),
                    str(env_path_prefix / "**" / "*.yml")
                ]
                
                for pattern in allowed_patterns:
                    if fnmatch.fnmatch(str(relative_path), pattern):
                        return True, f"Environment file modification allowed: {relative_path}"
                
                return False, f"File type not allowed for modification: {relative_path}"
            
            # Explicitly block framework files
            framework_prefixes = [
                "src/onesim/",
                "src/main.py",
                "config/",
                "requirements.txt",
                "Dockerfile",
                "docker-compose"
            ]
            
            for prefix in framework_prefixes:
                if str(relative_path).startswith(prefix) or str(relative_path) == prefix:
                    return False, f"Framework file modification forbidden: {relative_path}"
            
            return False, f"File {relative_path} is not within allowed environment directory {env_prefix}"
            
        except Exception as e:
            return False, f"Error checking file path: {e}"
    
    def setup_debug_logging(self, env_name: str):
        """Setup detailed debug logging"""
        if self.config.detailed_logging:
            debug_log_file = self.config.get_debug_log_file(env_name)
            self.debug_logger = logger.bind(session_id=self.session_id)
            
            # Add file handler for detailed logging
            logger.add(
                debug_log_file,
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {extra[session_id]} | {name}:{function}:{line} - {message}",
                filter=lambda record: record["extra"].get("session_id") == self.session_id
            )
            
            self.debug_logger.info(f"Debug session started: {self.session_id}")
            self.debug_logger.info(f"Environment: {env_name}")
            self.debug_logger.info(f"Model: {self.config.model_config_name}")
            self.debug_logger.info(f"Max iterations: {self.config.max_iterations}")
    
    def log_step(self, step: str, details: Optional[Dict[str, Any]] = None):
        """Log a debugging step with details"""
        if self.debug_logger:
            step_info = f"STEP: {step}"
            if details:
                step_info += f" | Details: {json.dumps(details, ensure_ascii=False, indent=2)}"
            self.debug_logger.info(step_info)
    
    def log_llm_call(self, prompt_type: str, prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """Log LLM call details with optimized response logging"""
        if self.debug_logger and self.config.log_llm_calls:
            call_id = str(uuid.uuid4())[:8]
            
            self.debug_logger.info(f"LLM_CALL_START [{call_id}] Type: {prompt_type}")
            if metadata:
                self.debug_logger.info(f"LLM_CALL_META [{call_id}] {json.dumps(metadata, ensure_ascii=False)}")
            
            self.debug_logger.info(f"LLM_CALL_PROMPT [{call_id}]:\n{prompt}")
            
            # Extract meaningful response content - avoid logging full responses
            if response:
                try:
                    # Try to parse as JSON first
                    import json as json_lib
                    parsed_response = json_lib.loads(response)
                    self.debug_logger.info(f"LLM_CALL_RESPONSE [{call_id}]:\n{json.dumps(parsed_response, ensure_ascii=False, indent=2)}")
                except:
                    # If not JSON, log first 500 characters + indication if truncated
                    response_text = response.strip()
                    self.debug_logger.info(f"LLM_CALL_RESPONSE [{call_id}]:\n{response_text}")
            
            self.debug_logger.info(f"LLM_CALL_END [{call_id}]")
    
   
    
    async def auto_debug(self, env_name: str) -> DebuggingResult:
        """Main auto-debugging workflow"""
        
        # Setup detailed logging and environment tracking
        self.current_env_name = env_name  # Set current environment for safety checks
        self.setup_debug_logging(env_name)
        
        logger.info(f"Starting auto-debugging environment: {env_name}")
        self.log_step("Starting auto-debug", {"env_name": env_name, "session_id": self.session_id})
        
        # Prepare container
        self.log_step("Preparing Docker container")
        if not await self.docker_runner.prepare_container(env_name):
            logger.error(f"Failed to prepare container for environment: {env_name}")
            self.log_step("Container preparation failed", {"env_name": env_name})
            return DebuggingResult(
                success=False,
                iterations=0,
                final_error="Failed to prepare Docker container"
            )
        self.log_step("Container prepared successfully")
        
        # Initialize unified experience manager with dual repositories
        env_storage = self.config.get_env_specific_history_dir(env_name)
        global_storage = self.config.get_global_experience_storage_path() if self.config.enable_global_experience else None
        
        self.experience_manager = ExperienceManager(
            local_storage_path=env_storage,
            global_storage_path=global_storage,
            model=self.model,
            use_global_experience=self.config.enable_global_experience
        )
        logger.info(f"Experience manager initialized - Local: {env_storage}, Global: {global_storage}, Use Global: {self.config.enable_global_experience}")
        
        # Initialize components
        self.context_manager = ContextManager(model=self.model, debugging_agent=self)
        self.context_manager.set_experience_manager(self.experience_manager)
        self.log_step("Experience manager updated for environment", {"local_history_path": env_storage, "global_path": global_storage})
        
        # Main debugging loop
        for iteration in range(self.config.max_iterations):
            iteration_num = iteration + 1
            logger.info(f"Starting debugging iteration {iteration_num}")
            self.log_step(f"Starting iteration {iteration_num}")
            
            # 1. Run and detect errors
            self.log_step("Running simulation", {"iteration": iteration_num})
            run_result = await self.run_simulation(env_name)
            
            if run_result.success:
                logger.info("Simulation successful, debugging completed")
                self.log_step("Simulation successful - debugging completed", {
                    "iteration": iteration_num,
                    "total_iterations": iteration_num
                })
                
                # Extract global patterns after successful debugging
                if self.config.enable_global_experience:
                    await self.extract_global_patterns_after_debug(env_name)
                
                return DebuggingResult(success=True, iterations=iteration_num)
            
            self.log_step("Simulation failed, analyzing errors", {
                "return_code": run_result.return_code,
                "stderr_length": len(run_result.stderr),
                "stdout_length": len(run_result.stdout)
            })
            
            # 2. Process errors with enhanced pipeline
            self.log_step("Processing errors with enhanced pipeline")
            errors = await self.process_errors_enhanced(run_result)
            if not errors:
                logger.warning("Unable to parse error information")
                self.log_step("No parseable errors found", {
                    "stderr": run_result.stderr
                })
                break
            
            self.log_step("Errors parsed", {
                "error_count": len(errors),
                "error_types": [str(e.error_type) for e in errors]
            })
            
            # Process each error sequentially in this iteration
            self.log_step("Starting sequential error processing", {
                "total_errors_to_process": len(errors),
                "iteration": iteration_num
            })
            
            errors_fixed = 0
            
            # Inner loop to process all errors one by one
            for error_index, current_error in enumerate(errors):
                self.log_step(f"Processing error {error_index + 1}/{len(errors)}", {
                    "error_type": str(current_error.error_type),
                    "error_message": current_error.error_message,
                    "file_path": current_error.file_path,
                    "line_number": current_error.line_number
                })
                
                # Get repair context for current error
                self.log_step(f"Getting repair context for error {error_index + 1}")
                repair_context = await self.context_manager.get_repair_context(
                    current_error, self.get_code_structure(env_name)
                )
                
                # Add experience patterns to repair context
                if self.experience_manager:
                    pattern_guidance = self.experience_manager.get_pattern_guidance(current_error)
                    if pattern_guidance:
                        repair_context.experience_guidance = pattern_guidance
                        self.log_step(f"Experience guidance added for error {error_index + 1}", {
                            "guidance_length": len(pattern_guidance)
                        })
                
                self.log_step(f"Repair context obtained for error {error_index + 1}", {
                    "has_historical_fixes": len(repair_context.historical_fixes) > 0,
                    "selected_agents": repair_context.selected_context.selected_agents,
                    "selected_events": repair_context.selected_context.selected_events,
                    "has_experience_guidance": hasattr(repair_context, 'experience_guidance')
                })
                
                # Execute repair using adaptive strategy based on classification
                classification = self.get_error_classification(current_error)
                self.log_step(f"Executing repair for error {error_index + 1}", {
                    "complexity": classification.complexity,
                    "repair_strategy": classification.repair_strategy,
                    "confidence": classification.confidence
                })
                
                repair_success = False
                
                if classification.repair_strategy == "react" and self.config.enable_react_debugging and self.react_agent:
                    # Use ReAct approach for complex errors
                    self.log_step(f"Using ReAct debugging approach for complex error {error_index + 1}")
                    
                    react_state = await self.react_agent.debug_with_react(
                        current_error, 
                        repair_context
                    )
                    logger.info(f"ReAct state: {react_state}")
                    # 将ReAct结果转换为传统RepairResult格式
                    if react_state.success and react_state.fix_attempts:
                        repair_result = await self.convert_react_to_repair_result(
                            react_state, current_error
                        )
                    else:
                        repair_result = RepairResult(
                            success=False,
                            error_message="ReAct debugging failed to generate valid fix"
                        )
                else:
                    # 使用简单修复方式（字符串替换）
                    self.log_step(f"Using string replacement approach for simple error {error_index + 1}")
                    repair_result = await self.repairer.repair_with_replacement(
                        current_error, repair_context
                    )
                
                # 记录修复历史
                self.log_step(f"Recording repair attempt for error {error_index + 1}")
                await self.record_repair_attempt(current_error, repair_result)
                
                if not repair_result.success:
                    logger.warning(f"修复失败 for error {error_index + 1}: {repair_result.error_message}")
                    self.log_step(f"Repair failed for error {error_index + 1}", {
                        "error_message": repair_result.error_message,
                        "error_type": str(current_error.error_type)
                    })
                else:
                    logger.info(f"修复成功 for error {error_index + 1}: {repair_result.repaired_file}")
                    self.log_step(f"Repair successful for error {error_index + 1}", {
                        "repaired_file": repair_result.repaired_file,
                        "changes_made": repair_result.changes_made,
                        "error_type": str(current_error.error_type)
                    })
                    repair_success = True
                    errors_fixed += 1
                
                # Log completion of this error's processing
                self.log_step(f"Completed processing error {error_index + 1}/{len(errors)}", {
                    "success": repair_success,
                    "error_type": str(current_error.error_type)
                })
            
            # Log summary of this iteration's error processing
            self.log_step("Sequential error processing completed", {
                "total_errors_processed": len(errors),
                "errors_fixed": errors_fixed,
                "errors_remaining": len(errors) - errors_fixed,
                "iteration": iteration_num
            })
            
            # If all errors were fixed, we're done
            if errors_fixed == len(errors):
                logger.info(f"All {errors_fixed} errors fixed successfully")
                self.log_step("All errors fixed successfully", {
                    "total_errors_fixed": errors_fixed,
                    "iteration": iteration_num
                })
                
                # Extract global patterns after successful debugging
                if self.config.enable_global_experience:
                    await self.extract_global_patterns_after_debug(env_name)
                
                return DebuggingResult(success=True, iterations=iteration_num)
            elif errors_fixed > 0:
                logger.info(f"Partial success: {errors_fixed}/{len(errors)} errors fixed")
                self.log_step("Partial success - continuing to next iteration", {
                    "errors_fixed_this_iteration": errors_fixed,
                    "total_errors": len(errors)
                })
                # Continue to next iteration to re-run simulation and check remaining errors
            else:
                logger.warning("No errors were fixed in this iteration")
                self.log_step("No errors fixed in this iteration", {
                    "iteration": iteration_num,
                    "total_errors_attempted": len(errors)
                })
                # Continue to next iteration anyway - simulation might reveal different errors
        
        final_result = DebuggingResult(
            success=False,
            iterations=self.config.max_iterations,
            final_error="达到最大迭代次数",
            react_actions=getattr(self, '_react_actions', [])
        )
        self.log_step("Debugging completed", {
            "success": final_result.success,
            "total_iterations": final_result.iterations,
            "final_error": final_result.final_error
        })
        
        # 调试完成后自动提取全局模式（仅当启用全局经验时）
        if self.config.enable_global_experience:
            await self.extract_global_patterns_after_debug(env_name)
        
        return final_result
    
    async def convert_react_to_repair_result(self, react_state: ReActState, 
                                            error: ParsedError) -> RepairResult:
        """将ReAct结果转换为传统RepairResult格式"""
        
        if not react_state.fix_attempts:
            return RepairResult(
                success=False,
                error_message="No fix attempts from ReAct agent"
            )
        
        # 获取最后一个修复尝试
        last_fix = react_state.fix_attempts[-1]
        
        try:
            # 执行修复
            if 'repairs' in last_fix and last_fix['repairs']:
                repair_params = RepairParams(
                    repair_strategy=last_fix.get('repair_strategy', 'string_replacement'),
                    repairs=[
                        RepairItem(
                            file_path=repair.get('file_path', ''),
                            old_string=repair['old_string'],
                            new_string=repair['new_string'],
                            expected_replacements=repair.get('expected_replacements', 1),
                            reasoning=repair.get('reasoning', '')
                        )
                        for repair in last_fix['repairs']
                    ],
                    backup_strategy=last_fix.get('backup_strategy', '')
                )
                
                # 收集所有需要的文件内容
                files_content = {}
                for repair in repair_params.repairs:
                    if repair.file_path and repair.file_path not in files_content:
                        file_content = self.repairer.read_file(repair.file_path)
                        if file_content:
                            files_content[repair.file_path] = file_content
                
                if not files_content:
                    return RepairResult(
                        success=False,
                        error_message="Cannot read any target files for repair"
                    )
                
                result = await self.repairer.execute_multi_file_replacement(repair_params, files_content)
                
                # 记录ReAct动作
                if not hasattr(self, '_react_actions'):
                    self._react_actions = []
                self._react_actions.extend([
                    f"{action.action_type}({action.target})" 
                    for action in react_state.actions_taken
                ])
                
                return result
            else:
                return RepairResult(
                    success=False,
                    error_message="ReAct fix attempt has no repairs"
                )
        except Exception as e:
            return RepairResult(
                success=False,
                error_message=f"ReAct repair execution failed: {e}"
            )
    
    async def run_simulation(self, env_name: str) -> SimulationResult:
        """运行模拟并返回结果"""
        
        try:
            result = await self.docker_runner.run_simulation(env_name)
            return SimulationResult(
                success=result.success,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.return_code,
                execution_time=result.execution_time
            )
        except Exception as e:
            logger.error(f"Simulation run failed: {e}")
            return SimulationResult(success=False, stderr=str(e))
    
    async def process_errors_enhanced(self, run_result: SimulationResult) -> List[ParsedError]:
        """统一的错误处理流程：提取 -> 去重 -> 分类 -> 排序"""
        # 1. 错误提取 (保留LLM增强功能)
        error_parser = ErrorParser(model=self.model)
        raw_errors = await error_parser.parse_error_output_enhanced(
            run_result.stderr, run_result.stdout
        )
        
        if not raw_errors:
            # LLM fallback extraction
            raw_errors = await self._extract_errors_with_llm_fallback(run_result)
        
        if not raw_errors:
            return []
        
        # 2. 错误去重和分析
        deduplicated_errors = await self._deduplicate_and_analyze_errors(raw_errors, run_result)
        
        # 3. 分类和排序
        classified_errors = await self._classify_and_prioritize_errors(deduplicated_errors)
        
        self.log_step("Error processing completed", {
            "original_count": len(raw_errors),
            "deduplicated_count": len(classified_errors),
            "error_types": [str(e.error_type) for e in classified_errors]
        })
        
        return classified_errors
    
    async def _extract_errors_with_llm_fallback(self, run_result: SimulationResult) -> List[ParsedError]:
        """LLM fallback错误提取"""
        try:
            extraction_prompt = f"""
# 错误提取任务

## 运行输出
STDERR:
{run_result.stderr}

STDOUT:
{run_result.stdout}

## 任务
请从上述输出中提取所有错误信息，返回JSON格式：
{{
  "errors": [
    {{
      "error_type": "错误类型",
      "error_message": "错误信息",
      "file_path": "文件路径",
      "line_number": 行号,
      "function_name": "函数名",
      "code_context": "相关代码"
    }}
  ]
}}
"""
            
            formatted_prompt = self.model.format(Message("user", extraction_prompt, role="user"))
            result = await self.model.acall(formatted_prompt)
            
            parser = JsonBlockParser()
            parsed = parser.parse(result)
            
            errors = []
            if parsed.parsed:
                for error_data in parsed.parsed.get("errors", []):
                    errors.append(ParsedError(
                        error_type=error_data.get("error_type", "UnknownError"),
                        error_message=error_data.get("error_message", ""),
                        file_path=error_data.get("file_path", ""),
                        line_number=error_data.get("line_number", 0),
                        function_name=error_data.get("function_name", ""),
                        code_context=error_data.get("code_context", "")
                    ))
            
            return errors
            
        except Exception as e:
            logger.error(f"LLM fallback extraction failed: {e}")
            return []
    
    async def _deduplicate_and_analyze_errors(self, errors: List[ParsedError], run_result: SimulationResult) -> List[ParsedError]:
        """LLM去重和深度分析错误"""
        if len(errors) <= 1:
            return errors
        
        try:
            # 构建去重分析提示
            errors_summary = "\n".join([
                f"{i+1}. {e.error_type}: {e.error_message}\n{e.full_traceback}"
                for i, e in enumerate(errors)
            ])
            
            dedup_prompt = f"""
# 错误去重和分析任务

## 错误列表
{errors_summary}


## 任务
1. 识别重复或本质相同的错误
2. 分析错误的真实根源
3. 返回去重后的错误列表和分析

请返回JSON格式：
```json
{{
  "deduplicated_errors": [
    {{
      "index": 原始错误索引,
      "merged_indices": [合并的其他错误索引],
      "enhanced_message": "增强后的错误信息",
      "root_cause_analysis": "根因分析"
    }}
  ]
}}
```
"""
            
            formatted_prompt = self.model.format(Message("user", dedup_prompt, role="user"))
            result = await self.model.acall(formatted_prompt)
            
            parser = JsonBlockParser()
            parsed = parser.parse(result).parsed
            
            if self.debug_logger:
                self.debug_logger.info(f"Deduplication prompt: {dedup_prompt}")
                self.debug_logger.info(f"Deduplication result: {parsed}")
            # 构建去重后的错误列表
            deduplicated = []
            for dedup_info in parsed.get("deduplicated_errors", []):
                idx = dedup_info.get("index", 0)
                if 0 <= idx < len(errors):
                    original_error = errors[idx]
                    enhanced_message = dedup_info.get("enhanced_message", original_error.error_message)
                    root_cause = dedup_info.get("root_cause_analysis", "")
                    
                    # 处理框架文件路径：onesim/ -> src/onesim/
                    if original_error.file_path.startswith('onesim/'):
                        original_error.file_path = f"src/{original_error.file_path}"
                    # 增强错误信息
                    enhanced_error = ParsedError(
                        error_type=original_error.error_type,
                        error_message=f"{enhanced_message}\n\n根因分析: {root_cause}",
                        file_path=original_error.file_path,
                        line_number=original_error.line_number,
                        function_name=original_error.function_name,
                        code_context=original_error.code_context
                    )
                    deduplicated.append(enhanced_error)
            
            return deduplicated if deduplicated else errors
            
        except Exception as e:
            logger.error(f"Error deduplication failed: {e}")
            return errors
    
    async def _classify_and_prioritize_errors(self, errors: List[ParsedError]) -> List[ParsedError]:
        """LLM增强的错误分类和优先级排序"""
        if not errors:
            return []
        
        try:
            # 构建分类分析提示
            errors_summary = "\n".join([
                f"{i+1}. {e.error_type}: {e.error_message} (文件: {e.file_path})"
                for i, e in enumerate(errors)
            ])
            
            classification_prompt = f"""
# 错误分类和修复策略选择

## 分类标准

### Simple (使用 string_replacement):
- **单文件内修复**: 错误和修复都在同一个文件内，无需跨文件分析
- **明显的语法/导入错误**: 缺失导入、拼写错误、语法问题
- **直接的参数修复**: 明确知道缺失参数或参数错误
- **简单的类型错误**: 类型转换、简单的变量错误
- **数据类型处理错误**: 字符串/列表混用、类型转换等明确的类型问题
- **函数参数匹配错误**: 构造函数参数不匹配、参数顺序错误等

### Complex (使用 react):  
- **跨文件依赖**: 需要读取多个文件理解上下文
- **逻辑分析需求**: 需要理解代码逻辑才能确定修复方案
- **上下文依赖修复**: 需要分析调用关系、数据流等
- **不确定性高**: 错误信息模糊，需要探索性分析

## 错误列表
{errors_summary}

## 分类任务
对每个错误分析:
1. **信息完整性**: 当前错误信息是否足够定位和修复问题
2. **修复范围**: 修复是否需要跨文件或跨模块分析  
3. **上下文需求**: 是否需要理解复杂的调用关系或数据流
4. **确定性程度**: 修复方案的明确程度

## 特殊考虑
- 如果错误信息明确指出问题且修复方案直观 → simple
- 需要理解业务逻辑或多文件交互才能修复 → complex
- 错误涉及框架内部但可通过环境文件适配 → simple

返回JSON格式:
{{
  "classifications": [
    {{
      "index": 错误索引(从1开始),
      "complexity": "simple/complex", 
      "repair_strategy": "string_replacement/react",
      "priority_score": 1-10（数字越小优先级越高）,
      "confidence": 0.0-1.0,
      "reasoning": "详细的分类理由，说明为什么选择此策略"
    }}
  ]
}}
"""
            
            formatted_prompt = self.model.format(Message("user", classification_prompt, role="user"))
            result = await self.model.acall(formatted_prompt)
            
            parser = JsonBlockParser()
            parsed = parser.parse(result).parsed
            
            if self.debug_logger:
                self.debug_logger.info(f"Classification prompt: {classification_prompt}")
                self.debug_logger.info(f"Classification result: {parsed}")
            
            # 为每个错误添加分类信息
            classified_errors = []
            classifications = parsed.get("classifications", [])
            
            for i, error in enumerate(errors):
                # 找到对应的分类信息
                classification_info = None
                for cls in classifications:
                    if cls.get("index") == i + 1:
                        classification_info = cls
                        break
                
                # 使用分类信息或默认分类
                if classification_info:
                    priority_score = classification_info.get("priority_score", 5)
                    complexity = classification_info.get("complexity", "simple")
                    strategy = classification_info.get("repair_strategy", "string_replacement")
                else:
                    priority_score = self._get_default_priority(error.error_type.value)
                    complexity = "simple" if priority_score <= 3 else "complex"
                    strategy = "string_replacement" if complexity == "simple" else "react"
                
                # 创建分类对象
                classification = ErrorClassification(
                    error_type=error.error_type.value,
                    complexity=complexity,
                    repair_strategy=strategy,
                    confidence=classification_info.get("confidence", 0.7) if classification_info else 0.6,
                    reasoning=classification_info.get("reasoning", "Default classification") if classification_info else "Default classification"
                )
                
                # 将分类信息存储到错误对象中
                error.classification = classification
                error.priority_score = priority_score
                
                classified_errors.append(error)
            
            # 按优先级排序
            sorted_errors = sorted(classified_errors, key=lambda e: e.priority_score or 5)
            
            self.log_step("Error classification and prioritization completed", {
                "total_errors": len(sorted_errors),
                "simple_errors": len([e for e in sorted_errors if e.classification and e.classification.complexity == "simple"]),
                "complex_errors": len([e for e in sorted_errors if e.classification and e.classification.complexity == "complex"]),
                "priority_order": [str(e.error_type) for e in sorted_errors]
            })
            
            return sorted_errors
            
        except Exception as e:
            logger.error(f"Error classification failed: {e}")
            # Fallback to basic prioritization
            return self._basic_error_prioritization(errors)
    
    def _get_default_priority(self, error_type: str) -> int:
        """获取默认优先级分数"""
        priority_map = {
            'SyntaxError': 1,
            'IndentationError': 1, 
            'ImportError': 2,
            'NameError': 3,
            'AttributeError': 4,
            'TypeError': 5,
            'ValueError': 6,
            'LogError': 7
        }
        return priority_map.get(error_type, 8)
    
    def _basic_error_prioritization(self, errors: List[ParsedError]) -> List[ParsedError]:
        """基础错误优先级排序（fallback）"""
        return sorted(errors, key=lambda e: self._get_default_priority(e.error_type.value))
    
    def get_error_classification(self, error: ParsedError) -> ErrorClassification:
        """获取错误分类信息"""
        if error.classification:
            return error.classification
        
        # Fallback classification
        priority = self._get_default_priority(error.error_type.value)
        return ErrorClassification(
            error_type=error.error_type.value,
            complexity="simple" if priority <= 3 else "complex",
            repair_strategy="string_replacement" if priority <= 3 else "react",
            confidence=0.6,
            reasoning="Fallback classification based on error type"
        )
    
    
    def get_code_structure(self, env_name: str) -> Dict[str, Any]:
        """获取代码结构信息，包含生成上下文"""
        structure = {}
        
        try:
            env_path = Path("src") / "envs" / env_name
            
            # 读取actions.json
            actions_file = env_path / "actions.json"
            if actions_file.exists():
                with open(actions_file, 'r', encoding='utf-8') as f:
                    structure["agents"] = json.load(f)
            
            # 读取events.json
            events_file = env_path / "events.json"
            if events_file.exists():
                with open(events_file, 'r', encoding='utf-8') as f:
                    structure["events"] = json.load(f)
            
            # 读取description.txt (原始需求描述)
            scene_info_file = env_path / "scene_info.json"
            if scene_info_file.exists():
                with open(scene_info_file, 'r', encoding='utf-8') as f:
                    structure["scene_info"] = json.load(f)
                    structure["odd_protocol"] = structure["scene_info"]["odd_protocol"]
            
            # 构建动作图谱(用于理解工作流设计)
            if "agents" in structure and "events" in structure:
                structure["action_graph"] = self._build_action_graph(structure["agents"], structure["events"])
                structure["event_names"] = [event['event_name'] for event in structure["events"].values()]
                    
        except Exception as e:
            logger.warning(f"Failed to read code structure: {e}")
        
        return structure
    
    def _build_action_graph(self, actions: Dict[str, List[Dict]], events: Dict[int, Dict]) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        """构建动作图谱，用于理解agent间的交互关系"""
        action_graph = {}
        for event in events.values():
            from_agent = event['from_agent_type']
            from_action = event['from_action_name']
            to_agent = event['to_agent_type']
            to_action = event['to_action_name']
            key = (from_agent, from_action)
            action_graph.setdefault(key, []).append((to_agent, to_action))
        return action_graph
    
    async def record_repair_attempt(self, error: ParsedError,
                                  repair_result: RepairResult):
        """记录修复尝试"""
        
        # 生成更详细的推理信息
        reasoning = f"Repair attempt for {error.error_type.value} in {error.file_path}"
        if repair_result.success:
            reasoning += f" - Successfully applied {repair_result.changes_made} changes"
            if repair_result.repair_details:
                reasoning += f" using {repair_result.repair_details.get('repair_strategy', 'string_replacement')} strategy"
        else:
            reasoning += f" - Failed: {repair_result.error_message}"
        
        repair_record = RepairRecord(
            error_type=error.error_type.value,
            error_message=error.error_message,
            file_path=error.file_path or "",
            function_name=error.function_name or "",
            original_code=repair_result.original_code,
            fixed_code=repair_result.fixed_code if repair_result.success else "",
            repair_strategy="string_replacement",
            success=repair_result.success,
            timestamp=datetime.now(),
            context_hash=hashlib.md5(f"{error.file_path}:{error.line_number}".encode()).hexdigest(),
            reasoning=reasoning
        )
        
        await self.experience_manager.store_repair(repair_record)
        
        # 如果修复成功，记录模式使用情况
        if repair_result.success and self.experience_manager:
            # Record local pattern usage
            local_patterns = self.experience_manager.find_applicable_local_patterns(error)
            for pattern in local_patterns:
                self.experience_manager.record_local_pattern_usage(pattern.pattern_id, True)
                self.log_step("Recorded local pattern usage", {
                    "pattern_id": pattern.pattern_id,
                    "pattern_name": pattern.pattern_name,
                    "success": True
                })
            
            # Record global pattern usage if enabled
            if self.config.enable_global_experience:
                global_patterns = self.experience_manager.find_applicable_global_patterns(error)
                for pattern in global_patterns:
                    self.experience_manager.record_global_pattern_usage(pattern.pattern_id, True)
                    self.log_step("Recorded global pattern usage", {
                        "pattern_id": pattern.pattern_id,
                        "pattern_name": pattern.pattern_name,
                        "success": True
                    })
    
    async def extract_global_patterns_after_debug(self, env_name: str):
        """调试完成后提取全局模式"""
        
        if not self.experience_manager or not self.config.enable_global_experience:
            return
        
        try:
            # 获取本地修复记录
            local_records = self.experience_manager.load_records()
            
            # 转换为RepairRecord对象
            repair_records = []
            for record_dict in local_records:
                repair_record = self.experience_manager.record_from_dict(record_dict)
                repair_records.append(repair_record)
            
            # 提取并更新本地模式
            await self.experience_manager.learn_local_patterns_from_repairs(repair_records)
            
            # 从本地模式中抽象提取全局模式
            local_patterns = list(self.experience_manager.local_patterns.values())
            await self.experience_manager.learn_global_patterns_from_local_patterns(local_patterns)
            
            # 获取统计信息
            summary = self.experience_manager.get_summary()
            
            logger.info(f"Global patterns updated. Summary: {summary}")
            self.log_step("Global patterns extraction completed", summary)
            
        except Exception as e:
            logger.error(f"Failed to extract global patterns: {e}")
            self.log_step("Global patterns extraction failed", {"error": str(e)})
    
    def get_global_experience_summary(self) -> Dict[str, Any]:
        """获取全局经验库摘要"""
        if not self.experience_manager:
            return {"global_experience_enabled": False}
        
        summary = self.experience_manager.get_summary()
        summary["global_experience_enabled"] = True
        return summary
    
    def list_global_patterns(self) -> List[Dict[str, Any]]:
        """列出所有全局模式"""
        if not self.experience_manager:
            return []
        
        patterns_info = []
        for pattern in self.experience_manager.global_patterns.values():
            patterns_info.append({
                "pattern_id": pattern.pattern_id,
                "pattern_name": pattern.pattern_name,
                "description": pattern.description,
                "error_type": pattern.error_type,
                "success_rate": pattern.success_rate,
                "usage_count": pattern.usage_count,
                "created_at": pattern.created_at.isoformat(),
                "last_updated": pattern.last_updated.isoformat()
            })
        
        # 按使用次数排序
        patterns_info.sort(key=lambda x: x["usage_count"], reverse=True)
        return patterns_info