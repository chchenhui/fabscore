# -*- coding: utf-8 -*-
"""
Context Selection Utilities
Handles intelligent context selection and file reading for debugging.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from loguru import logger

from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from .debugging_data_structures import (
    SelectedContext, ExtractedContent, AvailableContext
)


class LLMContextSelector:
    """LLM-driven context selector - directly use LLM to decide the most relevant context"""
    
    def __init__(self, model, debugging_agent=None, max_context_tokens: int = 6000):
        self.model = model
        self.debugging_agent = debugging_agent
        self.max_context_tokens = max_context_tokens
        self.json_parser = JsonBlockParser()
    
    async def select_context(self, error, code_structure: Dict) -> SelectedContext:
        """Use LLM to intelligently select context"""
        
        # 1. Basic context collection
        available_context = self.collect_available_context(error, code_structure)
        
        # 2. LLM context selection
        context_selection_prompt = self.build_context_selection_prompt(
            error, available_context
        )
        
        try:
            formatted_prompt = self.model.format(
                Message("user", context_selection_prompt, role="user")
            )
            
            # Log LLM call if debugging agent is available
            if self.debugging_agent:
                self.debugging_agent.log_llm_call(
                    "context_selection",
                    context_selection_prompt,
                    "",  # Will be filled after response
                    {"error_type": str(error.error_type), "file_path": error.file_path}
                )
            
            selection_result = await self.model.acall(formatted_prompt)
            
            # Log response
            if self.debugging_agent:
                self.debugging_agent.log_llm_call(
                    "context_selection",
                    context_selection_prompt,
                    str(selection_result.text),
                    {"error_type": str(error.error_type), "file_path": error.file_path}
                )
            
            try:
                selected_items = self.json_parser.parse(selection_result).parsed.get('selected_context', {})
            except Exception as e:
                logger.warning(f"Failed to parse context selection: {e}")
                selected_items = {}
            
            # 3. Build final context
            final_context = self.build_final_context(
                selected_items, available_context
            )
            
            return final_context
            
        except Exception as e:
            logger.warning(f"LLM context selection failed: {e}, using fallback")
            return self.build_fallback_context(available_context)
    
    def collect_available_context(self, error, code_structure: Dict) -> AvailableContext:
        """Collect all available context information"""
        context = AvailableContext()
        
        # Error directly related file content
        if error.file_path:
            context.error_file = self.read_file_with_context(error.file_path, error.line_number)
        
        # Extract relevant information from code_structure
        context.agents = code_structure.get('agents', {})
        context.events = code_structure.get('events', {})
        
        # Discover files in environment
        if self.debugging_agent:
            env_files, config_files = self.discover_environment_files()
            context.environment_files = env_files
            context.config_files = config_files
        
        return context
    
    def read_file_with_context(self, file_path: str, line_number: Optional[int] = None) -> str:
        """读取文件内容，包含错误行周围的上下文"""
        try:
            # Convert container path to host path if necessary
            actual_file_path = file_path
            if self.debugging_agent:
                actual_file_path = self.debugging_agent.container_path_to_host_path(file_path)
            
            with open(actual_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if line_number is not None and 1 <= line_number <= len(lines):
                # 提供错误行前后10行的上下文
                start = max(0, line_number - 11)
                end = min(len(lines), line_number + 10)
                context_lines = lines[start:end]
                
                # 标记错误行
                for i, line in enumerate(context_lines):
                    line_num = start + i + 1
                    if line_num == line_number:
                        context_lines[i] = f">>> {line_num}: {line}"  # 标记错误行
                    else:
                        context_lines[i] = f"    {line_num}: {line}"
                
                return "".join(context_lines)
            else:
                return "".join(lines[:50])  # 返回前50行
                
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return ""
    
    
    def build_context_selection_prompt(self, error, available_context: AvailableContext) -> str:
        """构建上下文选择提示"""
        
        # Get current environment name from debugging agent
        env_name = getattr(self.debugging_agent, 'current_env_name', 'current_env') if self.debugging_agent else 'current_env'
        
        # Organize files by categories
        file_structure = self._organize_files_by_structure(available_context, env_name)
        
        prompt = f"""
You are a professional code debugging assistant. Please select the most relevant information from the following available context to help fix this error.

## Error Information
Error type: {error.error_type}
Error message: {error.error_message}
File path: {error.file_path}
Function name: {error.function_name}
Line number: {error.line_number}

## Environment Structure: src/envs/{env_name}/

### Core Implementation (/code/)
{file_structure['code_files']}

### Agent Configuration (/profile/)
{file_structure['profile_files']}

### System Configuration (root level)
{file_structure['config_files']}

## Error file content
```python
{available_context.error_file}
```

## System Definitions
### Available Agents: {list(available_context.agents.keys())}
### Available Events: {list(available_context.events.get('definitions', {}).keys()) if available_context.events else []}

## File Modification Rules
- **Can modify**: All files under src/envs/{env_name}/
- **Cannot modify**: Framework files (src/onesim/), system configs outside env
- **Target for metric/monitor errors**: src/envs/{env_name}/code/metrics/metrics.py

## Context Selection Task

Please select from the above information the content that is most helpful for understanding and fixing the error. Return in JSON format:

```json
{{
    "selected_context": {{
        "include_file_content": true,
        "selected_agents": ["Agent1", "Agent2"],
        "selected_events": ["Event1", "Event2"], 
        "files_to_read": ["src/envs/{env_name}/code/Agent.py", "src/envs/{env_name}/scene_info.json"],
        "reasoning": "Reason for selecting these files and contexts"
    }}
}}
```

**Selection Priority:**
1. Files directly mentioned in error traceback
2. Related agent/event definitions for understanding workflow  
3. Configuration files for understanding system setup
"""
        return prompt
    
    def _organize_files_by_structure(self, available_context: AvailableContext, env_name: str) -> Dict[str, str]:
        """Organize files by environment structure"""
        env_path = f"src/envs/{env_name}"
        
        # Categorize files
        code_files = []
        profile_files = []
        config_files = []
        
        for file_path in available_context.environment_files:
            full_path = f"{env_path}/{file_path}"
            if file_path.startswith('code/'):
                code_files.append(full_path)
        
        for file_path in available_context.config_files:
            full_path = f"{env_path}/{file_path}"
            if full_path.startswith('profile/'):
                profile_files.append(full_path)
            else:
                config_files.append(full_path)
        
        # Format descriptions
        code_desc = self._format_code_files(code_files)
        profile_desc = self._format_profile_files(profile_files)
        config_desc = self._format_config_files(config_files)
        
        return {
            'code_files': code_desc,
            'profile_files': profile_desc,
            'config_files': config_desc
        }
    
    def _format_code_files(self, files: List[str]) -> str:
        """Format code files with descriptions"""
        descriptions = []
        
        for file_path in files:
            if 'metrics/metrics.py' in file_path:
                descriptions.append(f"{file_path} - Monitor/metric calculations during simulation")
            elif file_path.endswith('Agent.py'):
                agent_name = file_path.split('/')[-1].replace('.py', '')
                descriptions.append(f"{file_path} - {agent_name} class and action handlers")
            elif 'events.py' in file_path:
                descriptions.append(f"{file_path} - All event class definitions")
            elif 'SimEnv.py' in file_path:
                descriptions.append(f"{file_path} - Simulation environment definition")
            else:
                descriptions.append(f"{file_path} - Code implementation")
        
        return "\n".join(descriptions) if descriptions else "No code files found"
    
    def _format_profile_files(self, files: List[str]) -> str:
        """Format profile files with descriptions"""
        descriptions = []
        
        for file_path in files:
            if '/data/' in file_path:
                descriptions.append(f"{file_path} - Agent profile data")
            elif '/schema/' in file_path:
                descriptions.append(f"{file_path} - Agent profile schema definition")
            else:
                descriptions.append(f"{file_path} - Agent profile configuration")
        
        return "\n".join(descriptions) if descriptions else "No profile files found"
    
    def _format_config_files(self, files: List[str]) -> str:
        """Format config files with descriptions"""
        descriptions = []
        
        for file_path in files:
            if 'scene_info.json' in file_path:
                descriptions.append(f"{file_path} - Simulation scenario and metric definitions")
            elif 'actions.json' in file_path:
                descriptions.append(f"{file_path} - Action definitions for workflow")
            elif 'events.json' in file_path:
                descriptions.append(f"{file_path} - Event definitions for workflow")
            elif 'system_data_model.json' in file_path:
                descriptions.append(f"{file_path} - Data models used in simulation")
            elif 'env_data.json' in file_path:
                descriptions.append(f"{file_path} - Environment data loaded before simulation")
            else:
                descriptions.append(f"{file_path} - Configuration file")
        
        return "\n".join(descriptions) if descriptions else "No config files found"
    
    def build_final_context(self, selected_items: Dict[str, Any], 
                           available_context: AvailableContext) -> SelectedContext:
        """构建最终上下文"""
        return SelectedContext(
            error_file=available_context.error_file if selected_items.get('include_file_content') else "",
            selected_agents=selected_items.get('selected_agents', []),
            selected_events=selected_items.get('selected_events', []),
            selected_related_files=selected_items.get('selected_related_files', []),
            files_to_read=selected_items.get('files_to_read', []),
            reasoning=selected_items.get('reasoning', "")
        )
    
    def build_fallback_context(self, available_context: AvailableContext) -> SelectedContext:
        """构建fallback上下文"""
        return SelectedContext(
            error_file=available_context.error_file,
            selected_agents=list(available_context.agents.keys())[:3],  # 前3个agents
            selected_events=list(available_context.events.get('definitions', {}).keys())[:3] if available_context.events else [],
            selected_related_files=available_context.environment_files[:3],  # 前3个环境文件
            files_to_read=available_context.environment_files[:3],  # 前3个环境文件
            reasoning="Fallback context selection due to LLM parsing failure"
        )
    
    def discover_environment_files(self) -> Tuple[List[str], List[str]]:
        """发现环境中的有用文件，按照预定义的模式主动获取，避免包含log等无关文件"""
        env_files = []
        config_files = []
        
        if not self.debugging_agent:
            return env_files, config_files
        
        try:
            env_name = getattr(self.debugging_agent, 'current_env_name', None)
            if not env_name:
                return env_files, config_files
            
            # 获取环境目录路径
            env_path = Path(self.debugging_agent.config.host_project_path) / "src" / "envs" / env_name
            if not env_path.exists():
                return env_files, config_files
            
            # 主动搜索 /code/ 目录下的 .py 文件
            code_path = env_path / "code"
            if code_path.exists() and code_path.is_dir():
                for file_path in code_path.rglob("*.py"):
                    if file_path.is_file() and self._is_useful_py_file(file_path):
                        rel_path = str(file_path.relative_to(env_path))
                        env_files.append(rel_path)
            
            # 主动搜索 /profile/ 目录下的schema文件
            profile_schema_path = env_path / "profile" / "schema"
            if profile_schema_path.exists() and profile_schema_path.is_dir():
                for file_path in profile_schema_path.rglob("*"):
                    if file_path.is_file() and self._is_useful_profile_file(file_path):
                        rel_path = str(file_path.relative_to(env_path))
                        if file_path.suffix == '.json':
                            config_files.append(rel_path)
            
            # 主动搜索根目录下的特定配置文件
            root_config_files = [
                'scene_info.json',
                'actions.json', 
                'events.json',
                'system_data_model.json',
                'env_data.json'
            ]
            
            for config_file in root_config_files:
                config_path = env_path / config_file
                if config_path.exists() and config_path.is_file():
                    config_files.append(config_file)
            
            # # 搜索根目录下其他可能的配置文件(排除log和临时文件)
            # for file_path in env_path.iterdir():
            #     if file_path.is_file() and self._is_useful_root_config_file(file_path):
            #         rel_path = str(file_path.relative_to(env_path))
            #         if rel_path not in config_files:  # 避免重复
            #             config_files.append(rel_path)
            
            logger.debug(f"Discovered {len(env_files)} Python files and {len(config_files)} config files in environment {env_name}")
            return sorted(env_files), sorted(config_files)
            
        except Exception as e:
            logger.error(f"Failed to discover environment files: {e}")
            return [], []
    
    def _is_useful_py_file(self, file_path: Path) -> bool:
        """判断是否是有用的Python文件，排除临时文件和log文件"""
        file_name = file_path.name
        
        # 排除临时文件和缓存文件
        if file_name.startswith('.') or file_name.startswith('__pycache__'):
            return False
        
        # 排除常见的临时和log文件模式
        exclude_patterns = [
            'log', 'temp', 'tmp', 'backup', 'bak', 
            'test_', 'debug_', 'run_', 'output_'
        ]
        
        file_name_lower = file_name.lower()
        for pattern in exclude_patterns:
            if pattern in file_name_lower:
                return False
        
        return True
    
    def _is_useful_profile_file(self, file_path: Path) -> bool:
        """判断是否是有用的profile文件"""
        file_name = file_path.name
        
        # 排除临时文件
        if file_name.startswith('.') or file_name.startswith('__pycache__'):
            return False
        
        # 只保留配置相关的文件
        if file_path.suffix == '.json':
            # 排除log文件
            if 'log' in file_name.lower():
                return False
            return True
        
        return False
    
    def _is_useful_root_config_file(self, file_path: Path) -> bool:
        """判断是否是有用的根目录配置文件"""
        file_name = file_path.name
        
        # 只考虑配置文件格式
        if file_path.suffix not in ['.json', '.yaml', '.yml', '.txt', '.md']:
            return False
        
        # 排除log文件和临时文件
        exclude_patterns = [
            'log', 'temp', 'tmp', 'backup', 'bak',
            'output', 'result', 'debug', 'test'
        ]
        
        file_name_lower = file_name.lower()
        for pattern in exclude_patterns:
            if pattern in file_name_lower:
                return False
        
        # 排除隐藏文件
        if file_name.startswith('.'):
            return False
        
        return True
    


class ContextualFileReader:
    """智能文件读取和上下文处理"""
    
    def __init__(self, model, debugging_agent):
        self.model = model
        self.debugging_agent = debugging_agent
        self.large_file_threshold = 1000
        self.context_window = 50
        # 存储提取的内容用作上下文和参考
        self.extracted_content_cache: Dict[str, ExtractedContent] = {}
        
    async def read_file_with_context(self, file_path: str, error_line: Optional[int] = None, context_window: Optional[int] = None) -> str:
        """读取文件并提供错误位置的上下文"""
        if context_window is None:
            context_window = self.context_window
            
        try:
            actual_path = self.debugging_agent.container_path_to_host_path(file_path)
            with open(actual_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
            
            lines = full_content.split('\n')
            
            # 如果是大文件且有错误行号，使用LLM指导读取
            if len(lines) > self.large_file_threshold and error_line:
                return await self._read_large_file_with_llm_guidance(file_path, error_line, full_content)
            elif error_line:
                # 小文件或有错误行号，返回上下文窗口
                return self._get_context_window(full_content, error_line, context_window)
            else:
                # 无错误行号，返回完整内容或截断内容
                if len(lines) > self.large_file_threshold:
                    return f"File too large ({len(lines)} lines). First {context_window} lines:\n" + \
                           '\n'.join(lines[:context_window])
                return full_content
                
        except Exception as e:
            return f"Failed to read {file_path}: {e}"
    
    async def _read_large_file_with_llm_guidance(self, file_path: str, error_line: int, content: str) -> str:
        """使用LLM指导大文件的相关部分读取"""
        
        lines = content.split('\n')
        error_context = self._get_context_window(content, error_line, 50)
        
        # 使用LLM分析需要哪些其他部分
        analysis_prompt = f"""
文件 {file_path} 第 {error_line} 行出现错误。

错误附近代码:
{error_context}

文件总长度: {len(lines)} 行

请分析为了理解和修复这个错误，还需要文件的哪些部分:
1. 函数定义位置
2. 相关类定义  
3. 导入语句
4. 相关变量定义

返回需要读取的行号范围，格式: start-end，最多3个范围。
只返回行号范围，不要其他解释。

示例输出:
1-20
45-67
120-150
"""
        
        try:
            prompt = self.model.format(
                Message("user", analysis_prompt, role="user")
            )
            response = await self.model.acall(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            additional_ranges = self._parse_line_ranges(response_text)
            
            # 组合所有需要的内容
            combined_content = f"# Error context around line {error_line}:\n{error_context}"
            
            for start, end in additional_ranges:
                start = max(1, start)
                end = min(len(lines), end)
                section = '\n'.join(lines[start-1:end])
                combined_content += f"\n\n# Lines {start}-{end}:\n{section}"
                
            return combined_content
            
        except Exception as e:
            # Fallback to error context only
            return error_context
    
    def _get_context_window(self, content: str, error_line: int, window: int) -> str:
        """获取错误位置的上下文窗口"""
        lines = content.split('\n')
        start = max(0, error_line - window // 2)
        end = min(len(lines), error_line + window // 2)
        
        context_lines = []
        for i in range(start, end):
            marker = " >> " if i + 1 == error_line else "    "
            context_lines.append(f"{i+1:4d}{marker}{lines[i]}")
            
        return '\n'.join(context_lines)
    
    def _parse_line_ranges(self, response: str) -> List[Tuple[int, int]]:
        """解析LLM返回的行号范围"""
        ranges = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if '-' in line and line.replace('-', '').replace(' ', '').isdigit():
                try:
                    start, end = line.split('-')
                    start = int(start.strip())
                    end = int(end.strip())
                    if start <= end:
                        ranges.append((start, end))
                except ValueError:
                    continue
                    
        return ranges[:3]  # 最多3个范围

    async def extract_relevant_content_with_llm(self, file_path: str, purpose: str, 
                                               max_ranges: int = 5) -> ExtractedContent:
        """
        使用LLM分析整个文件并提取相关内容
        Args:
            file_path: 文件路径
            purpose: 提取目的（比如"debug error", "understand function", etc.）
            max_ranges: 最大返回范围数量
        Returns:
            ExtractedContent: 包含提取的内容和行号范围
        """
        try:
            actual_path = self.debugging_agent.container_path_to_host_path(file_path)
            with open(actual_path, 'r', encoding='utf-8') as f:
                full_content = f.read()
            
            lines = full_content.split('\n')
            total_lines = len(lines)
            
            # 使用LLM分析文件并提取相关部分
            analysis_prompt = f"""
请分析以下文件内容并根据目的提取相关部分。

文件路径: {file_path}
分析目的: {purpose}
文件总行数: {total_lines}

文件内容:
{full_content}

请分析并提取与目的最相关的、完整的代码部分，例如整个函数或整个类。提取时请确保代码块的完整性。

返回格式示例：
```json
{{
    "summary": "对提取内容的简要说明",
    "ranges": [
        {{"start": 45, "end": 87, "description": "完整的函数定义"}},
        {{"start": xxx, "end": xxx, "description": "xxxx"}}
    ]
}}
```

注意：
- 最多返回{max_ranges}个范围
- 行号从1开始
- 只返回JSON格式，不要其他解释

"""
            
            prompt = self.model.format(
                Message("user", analysis_prompt, role="user")
            )
            response = await self.model.acall(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            # 解析LLM返回的JSON结果
            extracted_data = self._parse_extraction_response(response_text)
            
            # 提取实际内容
            content_sections = []
            extracted_ranges = []
            
            for range_info in extracted_data.get('ranges', []):
                start = max(1, range_info.get('start', 1))
                end = min(total_lines, range_info.get('end', start))
                
                if start <= end:
                    section = '\n'.join(lines[start-1:end])
                    content_sections.append(f"# Lines {start}-{end}: {range_info.get('description', 'Unknown')}\n{section}")
                    extracted_ranges.append((start, end))
            
            # 创建提取结果
            extracted_content = ExtractedContent(
                file_path=file_path,
                purpose=purpose,
                extracted_ranges=extracted_ranges,
                content_sections=content_sections,
                summary=extracted_data.get('summary', 'LLM extraction completed')
            )
            
            # 保存到缓存
            cache_key = f"{file_path}:{purpose}"
            self.extracted_content_cache[cache_key] = extracted_content
            
            return extracted_content
            
        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {e}")
            return ExtractedContent(
                file_path=file_path,
                purpose=purpose,
                summary=f"Failed to extract content: {e}"
            )
    
    def _parse_extraction_response(self, response: str) -> Dict[str, Any]:
        """解析LLM返回的提取结果"""
        try:
            # 尝试从响应中提取JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback: 尝试直接解析整个响应
                return json.loads(response.strip())
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM extraction response as JSON")
            # Fallback: 尝试从文本中提取行号范围
            ranges = self._parse_line_ranges(response)
            return {
                'summary': 'Extracted using fallback parsing',
                'ranges': [{'start': start, 'end': end, 'description': 'Auto-extracted'} 
                          for start, end in ranges]
            }
    
    def save_extracted_content(self, extracted_content: ExtractedContent) -> str:
        """保存提取的内容到缓存"""
        cache_key = f"{extracted_content.file_path}:{extracted_content.purpose}"
        self.extracted_content_cache[cache_key] = extracted_content
        return cache_key
    
    def get_saved_content(self, file_path: str, purpose: str) -> Optional[ExtractedContent]:
        """获取保存的提取内容"""
        cache_key = f"{file_path}:{purpose}"
        return self.extracted_content_cache.get(cache_key)
    
    def get_all_saved_content(self) -> Dict[str, ExtractedContent]:
        """获取所有保存的提取内容"""
        return self.extracted_content_cache.copy()
    
    def clear_saved_content(self, file_path: Optional[str] = None):
        """清理保存的内容"""
        if file_path:
            # 清理特定文件的内容
            keys_to_remove = [key for key in self.extracted_content_cache.keys() 
                             if key.startswith(f"{file_path}:")]
            for key in keys_to_remove:
                del self.extracted_content_cache[key]
        else:
            # 清理所有内容
            self.extracted_content_cache.clear()