# -*- coding: utf-8 -*-
"""
错误解析器模块
用于解析运行时错误并提取关键信息
"""

import re
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from loguru import logger
import json
from onesim.models import JsonBlockParser,UserMessage

if TYPE_CHECKING:
    from onesim.utils.debugging_data_structures import ErrorClassification

class ErrorType(Enum):
    """错误类型枚举"""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    NAME_ERROR = "name_error"
    ATTRIBUTE_ERROR = "attribute_error"
    TYPE_ERROR = "type_error"
    VALUE_ERROR = "value_error"
    KEY_ERROR = "key_error"
    INDEX_ERROR = "index_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT_ERROR = "timeout_error"
    LOG_ERROR = "log_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ParsedError:
    """解析后的错误信息"""
    error_type: ErrorType
    error_message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    function_name: Optional[str] = None
    code_context: Optional[str] = None
    full_traceback: str = ""
    suggestions: List[str] = None
    
    # Debugging metadata (added for classification and prioritization)
    classification: Optional['ErrorClassification'] = None
    priority_score: Optional[int] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class ErrorParser:
    """错误解析器"""
    
    def __init__(self, model=None):
        self.model = model
        # 常见错误模式的正则表达式
        self.error_patterns = {
            ErrorType.SYNTAX_ERROR: [
                r"SyntaxError: (.+)",
                r"IndentationError: (.+)",
                r"invalid syntax.*line (\d+)"
            ],
            ErrorType.IMPORT_ERROR: [
                r"ImportError: (.+)",
                r"ModuleNotFoundError: No module named '(.+)'",
                r"cannot import name '(.+)'"
            ],
            ErrorType.NAME_ERROR: [
                r"NameError: name '(.+)' is not defined",
                r"NameError: (.+)"
            ],
            ErrorType.ATTRIBUTE_ERROR: [
                r"AttributeError: '(.+)' object has no attribute '(.+)'",
                r"AttributeError: (.+)"
            ],
            ErrorType.TYPE_ERROR: [
                r"TypeError: (.+)",
                r"unsupported operand type\(s\) for (.+)"
            ],
            ErrorType.VALUE_ERROR: [
                r"ValueError: (.+)"
            ],
            ErrorType.KEY_ERROR: [
                r"KeyError: '(.+)'",
                r"KeyError: (.+)"
            ],
            ErrorType.INDEX_ERROR: [
                r"IndexError: (.+)"
            ]
        }
        
        # 文件路径和行号的正则表达式
        self.traceback_pattern = r'File "(.+)", line (\d+)(?:, in (.+))?'
        self.code_context_pattern = r'^\s*(.+)$'
        
        # Log error regex patterns
        self.log_error_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) \| (ERROR|CRITICAL) +\| ([^:]+):([^:]+):(\d+) - (.+)'
        # Alternative log patterns for different formats
        self.alt_log_patterns = [
            r'(ERROR|CRITICAL)\s+\|\s+([^:]+):([^:]+):(\d+)\s+-\s+(.+)',  # Without timestamp
            r'\[([^\]]+)\]\s+(ERROR|CRITICAL):\s+(.+)',  # Standard format like [timestamp] ERROR: message
            r'(ERROR|CRITICAL):\s+(.+)',  # Simple ERROR: message format
        ]
        
    def parse_error_output(self, stderr: str, stdout: str = "") -> List[ParsedError]:
        """
        解析错误输出并提取错误信息
        
        Args:
            stderr: 标准错误输出
            stdout: 标准输出
            
        Returns:
            List[ParsedError]: 解析后的错误列表
        """
        errors = []
        
        # 首先尝试解析日志错误
        if stderr:
            log_errors = self._parse_log_errors(stderr)
            errors.extend(log_errors)
            
        # 尝试解析Python异常
        if stderr:
            python_errors = self._parse_python_traceback(stderr)
            errors.extend(python_errors)
        
        # 解析其他类型的错误信息
        if not errors:
            other_errors = self._parse_other_errors(stderr, stdout)
            errors.extend(other_errors)
            
        # 如果没有找到任何错误但有stderr输出，创建一个通用错误
        if not errors and stderr.strip():
            errors.append(ParsedError(
                error_type=ErrorType.RUNTIME_ERROR,
                error_message=stderr.strip(),
                full_traceback=stderr
            ))
        return errors
    
    def _parse_log_errors(self, stderr: str) -> List[ParsedError]:
        """Parse loguru-style log errors"""
        errors = []
        lines = stderr.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Try primary log error pattern first
            match = re.match(self.log_error_pattern, line)
            if match:
                _, _, module, function, line_num, message = match.groups()
                
                # Parse error information
                error_type, parsed_message = self._parse_log_error_message(message)
                if ('agent_registry' in parsed_message) or ('registered' in parsed_message):
                    continue

                # Construct file path
                file_path = module.replace('.', '/') + '.py'
                
                error = ParsedError(
                    error_type=error_type,
                    error_message=parsed_message,
                    file_path=file_path,
                    line_number=int(line_num),
                    function_name=function,
                    full_traceback=line
                )
                
                # Generate fix suggestions
                error.suggestions = self._generate_log_error_suggestions(error, message)
                errors.append(error)
                continue
            
            # Try alternative log patterns
            for alt_pattern in self.alt_log_patterns:
                alt_match = re.match(alt_pattern, line)
                if alt_match:
                    groups = alt_match.groups()
                    
                    # Handle different pattern formats
                    if len(groups) >= 5:  # Pattern with module, function, line_num
                        _, module, function, line_num, message = groups
                        file_path = module.replace('.', '/') + '.py'
                        line_number = int(line_num)
                    elif len(groups) == 3:  # Pattern like [timestamp] ERROR: message
                        _, _, message = groups
                        file_path = None
                        line_number = None
                        function = None
                    else:  # Simple ERROR: message format
                        _, message = groups
                        file_path = None
                        line_number = None
                        function = None
                    
                    # Parse error information
                    error_type, parsed_message = self._parse_log_error_message(message)
                    if 'agent_registry' in parsed_message:
                        continue

                    error = ParsedError(
                        error_type=error_type,
                        error_message=parsed_message,
                        file_path=file_path,
                        line_number=line_number,
                        function_name=function,
                        full_traceback=line
                    )
                    
                    # Generate fix suggestions
                    error.suggestions = self._generate_log_error_suggestions(error, message)
                    errors.append(error)
                    break
        
        return errors
    
    def _parse_log_error_message(self, message: str) -> Tuple[ErrorType, str]:
        """Parse log error message to extract error type and core information"""
        # Common Python error patterns in log messages
        python_error_patterns = {
            ErrorType.TYPE_ERROR: [
                r'missing \d+ required.*argument',
                r'takes \d+ positional argument',
                r'TypeError',
                r'unsupported operand type',
                r'argument of type .* is not iterable'
            ],
            ErrorType.NAME_ERROR: [
                r'name \'.*\' is not defined',
                r'NameError',
                r'global name .* is not defined'
            ],
            ErrorType.ATTRIBUTE_ERROR: [
                r'has no attribute',
                r'AttributeError',
                r'object has no attribute',
                r'module .* has no attribute'
            ],
            ErrorType.KEY_ERROR: [
                r'KeyError',
                r'key not found',
                r'dictionary key .* not found'
            ],
            ErrorType.VALUE_ERROR: [
                r'ValueError',
                r'invalid literal',
                r'could not convert string to float',
                r'invalid value'
            ],
            ErrorType.INDEX_ERROR: [
                r'IndexError',
                r'list index out of range',
                r'tuple index out of range',
                r'string index out of range'
            ],
            ErrorType.IMPORT_ERROR: [
                r'ImportError',
                r'ModuleNotFoundError',
                r'No module named',
                r'cannot import name'
            ]
        }
        
        # Check for specific Python error types
        for error_type, patterns in python_error_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message, re.IGNORECASE):
                    return error_type, message
        
        # Common application-level error patterns
        if re.search(r'(calculation|computing|calculating).*error', message, re.IGNORECASE):
            return ErrorType.RUNTIME_ERROR, message
        
        if re.search(r'(connection|network|timeout).*error', message, re.IGNORECASE):
            return ErrorType.TIMEOUT_ERROR, message
        
        # Default to LOG_ERROR for unmatched patterns
        return ErrorType.LOG_ERROR, message
    
    def _generate_log_error_suggestions(self, error: ParsedError, original_message: str) -> List[str]:
        """Generate fix suggestions for log errors"""
        suggestions = []
        
        # Generate suggestions based on error type
        if error.error_type == ErrorType.TYPE_ERROR:
            if 'missing' in original_message and 'argument' in original_message:
                suggestions.extend([
                    "Check if function call is missing required arguments",
                    "Verify the number and type of parameters passed",
                    "Review function definition parameter signature"
                ])
            else:
                suggestions.extend([
                    "Check if parameter types are correct",
                    "Verify function call syntax",
                    "Ensure compatible data types for operations"
                ])
        
        elif error.error_type == ErrorType.KEY_ERROR:
            suggestions.extend([
                "Check if dictionary key exists before accessing",
                "Use .get() method for safe dictionary access",
                "Validate data structure integrity",
                "Add key existence checks in code"
            ])
        
        elif error.error_type == ErrorType.ATTRIBUTE_ERROR:
            suggestions.extend([
                "Check if object has the required attribute",
                "Verify object type is correct",
                "Ensure object is properly initialized",
                "Add attribute existence checks"
            ])
        
        elif error.error_type == ErrorType.NAME_ERROR:
            suggestions.extend([
                "Check if variable is defined before use",
                "Verify variable name spelling",
                "Ensure variable is in correct scope",
                "Check for import statements"
            ])
        
        elif error.error_type == ErrorType.VALUE_ERROR:
            suggestions.extend([
                "Validate input data format",
                "Check data conversion operations",
                "Add input validation checks",
                "Handle edge cases in data processing"
            ])
        
        elif error.error_type == ErrorType.INDEX_ERROR:
            suggestions.extend([
                "Check list/array bounds before accessing",
                "Validate index values",
                "Add boundary checks",
                "Handle empty collection cases"
            ])
        
        elif error.error_type == ErrorType.IMPORT_ERROR:
            suggestions.extend([
                "Check if module is installed",
                "Verify module name spelling",
                "Check import path correctness",
                "Resolve circular import issues"
            ])
        
        elif error.error_type == ErrorType.RUNTIME_ERROR:
            suggestions.extend([
                "Add error handling for runtime conditions",
                "Validate input data before processing",
                "Check system resource availability",
                "Add logging for debugging"
            ])
        
        elif error.error_type == ErrorType.TIMEOUT_ERROR:
            suggestions.extend([
                "Increase timeout duration",
                "Optimize operation performance",
                "Add retry mechanism",
                "Check network connectivity"
            ])
        
        elif error.error_type == ErrorType.LOG_ERROR:
            suggestions.extend([
                "Check log context information",
                "Validate related data integrity",
                "Review detailed error stack trace",
                "Add more specific error handling"
            ])
        
        # Add specific suggestions based on message content
        if 'safe_get' in original_message:
            suggestions.append("Check safe_get function usage - ensure 'key' parameter is provided")
        
        if 'context' in original_message.lower():
            suggestions.append("Review error context information for debugging clues")
        
        if 'calculation' in original_message.lower() or 'computing' in original_message.lower():
            suggestions.extend([
                "Add input validation before calculations",
                "Handle division by zero cases",
                "Check for null/empty data in calculations"
            ])
        
        # Add function-specific suggestions
        if error.function_name:
            suggestions.append(f"Add more detailed logging in function {error.function_name}")
        
        return suggestions
    
    def _parse_python_traceback(self, stderr: str) -> List[ParsedError]:
        """解析Python异常回溯信息"""
        errors = []
        
        # 分割多个traceback
        traceback_blocks = self._split_tracebacks(stderr)
        
        for block in traceback_blocks:
            error = self._parse_single_traceback(block)
            if error:
                errors.append(error)
                
        return errors
    
    def _split_tracebacks(self, stderr: str) -> List[str]:
        """分割多个traceback块"""
        # 简单的分割方法，基于"Traceback"关键字
        blocks = []
        current_block = []
        
        lines = stderr.split('\n')
        
        for line in lines:
            if line.strip().startswith('Traceback'):
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
            current_block.append(line)
            
        if current_block:
            blocks.append('\n'.join(current_block))
            
        return [block for block in blocks if block.strip()]
    
    def _parse_single_traceback(self, traceback_text: str) -> Optional[ParsedError]:
        """解析单个traceback"""
        lines = traceback_text.split('\n')
        
        # 查找错误类型和消息
        error_line = None
        for line in reversed(lines):
            line = line.strip()
            if line and ':' in line and any(
                line.startswith(error_name) for error_name in [
                    'SyntaxError', 'ImportError', 'ModuleNotFoundError',
                    'NameError', 'AttributeError', 'TypeError', 'ValueError',
                    'KeyError', 'IndexError', 'RuntimeError', 'Exception'
                ]
            ):
                error_line = line
                break
        
        if not error_line:
            return None
            
        # 解析错误类型和消息
        error_type, error_message = self._classify_error(error_line)
        
        # 查找文件信息
        file_info = self._extract_file_info(traceback_text)
        
        error = ParsedError(
            error_type=error_type,
            error_message=error_message,
            file_path=file_info.get('file_path'),
            line_number=file_info.get('line_number'),
            function_name=file_info.get('function_name'),
            code_context=file_info.get('code_context'),
            full_traceback=traceback_text
        )
        
        # 生成修复建议
        error.suggestions = self._generate_suggestions(error)
        
        return error
    
    def _classify_error(self, error_line: str) -> Tuple[ErrorType, str]:
        """分类错误并提取消息"""
        # 按照优先级检查错误类型
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, error_line, re.IGNORECASE)
                if match:
                    message = match.group(1) if match.groups() else error_line
                    return error_type, message.strip()
        
        # 默认分类
        if ':' in error_line:
            parts = error_line.split(':', 1)
            message = parts[1].strip() if len(parts) > 1 else error_line
        else:
            message = error_line
            
        return ErrorType.UNKNOWN_ERROR, message
    
    def _extract_file_info(self, traceback_text: str) -> Dict[str, Any]:
        """从traceback中提取文件信息"""
        file_info = {}
        lines = traceback_text.split('\n')
        
        # 查找最后一个有效的文件引用
        for i, line in enumerate(lines):
            match = re.search(self.traceback_pattern, line)
            if match:
                file_info['file_path'] = match.group(1)
                file_info['line_number'] = int(match.group(2))
                if match.group(3):
                    file_info['function_name'] = match.group(3)
                
                # 尝试获取代码上下文
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith(('File', 'Traceback')):
                        file_info['code_context'] = next_line
        
        return file_info
    
    async def _enhance_error_with_llm(self, error: ParsedError, original_traceback: str) -> ParsedError:
        """
        Use LLM to enhance error parsing when rule-based parsing fails
        Only enhances if critical info like file_path is missing
        """
        if not self.model:
            logger.debug("No model available for LLM-enhanced error parsing")
            return error
            
        # Check if we need LLM enhancement (only when critical info is missing)
        needs_enhancement = (
            not error.file_path or 
            error.file_path == "" or
            not error.line_number or
            error.error_type == ErrorType.UNKNOWN_ERROR
        )
        
        if not needs_enhancement:
            logger.debug("Error parsing successful, no LLM enhancement needed")
            return error
            
        logger.info("Enhancing error parsing with LLM due to missing file_path or other critical info")
        
        try:
            # Prepare optimized LLM prompt for error analysis
            prompt = f"""Analyze the following Python error output and extract key information. Return the result in JSON format.

Error Output:
```
{original_traceback}
```

Extract and return the following information in JSON format:
- "file_path": The file path where the error occurred (if available)
- "line_number": The line number where the error occurred (if available, as integer)
- "error_type": The error type (e.g., "syntax_error", "import_error", "name_error", "attribute_error", "type_error", "value_error", "key_error", "index_error", "runtime_error", "timeout_error", "unknown_error")
- "error_message": A concise error description
- "function_name": The function name where the error occurred (if available)
- "code_context": The relevant code snippet (if available)
- "suggestions": A list of fix suggestions

Return the result in the following format:
```json
{{
  "file_path": "/path/to/file.py",
  "line_number": 42,
  "error_type": "name_error",
  "error_message": "name 'variable' is not defined",
  "function_name": "my_function",
  "code_context": "print(variable)",
  "suggestions": ["Check if variable is defined", "Check variable spelling"]
}}
```

Only return the JSON, no additional explanation."""
            
            # Format the prompt as a Message
            formatted_messages = self.model.format(UserMessage(prompt))
            
            # Call LLM
            logger.debug(f"Calling LLM for enhanced error parsing")
            response = await self.model.acall(formatted_messages)
            
            # Parse LLM response using JsonBlockParser
            try:
                parser = JsonBlockParser()
                llm_result = parser.parse(response).parsed
                
                if not llm_result:
                    logger.warning("JsonBlockParser failed to extract JSON from LLM response")
                    return error
                
                # Update error with LLM results if they're better than existing
                if llm_result.get('file_path') and not error.file_path:
                    error.file_path = llm_result['file_path']
                    
                if llm_result.get('line_number') and not error.line_number:
                    try:
                        error.line_number = int(llm_result['line_number'])
                    except (ValueError, TypeError):
                        pass
                        
                if llm_result.get('function_name') and not error.function_name:
                    error.function_name = llm_result['function_name']
                    
                if llm_result.get('code_context') and not error.code_context:
                    error.code_context = llm_result['code_context']
                    
                # Update error type if it was unknown
                if error.error_type == ErrorType.UNKNOWN_ERROR and llm_result.get('error_type'):
                    try:
                        error.error_type = ErrorType(llm_result['error_type'])
                    except ValueError:
                        pass  # Keep original if not a valid enum value
                        
                # Enhance error message if LLM provides a better one
                if llm_result.get('error_message') and len(llm_result['error_message']) < len(error.error_message):
                    error.error_message = llm_result['error_message']
                    
                # Add LLM suggestions
                if llm_result.get('suggestions') and isinstance(llm_result['suggestions'], list):
                    error.suggestions.extend(llm_result['suggestions'])
                    
                logger.info(f"Successfully enhanced error parsing with LLM: file_path={error.file_path}, line_number={error.line_number}")
                
            except Exception as e:
                logger.warning(f"Error processing LLM response with JsonBlockParser: {e}")
                
        except Exception as e:
            logger.error(f"Failed to enhance error parsing with LLM: {e}")
            
        return error
    
    def _parse_other_errors(self, stderr: str, stdout: str = "") -> List[ParsedError]:
        """解析非Python异常的其他错误"""
        errors = []
        
        # 检查超时错误
        if "timeout" in stderr.lower() or "time out" in stderr.lower():
            errors.append(ParsedError(
                error_type=ErrorType.TIMEOUT_ERROR,
                error_message="执行超时",
                full_traceback=stderr,
                suggestions=["增加超时时间", "优化代码性能", "检查是否存在死循环"]
            ))
        
        # 检查Docker相关错误
        if "docker" in stderr.lower():
            errors.append(ParsedError(
                error_type=ErrorType.RUNTIME_ERROR,
                error_message="Docker执行错误",
                full_traceback=stderr,
                suggestions=["检查Docker容器状态", "确认容器权限", "检查挂载路径"]
            ))
        
        return errors
    
    def _generate_suggestions(self, error: ParsedError) -> List[str]:
        """根据错误类型生成修复建议"""
        suggestions = []
        
        if error.error_type == ErrorType.IMPORT_ERROR:
            if "module named" in error.error_message.lower():
                module_name = re.search(r"'([^']+)'", error.error_message)
                if module_name:
                    suggestions.extend([
                        f"检查模块 {module_name.group(1)} 是否已安装",
                        "检查模块名称拼写是否正确",
                        "确认模块路径是否正确"
                    ])
            else:
                suggestions.extend([
                    "检查导入语句的语法",
                    "确认被导入的对象是否存在",
                    "检查循环导入问题"
                ])
        
        elif error.error_type == ErrorType.NAME_ERROR:
            var_name = re.search(r"'([^']+)'", error.error_message)
            if var_name:
                suggestions.extend([
                    f"检查变量 {var_name.group(1)} 是否已定义",
                    "检查变量名拼写是否正确",
                    "确认变量作用域是否正确"
                ])
        
        elif error.error_type == ErrorType.ATTRIBUTE_ERROR:
            match = re.search(r"'([^']+)' object has no attribute '([^']+)'", error.error_message)
            if match:
                obj_type, attr_name = match.groups()
                suggestions.extend([
                    f"检查 {obj_type} 对象是否有 {attr_name} 属性",
                    f"检查 {attr_name} 属性名拼写是否正确",
                    f"确认 {obj_type} 对象是否为预期类型"
                ])
        
        elif error.error_type == ErrorType.TYPE_ERROR:
            suggestions.extend([
                "检查函数参数类型是否正确",
                "确认操作数类型是否兼容",
                "检查函数调用的参数数量"
            ])
        
        elif error.error_type == ErrorType.SYNTAX_ERROR:
            suggestions.extend([
                "检查语法错误的行",
                "确认括号、引号是否匹配",
                "检查缩进是否正确"
            ])
        
        elif error.error_type == ErrorType.KEY_ERROR:
            key_name = re.search(r"'([^']+)'", error.error_message)
            if key_name:
                suggestions.extend([
                    f"检查字典是否包含键 {key_name.group(1)}",
                    "使用 get() 方法安全获取字典值",
                    "检查键名拼写是否正确"
                ])
        
        elif error.error_type == ErrorType.LOG_ERROR:
            suggestions.extend([
                "Check log context information",
                "Validate related data integrity",
                "Review detailed error stack trace",
                "Add more specific error handling"
            ])
        
        return suggestions
    
    async def parse_error_output_enhanced(self, stderr: str, stdout: str = "") -> List[ParsedError]:
        """
        Enhanced error parsing that uses LLM when rule-based parsing fails
        Only applies LLM enhancement to errors with missing critical information
        
        Args:
            stderr: 标准错误输出
            stdout: 标准输出
            
        Returns:
            List[ParsedError]: 解析后的错误列表（LLM增强）
        """
        # First try rule-based parsing
        errors = self.parse_error_output(stderr, stdout)
        
        if not self.model:
            logger.debug("No model available for LLM enhancement, returning rule-based results")
            return errors
        
        # Enhance only errors that need LLM assistance (missing critical info)
        enhanced_errors = []
        for error in errors:
            # Check if this error needs enhancement
            needs_enhancement = (
                not error.file_path or 
                error.file_path == "" or
                not error.line_number or
                error.error_type == ErrorType.UNKNOWN_ERROR
            )
            
            if needs_enhancement:
                enhanced_error = await self._enhance_error_with_llm(error, stderr)
                if enhanced_error.file_path.startswith("/app"):
                    enhanced_error.file_path = enhanced_error.file_path[len("/app"):]
                enhanced_errors.append(enhanced_error)
            else:
                # No enhancement needed, use original
                if error.file_path.startswith("/app/"):
                    error.file_path = error.file_path[len("/app/"):]
                enhanced_errors.append(error)
            
        return enhanced_errors
    
    def extract_relevant_context(self, error: ParsedError, 
                               source_code: str) -> Dict[str, Any]:
        """提取错误相关的代码上下文"""
        context = {
            'error_line': None,
            'surrounding_lines': [],
            'function_context': None
        }
        
        if not error.line_number or not source_code:
            return context
        
        lines = source_code.split('\n')
        error_line_idx = error.line_number - 1  # 转换为0索引
        
        if 0 <= error_line_idx < len(lines):
            context['error_line'] = lines[error_line_idx]
            
            # 获取周围的行
            start = max(0, error_line_idx - 5)
            end = min(len(lines), error_line_idx + 5)
            context['surrounding_lines'] = [
                (i + 1, lines[i]) for i in range(start, end)
            ]
            
            # 尝试找到函数上下文
            context['function_context'] = self._find_function_context(
                lines, error_line_idx
            )
        
        return context
    
    def _find_function_context(self, lines: List[str], 
                             error_line_idx: int) -> Optional[str]:
        """查找错误所在的函数上下文"""
        # 向上查找函数定义
        for i in range(error_line_idx, -1, -1):
            line = lines[i].strip()
            if line.startswith('def ') or line.startswith('async def '):
                return line
        return None


# 便捷函数
def parse_docker_error(stderr: str, stdout: str = "") -> List[ParsedError]:
    """
    快速解析Docker运行错误的便捷函数
    
    Args:
        stderr: 标准错误输出
        stdout: 标准输出
        
    Returns:
        List[ParsedError]: 解析后的错误列表
    """
    parser = ErrorParser()
    return parser.parse_error_output(stderr, stdout)


async def parse_docker_error_enhanced(stderr: str, stdout: str = "", model=None) -> List[ParsedError]:
    """
    快速解析Docker运行错误的便捷函数（LLM增强）
    
    Args:
        stderr: 标准错误输出
        stdout: 标准输出
        model: LLM模型客户端
        
    Returns:
        List[ParsedError]: 解析后的错误列表
    """
    parser = ErrorParser(model=model)
    return await parser.parse_error_output_enhanced(stderr, stdout)


def get_error_summary(errors: List[ParsedError]) -> str:
    """
    获取错误摘要的便捷函数
    
    Args:
        errors: 错误列表
        
    Returns:
        str: 错误摘要
    """
    if not errors:
        return "没有发现错误"
    
    summary_lines = []
    for i, error in enumerate(errors, 1):
        summary_lines.append(f"错误 {i}: {error.error_type.value}")
        summary_lines.append(f"  消息: {error.error_message}")
        if error.file_path:
            summary_lines.append(f"  文件: {error.file_path}")
        if error.line_number:
            summary_lines.append(f"  行号: {error.line_number}")
        if error.suggestions:
            summary_lines.append(f"  建议: {'; '.join(error.suggestions[:2])}")
        summary_lines.append("")
    
    return '\n'.join(summary_lines)