# -*- coding: utf-8 -*-
"""
Repair Corrector Utilities
Handles intelligent correction of mismatched repair operations.
"""

from typing import Optional, Dict
from loguru import logger

from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from .debugging_data_structures import RepairItem


class IntelligentCorrector:
    """智能纠错器 - 处理字符串匹配失败的情况"""
    
    def __init__(self, model):
        self.model = model
        self.json_parser = JsonBlockParser()
    
    async def correct_mismatch(self, repair: RepairItem,
                             file_content: str) -> Optional[RepairItem]:
        """纠正匹配失败的修复"""
        
        # 1. 转义修复
        unescaped_old_string = self.unescape_string(repair.old_string)
        if self.count_occurrences(file_content, unescaped_old_string) > 0:
            return RepairItem(
                file_path=repair.file_path,
                old_string=unescaped_old_string,
                new_string=repair.new_string,
                expected_replacements=repair.expected_replacements
            )
        
        # 2. LLM辅助修正
        corrected_repair = await self.llm_correct_mismatch(repair, file_content)
        
        return corrected_repair
    
    def unescape_string(self, string: str) -> str:
        """修正过度转义的字符串"""
        return string.replace('\\\\n', '\n').replace('\\\\t', '\t').replace('\\\\"', '"')
    
    def count_occurrences(self, content: str, substring: str) -> int:
        """计算子串出现次数"""
        return content.count(substring)
    
    async def llm_correct_mismatch(self, repair: RepairItem,
                                 file_content: str) -> Optional[RepairItem]:
        """使用LLM修正无法匹配的代码片段"""
        
        correction_prompt = f"""
无法在文件中找到完全匹配的代码片段。请帮助修正匹配。

## 要匹配的代码片段
```python
{repair.old_string}
```

## 实际文件内容
```python
{file_content}
```

请找到文件中最相似的代码片段，返回修正后的匹配参数：

```json
{{
    "corrected_old_string": "实际文件中的代码片段",
    "reasoning": "修正原因"
}}
```
"""
        
        try:
            formatted_prompt = self.model.format(
                Message("user", correction_prompt, role="user")
            )
            response = await self.model.acall(formatted_prompt)
            correction_result = self.parse_correction_result(response)
            
            if correction_result and correction_result.get('corrected_old_string'):
                return RepairItem(
                    file_path=repair.file_path,
                    old_string=correction_result['corrected_old_string'],
                    new_string=repair.new_string,
                    expected_replacements=repair.expected_replacements
                )
                
        except Exception as e:
            logger.warning(f"LLM correction failed: {e}")
        
        return None
    
    def parse_correction_result(self, response) -> Optional[Dict[str, str]]:
        """解析修正结果"""
        try:
            # Create a ModelResponse object if we have a string
            if isinstance(response, str):
                from onesim.models.core.model_response import ModelResponse
                model_response = ModelResponse(text=response)
            else:
                model_response = response
                
            parsed = self.json_parser.parse(model_response)
            if parsed.parsed:
                return parsed.parsed
        except Exception as e:
            logger.warning(f"Failed to parse correction result: {e}")
        
        return None