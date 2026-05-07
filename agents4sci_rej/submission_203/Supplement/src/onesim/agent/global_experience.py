# -*- coding: utf-8 -*-
"""
Global Experience Repository - Minimal Version
全局系统级修复模式的提取、存储和使用
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass
from loguru import logger

from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from onesim.utils.error_parser import ParsedError


@dataclass
class GlobalRepairPattern:
    """全局修复模式 - 极简版"""
    
    # 基础信息
    pattern_id: str
    pattern_name: str
    description: str
    
    # 错误相关
    error_type: str                    # 错误类型
    error_keywords: List[str]          # 错误关键词
    
    # 修复相关  
    solution_template: str             # 解决方案模板
    key_insights: List[str]            # 关键洞察
    
    # 质量指标
    success_rate: float               # 成功率
    usage_count: int                  # 使用次数
    
    # 时间信息
    created_at: datetime
    last_updated: datetime


@dataclass
class GlobalExperienceRepository:
    """全局经验库 - 极简版"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.patterns_file = self.storage_path / "global_patterns.json"
        self.patterns: Dict[str, GlobalRepairPattern] = {}
        self.error_index: Dict[str, List[str]] = {}  # 错误类型 -> 模式ID列表
        # Note: load_patterns will be called by SimpleGlobalExperienceManager


class SimpleGlobalExperienceManager:
    """简化的全局经验管理器"""
    
    def __init__(self, storage_path: str, model=None):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.repository = GlobalExperienceRepository(storage_path)
        self.json_parser = JsonBlockParser()
        # Load patterns after repository is created
        self.load_patterns()
    
    async def extract_and_update_patterns(self, repair_records: List[Any]):
        """从修复记录中提取并更新全局模式"""
        
        if not self.model or not repair_records:
            return
        
        # 只处理成功的修复记录
        successful_repairs = [r for r in repair_records if getattr(r, 'success', False)]
        
        if len(successful_repairs) < 2:  # 至少需要2个成功案例
            return
        
        # 按错误类型分组
        error_groups = {}
        for repair in successful_repairs:
            error_type = getattr(repair, 'error_type', 'Unknown')
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(repair)
        
        # 为每个错误类型提取模式
        for error_type, repairs in error_groups.items():
            if len(repairs) >= 2:  # 至少2个同类型的成功修复
                pattern = await self.extract_pattern_from_repairs(error_type, repairs)
                if pattern:
                    self.add_or_update_pattern(pattern)
        
        # 保存更新
        self.save_patterns()
    
    async def extract_pattern_from_repairs(self, error_type: str, repairs: List[Any]) -> Optional[GlobalRepairPattern]:
        """从修复记录中提取模式"""
        
        prompt = f"""
基于以下{error_type}错误的成功修复案例，提取一个通用的修复模式：

修复案例：
{self.format_repairs_for_prompt(repairs)}

请提取通用模式，返回JSON格式：

```json
{{
    "pattern_name": "简洁的模式名称",
    "description": "一句话描述这个修复模式",
    "error_keywords": ["关键词1", "关键词2"],
    "solution_template": "通用解决方案描述",
    "key_insights": ["洞察1", "洞察2", "洞察3"]
}}
```

要求：
1. 模式名称要简洁明了
2. 解决方案要通用，不要包含具体的代码细节
3. 关键洞察要精炼，最多3个
4. 只提取真正有价值的通用经验
"""
        
        try:
            response = await self.model.acall(
                self.model.format(Message("user", prompt, role="user"))
            )
            
            parsed = self.json_parser.parse(response)
            if not parsed.parsed:
                return None
            
            data = parsed.parsed
            success_rate = len(repairs) / len(repairs)  # 都是成功的案例
            
            return GlobalRepairPattern(
                pattern_id=self.generate_pattern_id(error_type, data.get("pattern_name", "")),
                pattern_name=data.get("pattern_name", ""),
                description=data.get("description", ""),
                error_type=error_type,
                error_keywords=data.get("error_keywords", []),
                solution_template=data.get("solution_template", ""),
                key_insights=data.get("key_insights", [])[:3],  # 最多3个
                success_rate=success_rate,
                usage_count=0,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to extract pattern for {error_type}: {e}")
            return None
    
    def format_repairs_for_prompt(self, repairs: List[Any]) -> str:
        """格式化修复记录用于提示"""
        formatted = []
        for i, repair in enumerate(repairs[:5], 1):  # 最多5个案例
            error_message = getattr(repair, 'error_message', '')
            original_code = getattr(repair, 'original_code', '')
            fixed_code = getattr(repair, 'fixed_code', '')
            reasoning = getattr(repair, 'reasoning', '')
            
            formatted.append(f"""
案例{i}:
- 错误信息: {error_message}
- 原始代码: {original_code[:100]}...
- 修复代码: {fixed_code[:100]}...
- 修复原因: {reasoning[:100]}...
""")
        return "\n".join(formatted)
    
    def generate_pattern_id(self, error_type: str, pattern_name: str) -> str:
        """生成模式ID"""
        content = f"{error_type}:{pattern_name}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def add_or_update_pattern(self, new_pattern: GlobalRepairPattern):
        """添加或更新模式"""
        
        existing_pattern = self.repository.patterns.get(new_pattern.pattern_id)
        
        if existing_pattern:
            # 更新现有模式
            existing_pattern.description = new_pattern.description
            existing_pattern.solution_template = new_pattern.solution_template
            existing_pattern.key_insights = new_pattern.key_insights
            existing_pattern.last_updated = datetime.now()
        else:
            # 添加新模式
            self.repository.patterns[new_pattern.pattern_id] = new_pattern
            
            # 更新索引
            error_type = new_pattern.error_type
            if error_type not in self.repository.error_index:
                self.repository.error_index[error_type] = []
            self.repository.error_index[error_type].append(new_pattern.pattern_id)
    
    def find_applicable_patterns(self, error: ParsedError) -> List[GlobalRepairPattern]:
        """查找适用的模式"""
        
        applicable = []
        error_type = error.error_type.value
        
        # 1. 按错误类型匹配
        if error_type in self.repository.error_index:
            pattern_ids = self.repository.error_index[error_type]
            for pattern_id in pattern_ids:
                if pattern_id in self.repository.patterns:
                    applicable.append(self.repository.patterns[pattern_id])
        
        # 2. 按关键词匹配
        error_message_lower = error.error_message.lower()
        for pattern in self.repository.patterns.values():
            for keyword in pattern.error_keywords:
                if keyword.lower() in error_message_lower:
                    if pattern not in applicable:
                        applicable.append(pattern)
        
        # 按使用次数和成功率排序
        applicable.sort(
            key=lambda p: (p.success_rate, p.usage_count),
            reverse=True
        )
        
        return applicable[:2]  # 返回最多2个模式
    
    def record_pattern_usage(self, pattern_id: str, success: bool):
        """记录模式使用情况"""
        
        if pattern_id not in self.repository.patterns:
            return
        
        pattern = self.repository.patterns[pattern_id]
        pattern.usage_count += 1
        
        # 更新成功率
        if success:
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 1) / pattern.usage_count
        else:
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1)) / pattern.usage_count
        
        self.save_patterns()
    
    def save_patterns(self):
        """保存模式到文件"""
        try:
            patterns_data = {}
            for pattern_id, pattern in self.repository.patterns.items():
                patterns_data[pattern_id] = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_name": pattern.pattern_name,
                    "description": pattern.description,
                    "error_type": pattern.error_type,
                    "error_keywords": pattern.error_keywords,
                    "solution_template": pattern.solution_template,
                    "key_insights": pattern.key_insights,
                    "success_rate": pattern.success_rate,
                    "usage_count": pattern.usage_count,
                    "created_at": pattern.created_at.isoformat(),
                    "last_updated": pattern.last_updated.isoformat()
                }
            
            # 保存模式和索引到同一个文件
            save_data = {
                "patterns": patterns_data,
                "error_index": self.repository.error_index
            }
            
            with open(self.repository.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    def load_patterns(self):
        """从文件加载模式"""
        try:
            if not self.repository.patterns_file.exists():
                return
            
            with open(self.repository.patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载模式
            patterns_data = data.get("patterns", {})
            for pattern_id, pattern_data in patterns_data.items():
                self.repository.patterns[pattern_id] = GlobalRepairPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_name=pattern_data["pattern_name"],
                    description=pattern_data["description"],
                    error_type=pattern_data["error_type"],
                    error_keywords=pattern_data["error_keywords"],
                    solution_template=pattern_data["solution_template"],
                    key_insights=pattern_data["key_insights"],
                    success_rate=pattern_data["success_rate"],
                    usage_count=pattern_data["usage_count"],
                    created_at=datetime.fromisoformat(pattern_data["created_at"]),
                    last_updated=datetime.fromisoformat(pattern_data["last_updated"])
                )
            
            # 加载索引
            self.repository.error_index = data.get("error_index", {})
            
            logger.info(f"Loaded {len(self.repository.patterns)} patterns from storage")
            
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")

    def get_patterns_summary(self) -> Dict[str, Any]:
        """获取模式库摘要信息"""
        patterns = list(self.repository.patterns.values())
        if not patterns:
            return {"total_patterns": 0}
        
        return {
            "total_patterns": len(patterns),
            "error_types": list(self.repository.error_index.keys()),
            "most_used_pattern": max(patterns, key=lambda p: p.usage_count).pattern_name if patterns else None,
            "highest_success_rate": max(patterns, key=lambda p: p.success_rate).success_rate if patterns else 0.0,
            "total_usage": sum(p.usage_count for p in patterns)
        }