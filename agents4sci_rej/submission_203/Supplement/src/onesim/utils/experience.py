# -*- coding: utf-8 -*-
"""
Experience Management System
Unified experience repository for both local repair records and global patterns.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger

from onesim.models.core.message import Message
from onesim.models import JsonBlockParser
from onesim.utils.error_parser import ParsedError


@dataclass
class LocalRepairPattern:
    """Local repair pattern - patterns extracted from repair records"""
    
    # Basic information
    pattern_id: str
    pattern_name: str
    description: str
    
    # Error related
    error_type: str
    error_keywords: List[str]
    
    # Repair related
    solution_template: str
    key_insights: List[str]
    
    # Source information
    source_records: List[str]  # IDs of repair records this pattern was extracted from
    
    # Quality metrics
    success_rate: float
    usage_count: int
    
    # Time information
    created_at: datetime
    last_updated: datetime


@dataclass
class GlobalRepairPattern:
    """Global repair pattern - abstracted from local patterns"""
    
    # Basic information
    pattern_id: str
    pattern_name: str
    description: str
    
    # Error related
    error_type: str
    error_keywords: List[str]
    
    # Repair related
    solution_template: str
    key_insights: List[str]
    
    # Source information
    source_patterns: List[str]  # IDs of local patterns this was abstracted from
    
    # Quality metrics
    success_rate: float
    usage_count: int
    
    # Time information
    created_at: datetime
    last_updated: datetime


@dataclass
class RepairRecord:
    """Historical repair record"""
    error_type: str
    error_message: str
    file_path: str
    function_name: str = ""
    original_code: str = ""
    fixed_code: str = ""
    repair_strategy: str = "string_replacement"
    success: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    context_hash: str = ""
    reasoning: str = ""


class ExperienceRepository:
    """Three-layer experience repository: records -> local patterns -> global patterns"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Layer 1: Local repair records storage
        self.records_file = self.storage_path / "repair_records.json"
        self.records: List[RepairRecord] = []
        
        # Layer 2: Local patterns storage
        self.local_patterns_file = self.storage_path / "local_patterns.json"
        self.local_patterns: Dict[str, LocalRepairPattern] = {}
        self.local_error_index: Dict[str, List[str]] = {}
        
        # Layer 3: Global patterns storage
        self.global_patterns_file = self.storage_path / "global_patterns.json"
        self.global_patterns: Dict[str, GlobalRepairPattern] = {}
        self.global_error_index: Dict[str, List[str]] = {}
        
        # Load data on initialization
        self._load_records()
        self._load_local_patterns()
        self._load_global_patterns()
    
    def load_records(self) -> List[Dict]:
        """Load repair records as dict format for compatibility"""
        try:
            if not self.records_file.exists():
                return []
            
            with open(self.records_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load repair records: {e}")
            return []
    
    def record_from_dict(self, record_dict: Dict) -> RepairRecord:
        """Convert dict to RepairRecord for compatibility"""
        return RepairRecord(
            error_type=record_dict.get('error_type', ''),
            error_message=record_dict.get('error_message', ''),
            file_path=record_dict.get('file_path', ''),
            function_name=record_dict.get('function_name', ''),
            original_code=record_dict.get('original_code', ''),
            fixed_code=record_dict.get('fixed_code', ''),
            repair_strategy=record_dict.get('repair_strategy', 'string_replacement'),
            success=record_dict.get('success', False),
            timestamp=datetime.fromisoformat(record_dict.get('timestamp', datetime.now().isoformat())),
            context_hash=record_dict.get('context_hash', ''),
            reasoning=record_dict.get('reasoning', '')
        )
    
    def _load_records(self):
        """Load repair records from file"""
        try:
            if not self.records_file.exists():
                return
            
            with open(self.records_file, 'r', encoding='utf-8') as f:
                records_data = json.load(f)
            
            self.records = []
            for record_data in records_data:
                self.records.append(RepairRecord(
                    error_type=record_data.get('error_type', ''),
                    error_message=record_data.get('error_message', ''),
                    file_path=record_data.get('file_path', ''),
                    function_name=record_data.get('function_name', ''),
                    original_code=record_data.get('original_code', ''),
                    fixed_code=record_data.get('fixed_code', ''),
                    repair_strategy=record_data.get('repair_strategy', 'string_replacement'),
                    success=record_data.get('success', False),
                    timestamp=datetime.fromisoformat(record_data.get('timestamp', datetime.now().isoformat())),
                    context_hash=record_data.get('context_hash', ''),
                    reasoning=record_data.get('reasoning', '')
                ))
            
            logger.info(f"Loaded {len(self.records)} repair records from storage")
            
        except Exception as e:
            logger.error(f"Failed to load repair records: {e}")
    
    def _load_local_patterns(self):
        """Load local patterns from file"""
        try:
            if not self.local_patterns_file.exists():
                return
            
            with open(self.local_patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load local patterns
            patterns_data = data.get("patterns", {})
            for pattern_id, pattern_data in patterns_data.items():
                self.local_patterns[pattern_id] = LocalRepairPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_name=pattern_data["pattern_name"],
                    description=pattern_data["description"],
                    error_type=pattern_data["error_type"],
                    error_keywords=pattern_data["error_keywords"],
                    solution_template=pattern_data["solution_template"],
                    key_insights=pattern_data["key_insights"],
                    source_records=pattern_data.get("source_records", []),
                    success_rate=pattern_data["success_rate"],
                    usage_count=pattern_data["usage_count"],
                    created_at=datetime.fromisoformat(pattern_data["created_at"]),
                    last_updated=datetime.fromisoformat(pattern_data["last_updated"])
                )
            
            # Load error index
            self.local_error_index = data.get("error_index", {})
            
            logger.info(f"Loaded {len(self.local_patterns)} local patterns from storage")
            
        except Exception as e:
            logger.error(f"Failed to load local patterns: {e}")
    
    def _load_global_patterns(self):
        """Load global patterns from file"""
        try:
            if not self.global_patterns_file.exists():
                return
            
            with open(self.global_patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Load global patterns
            patterns_data = data.get("patterns", {})
            for pattern_id, pattern_data in patterns_data.items():
                self.global_patterns[pattern_id] = GlobalRepairPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_name=pattern_data["pattern_name"],
                    description=pattern_data["description"],
                    error_type=pattern_data["error_type"],
                    error_keywords=pattern_data["error_keywords"],
                    solution_template=pattern_data["solution_template"],
                    key_insights=pattern_data["key_insights"],
                    source_patterns=pattern_data.get("source_patterns", []),
                    success_rate=pattern_data["success_rate"],
                    usage_count=pattern_data["usage_count"],
                    created_at=datetime.fromisoformat(pattern_data["created_at"]),
                    last_updated=datetime.fromisoformat(pattern_data["last_updated"])
                )
            
            # Load error index
            self.global_error_index = data.get("error_index", {})
            
            logger.info(f"Loaded {len(self.global_patterns)} global patterns from storage")
            
        except Exception as e:
            logger.error(f"Failed to load global patterns: {e}")
    
    def save_records(self):
        """Save repair records to file"""
        try:
            records_data = []
            for record in self.records:
                records_data.append({
                    "error_type": record.error_type,
                    "error_message": record.error_message,
                    "file_path": record.file_path,
                    "function_name": record.function_name,
                    "original_code": record.original_code,
                    "fixed_code": record.fixed_code,
                    "repair_strategy": record.repair_strategy,
                    "success": record.success,
                    "timestamp": record.timestamp.isoformat(),
                    "context_hash": record.context_hash,
                    "reasoning": record.reasoning
                })
            
            with open(self.records_file, 'w', encoding='utf-8') as f:
                json.dump(records_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save records: {e}")
    
    def save_local_patterns(self):
        """Save local patterns to file"""
        try:
            patterns_data = {}
            for pattern_id, pattern in self.local_patterns.items():
                patterns_data[pattern_id] = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_name": pattern.pattern_name,
                    "description": pattern.description,
                    "error_type": pattern.error_type,
                    "error_keywords": pattern.error_keywords,
                    "solution_template": pattern.solution_template,
                    "key_insights": pattern.key_insights,
                    "source_records": pattern.source_records,
                    "success_rate": pattern.success_rate,
                    "usage_count": pattern.usage_count,
                    "created_at": pattern.created_at.isoformat(),
                    "last_updated": pattern.last_updated.isoformat()
                }
            
            save_data = {
                "patterns": patterns_data,
                "error_index": self.local_error_index
            }
            
            with open(self.local_patterns_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save local patterns: {e}")
    
    def save_global_patterns(self):
        """Save global patterns to file"""
        try:
            patterns_data = {}
            for pattern_id, pattern in self.global_patterns.items():
                patterns_data[pattern_id] = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_name": pattern.pattern_name,
                    "description": pattern.description,
                    "error_type": pattern.error_type,
                    "error_keywords": pattern.error_keywords,
                    "solution_template": pattern.solution_template,
                    "key_insights": pattern.key_insights,
                    "source_patterns": pattern.source_patterns,
                    "success_rate": pattern.success_rate,
                    "usage_count": pattern.usage_count,
                    "created_at": pattern.created_at.isoformat(),
                    "last_updated": pattern.last_updated.isoformat()
                }
            
            save_data = {
                "patterns": patterns_data,
                "error_index": self.global_error_index
            }
            
            with open(self.global_patterns_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save global patterns: {e}")
    
    def add_record(self, record: RepairRecord):
        """Add a repair record"""
        self.records.append(record)
        self.save_records()
    
    def add_local_pattern(self, pattern: LocalRepairPattern):
        """Add or update a local pattern"""
        self.local_patterns[pattern.pattern_id] = pattern
        
        # Update error index
        if pattern.error_type not in self.local_error_index:
            self.local_error_index[pattern.error_type] = []
        if pattern.pattern_id not in self.local_error_index[pattern.error_type]:
            self.local_error_index[pattern.error_type].append(pattern.pattern_id)
        
        self.save_local_patterns()
    
    def add_global_pattern(self, pattern: GlobalRepairPattern):
        """Add or update a global pattern"""
        self.global_patterns[pattern.pattern_id] = pattern
        
        # Update error index
        if pattern.error_type not in self.global_error_index:
            self.global_error_index[pattern.error_type] = []
        if pattern.pattern_id not in self.global_error_index[pattern.error_type]:
            self.global_error_index[pattern.error_type].append(pattern.pattern_id)
        
        self.save_global_patterns()
    
    def find_similar_repairs(self, error: ParsedError, limit: int = 5) -> List[RepairRecord]:
        """Find similar successful repair records"""
        similar_records = []
        
        # Filter by error type first
        for record in self.records:
            if (record.error_type == error.error_type.value and 
                record.success and 
                len(similar_records) < limit):
                similar_records.append(record)
        
        # Filter by error message similarity if needed
        if len(similar_records) < limit:
            error_msg_partial = error.error_message[:50]
            for record in self.records:
                if (error_msg_partial in record.error_message and 
                    record.success and 
                    record.error_type != error.error_type.value and
                    len(similar_records) < limit):
                    similar_records.append(record)
        
        # Sort by timestamp (most recent first)
        similar_records.sort(key=lambda x: x.timestamp, reverse=True)
        
        return similar_records[:limit]
    
    def find_applicable_local_patterns(self, error: ParsedError) -> List[LocalRepairPattern]:
        """Find applicable local patterns for an error"""
        applicable = []
        error_type = error.error_type.value
        
        # Match by error type
        if error_type in self.local_error_index:
            pattern_ids = self.local_error_index[error_type]
            for pattern_id in pattern_ids:
                if pattern_id in self.local_patterns:
                    applicable.append(self.local_patterns[pattern_id])
        
        # Match by keywords
        error_message_lower = error.error_message.lower()
        for pattern in self.local_patterns.values():
            for keyword in pattern.error_keywords:
                if keyword.lower() in error_message_lower:
                    if pattern not in applicable:
                        applicable.append(pattern)
        
        # Sort by success rate and usage count
        applicable.sort(
            key=lambda p: (p.success_rate, p.usage_count),
            reverse=True
        )
        
        return applicable[:3]  # Return top 3 patterns
    
    def find_applicable_global_patterns(self, error: ParsedError) -> List[GlobalRepairPattern]:
        """Find applicable global patterns for an error"""
        applicable = []
        error_type = error.error_type.value
        
        # Match by error type
        if error_type in self.global_error_index:
            pattern_ids = self.global_error_index[error_type]
            for pattern_id in pattern_ids:
                if pattern_id in self.global_patterns:
                    applicable.append(self.global_patterns[pattern_id])
        
        # Match by keywords
        error_message_lower = error.error_message.lower()
        for pattern in self.global_patterns.values():
            for keyword in pattern.error_keywords:
                if keyword.lower() in error_message_lower:
                    if pattern not in applicable:
                        applicable.append(pattern)
        
        # Sort by success rate and usage count
        applicable.sort(
            key=lambda p: (p.success_rate, p.usage_count),
            reverse=True
        )
        
        return applicable[:3]  # Return top 3 patterns
    
    def update_local_pattern_usage(self, pattern_id: str, success: bool):
        """Update local pattern usage statistics"""
        if pattern_id not in self.local_patterns:
            return
        
        pattern = self.local_patterns[pattern_id]
        pattern.usage_count += 1
        
        # Update success rate
        if success:
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 1) / pattern.usage_count
        else:
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1)) / pattern.usage_count
        
        pattern.last_updated = datetime.now()
        self.save_local_patterns()
    
    def update_global_pattern_usage(self, pattern_id: str, success: bool):
        """Update global pattern usage statistics"""
        if pattern_id not in self.global_patterns:
            return
        
        pattern = self.global_patterns[pattern_id]
        pattern.usage_count += 1
        
        # Update success rate
        if success:
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1) + 1) / pattern.usage_count
        else:
            pattern.success_rate = (pattern.success_rate * (pattern.usage_count - 1)) / pattern.usage_count
        
        pattern.last_updated = datetime.now()
        self.save_global_patterns()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get repository summary"""
        local_patterns = list(self.local_patterns.values())
        global_patterns = list(self.global_patterns.values())
        
        return {
            "total_records": len(self.records),
            "total_local_patterns": len(local_patterns),
            "total_global_patterns": len(global_patterns),
            "local_error_types": list(self.local_error_index.keys()),
            "global_error_types": list(self.global_error_index.keys()),
            "most_used_local_pattern": max(local_patterns, key=lambda p: p.usage_count).pattern_name if local_patterns else None,
            "most_used_global_pattern": max(global_patterns, key=lambda p: p.usage_count).pattern_name if global_patterns else None,
            "highest_success_rate_local": max(local_patterns, key=lambda p: p.success_rate).success_rate if local_patterns else 0.0,
            "highest_success_rate_global": max(global_patterns, key=lambda p: p.success_rate).success_rate if global_patterns else 0.0,
            "total_local_usage": sum(p.usage_count for p in local_patterns),
            "total_global_usage": sum(p.usage_count for p in global_patterns)
        }


class ExperienceManager:
    """Unified experience manager supporting both local and global repositories simultaneously"""
    
    def __init__(self, local_storage_path: str, global_storage_path: Optional[str] = None, model=None, use_global_experience: bool = False):
        self.local_storage_path = Path(local_storage_path)
        self.global_storage_path = Path(global_storage_path) if global_storage_path else None
        self.model = model
        self.use_global_experience = use_global_experience
        self.json_parser = JsonBlockParser()
        
        # Always create local repository
        self.local_repository = ExperienceRepository(str(self.local_storage_path))
        
        # Create global repository if configured
        if self.use_global_experience and self.global_storage_path:
            self.global_repository = ExperienceRepository(str(self.global_storage_path))
            logger.info(f"Global experience repository initialized at: {self.global_storage_path}")
        else:
            self.global_repository = None
            logger.info("Global experience repository disabled")
        
        logger.info(f"Local experience repository initialized at: {self.local_storage_path}")
    
    def load_records(self) -> List[Dict]:
        """Load records for compatibility with old code - returns local records"""
        return self.local_repository.load_records()
    
    def record_from_dict(self, record_dict: Dict) -> RepairRecord:
        """Convert dict to RepairRecord for compatibility"""
        return self.local_repository.record_from_dict(record_dict)
    
    async def learn_local_patterns_from_repairs(self, repair_records: List[RepairRecord]):
        """Learn local patterns from successful repair records"""
        if not self.model or not repair_records:
            return
        
        # Filter successful repairs
        successful_repairs = [r for r in repair_records if r.success]
        
        if len(successful_repairs) < 2:
            return
        
        # Group by error type
        error_groups = {}
        for repair in successful_repairs:
            error_type = repair.error_type
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(repair)
        
        # Extract local patterns for each error type
        for error_type, repairs in error_groups.items():
            if len(repairs) >= 2:
                pattern = await self._extract_local_pattern_from_repairs(error_type, repairs)
                if pattern:
                    self.local_repository.add_local_pattern(pattern)
    
    async def learn_global_patterns_from_local_patterns(self, local_patterns: List[LocalRepairPattern]):
        """Learn global patterns from local patterns - abstraction and summarization"""
        if not self.model or not local_patterns or not self.global_repository:
            return
        
        # Group by error type
        error_groups = {}
        for pattern in local_patterns:
            error_type = pattern.error_type
            if error_type not in error_groups:
                error_groups[error_type] = []
            error_groups[error_type].append(pattern)
        
        # Extract global patterns for each error type
        for error_type, patterns in error_groups.items():
            if len(patterns) >= 2:
                global_pattern = await self._extract_global_pattern_from_local_patterns(error_type, patterns)
                if global_pattern:
                    self.global_repository.add_global_pattern(global_pattern)
    
    async def _extract_local_pattern_from_repairs(self, error_type: str, repairs: List[RepairRecord]) -> Optional[LocalRepairPattern]:
        """Extract a local pattern from repair records using LLM"""
        prompt = f"""
Based on the following successful repair cases for {error_type} errors, extract a LOCAL repair pattern:

Repair Cases:
{self._format_repairs_for_prompt(repairs)}

Please extract a LOCAL pattern and return JSON format:

```json
{{
    "pattern_name": "Local pattern name describing the specific repair approach",
    "description": "Description of this repair pattern based on the specific cases",
    "error_keywords": ["keyword1", "keyword2"],
    "solution_template": "Solution template based on these specific cases",
    "key_insights": ["Insight 1", "Insight 2", "Insight 3"]
}}
```

Requirements:
1. Pattern should be specific to these repair cases
2. Solution template should describe the approach used in these cases
3. Key insights should be derived from these specific repairs
4. Focus on the concrete repair techniques used
"""
        
        try:
            if not self.model:
                return None
            response = await self.model.acall(
                self.model.format(Message("user", prompt, role="user"))
            )
            
            parsed = self.json_parser.parse(response)
            if not parsed or not parsed.parsed:
                return None
            
            data = parsed.parsed
            success_rate = 1.0  # All are successful cases
            
            # Generate record IDs for source tracking
            record_ids = [self._generate_record_id(repair) for repair in repairs]
            
            return LocalRepairPattern(
                pattern_id=self._generate_pattern_id(error_type, data.get("pattern_name", "")),
                pattern_name=data.get("pattern_name", ""),
                description=data.get("description", ""),
                error_type=error_type,
                error_keywords=data.get("error_keywords", []),
                solution_template=data.get("solution_template", ""),
                key_insights=data.get("key_insights", [])[:3],
                source_records=record_ids,
                success_rate=success_rate,
                usage_count=0,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to extract local pattern for {error_type}: {e}")
            return None
    
    async def _extract_global_pattern_from_local_patterns(self, error_type: str, local_patterns: List[LocalRepairPattern]) -> Optional[GlobalRepairPattern]:
        """Extract a global pattern from local patterns using LLM - abstraction and summarization"""
        prompt = f"""
Based on the following local repair patterns for {error_type} errors, extract a GLOBAL repair pattern that abstracts and summarizes the common principles:

Local Patterns:
{self._format_local_patterns_for_prompt(local_patterns)}

Please extract a GLOBAL pattern and return JSON format:

```json
{{
    "pattern_name": "Global pattern name (abstracted from local patterns)",
    "description": "Abstract description summarizing the universal principle",
    "error_keywords": ["keyword1", "keyword2"],
    "solution_template": "Abstract solution template applicable across scenarios",
    "key_insights": ["Universal insight 1", "Universal insight 2", "Universal insight 3"]
}}
```

Requirements:
1. Pattern should be abstracted from the local patterns
2. Solution template should describe the general approach
3. Focus on the underlying principles common across local patterns
4. Key insights should be universally valuable wisdom
5. Synthesize the common themes from local patterns
"""
        
        try:
            if not self.model:
                return None
            response = await self.model.acall(
                self.model.format(Message("user", prompt, role="user"))
            )
            
            parsed = self.json_parser.parse(response)
            if not parsed or not parsed.parsed:
                return None
            
            data = parsed.parsed
            success_rate = sum(p.success_rate for p in local_patterns) / len(local_patterns)
            
            # Generate pattern IDs for source tracking
            pattern_ids = [pattern.pattern_id for pattern in local_patterns]
            
            return GlobalRepairPattern(
                pattern_id=self._generate_pattern_id(error_type + "_global", data.get("pattern_name", "")),
                pattern_name=data.get("pattern_name", ""),
                description=data.get("description", ""),
                error_type=error_type,
                error_keywords=data.get("error_keywords", []),
                solution_template=data.get("solution_template", ""),
                key_insights=data.get("key_insights", [])[:3],
                source_patterns=pattern_ids,
                success_rate=success_rate,
                usage_count=0,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to extract global pattern for {error_type}: {e}")
            return None
    
    def _format_repairs_for_prompt(self, repairs: List[RepairRecord]) -> str:
        """Format repair records for LLM prompt"""
        formatted = []
        for i, repair in enumerate(repairs[:5], 1):
            formatted.append(f"""
Case {i}:
- Error message: {repair.error_message}
- Original code: {repair.original_code[:100]}...
- Fixed code: {repair.fixed_code[:100]}...
- Reasoning: {repair.reasoning[:100]}...
""")
        return "\n".join(formatted)
    
    def _format_local_patterns_for_prompt(self, patterns: List[LocalRepairPattern]) -> str:
        """Format local patterns for LLM prompt"""
        formatted = []
        for i, pattern in enumerate(patterns[:5], 1):
            formatted.append(f"""
Pattern {i}:
- Pattern name: {pattern.pattern_name}
- Description: {pattern.description}
- Solution template: {pattern.solution_template[:100]}...
- Key insights: {'; '.join(pattern.key_insights)}
- Success rate: {pattern.success_rate:.2f}
""")
        return "\n".join(formatted)
    
    def _generate_record_id(self, repair: RepairRecord) -> str:
        """Generate a unique record ID"""
        content = f"{repair.error_type}:{repair.file_path}:{repair.timestamp.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _generate_pattern_id(self, error_type: str, pattern_name: str) -> str:
        """Generate a unique pattern ID"""
        content = f"{error_type}:{pattern_name}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def find_applicable_local_patterns(self, error: ParsedError) -> List[LocalRepairPattern]:
        """Find applicable local patterns for an error"""
        return self.local_repository.find_applicable_local_patterns(error)
    
    def find_applicable_global_patterns(self, error: ParsedError) -> List[GlobalRepairPattern]:
        """Find applicable global patterns for an error"""
        if self.global_repository:
            return self.global_repository.find_applicable_global_patterns(error)
        return []
    
    def find_applicable_patterns(self, error: ParsedError) -> List[GlobalRepairPattern]:
        """Find applicable global patterns for an error (compatibility)"""
        return self.find_applicable_global_patterns(error)
    
    def find_combined_patterns(self, error: ParsedError) -> Tuple[List[LocalRepairPattern], List[GlobalRepairPattern]]:
        """Find both local and global patterns for an error"""
        local_patterns = self.find_applicable_local_patterns(error)
        global_patterns = self.find_applicable_global_patterns(error) if self.use_global_experience else []
        return local_patterns, global_patterns
    
    def find_similar_repairs(self, error: ParsedError, limit: int = 5) -> List[RepairRecord]:
        """Find similar repair records from local repository"""
        return self.local_repository.find_similar_repairs(error, limit)
    
    def record_local_pattern_usage(self, pattern_id: str, success: bool):
        """Record local pattern usage and update statistics"""
        self.local_repository.update_local_pattern_usage(pattern_id, success)
    
    def record_global_pattern_usage(self, pattern_id: str, success: bool):
        """Record global pattern usage and update statistics"""
        if self.global_repository:
            self.global_repository.update_global_pattern_usage(pattern_id, success)
    
    def record_pattern_usage(self, pattern_id: str, success: bool):
        """Record pattern usage and update statistics (compatibility)"""
        # Try global first, then local
        if self.global_repository and pattern_id in self.global_repository.global_patterns:
            self.global_repository.update_global_pattern_usage(pattern_id, success)
        elif pattern_id in self.local_repository.local_patterns:
            self.local_repository.update_local_pattern_usage(pattern_id, success)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get experience manager summary combining both repositories"""
        local_summary = self.local_repository.get_summary()
        if self.global_repository:
            global_summary = self.global_repository.get_summary()
            return {
                "local": local_summary,
                "global": global_summary,
                "combined_total_patterns": local_summary["total_local_patterns"] + global_summary["total_global_patterns"],
                "use_global_experience": self.use_global_experience
            }
        else:
            return {
                "local": local_summary,
                "global": {"total_records": 0, "total_local_patterns": 0, "total_global_patterns": 0},
                "combined_total_patterns": local_summary["total_local_patterns"],
                "use_global_experience": self.use_global_experience
            }
    
    async def store_repair(self, repair_record: RepairRecord):
        """Store a repair record in local repository"""
        self.local_repository.add_record(repair_record)
        
        # If successful, potentially learn patterns from recent repairs
        if repair_record.success and self.model:
            # Check if we have enough similar successful repairs to learn a local pattern
            recent_repairs = [r for r in self.local_repository.records if r.success and r.error_type == repair_record.error_type]
            if len(recent_repairs) >= 3:  # Need at least 3 similar successful repairs
                await self.learn_local_patterns_from_repairs(recent_repairs)
                
                # Check if we have enough local patterns to learn a global pattern
                local_patterns = [p for p in self.local_repository.local_patterns.values() if p.error_type == repair_record.error_type]
                if len(local_patterns) >= 2 and self.global_repository:  # Need at least 2 local patterns
                    await self.learn_global_patterns_from_local_patterns(local_patterns)
    
    @property
    def patterns(self) -> Dict[str, GlobalRepairPattern]:
        """Get global patterns for compatibility"""
        return self.global_repository.global_patterns if self.global_repository else {}
    
    @property
    def local_patterns(self) -> Dict[str, LocalRepairPattern]:
        """Get local patterns"""
        return self.local_repository.local_patterns
    
    @property
    def global_patterns(self) -> Dict[str, GlobalRepairPattern]:
        """Get global patterns"""
        return self.global_repository.global_patterns if self.global_repository else {}
    
    def format_patterns_for_llm(self, error: ParsedError, max_patterns: int = 3) -> str:
        """Format applicable patterns for LLM-friendly prompts"""
        local_patterns, global_patterns = self.find_combined_patterns(error)
        
        if not local_patterns and not global_patterns:
            return "No relevant experience patterns found for this error type."
        
        formatted_output = []
        
        # Format local patterns
        if local_patterns:
            formatted_output.append("### 🔧 Local Experience Patterns (Environment-Specific)")
            for i, pattern in enumerate(local_patterns[:max_patterns], 1):
                formatted_output.append(f"""
**Pattern {i}: {pattern.pattern_name}**
- **Error Type**: {pattern.error_type}
- **Success Rate**: {pattern.success_rate:.1%} (used {pattern.usage_count} times)
- **Solution Template**: {pattern.solution_template}
- **Key Insights**: {'; '.join(pattern.key_insights)}
- **Keywords**: {', '.join(pattern.error_keywords)}
""")
        
        # Format global patterns
        if global_patterns:
            formatted_output.append("### 🌍 Global Experience Patterns (Cross-Environment)")
            for i, pattern in enumerate(global_patterns[:max_patterns], 1):
                formatted_output.append(f"""
**Pattern {i}: {pattern.pattern_name}**
- **Error Type**: {pattern.error_type}
- **Success Rate**: {pattern.success_rate:.1%} (used {pattern.usage_count} times)
- **Solution Template**: {pattern.solution_template}
- **Key Insights**: {'; '.join(pattern.key_insights)}
- **Keywords**: {', '.join(pattern.error_keywords)}
""")
        
        return "\n".join(formatted_output)
    
    def get_pattern_guidance(self, error: ParsedError) -> str:
        """Get concise pattern-based guidance for LLM"""
        local_patterns, global_patterns = self.find_combined_patterns(error)
        
        if not local_patterns and not global_patterns:
            return "No prior experience with this error type."
        
        guidance_parts = []
        
        # Combine top insights from both pattern types
        all_insights = []
        for pattern in (local_patterns[:2] + global_patterns[:2]):
            all_insights.extend(pattern.key_insights)
        
        # Get top solution approaches
        solution_approaches = []
        for pattern in (local_patterns[:1] + global_patterns[:1]):
            if pattern.solution_template:
                solution_approaches.append(pattern.solution_template)
        
        if all_insights:
            guidance_parts.append(f"**Key Insights**: {'; '.join(all_insights[:3])}")
        
        if solution_approaches:
            guidance_parts.append(f"**Recommended Approach**: {solution_approaches[0]}")
        
        return "\n".join(guidance_parts)