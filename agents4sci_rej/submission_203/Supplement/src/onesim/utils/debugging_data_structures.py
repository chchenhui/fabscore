# -*- coding: utf-8 -*-
"""
Debugging Agent Data Structures
Common data structures used across debugging components.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from onesim.utils.error_parser import ParsedError
    from onesim.utils.experience import RepairRecord


@dataclass
class SelectedContext:
    """Selected context for repair"""
    error_file: str = ""
    selected_agents: List[str] = field(default_factory=list)
    selected_events: List[str] = field(default_factory=list)
    selected_related_files: List[str] = field(default_factory=list)
    files_to_read: List[str] = field(default_factory=list)  # Files to read for repair context
    reasoning: str = ""

    def get_formatted_context(self) -> str:
        """Get formatted context for LLM"""
        context_parts = []
        
        if self.error_file:
            context_parts.append(f"Error File Content:\n{self.error_file}")
            
        if self.selected_agents:
            context_parts.append(f"Related Agents: {', '.join(self.selected_agents)}")
            
        if self.selected_events:
            context_parts.append(f"Related Events: {', '.join(self.selected_events)}")
            
        if self.selected_related_files:
            context_parts.append(f"Related Files: {', '.join(self.selected_related_files)}")
            
        if self.files_to_read:
            context_parts.append(f"Files to Read: {', '.join(self.files_to_read)}")
            
        if self.reasoning:
            context_parts.append(f"Reasoning: {self.reasoning}")
        
        return "\n\n".join(context_parts)

@dataclass
class ExtractedContent:
    """Extracted content from LLM analysis"""
    file_path: str
    purpose: str
    extracted_ranges: List[Tuple[int, int]] = field(default_factory=list)  # (start_line, end_line) pairs
    content_sections: List[str] = field(default_factory=list)  # Actual extracted content
    summary: str = ""  # LLM summary of what was extracted
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AvailableContext:
    """Available context information for debugging"""
    # Core context
    error_file: str = ""
    agents: Dict[str, Any] = field(default_factory=dict)
    events: Dict[str, Any] = field(default_factory=dict)
    
    # File context (organized by categories)
    environment_files: List[str] = field(default_factory=list)  # All files in environment
    config_files: List[str] = field(default_factory=list)  # Config files (*.json, *.yaml)
    
    def get_all_files(self) -> List[str]:
        """Get all available files (deduplicated)"""
        all_files = set()
        if self.error_file:
            all_files.add(self.error_file)
        all_files.update(self.environment_files)
        all_files.update(self.config_files)
        return sorted(list(all_files))
    
    def get_files_by_type(self, file_type: str) -> List[str]:
        """Get files by type (py, json, yaml, etc.)"""
        all_files = self.get_all_files()
        if file_type == "py":
            return [f for f in all_files if f.endswith('.py')]
        elif file_type == "config":
            return [f for f in all_files if f.endswith(('.json', '.yaml', '.yml'))]
        return all_files


@dataclass
class RepairParams:
    """Repair parameters for string replacement"""
    repair_strategy: str = "string_replacement"
    repairs: List['RepairItem'] = field(default_factory=list)
    backup_strategy: str = ""


@dataclass
class RepairItem:
    """Individual repair item"""
    file_path: str  # Target file path for this repair
    old_string: str
    new_string: str
    expected_replacements: int = 1
    reasoning: str = ""


@dataclass
class ErrorClassification:
    """Error classification for repair strategy selection"""
    error_type: str
    complexity: str  # "simple" or "complex"
    repair_strategy: str  # "string_replacement" or "react"
    confidence: float
    reasoning: str


@dataclass
class MultiProblemContext:
    """Context for fixing multiple problems in one iteration"""
    errors: List['ParsedError'] = field(default_factory=list)
    error_classifications: List[ErrorClassification] = field(default_factory=list)
    simple_errors: List['ParsedError'] = field(default_factory=list)
    complex_errors: List['ParsedError'] = field(default_factory=list)


@dataclass
class RepairResult:
    """Result of a repair operation"""
    success: bool
    repaired_file: str = ""
    changes_made: int = 0
    error_message: str = ""
    original_code: str = ""  # Code before repair
    fixed_code: str = ""     # Code after repair
    repair_details: Dict[str, Any] = field(default_factory=dict)  # Additional repair metadata
    message: str = ""  # Success message


@dataclass
class ReActAction:
    """ReAct agent action"""
    action_type: str  # 'read_file', 'modify_file', 'generate_fix'
    target: str  # file path or analysis target
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = False
    reasoning: str = ""
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ReActState:
    """ReAct agent state"""
    current_error: Optional['ParsedError'] = None
    thinking_history: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    actions_taken: List[ReActAction] = field(default_factory=list)
    files_read: Dict[str, str] = field(default_factory=dict)  # path -> content
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    fix_attempts: List[Dict[str, Any]] = field(default_factory=list)
    confidence_progression: List[float] = field(default_factory=list)
    completed: bool = False
    success: bool = False
    termination_reason: str = ""
    failure_patterns: Dict[str, List[str]] = field(default_factory=dict)  # Track failure patterns for learning
    
    def update_with_observation(self, action: ReActAction, observation: str, thinking: str):
        """Update state with new cycle results"""
        self.actions_taken.append(action)
        self.observations.append(observation)
        self.thinking_history.append(thinking)
        
        if action.success and hasattr(action, 'confidence'):
            self.confidence_progression.append(action.confidence)


@dataclass
class ThinkingResult:
    """Think阶段的结果"""
    analysis: str
    information_needs: str
    recommended_action: str
    action_target: str
    confidence: float
    reasoning: str


@dataclass
class RepairStrategy:
    """Repair strategy definition"""
    type: str = "string_replacement"
    scope: str = "local"
    priority: str = "medium"


@dataclass
class RepairContext:
    """Complete repair context"""
    error_info: 'ParsedError'
    selected_context: SelectedContext
    historical_fixes: List['RepairRecord'] = field(default_factory=list)
    code_structure_snippet: Dict[str, Any] = field(default_factory=dict)
    experience_guidance: Optional[str] = None  # Experience patterns guidance for LLM
    
    def get_formatted_context(self) -> str:
        """Get formatted context for LLM"""
        context_parts = []
        
        if self.selected_context:
            context_parts.append(self.selected_context.get_formatted_context())

        if self.historical_fixes:
            context_parts.append(self.get_historical_fixes_summary())
        
        if self.experience_guidance:
            context_parts.append(f"Experience Guidance:\n{self.experience_guidance}")

        return "\n\n".join(context_parts)
    
    def get_historical_fixes_summary(self) -> str:
        """Get summary of historical fixes"""
        if not self.historical_fixes:
            return "No historical fixes available."
        
        summaries = []
        for i, fix in enumerate(self.historical_fixes[:3], 1):  # Limit to top 3
            summary = f"{i}. {fix.error_type} error:"
            if fix.original_code and fix.fixed_code:
                summary += f"\n   Original code: {fix.original_code}"
                summary += f"\n   Fixed code: {fix.fixed_code}"
            summary += f"\n   Experience: {fix.reasoning}"
            summaries.append(summary)
            
        return "\n\n".join(summaries)


@dataclass
class DebuggingResult:
    """Final debugging result"""
    success: bool
    iterations: int = 0
    final_error: str = ""
    react_actions: List[str] = field(default_factory=list)  # ReAct actions taken


@dataclass
class SimulationResult:
    """Simulation execution result"""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.0