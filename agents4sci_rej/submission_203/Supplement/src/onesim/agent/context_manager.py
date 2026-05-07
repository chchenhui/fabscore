# -*- coding: utf-8 -*-
"""
Context Manager for Debugging Agent
Handles context collection and repair context preparation.
"""

from typing import Dict, Any

from onesim.utils.debugging_data_structures import RepairContext, SelectedContext
from onesim.utils.context_selector import LLMContextSelector


class ContextManager:
    """上下文管理器"""
    
    def __init__(self, model, debugging_agent=None):
        self.llm_selector = LLMContextSelector(model, debugging_agent)
        self.experience_manager = None  # Will be set later
        self.debugging_agent = debugging_agent
    
    def set_experience_manager(self, experience_manager):
        """Set the experience manager reference"""
        self.experience_manager = experience_manager
    
    async def get_repair_context(self, error,
                               code_structure: Dict) -> RepairContext:
        """获取修复所需的上下文"""
        
        # 1. LLM选择核心上下文
        selected_context = await self.llm_selector.select_context(error, code_structure)
        
        # 2. 查询历史修复经验和模式
        historical_fixes = []
        if self.experience_manager:
            historical_fixes = self.experience_manager.find_similar_repairs(error, limit=3)
            
            # 查询全局修复模式
            global_patterns = self.experience_manager.find_applicable_global_patterns(error)
            
            if global_patterns:
                # 将全局模式转换为RepairRecord格式
                global_records = []
                for pattern in global_patterns:
                    # Import here to avoid circular imports
                    from onesim.utils.experience import RepairRecord
                    
                    global_record = RepairRecord(
                        error_type=pattern.error_type,
                        error_message=f"全局经验: {pattern.pattern_name}",
                        file_path="",
                        function_name="",
                        original_code="",
                        fixed_code=pattern.solution_template,
                        repair_strategy="global_pattern",
                        success=True,
                        reasoning=f"{pattern.description}。关键洞察: {'; '.join(pattern.key_insights)}"
                    )
                    global_records.append(global_record)
                
                # 将全局经验放在历史修复的前面
                historical_fixes = global_records + historical_fixes[:2]  # 保持总数不要太多
        
        # 4. 组合最终上下文
        repair_context = RepairContext(
            error_info=error,
            selected_context=selected_context,
            historical_fixes=historical_fixes,
            code_structure_snippet=self.extract_relevant_structure(
                code_structure, selected_context
            )
        )
        
        return repair_context
    
    def extract_relevant_structure(self, code_structure: Dict, 
                                 selected_context: SelectedContext) -> Dict[str, Any]:
        """提取相关的结构信息"""
        relevant = {}
        
        # 提取选中的agents信息
        if selected_context.selected_agents and 'agents' in code_structure:
            relevant['agents'] = {
                agent: code_structure['agents'][agent] 
                for agent in selected_context.selected_agents 
                if agent in code_structure['agents']
            }
        
        # 提取选中的events信息
        if selected_context.selected_events and 'events' in code_structure:
            events_def = code_structure['events'].get('definitions', {})
            relevant['events'] = {
                event: events_def[event] 
                for event in selected_context.selected_events 
                if event in events_def
            }
        
        return relevant