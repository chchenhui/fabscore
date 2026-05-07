import json
import asyncio
import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

# 导入onesim模型模块
from onesim.models.core.model_manager import ModelManager
from onesim.models.core.message import Message, SystemMessage, UserMessage

from .tool_registry import TOOLS
# 删除这行：from .agent_core import route_and_run

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolSelectionStrategy(Enum):
    """Tool selection strategy enumeration"""
    DIRECT = "direct"  # Direct selection
    RANKING = "ranking"  # Ranking selection
    ENSEMBLE = "ensemble"  # Ensemble selection
    REASONING = "reasoning"  # Reasoning selection


@dataclass
class ToolCandidate:
    """Tool candidate"""
    name: str
    description: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]


@dataclass
class ToolSelectionResult:
    """Tool selection result"""
    selected_tool: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]
    alternatives: List[ToolCandidate]
    execution_plan: str


class EnhancedStatAgentLLMAdapter:
    def __init__(
        self,
        config_path: str = None,
        config_name: str = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        generate_args: Dict[str, Any] = None,
        tool_selection_strategy: str = "reasoning"
    ):
        """Initialize the enhanced LLM adapter for statistical analysis"""
        import os
        from pathlib import Path
        
        # Use relative path or environment variable for config
        if config_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            config_path = os.getenv('ONESIM_MODEL_CONFIG_PATH', project_root / "config" / "model_config.json")
        
        self.config_path = str(config_path)
        self.generate_args = generate_args or {}
        self.tool_selection_strategy = tool_selection_strategy
        
        # initalize model_manager
        self.model_manager = ModelManager.get_instance()
        self.model_manager.initialize(self.config_path)
        
        # gain model instance
        if config_name:
            logger.info(f"Getting model instance for config_name: {config_name} from config {self.config_path}")
            self.model = self.model_manager.get_model(config_name=config_name)
            # gain model name
            self.model_name = getattr(self.model, 'model_name', config_name)
        elif model_name:
            logger.info(f"Getting model instance for model_name: {model_name} from config {self.config_path}")
            self.model = self.model_manager.get_model(model_name=model_name)
            self.model_name = model_name
        else:
            raise ValueError("Either config_name or model_name must be provided")
        
        # gain client
        self.client = getattr(self.model, 'client', None)
        self.async_client = getattr(self.model, 'async_client', None)
        
        logger.info(f"Model instance initialized: {self.model_name}")
        logger.info(f"Client available: {self.client is not None}")
        logger.info(f"Async client available: {self.async_client is not None}")
    
        # Tool selection history
        self.tool_selection_history = []
        
        # Initialize tool categorization
        self._categorize_tools()

    def _categorize_tools(self):
        """Categorize tools by functionality"""
        self.tool_categories = {
            "parametric_tests": [],
            "non_parametric_tests": [],
            "time_series": [],
            "bayesian": [],
            "mixed_effects": [],
            "multifactor": [],
            "repeated_measures": []
        }
        
        for tool_name, tool_info in TOOLS.items():
            if "t_test" in tool_name or "anova" in tool_name:
                self.tool_categories["parametric_tests"].append(tool_name)
            elif "mann_whitney" in tool_name or "wilcoxon" in tool_name or "kruskal" in tool_name:
                self.tool_categories["non_parametric_tests"].append(tool_name)
            elif "granger" in tool_name or "var" in tool_name or "trend" in tool_name:
                self.tool_categories["time_series"].append(tool_name)
            elif "bayes" in tool_name:
                self.tool_categories["bayesian"].append(tool_name)
            elif "mixed" in tool_name:
                self.tool_categories["mixed_effects"].append(tool_name)
            elif "factorial" in tool_name or "manova" in tool_name:
                self.tool_categories["multifactor"].append(tool_name)
            elif "repeated" in tool_name or "friedman" in tool_name or "ancova" in tool_name:
                self.tool_categories["repeated_measures"].append(tool_name)

    def build_tool_selection_prompt(self, user_query: str) -> str:
        """Build system prompt for tool selection"""
        tools_desc = "\n".join([
            f"- {name}: {tool['description']}" for name, tool in TOOLS.items()
        ])
        
        categories_desc = "\n".join([
            f"## {category.replace('_', ' ').title()}:\n" + 
            "\n".join([f"  - {tool}: {TOOLS[tool]['description']}" for tool in tools])
            for category, tools in self.tool_categories.items() if tools
        ])
        
        system_prompt = f"""You are an expert statistical analysis assistant. Your task is to analyze user queries and select the most appropriate statistical tools.

## Available Statistical Tools by Category:

{categories_desc}

## Tool Selection Guidelines:

1. **Data Type Analysis**: First determine if the data is continuous, categorical, or time series
2. **Sample Size Consideration**: For small samples (<30), prefer non-parametric tests
3. **Distribution Assumptions**: Check if normality and homogeneity of variance assumptions are met
4. **Research Design**: Consider if it's between-subjects, within-subjects, or mixed design
5. **Number of Groups**: One group (descriptive), two groups (t-tests), three+ groups (ANOVA)
6. **Dependent Variables**: Single (univariate) vs multiple (multivariate)

## Parameter Extraction Rules:
- Extract numerical data from user query using regex patterns
- For t-tests: extract two lists as sample1 and sample2
- For ANOVA: extract multiple lists as *samples
- For time series: extract data as 2D array
- For Bayesian tests: extract two lists as x and y
- Convert all numbers to float type
- Handle missing parameters gracefully

## Response Format:
Return a JSON object with the following structure:
{{
    "selected_tool": "tool_name",
    "confidence": 0.95,
    "reasoning": "Detailed explanation of why this tool was chosen",
    "parameters": {{
        "sample1": [1.0, 2.0, 3.0],
        "sample2": [4.0, 5.0, 6.0]
    }},
    "alternatives": [
        {{
            "name": "alternative_tool",
            "confidence": 0.7,
            "reasoning": "Why this alternative was considered"
        }}
    ],
    "execution_plan": "Step-by-step execution plan"
}}

## Important Notes:
- Always consider the research question and data characteristics
- Provide confidence scores (0-1) for your selection
- Include alternative tools that could also be appropriate
- Explain your reasoning clearly
- Ensure parameters match the tool's requirements
- Extract data from user query using regex patterns
"""
        return system_prompt

    def build_reasoning_prompt(self, user_query: str, selected_tool: str) -> str:
        """Build reasoning prompt for detailed tool selection analysis"""
        tool_info = TOOLS.get(selected_tool, {})
        
        reasoning_prompt = f"""You are a statistical expert. Analyze the following query and explain why the selected tool is appropriate.

**User Query**: {user_query}
**Selected Tool**: {selected_tool}
**Tool Description**: {tool_info.get('description', 'N/A')}

Please provide a detailed analysis including:

1. **Data Characteristics**: What type of data is being analyzed?
2. **Statistical Assumptions**: What assumptions does this test make and are they met?
3. **Research Design**: What type of experimental design is this?
4. **Alternative Considerations**: What other tools could be used and why weren't they chosen?
5. **Interpretation Guidance**: How should the results be interpreted?

Provide a comprehensive analysis that demonstrates deep statistical knowledge.
"""
        return reasoning_prompt

    def extract_parameters_from_query(self, user_query: str) -> Dict[str, Any]:
        """Extract parameters from user query"""
        parameters = {}
        
        # Extract number list patterns
        number_pattern = r'\[([\d\.,\s]+)\]'
        lists = re.findall(number_pattern, user_query)
        
        # Convert to float lists
        float_lists = []
        for list_str in lists:
            try:
                # Split and convert to float
                numbers = [float(x.strip()) for x in list_str.split(',')]
                float_lists.append(numbers)
            except ValueError:
                continue
        
        # Assign parameters based on tool type
        if len(float_lists) >= 2:
            query_lower = user_query.lower()
            
            if "paired" in query_lower or "dependent" in query_lower:
                parameters["sample1"] = float_lists[0]
                parameters["sample2"] = float_lists[1]
            elif "anova" in query_lower or "three" in query_lower or "multiple" in query_lower or "groups" in query_lower:
                parameters["*samples"] = float_lists
            elif "bayes" in query_lower:
                parameters["x"] = float_lists[0]
                parameters["y"] = float_lists[1]
            elif "time" in query_lower or "series" in query_lower or "granger" in query_lower:
                # Time series data
                if len(float_lists) == 2:
                    import numpy as np
                    data = np.column_stack(float_lists)
                    parameters["data"] = data.tolist()
            elif "mann" in query_lower or "whitney" in query_lower:
                parameters["sample1"] = float_lists[0]
                parameters["sample2"] = float_lists[1]
            elif "wilcoxon" in query_lower:
                parameters["sample1"] = float_lists[0]
                parameters["sample2"] = float_lists[1]
            elif "kruskal" in query_lower:
                parameters["*samples"] = float_lists
            else:
                # Default t-test
                parameters["sample1"] = float_lists[0]
                parameters["sample2"] = float_lists[1]
        
        return parameters

    def select_tools_with_reasoning(self, user_query: str) -> ToolSelectionResult:
        """Tool selection with reasoning"""
        logger.info("Performing tool selection with reasoning...")
        
        # Extract parameters
        extracted_params = self.extract_parameters_from_query(user_query)
        
        # Build selection prompt
        selection_prompt = self.build_tool_selection_prompt(user_query)
        
        messages = [
            {"role": "system", "content": selection_prompt},
            {"role": "user", "content": user_query}
        ]
        
        call_kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        call_kwargs.update(self.generate_args)
        
        try:
            resp = self.client.chat.completions.create(**call_kwargs)
            content = resp.choices[0].message.content
            
            try:
                selection_result = json.loads(content)
            except json.JSONDecodeError:
                selection_result = self._fallback_tool_selection(user_query)
            
            # Merge extracted parameters with LLM-generated parameters
            if extracted_params:
                selection_result["parameters"] = {**selection_result.get("parameters", {}), **extracted_params}
            
            # Detailed reasoning
            reasoning_prompt = self.build_reasoning_prompt(
                user_query, 
                selection_result.get("selected_tool", "")
            )
            
            reasoning_messages = [
                {"role": "system", "content": reasoning_prompt},
                {"role": "user", "content": user_query}
            ]
            
            reasoning_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=reasoning_messages,
                temperature=0.3
            )
            
            detailed_reasoning = reasoning_resp.choices[0].message.content
            
            result = ToolSelectionResult(
                selected_tool=selection_result.get("selected_tool", ""),
                confidence=selection_result.get("confidence", 0.5),
                reasoning=detailed_reasoning,
                parameters=selection_result.get("parameters", {}),
                alternatives=[
                    ToolCandidate(
                        name=alt.get("name", ""),
                        description=TOOLS.get(alt.get("name", ""), {}).get("description", ""),
                        confidence=alt.get("confidence", 0.0),
                        reasoning=alt.get("reasoning", ""),
                        parameters=alt.get("parameters", {})
                    )
                    for alt in selection_result.get("alternatives", [])
                ],
                execution_plan=selection_result.get("execution_plan", "")
            )
            
            # Record to history
            self.tool_selection_history.append({
                "query": user_query,
                "selected_tool": result.selected_tool,
                "confidence": result.confidence,
                "reasoning": result.reasoning[:200] + "..." if len(result.reasoning) > 200 else result.reasoning
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return self._fallback_tool_selection(user_query)

    def _fallback_tool_selection(self, user_query: str) -> ToolSelectionResult:
        """Fallback tool selection"""
        logger.warning("Using fallback tool selection")
        
        # Simple keyword matching
        query_lower = user_query.lower()
        
        if "t-test" in query_lower or "t test" in query_lower:
            if "paired" in query_lower or "dependent" in query_lower:
                selected_tool = "paired_t_test"
            else:
                selected_tool = "independent_t_test"
        elif "anova" in query_lower:
            selected_tool = "one_way_anova"
        elif "mann" in query_lower or "whitney" in query_lower:
            selected_tool = "mann_whitney_u_test"
        elif "wilcoxon" in query_lower:
            selected_tool = "wilcoxon_test"
        elif "kruskal" in query_lower:
            selected_tool = "kruskal_wallis_test"
        elif "bayes" in query_lower:
            selected_tool = "bayes_factor_ttest"
        elif "granger" in query_lower or "causality" in query_lower:
            selected_tool = "granger_causality"
        else:
            selected_tool = "independent_t_test"
        
        # Extract parameters
        extracted_params = self.extract_parameters_from_query(user_query)
        
        return ToolSelectionResult(
            selected_tool=selected_tool,
            confidence=0.5,
            reasoning=f"Fallback selection based on keywords in query: {user_query}",
            parameters=extracted_params,
            alternatives=[],
            execution_plan="Execute the selected tool with extracted parameters"
        )

    def validate_tool_parameters(self, tool_name: str, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate tool parameters"""
        if tool_name not in TOOLS:
            return False, f"Unknown tool: {tool_name}"
        
        tool_info = TOOLS[tool_name]
        required_params = tool_info.get("args", [])
        
        # Check required parameters
        for param in required_params:
            if param not in parameters:
                return False, f"Missing required parameter: {param} ({tool_info.get('arg_types', {}).get(param, 'unknown type')})"
        
        return True, "Parameters validated successfully"

    def execute_with_validation(self, tool_selection: ToolSelectionResult) -> Dict[str, Any]:
        """Execute and validate"""
        # Validate parameters
        is_valid, validation_msg = self.validate_tool_parameters(
            tool_selection.selected_tool, 
            tool_selection.parameters
        )
        
        if not is_valid:
            return {
                "error": validation_msg,
                "tool_selection": tool_selection
            }
        
        # Execute tool
        try:
            tool_info = TOOLS[tool_selection.selected_tool]
            func = tool_info["func"]
            
            # Build call parameters based on parameter type
            if "*samples" in tool_selection.parameters:
                # Handle variable arguments
                samples = tool_selection.parameters["*samples"]
                result = func(*samples)
            elif "data" in tool_selection.parameters:
                # Handle data parameters
                data = tool_selection.parameters["data"]
                result = func(data)
            else:
                # Handle regular parameters
                result = func(**tool_selection.parameters)
            
            return {
                "success": True,
                "tool_selection": tool_selection,
                "execution_result": result,
                "validation": validation_msg
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "tool_selection": tool_selection,
                "validation": validation_msg
            }

    def __call__(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """Sync call: user_query -> tool selection reasoning -> execution -> result"""
        logger.info(f"Processing query: {user_query}")
        
        # Tool selection reasoning
        tool_selection = self.select_tools_with_reasoning(user_query)
        
        # Execute and validate
        execution_result = self.execute_with_validation(tool_selection)
        
        return {
            "query": user_query,
            "tool_selection": tool_selection,
            "execution_result": execution_result,
            "confidence": tool_selection.confidence,
            "reasoning": tool_selection.reasoning
        }

    async def acall(self, user_query: str, **kwargs) -> Dict[str, Any]:
        """Async call"""
        logger.info(f"Async processing query: {user_query}")
        
        # Check if in main thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Async tool selection reasoning
        tool_selection = await self._async_select_tools_with_reasoning(user_query)
        
        # Async execute and validate
        execution_result = await self._async_execute_with_validation(tool_selection)
        
        return {
            "query": user_query,
            "tool_selection": tool_selection,
            "execution_result": execution_result,
            "confidence": tool_selection.confidence,
            "reasoning": tool_selection.reasoning
        }

    async def _async_select_tools_with_reasoning(self, user_query: str) -> ToolSelectionResult:
        """Async tool selection reasoning"""
        # Extract parameters
        extracted_params = self.extract_parameters_from_query(user_query)
        
        selection_prompt = self.build_tool_selection_prompt(user_query)
        
        messages = [
            {"role": "system", "content": selection_prompt},
            {"role": "user", "content": user_query}
        ]
        
        call_kwargs = dict(
            model=self.model_name,
            messages=messages,
            temperature=0.1
        )
        call_kwargs.update(self.generate_args)
        
        # 检查是否有async_client可用
        if self.async_client:
            resp = await self.async_client.chat.completions.create(**call_kwargs)
        elif self.client:
            # 使用线程池执行同步调用
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None, 
                lambda: self.client.chat.completions.create(**call_kwargs)
            )
        else:
            # 如果都没有，使用模型的调用方法
            loop = asyncio.get_event_loop()
            model_response = await loop.run_in_executor(
                None,
                lambda: self.model(messages, **self.generate_args)
            )
            # 模拟OpenAI响应格式
            class MockResponse:
                def __init__(self, content):
                    self.choices = [type('obj', (object,), {
                        'message': type('obj', (object,), {'content': content})()
                    })()]
            
            resp = MockResponse(model_response.text)
        
        content = resp.choices[0].message.content
        
        try:
            selection_result = json.loads(content)
        except json.JSONDecodeError:
            selection_result = self._fallback_tool_selection(user_query)
        
        # Merge extracted parameters with LLM-generated parameters
        if extracted_params:
            selection_result["parameters"] = {**selection_result.get("parameters", {}), **extracted_params}
        
        # Async detailed reasoning
        reasoning_prompt = self.build_reasoning_prompt(
            user_query, 
            selection_result.get("selected_tool", "")
        )
        
        reasoning_messages = [
            {"role": "system", "content": reasoning_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # 同样的逻辑处理reasoning调用
        if self.async_client:
            reasoning_resp = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=reasoning_messages,
                temperature=0.3
            )
        elif self.client:
            loop = asyncio.get_event_loop()
            reasoning_resp = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=reasoning_messages,
                    temperature=0.3
                )
            )
        else:
            loop = asyncio.get_event_loop()
            model_response = await loop.run_in_executor(
                None,
                lambda: self.model(reasoning_messages, temperature=0.3)
            )
            reasoning_resp = MockResponse(model_response.text)
        
        detailed_reasoning = reasoning_resp.choices[0].message.content
        
        result = ToolSelectionResult(
            selected_tool=selection_result.get("selected_tool", ""),
            confidence=selection_result.get("confidence", 0.5),
            reasoning=detailed_reasoning,
            parameters=selection_result.get("parameters", {}),
            alternatives=[
                ToolCandidate(
                    name=alt.get("name", ""),
                    description=TOOLS.get(alt.get("name", ""), {}).get("description", ""),
                    confidence=alt.get("confidence", 0.0),
                    reasoning=alt.get("reasoning", ""),
                    parameters=alt.get("parameters", {})
                )
                for alt in selection_result.get("alternatives", [])
            ],
            execution_plan=selection_result.get("execution_plan", "")
        )
        
        # Record to history
        self.tool_selection_history.append({
            "query": user_query,
            "selected_tool": result.selected_tool,
            "confidence": result.confidence,
            "reasoning": result.reasoning[:200] + "..." if len(result.reasoning) > 200 else result.reasoning
        })
        
        return result

    async def _async_execute_with_validation(self, tool_selection: ToolSelectionResult) -> Dict[str, Any]:
        """Async execute and validate"""
        # Validate parameters
        is_valid, validation_msg = self.validate_tool_parameters(
            tool_selection.selected_tool, 
            tool_selection.parameters
        )
        
        if not is_valid:
            return {
                "error": validation_msg,
                "tool_selection": tool_selection
            }
        
        # Async execute tool
        try:
            tool_info = TOOLS[tool_selection.selected_tool]
            func = tool_info["func"]
            
            # Use thread pool to execute sync functions
            loop = asyncio.get_event_loop()
            
            if "*samples" in tool_selection.parameters:
                # Handle variable arguments
                samples = tool_selection.parameters["*samples"]
                result = await loop.run_in_executor(None, lambda: func(*samples))
            elif "data" in tool_selection.parameters:
                # Handle data parameters
                data = tool_selection.parameters["data"]
                result = await loop.run_in_executor(None, lambda: func(data))
            else:
                # Handle regular parameters
                result = await loop.run_in_executor(None, lambda: func(**tool_selection.parameters))
            
            return {
                "success": True,
                "tool_selection": tool_selection,
                "execution_result": result,
                "validation": validation_msg
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "tool_selection": tool_selection,
                "validation": validation_msg
            }

    def get_tool_selection_history(self) -> List[Dict[str, Any]]:
        """Get tool selection history"""
        return self.tool_selection_history

    def get_tool_usage_statistics(self) -> Dict[str, int]:
        """Get tool usage statistics"""
        stats = {}
        for entry in self.tool_selection_history:
            tool = entry["selected_tool"]
            stats[tool] = stats.get(tool, 0) + 1
        return stats

    def analyze_data(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze provided data with a two-stage LLM-tool pipeline and return raw JSON text in 'analysis'."""
        try:
            # Stage 0: prepare data sampling for AI planning
            samples = self._collect_samples_for_prompt(data, max_samples=8)

            # Stage 1: Ask AI to plan tool calls (STRICT JSON ONLY, tools from TOOLS)
            planning_prompt = self._create_tool_planning_prompt(data, samples, context)
            planning_messages = [
                SystemMessage(content=planning_prompt),
                UserMessage(content="Plan statistical tool calls based on provided samples and dataset context.")
            ]

            # Call model to get tool plan (strict JSON text)
            if self.client:
                plan_resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": m.role, "content": m.content} for m in planning_messages],
                    temperature=0.1,
                    **self.generate_args
                )
                plan_text = plan_resp.choices[0].message.content
            else:
                # Fallback generic interfaces if available
                if hasattr(self.model, 'generate'):
                    plan_text = self.model.generate(planning_messages, **self.generate_args).content
                elif hasattr(self.model, 'chat'):
                    plan_text = self.model.chat(planning_messages, **self.generate_args).content
                else:
                    raise AttributeError("No suitable method found for planning response")

            # Parse tool calls JSON (internal use only for execution)
            try:
                plan_json = json.loads(plan_text)
                tool_calls = plan_json.get("tool_calls", [])
            except Exception:
                tool_calls = []

            # Execute tool calls and collect outputs
            executed = self.plan_and_execute_tools(tool_calls)

            # Stage 2: Ask AI to produce final analysis JSON (STRICT JSON ONLY)
            final_prompt = self._create_final_analysis_prompt(data, context, executed, samples)
            final_messages = [
                SystemMessage(content=final_prompt),
                UserMessage(content="Generate the final analysis JSON strictly following the requested shape.")
            ]

            if self.client:
                final_resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": m.role, "content": m.content} for m in final_messages],
                    temperature=0.2,
                    **self.generate_args
                )
                final_text = final_resp.choices[0].message.content
            else:
                if hasattr(self.model, 'generate'):
                    final_text = self.model.generate(final_messages, **self.generate_args).content
                elif hasattr(self.model, 'chat'):
                    final_text = self.model.chat(final_messages, **self.generate_args).content
                else:
                    raise AttributeError("No suitable method found for final response")

            # Return raw text in 'analysis' so the saver can write it directly
            return {
                "status": "success",
                "analysis": final_text,
                "metadata": {
                    "model_used": self.model_name,
                    "data_summary": self._summarize_data(data),
                    "context_provided": context is not None,
                    "tool_calls_count": len(tool_calls),
                }
            }

        except Exception as e:
            logger.error(f"Error in analyze_data: {e}")
            return {
                "status": "error",
                "error": str(e),
                "fallback_analysis": self._create_fallback_analysis(data, context)
            }
    
    def _collect_samples_for_prompt(self, data: Any, max_samples: int = 8) -> List[Dict[str, Any]]:
        """Recursively collect numeric lists from aggregated data for prompting."""
        samples = []

        def is_numeric_list(lst):
            try:
                if not isinstance(lst, list) or len(lst) < 3:
                    return False
                cnt = 0
                for x in lst:
                    if isinstance(x, (int, float)):
                        cnt += 1
                    else:
                        return False
                return cnt >= 3
            except Exception:
                return False

        def walk(obj, path="root"):
            nonlocal samples
            if len(samples) >= max_samples:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{path}.{k}")
                    if len(samples) >= max_samples:
                        return
            elif isinstance(obj, list):
                # record if numeric list
                if is_numeric_list(obj):
                    samples.append({"path": path, "values": obj[:50]})
                else:
                    for idx, v in enumerate(obj[:100]):
                        walk(v, f"{path}[{idx}]")
                        if len(samples) >= max_samples:
                            return

        # Prefer processed data buckets when present
        try:
            if isinstance(data, dict) and "processed" in data and isinstance(data["processed"], dict):
                for k, v in data["processed"].items():
                    walk(v, f"processed.{k}")
                    if len(samples) >= max_samples:
                        break
        except Exception:
            pass

        # Fallback: scan the entire data
        if len(samples) < max_samples:
            walk(data)

        # Deduplicate by values string
        dedup = []
        seen = set()
        for s in samples:
            key = str(s.get("values"))
            if key not in seen:
                seen.add(key)
                dedup.append(s)
        return dedup[:max_samples]

    def _create_tool_planning_prompt(self, data: Any, samples: List[Dict[str, Any]], context: Dict[str, Any] = None) -> str:
        """Prompt for planning tool calls strictly based on tool_registry TOOLS."""
        # Build tools catalog from TOOLS
        tool_specs = []
        for name, spec in TOOLS.items():
            args = spec.get("args", [])
            desc = spec.get("description", "")
            tool_specs.append(f"- {name} | args: {', '.join(args)} | {desc}")

        # Context notes
        overview = ""
        if context and isinstance(context, dict):
            ov = context.get("data_overview", {})
            overview = f"Total files: {ov.get('total_files')}, Categories: {', '.join(ov.get('categories', []))}"

        # Sample fragments for guidance
        sample_texts = []
        for i, s in enumerate(samples[:8], start=1):
            sample_texts.append(f"Sample {i} from {s['path']}: {s['values']}")

        prompt = (
            "You are a statistical planner. Based on the provided dataset context and numeric samples, "
            "plan a small set of tool invocations using ONLY the tools listed below. "
            "Return STRICT JSON ONLY with the shape:\n"
            "{\n"
            '  "tool_calls": [\n'
            '    {"tool_name": "<from TOOLS>", "parameters": { /* required args only; numeric arrays from samples */ } }\n'
            "  ]\n"
            "}\n"
            "Rules:\n"
            "- Use 1 to 4 tool calls.\n"
            "- Tools MUST be chosen from the following catalog and adhere to required args:\n"
            f"{chr(10).join(tool_specs)}\n"
            "- For parametric/non-parametric tests, pick relevant pairs or groups from the samples.\n"
            "- For time series tools, only if you can construct 2D data from samples.\n"
            "- DO NOT include markdown fences or any text outside JSON.\n"
            "- Parameters MUST be numeric lists drawn from the samples.\n"
            "- If insufficient samples for a specific tool, skip that tool.\n"
            "\n"
            f"Dataset overview: {overview}\n"
            "Numeric samples:\n"
            f"{chr(10).join(sample_texts)}\n"
        )
        return prompt

    def plan_and_execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and execute planned tool calls, returning their outputs."""
        executed = []
        for call in tool_calls:
            try:
                tname = call.get("tool_name")
                params = call.get("parameters", {}) or {}
                # validate
                valid, msg = self.validate_tool_parameters(tname, params)
                if not valid:
                    executed.append({
                        "tool_name": tname,
                        "parameters": params,
                        "status": "validation_error",
                        "message": msg
                    })
                    continue
                # execute
                result = self.execute_with_validation(
                    ToolSelectionResult(
                        selected_tool=tname,
                        confidence=0.8,
                        reasoning="Planned execution from AI tool calls",
                        parameters=params,
                        alternatives=[],
                        execution_plan=f"Execute {tname} with provided parameters"
                    )
                )
                executed.append({
                    "tool_name": tname,
                    "parameters": params,
                    "status": "success" if "success" in result else "error",
                    "result": result.get("execution_result", result),
                })
            except Exception as e:
                executed.append({
                    "tool_name": call.get("tool_name"),
                    "parameters": call.get("parameters", {}),
                    "status": "exception",
                    "message": str(e)
                })
        return executed

    def _create_final_analysis_prompt(
        self,
        data: Any,
        context: Dict[str, Any],
        executed: List[Dict[str, Any]],
        samples: List[Dict[str, Any]]
    ) -> str:
        """Prompt for final analysis JSON strictly following target shape and incorporating tool outputs."""
        # Compose a concise execution summary
        exec_lines = []
        for item in executed[:6]:
            t = item.get("tool_name")
            st = item.get("status")
            exec_lines.append(f"- {t}: {st}")

        prompt = (
            "You are an expert data analyst. Using the dataset context, numeric samples, and the executed tool outputs, "
            "produce the FINAL ANALYSIS as STRICT JSON ONLY. DO NOT include markdown fences or any prose outside JSON.\n"
            "The JSON must have top-level keys: info, data_summary, models, nonparametric, conclusions.\n"
            "Guidelines:\n"
            "- info: include generated_at (ISO time), endpoint_definition, last_k_steps (if unknown set to null), data_source (short description), methods (list).\n"
            "- data_summary: include overall (n, mean, std, min, max) when applicable, and optionally grouped summaries derived from samples.\n"
            "- models: if any parametric model-like interpretation emerges, summarize in a structured way; else {}.\n"
            "- nonparametric: include test names and key statistics (e.g., H, p_value, medians) from executed tools when available; else {}.\n"
            "- conclusions: provide primary conclusions and concise notes.\n"
            "- If a section does not apply, use an empty object.\n"
            "Inputs for your analysis (concise):\n"
            "Executed tool calls:\n"
            f"{chr(10).join(exec_lines)}\n"
            "You may integrate numeric results from the executed outputs logically.\n"
            "Return VALID JSON ONLY."
        )
        return prompt
    
    def _create_default_analysis_prompt(self, data: Any, context: Dict[str, Any] = None) -> str:
        """
        Create a default analysis prompt when none is provided in context
        """
        prompt = "Please analyze the following data and provide insights. When possible, structure your output similar to data_analysis.json (soft requirement).\n\n"
        
        # Add context information if available
        if context:
            if "data_overview" in context:
                overview = context["data_overview"]
                prompt += f"Data Overview:\n"
                prompt += f"- Total files: {overview.get('total_files', 'Unknown')}\n"
                prompt += f"- Categories: {', '.join(overview.get('categories', []))}\n\n"
            
            if "file_metadata" in context:
                prompt += "File Metadata:\n"
                for meta in context["file_metadata"][:5]:  # Limit to first 5 files
                    prompt += f"- {meta.get('name', 'Unknown')}: {meta.get('type', 'Unknown')} ({meta.get('category', 'Unknown')})\n"
                prompt += "\n"
        
        # Add data summary
        prompt += f"Data to analyze:\n{self._format_data_for_prompt(data)}\n\n"
        prompt += "Please provide:\n"
        prompt += "1. Key patterns and trends\n"
        prompt += "2. Statistical insights\n"
        prompt += "3. Potential issues or anomalies\n"
        prompt += "4. Recommendations for further analysis\n\n"
        prompt += (
            "Output guidance (soft): Return a JSON with top-level sections: 'info', 'data_summary', 'models', 'nonparametric', 'conclusions'. "
            "If sections do not apply, omit them and explain briefly."
        )
        
        return prompt
    
    def _format_data_for_prompt(self, data: Any) -> str:
        """
        Format data for inclusion in prompt
        """
        if isinstance(data, dict):
            # Limit the size of data shown in prompt
            if len(str(data)) > 2000:
                return f"Large dataset with {len(data)} top-level keys: {list(data.keys())[:10]}..."
            return json.dumps(data, indent=2)[:2000] + ("..." if len(json.dumps(data, indent=2)) > 2000 else "")
        elif isinstance(data, list):
            return f"List with {len(data)} items: {str(data[:5])}{'...' if len(data) > 5 else ''}"
        else:
            return str(data)[:1000] + ("..." if len(str(data)) > 1000 else "")
    
    def _summarize_data(self, data: Any) -> Dict[str, Any]:
        """
        Create a summary of the data for metadata
        """
        if isinstance(data, dict):
            return {
                "type": "dictionary",
                "keys_count": len(data),
                "top_keys": list(data.keys())[:10]
            }
        elif isinstance(data, list):
            return {
                "type": "list",
                "length": len(data),
                "sample_items": data[:3] if data else []
            }
        else:
            return {
                "type": type(data).__name__,
                "length": len(str(data))
            }
    
    def _create_fallback_analysis(self, data: Any, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a basic fallback analysis when LLM analysis fails
        """
        summary = self._summarize_data(data)
        
        analysis = {
            "basic_summary": summary,
            "note": "This is a fallback analysis due to LLM processing error"
        }
        
        if isinstance(data, dict):
            analysis["key_statistics"] = {
                "total_keys": len(data),
                "nested_structures": sum(1 for v in data.values() if isinstance(v, (dict, list)))
            }
        elif isinstance(data, list):
            analysis["list_statistics"] = {
                "total_items": len(data),
                "item_types": list(set(type(item).__name__ for item in data[:100]))
            }
        
        return analysis

def _load_json_file(path):
    try:
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def collect_project_analysis_data(project_dir: str) -> Dict[str, Any]:
    from pathlib import Path
    project_path = Path(project_dir).resolve()
    analysis_dir = project_path / "analysis"
    analysis_data_dir = analysis_dir / "data"
    processed_dir = analysis_data_dir / "processed"

    aggregated = {
        "analysis_results": _load_json_file(analysis_data_dir / "analysis_results.json"),
        "analysis_conclusions": _load_json_file(analysis_data_dir / "analysis_conclusions.json"),
        "collection_summary_all_groups": _load_json_file(analysis_data_dir / "collection_summary_all_groups.json"),
        "figures_analysis_combine": _load_json_file(analysis_data_dir / "figures_analysis_combine.json"),
        "figures_analysis_combine_en": _load_json_file(analysis_data_dir / "figures_analysis_combine_en.json"),
        "top_level_analysis_report": _load_json_file(analysis_dir / "analysis_report.json"),
        "processed": {},
    }

    if processed_dir.exists():
        for jf in sorted(processed_dir.glob("*.json")):
            name = jf.stem
            aggregated["processed"][name] = _load_json_file(jf)

    file_metadata = []
    categories = []
    def add_meta(path, category):
        from pathlib import Path
        p = Path(path)
        if p.exists():
            for jf in ([p] if p.is_file() else list(p.glob("*.json"))):
                file_metadata.append({
                    "name": jf.name,
                    "type": "json",
                    "category": category,
                    "size_bytes": jf.stat().st_size if jf.exists() else 0,
                })
            if category not in categories:
                categories.append(category)

    add_meta(analysis_data_dir / "analysis_results.json", "data")
    add_meta(analysis_data_dir / "analysis_conclusions.json", "data")
    add_meta(analysis_data_dir / "collection_summary_all_groups.json", "data")
    add_meta(analysis_data_dir / "figures_analysis_combine.json", "data")
    add_meta(analysis_data_dir / "figures_analysis_combine_en.json", "data")
    add_meta(analysis_data_dir / "analysis_report.json", "analysis")
    add_meta(processed_dir, "processed")

    context = {
        "data_overview": {
            "total_files": len(file_metadata),
            "categories": categories,
        },
        "file_metadata": file_metadata,
        "project_path": str(project_path),
        "analysis_data_dir": str(analysis_data_dir),
    }
    return {"data": aggregated, "context": context}

# module-level helpers (save function and main)
def save_llm_analysis_output(output_dir: str, analysis_result: Any, raw_json_only: bool = True) -> str:
    """
    Save the LLM output to a file.
    - If raw_json_only is True: write the content directly as text (expected to be strict JSON returned by the model).
    - Otherwise: dump the Python object as JSON (fallback).
    """
    from pathlib import Path
    import json
    from datetime import datetime

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.parent / f"data_analysis.json"

    if raw_json_only:
        # When raw_json_only, we write exactly what the model responded (no parsing).
        # Accept either a dict with 'analysis' str or a pure str.
        if isinstance(analysis_result, dict) and isinstance(analysis_result.get("analysis"), str):
            content = analysis_result["analysis"]
        elif isinstance(analysis_result, str):
            content = analysis_result
        else:
            # If model returned non-string, fall back to JSON dump (still not parsing the inner structure).
            content = json.dumps(analysis_result, ensure_ascii=False, indent=2)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(content)
    else:
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)

    return str(out_path)

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Run EnhancedStatAgentLLMAdapter on a project folder and output analysis JSON.")
    parser.add_argument("--project", required=True, help="Path to the project folder (e.g., /data/.../projects/social_dynamics_combine)")
    parser.add_argument("--config-name", default="openai-gpt4o", help="Model config name defined in config/model_config.json (default: openai-gpt4o)")
    parser.add_argument("--model-name", default=None, help="Alternative: specify a model_name if not using config-name")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max tokens for generation")
    parser.add_argument("--raw-json-only", action="store_true", default=True, help="If set, write model's JSON text directly without parsing (default: True)")
    args = parser.parse_args()

    project_path = Path(args.project).resolve()
    if not project_path.exists():
        raise FileNotFoundError(f"Project path not found: {project_path}")

    collected = collect_project_analysis_data(str(project_path))
    data = collected["data"]
    context = collected["context"]

    adapter_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "generate_args": {},
    }
    if args.config_name:
        adapter_kwargs["config_name"] = args.config_name
    if args.model_name:
        adapter_kwargs["model_name"] = args.model_name

    adapter = EnhancedStatAgentLLMAdapter(**adapter_kwargs)
    result = adapter.analyze_data(data, context=context)

    output_dir = project_path / "analysis" / "data"
    out_path = save_llm_analysis_output(str(output_dir), result, raw_json_only=args.raw_json_only)
    print(f"LLM analysis saved to: {out_path}")

if __name__ == "__main__":
    main()