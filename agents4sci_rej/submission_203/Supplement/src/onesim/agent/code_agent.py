from typing import Any, Dict, List, Tuple
import re
from loguru import logger
from .base import AgentBase
import json
from onesim.models import JsonBlockParser
from onesim.models.core.message import Message
import os
import time

class CodeAgent(AgentBase):
    """An agent that generates code for agents and events based on a natural language description, actions, and events definitions by invoking an LLM."""

    def __init__(
        self,
        model_config_name: str,
        sys_prompt: str = '',
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sys_prompt=sys_prompt,
            model_config_name=model_config_name,
        )
        if sys_prompt == '':
            self.sys_prompt = """You are an AI assistant that generates Python code for agent-based systems. Given a natural language description of a Multi-Agent system, and definitions of actions and events, your task is to produce agent classes, event classes, and handler functions that follow specified templates. Each agent and event should use the same framework, but with specific content and instructions unique to each. Use triple quotes for multi-line strings, and ensure the code is complete and ready to be executed."""
        if kwargs:
            logger.warning(
                f"Unused keyword arguments are provided: {kwargs}",
            )

        # Add token usage tracking
        self.token_usage = 0
        self.verification_issues = []
        
        self.sample_handler_code='''
async def receive_word(self, event: Event) -> List[Event]:
        chosen_word = event.target_word
        observation = f"Chosen word: {chosen_word}"
        instruction = """Please generate a description for the given word without directly mentioning it. 
    The description should be helpful for guessing the word but should not reveal it directly.
    Please return the information in the following JSON format:

    {
    "description": "<A detailed description of the word>",
    "target_ids": ["<The string ID of the Guesser agent>"]
    }
    """
        
        result = await self.generate_reaction(instruction, observation)
        
        description = result.get('description',None) 
        target_ids = result.get('target_ids',None)
        if not isinstance(target_ids,list):
            target_ids = [target_ids]

        # Prepare and send the DescriptionProvidedEvent to the Guesser
        events=[]
        for target_id in target_ids:
            description_event = DescriptionProvidedEvent(self.profile_id, target_id, description)
            events.append(description_event)
        
        return events'''
        self.event_imports = '''
from onesim.events import Event
from typing import Dict, List, Any        
from datetime import datetime
'''
        self.sample_event_code='''
class StartEvent(Event):
    def __init__(self, from_agent_id: str, to_agent_id: str):
        super().__init__(from_agent_id=from_agent_id,
                         to_agent_id=to_agent_id)
'''
        self.generated_event_classes = []

    def build_action_graph(self, actions: Dict[str, List[Dict]], events: Dict[int, Dict]) -> Dict[Tuple[str, str], List[Tuple[str, str]]]:
        action_graph = {}
        for event in events.values():
            from_agent = event['from_agent_type']
            from_action = event['from_action_name']
            to_agent = event['to_agent_type']
            to_action = event['to_action_name']
            key = (from_agent, from_action)
            action_graph.setdefault(key, []).append((to_agent, to_action))
        return action_graph

    def get_start_actions(self, action_graph: Dict[Tuple[str, str], List[Tuple[str, str]]], actions: Dict[str, List[Dict]]) -> List[Tuple[str, str]]:
        # 找出所有动作节点
        all_actions = set()
        for agent_type, action_list in actions.items():
            for action in action_list:
                all_actions.add((agent_type, action['name']))
        # 计算入度
        in_degree = {action: 0 for action in all_actions}
        for targets in action_graph.values():
            for target in targets:
                if target in in_degree:
                    in_degree[target] += 1
        # 入度为零的动作
        start_actions = [action for action, degree in in_degree.items() if degree == 0]
        return start_actions

    def generate_code_phased(self, description: str, actions: Dict[str, List[Dict]], events: Dict[int, Dict], env_path: str, status_dict: Dict[str, Any], max_iterations: int = 3) -> None:
        """
        Generate code in phases and update status to status_dict
        
        Args:
            description: System description
            actions: Action definitions
            events: Event definitions
            env_path: Environment path
            status_dict: Status dictionary for tracking and updating progress
            max_iterations: Maximum number of fix iterations
        """
        # Phase 1: Generate initial code
        status_dict["phase"] = 1
        status_dict["content"] = "Starting code generation...\n"
        
        try:
            agent_code_dict, event_code = self.generate_initial_code(
                description, actions, events, env_path, status_dict
            )
            
            status_dict["generated_code"] = agent_code_dict
            status_dict["event_code"] = event_code
            status_dict["progress"] = 0.4
            
            # Add generated code to content
            status_dict["content"] += "\nGenerated code:\n"
            for agent_type, code in agent_code_dict.items():
                status_dict["content"] += f"\n--- {agent_type} ---\n{code}\n"
            status_dict["content"] += f"\n--- Events ---\n{event_code}\n"
            
            # Phase 2: Validate and fix code
            status_dict["phase"] = 2
            if "all_content" not in status_dict:
                status_dict["all_content"] = ""
            status_dict["all_content"] += status_dict["content"]
            status_dict["content"] = "Starting code refinement...\n"
            
            verification_results = self.check_code(
                agent_code_dict, event_code, description, actions, events
            )
            
            has_issues = any(len(issues) > 0 for issues in verification_results.values())
            status_dict["issues"] = verification_results
            
            if not has_issues:
                status_dict["content"] += "Validation passed, no issues found\n"
                status_dict["progress"] = 0.8
            else:
                # Format issues for display
                status_dict["content"] += "Validation found the following issues:\n"
                for class_name, issues in verification_results.items():
                    if issues:
                        status_dict["content"] += f"- {class_name}:\n"
                        for issue in issues:
                            status_dict["content"] += f"  - {issue['type']}: {issue['description']}\n"
            
            # Fix code if necessary (part of phase 2)
            if has_issues:
                iteration = 0
                while iteration < max_iterations and has_issues:
                    status_dict["fix_iteration"] = iteration + 1
                    status_dict["content"] += f"Performing fix iteration {iteration + 1}...\n"
                    
                    # Perform fixes
                    agent_code_dict, event_code = self.fix_code(
                        agent_code_dict, event_code, verification_results, actions, events
                    )
                    
                    # Update status
                    status_dict["generated_code"] = agent_code_dict
                    status_dict["event_code"] = event_code
                    status_dict["progress"] = 0.6 + (0.4 * (iteration + 1) / max_iterations)
                    
                    # Add fixed code to content
                    status_dict["content"] += f"\nCode after fix iteration {iteration + 1}:\n"
                    for agent_type, code in agent_code_dict.items():
                        status_dict["content"] += f"\n--- {agent_type} ---\n{code}\n"
                    status_dict["content"] += f"\n--- Events ---\n{event_code}\n"
                    
                    # Verify again
                    verification_results = self.check_code(
                        agent_code_dict, event_code, description, actions, events
                    )
                    
                    has_issues = any(len(issues) > 0 for issues in verification_results.values())
                    status_dict["issues"] = verification_results
                    
                    if not has_issues:
                        status_dict["content"] += "Fix successful! Code validation passed\n"
                        break
                    else:
                        status_dict["content"] += "Fix failed! Code issues:\n"
                        # Format current issues
                        for class_name, issues in verification_results.items():
                            if issues:
                                status_dict["content"] += f"- {class_name}:\n"
                                for issue in issues:
                                    status_dict["content"] += f"  - {issue['type']}: {issue['description']}\n"
                    
                    iteration += 1
                
                if has_issues:
                    status_dict["content"] += f"Reached maximum fix iterations ({max_iterations}), unresolved issues remain\n"
            
            # Generate SimEnv code once
            start_event_code = next((code for code in self.generated_event_classes if "StartEvent" in code), "")
            simenv_code = self.generate_simenv_subclass_code(description, start_event_code)
            status_dict["simenv_code"] = simenv_code
            
            # Save code to files
            self.save_phased_code(agent_code_dict, event_code, env_path, actions, events, status_dict)
            
            # Add SimEnv code to content
            status_dict["content"] += f"\n--- SimEnv ---\n{simenv_code}\n"
            
            # Set phase to complete
            status_dict["phase"] = 0
            status_dict["progress"] = 1.0
            status_dict["content"] += "Code generation complete!"
            
        except Exception as e:
            status_dict["phase"] = -1  # Error, set to complete
            status_dict["content"] = f"Error occurred during code generation: {str(e)}"
            raise

    def generate_initial_code(self, description: str, actions: Dict[str, List[Dict]], events: Dict[int, Dict], env_path: str, status_dict: Dict[str, Any]) -> Tuple[Dict[str, str], str]:
        """Generate initial code, return agent_code_dict and event_code"""
        self.generated_event_classes = []  # Reset for each generation
        agent_code_dict = {}
        
        # Build topology graph
        action_graph = self.build_action_graph(actions, events)
        
        # Traverse graph and generate code
        visited_actions = set()
        queue = [('EnvAgent', 'start')]
        
        # Initialize agent code dictionary
        status_dict["content"] += "Initializing agent classes...\n"
        for agent_type, action_list in actions.items():
            agent_code_dict[agent_type] = self.initialize_agent_class(agent_type, action_list, events)
            status_dict["content"] += f"- Initialized {agent_type} agent class\n"
            status_dict["content"] += f"```python\n{agent_code_dict[agent_type]}\n```\n\n"
            status_dict["progress"] = 0.05
        
        start_event_code = ""
        total_actions = sum(len(action_list) for action_list in actions.values())
        processed_actions = 0
        
        while queue:
            current_agent_type, current_action_name = queue.pop(0)
            if (current_agent_type, current_action_name) in visited_actions:
                continue
            
            visited_actions.add((current_agent_type, current_action_name))
            status_dict["content"] += f"Processing {current_agent_type}.{current_action_name}...\n"
            
            # Collect relevant events
            incoming_events = [e for e in events.values() if e['to_agent_type'] == current_agent_type and e['to_action_name'] == current_action_name]
            outgoing_events = [e for e in events.values() if e['from_agent_type'] == current_agent_type and e['from_action_name'] == current_action_name]
            
            # Step 1: First generate next event classes
            next_event_classes = []
            for event_info in outgoing_events:
                status_dict["content"] += f"- Generating event class {event_info['event_name']}...\n"
                event_class_code = self.generate_event_class(event_info)
                status_dict["content"] += f"```python\n{event_class_code}\n```\n\n"
                if current_agent_type == 'EnvAgent' and current_action_name == 'start':
                    start_event_code += event_class_code + "\n"
                next_event_classes.append(event_class_code)
                self.generated_event_classes.append(event_class_code)
            
            if current_agent_type != 'EnvAgent':
                action_info = next((a for a in actions[current_agent_type] if a['name'] == current_action_name), None)
                if not action_info:
                    continue
                
                # Step 2: Use event information to generate handler
                status_dict["content"] += f"- Generating handler {action_info['name']}...\n"
                event_data_info = [self.extract_event_data(event_code) for event_code in next_event_classes]
                handler_code = self.generate_handler_code(description, current_agent_type, action_info, incoming_events, outgoing_events, event_data_info)
                status_dict["content"] += f"```python\n{handler_code}\n```\n\n"
                
                # Add handler code to agent class code
                agent_code_dict[current_agent_type] += f"\n{handler_code}\n"
                status_dict["content"] += f"- Updated {current_agent_type} agent class...\n"
                processed_actions += 1
                status_dict["progress"] = 0.05 + (0.35 * processed_actions / total_actions)
            
            # Add next actions to queue
            next_actions = action_graph.get((current_agent_type, current_action_name), [])
            for next_agent_type, next_action_name in next_actions:
                if (next_agent_type, next_action_name) not in visited_actions:
                    queue.append((next_agent_type, next_action_name))
        
        # Generate final event code
        event_code = self.generate_event_code()
        status_dict["content"] += "Generating final event code...\n"
        status_dict["content"] += f"```python\n{event_code}\n```\n\n"
        status_dict["content"] += "Initial code generation complete!\n"
        
        return agent_code_dict, event_code

    def check_code(self, agent_code_dict: Dict[str, str], event_code: str, description: str, actions: Dict[str, List[Dict]], events: Dict[int, Dict]) -> Dict[str, List[Dict[str, Any]]]:
        """验证生成的代码并返回问题"""
        # 合并代理代码
        all_code = ""
        for agent_type, code in agent_code_dict.items():
            all_code += agent_type + ":\n\n" + code + "\n\n"
        
        # 添加事件代码
        all_code += "Events:\n\n" + event_code
        
        # 执行验证
        verification_results = self.verify_code(all_code, description, actions, events)
        
        return verification_results

    def save_phased_code(self, agent_code_dict: Dict[str, str], event_code: str, env_path: str, actions: Dict[str, List[Dict]], events: Dict[int, Dict], status_dict: Dict[str, Any]) -> None:
        """Save generated code to files"""
        status_dict["content"] += "Saving code to files...\n"
        
        # Use the already generated SimEnv code
        simenv_code = status_dict.get("simenv_code")
        
        # Save code to files
        self.save_code_to_files(agent_code_dict, event_code, env_path, actions, events)
        
        # Save SimEnv code
        simenv_path = os.path.join(env_path, "SimEnv.py")
        with open(simenv_path, 'w', encoding='utf-8') as f:
            f.write(simenv_code)
        
        # Update code fields in actions and events
        self.update_workflow_with_code(agent_code_dict, event_code, actions, events)
        
        status_dict["content"] += "Code files saved successfully!\n"

    def update_workflow_with_code(self, agent_code_dict: Dict[str, str], event_code: str, actions: Dict[str, List[Dict]], events: Dict[int, Dict]) -> None:
        """Update code fields in actions and events dictionaries"""
        # Restructure code into ID-organized structure
        restructured = self.restructure_code(agent_code_dict, event_code, actions, events)
        
        # Update code in actions
        for agent_type, agent_actions in actions.items():
            if agent_type in restructured["agents"]:
                for action in agent_actions:
                    action_id = str(action.get("id"))
                    if action_id in restructured["agents"][agent_type]["handlers"]:
                        action["code"] = restructured["agents"][agent_type]["handlers"][action_id]["code"]
        
        # Update code in events
        for event_id, event_info in events.items():
            if str(event_id) in restructured["events"]["definitions"]:
                events[event_id]["code"] = restructured["events"]["definitions"][str(event_id)]["code"]

    def generate_code(self, description: str, actions: Dict[str, List[Dict]], events: Dict[int, Dict], env_path: str, max_iterations: int = 3) -> Dict[str, Any]:
        """
        原始的generate_code方法，现在实现为基于新的分阶段生成机制，保持兼容性
        """
        # 创建状态跟踪字典
        status_dict = {
            "phase": 1,
            "content": "",
            "generated_code": {},
            "event_code": "",
            "issues": {},
            "fix_iteration": 0,
            "progress": 0.0,
            "description": description
        }
        
        # 调用分阶段生成方法
        self.generate_code_phased(description, actions, events, env_path, status_dict, max_iterations)
        
        # 重构代码结果
        result = self.restructure_code(
            status_dict["generated_code"], status_dict["event_code"], actions, events
        )
        
        # 使用已经生成的SimEnv代码
        simenv_code = status_dict.get("simenv_code")
        if simenv_code is None:
            # Fallback in case it wasn't generated earlier
            start_event_code = next((code for code in self.generated_event_classes if "StartEvent" in code), "")
            simenv_code = self.generate_simenv_subclass_code(description, start_event_code)
        
        result['simenv'] = {'code': simenv_code}
        
        return result

    def generate_simenv_subclass_code(self, description: str, start_event_code: str) -> str:
        """
        调用LLM生成SimEnv(BasicSimEnv)子类的代码示例，包括_create_start_event方法。
        要求LLM根据上下文信息来实现_create_start_event从self.data中获取需要的信息（如有需求）。

        传入的description中包含对环境的描述以及StartEvent的上下文。
        """
        prompt = f"""
You are an AI programming assistant working on a multi-agent simulation system with a BasicSimEnv class and a predefined StartEvent event class.

Based on the following context, generate an example SimEnv class that inherits from BasicSimEnv. The class must implement the _create_start_event method with the following requirements:
- Extract necessary information from self.data (if required)
- Return a StartEvent instance
- Utilize the predefined StartEvent class: from .events import StartEvent

Context:
{description}

StartEvent Info:
{start_event_code}

Coding Guidelines:
1. Implement _create_start_event method using the context-specific requirements
2. Leverage information stored in self.data for event creation
3. Ensure type hints and proper event construction

Example Template:
```python
from onesim.simulator import BasicSimEnv
from onesim.events import Event
from .events import StartEvent

class SimEnv(BasicSimEnv):
    async def _create_start_event(self, target_id: str) -> Event:
        # Extract relevant information from self.data according to StartEvent
        source_id = self.data.get('id','ENV')
        return StartEvent(from_agent_id=source_id, to_agent_id=target_id)
```

Strict Generation Requirements:
- Output only the SimEnv class code block
- Wrap the code in markdown Python code block (```python)
- Do not include additional explanations or comments
- Ensure the code is concise, readable, and follows Python best practices
"""
        response = self.call_llm(prompt).text
        code_start = response.find("```python")
        code_end = response.find("```", code_start+1)
        if code_start != -1 and code_end != -1:
            simenv_class_code = response[code_start+len("```python"):code_end].strip()
            return simenv_class_code
        else:
            raise ValueError("LLM response does not contain valid Python code for SimEnv subclass.")
        



    def initialize_agent_class(self, agent_type: str, action_list: List[dict], events: Dict[int, Dict]) -> str:
        # 初始化代理类代码
        agent_code = f'''
from typing import Any, List,Optional
import json
import asyncio
from loguru import logger
from onesim.models import JsonBlockParser
from onesim.agent import GeneralAgent
from onesim.profile import AgentProfile
from onesim.memory import MemoryStrategy
from onesim.planning import PlanningBase
from onesim.events import *
from onesim.relationship import RelationshipManager
from .events import *


class {agent_type}(GeneralAgent):
    def __init__(self,
                 sys_prompt: str | None = None,
                 model_config_name: str = None,
                 event_bus_queue: asyncio.Queue = None,
                 profile: AgentProfile=None,
                 memory: MemoryStrategy=None,
                 planning: PlanningBase=None,
                 relationship_manager: RelationshipManager=None) -> None:
        super().__init__(sys_prompt, model_config_name, event_bus_queue, profile, memory, planning, relationship_manager)
'''
        events_for_agent = [event for event in events.values() if event['to_agent_type'] == agent_type]

        # Map each action to its corresponding event names
        action_to_events = {}
        for event in events_for_agent:
            to_action_name = event['to_action_name']
            event_name = event['event_name']
            action_to_events.setdefault(to_action_name, []).append(event_name)

        # Register events with their corresponding handlers
        for action in action_list:
            action_name = action['name']
            event_names = action_to_events.get(action_name, [])
            for event_name in event_names:
                agent_code += f"        self.register_event(\"{event_name}\", \"{action_name}\")\n"


        return agent_code

    def generate_event_class(self, event_info: dict) -> str:
        """
        Generates Python code for an Event subclass based on event_info.
        
        Args:
            event_info (dict): Dictionary containing event class information
            
        Returns:
            str: Generated Python code for the event class
        """
        # Generate imports
        imports = [
            "from typing import Any, List",
            "from datetime import datetime",
            "from onesim.events import Event",
            ""  # Empty line after imports
        ]
        
        event_name = event_info['event_name']
        fields = event_info.get('fields', [])

        # Base parameters that all events have
        base_params = [
            "from_agent_id: str",
            "to_agent_id: str"
        ]

        # Process additional fields
        extra_params = []
        init_assignments = []

        for field in fields:
            name = field['name']
            type_ = field.get('type', 'Any')
            default = field.get('default_value')

            # Convert type strings to Python types
            if type_ == 'string':
                type_ = 'str'
            elif type_ == 'datetime':
                type_ = 'datetime'
            elif type_ == 'list':
                type_ = 'List[Any]'
            elif type_=='integer':
                type_ = 'int'

            # Format default values
            if default == 'null':
                default_str = 'None'
            elif default == '[]':
                default_str = '[]'
            elif default == '':
                default_str = '""'
            elif default is not None:
                default_str = repr(default)
            else:
                default_str = None

            # Build parameter string
            if default_str is not None:
                extra_params.append(f"{name}: {type_} = {default_str}")
            else:
                extra_params.append(f"{name}: {type_}")
                
            init_assignments.append(f"        self.{name} = {name}")

        # Combine all parameters
        all_params = base_params + extra_params
        params_str = ",\n        ".join(all_params)

        # Generate class code
        class_lines = [
            f"class {event_name}(Event):",
            f"    def __init__(self,",
            f"        {params_str},",
            f"        **kwargs: Any",
            f"    ) -> None:",
            f"        super().__init__(from_agent_id=from_agent_id, to_agent_id=to_agent_id, **kwargs)"
        ]
        
        # Add field assignments
        if init_assignments:
            class_lines.extend(init_assignments)
        
        # Join all code parts
        code = "\n".join(imports + class_lines)
        
        return code


    def generate_handler_code(self, description: str, agent_type: str, action_info: Dict, incoming_events: List[Dict], outgoing_events: List[Dict], event_data_info: List[Dict]) -> str:
        thinking_prompt = f'''
You are the agent '{agent_type}'. You are analyzing how to handle the action '{action_info['name']}'.

This is for analysis only; do not generate code here.

Consider the following:

- The Multi-Agent system description: {description}
- The action details: {json.dumps(action_info, indent=4)}
- Relevant incoming events (events where '{action_info['name']}' is the target action):
{json.dumps(incoming_events, indent=4)}
- Relevant outgoing events (events where '{action_info['name']}' is the source action):
{json.dumps(outgoing_events, indent=4)}
- Next Event classes and their data structures:
{json.dumps(event_data_info, indent=4)}


### Action Type Consideration:
- **OR**: Multiple input events can trigger the action independently.
- **AND**: All specified events must be received before the action can be triggered.
- **XOR**: Only the first event is processed; subsequent events are ignored.

### Action Condition:
- Action condition: {action_info.get('condition', 'No specific condition')}
- If the condition is specified, the action should only be executed when this condition is met
- Consider how to check or validate this condition in the handler code


### Data Access Guidelines:
- If the action requires data from the agent's own state, use function `await self.get_data("var_name","default")`.
- If the action would update data of the agent's own state, use function `await self.update_data("var_name",new_value)`.
- If the action requires data from environment variables, use function `await self.get_env_data("var_name","default")`.
- If the action would update data of environment variables, use function `await self.update_env_data("var_name",new_value)`.
- If the action would requires data of other agent's state, use function `await self.get_agent_data(other_agent_id,"var_name","default")`.
- If the action would update data of other agent's state, use function `await self.update_agent_data(other_agent_id,"var_name",new_value)`.
- If the action requires data from the event payload, it should be available in the `event` object fields.

### State Management:
- Utilize existing meaningful data in `self.profile` to track the agent's current state.
- Do not introduce new flags or variables; instead, leverage the already defined `required_variables` and `output_updates` to infer and manage state.

### Type-Specific Guidance:
- **OR**: Handle any incoming event that matches, triggering the action independently.
- **AND**: Ensure all required events have been received by checking relevant data in `self.profile`.
- **XOR**: Process only the first incoming event by checking existing state in `self.profile` to determine if the action has already been triggered.


Based on the above, provide clear guidance for generating the handler in the following JSON format:


{{
    "event_tracking_strategy": "How to track event states based on the trigger type (OR/AND/XOR).",
    "event_validation_logic": "How to validate if action should proceed based on trigger type.",
    "condition_check_strategy": "How to implement the specific condition check for this action, if applicable.",
    "instruction_content": "Details to include in the 'instruction' for 'generate_reaction', including specifying that target_ids can be a single ID or a list of IDs.",
    "llm_return_content": "Details of what the LLM's returned content should include, such as target_ids (single ID or list).",
    "llm_return_handling": "Explanation of how to handle the LLM's returned results, including how to process target_ids whether it's a single ID or a list.",
    "action_decision": "Conditions under which different actions should be taken.",
    "event_sending_decision": "How to decide the next Event(s) to send to one or multiple targets, ensuring they are among the 'outgoing_events'."
}}
You should respond a json object in a json fenced code block as follows:
```json
Your JSON response here
```
'''
        thinking_response = self.call_llm(thinking_prompt)
        parser = JsonBlockParser() 
        try: 
            thinking_res = parser.parse(thinking_response) 
            thinking_res = thinking_res.parsed
        except json.JSONDecodeError: 
            raise ValueError("LLM response is not valid JSON.")
        
        print(f"Thinking response for action '{action_info['name']}':")
        print(thinking_res)

        handler_prompt=f'''
# Multi-Agent Action Handler Generation Prompt

## Context
- Agent Type: {agent_type}
- Action Name: {action_info['name']}
- Action Condition: {action_info.get('condition', 'No specific condition')}

## System Description
{description}

## Detailed Context
### Action Details
{json.dumps(action_info, indent=4)}

### Incoming Events
{json.dumps(incoming_events, indent=4)}

### Outgoing Events
{json.dumps(outgoing_events, indent=4)}

### Event Data Structures
{json.dumps(event_data_info, indent=4)}

## Initial Analysis
{thinking_res}

## Handler Generation Instructions

### Objective
Generate an asynchronous handler method for the action '{action_info['name']}' with the following requirements:

1. Method Signature
- Must be `async def {action_info['name']}(self, event: Event) -> List[Event]:`

2. Condition Check Implementation
- If the action has a condition: "{action_info.get('condition', 'No specific condition')}"
- Implement this condition check at the beginning of the handler
- Return an empty list if the condition is not met
- Only proceed with the handler logic if the condition is met or if there is no condition

3. Data Access:
- If a required variable context is 'env', retrieve it with: value = await self.get_env_data("var_name","default")
- If context is 'agent', retrieve it with: value = self.profile.get_data("var_name","default")
- If context is 'event', retrieve it from the event object, e.g. event.some_field
- To access all relationship data of related agents (if needed for specific handler logic):
  * Get all relationships: self.relationships = relationship_manager.get_all_relationships()
  * For each relationship, access target variable via: value=await self.get_agent_data(relationship.target_id,"var_name","default")

4. Data Modification:
- If modifying a variable in 'env', use: self.env.update_data("var_name", new_value)
- If modifying a variable in 'agent', use: self.profile.update_data("var_name", new_value)

5. Decision Making
CRITICAL: Use generate_reaction() for ALL decisions including target_ids selection:
- Method signature: generate_reaction(self, instruction: str, observation: str = None)
- Instructions MUST include an explicit requirement to return target_ids. This can be a single string ID or a list of IDs. The agent relationships will be provided in generate_reaction(), where the LLM will determine and select the appropriate target_id(s).
- NOTE: All agent relationships information is automatically provided within each generate_reaction() call for decision making - no need to explicitly pass relationship data
- Include current context as observation when needed
- Craft a clear, detailed instruction for the LLM that:
  * Explains the context of the action
  * Specifies the exact JSON response format
  * Includes key requirements like `target_ids` (which can be a single ID string or a list of ID strings). 
  * The LLM should strategically decide which agent(s) to target by analyzing the current context and selecting the most appropriate target_id(s) for optimal workflow (note: EnvAgent's ID is 'ENV', only use 'ENV' as target_id for terminal events)

6. Type-Specific Logic:
- **OR**:
  - Handle any incoming event that matches, triggering the action independently.
  - Utilize existing data in `self.profile` to manage state if necessary.
- **AND**:
  - Ensure all required events have been received by checking relevant data in `self.profile`.
  - Use existing data structures to verify the receipt of all necessary events before triggering the action.
- **XOR**:
  - Process only the first incoming event by checking existing state in `self.profile` to determine if the action has already been triggered.
  - Ensure that once an event has been processed, the handler does not respond to further events of the same type. 


7. Response Processing
- Parse the LLM's JSON response
- Validate that the response matches expected outgoing event types
- If target_ids is not a list, convert it to a list with a single element: `if not isinstance(target_ids,list): target_ids = [target_ids]`
- Iterate through target_ids to create and send appropriate Event(s) from `outgoing_events` to each target

### Example handler code:
```python
{self.sample_handler_code}
```

### Strict Generation Instructions
- CRITICAL: Generate ONLY the handler method 
- DO NOT include any class declaration, imports, or additional code
- ONLY output the exact Python function definition as specified

### Constraints
- ONLY generate the handler method. Other code has been provided.
- Ensure type safety and error handling
- Use type hints and follow Python best practices
- Use event.__class__.__name__ to determine the class type of event(if required)
- DO NOT use `isinstance` to determine the class type of event
- DO NOT log information in handler

Provide the complete handler method in a Python code block.
'''        
        handler_response = self.call_llm(handler_prompt).text

        # 从响应中提取代码
        code_start = handler_response.find("```python")
        code_end = handler_response.find("```", code_start + 1)
        if code_start != -1 and code_end != -1:
            handler_code = handler_response[code_start + len("```python"):code_end].strip()
            lines = handler_code.split("\n")
            if not lines[0].startswith("    "):
                lines[0] = "    " + lines[0]

            # 为剩下的每一行增加4格缩进
            lines = [lines[0]] + ["    " + line for line in lines[1:]]
            handler_code = "\n".join(lines)
            print(f"\n\nGenerated handler code for action '{action_info['name']}':")
            print(handler_code+"\n\n")
            return handler_code
        else:
            raise ValueError("LLM response does not contain valid Python code for handler.")
        
    def extract_event_data(self, event_code: str) -> Dict[str, Any]:
        """
        A simple parser to extract the event class name and its __init__ parameters.
        This is a placeholder; in practice, you'd use a proper Python code parser.
        """
        # This is a simplified extraction, assuming standard formatting
        lines = event_code.strip().split('\n')
        class_name_line = next((line for line in lines if line.strip().startswith('class ')), '')
        class_name = class_name_line.strip().split()[1].split('(')[0]

        init_line_index = next((i for i, line in enumerate(lines) if line.strip().startswith('def __init__')), None)
        if init_line_index is not None:
            init_line = lines[init_line_index]
            params = init_line[init_line.find('(')+1:init_line.find(')')].split(',')
            params = [param.strip() for param in params if param.strip() not in ('self',)]
        else:
            params = []

        return {'class_name': class_name, 'params': params}

    def call_llm(self, prompt: str) -> str:
        # This method abstracts the LLM call
        formatted_prompt = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt, role="user")
        )
        response = self.model(formatted_prompt)
        
        # Track token usage if available
        if hasattr(response, 'usage') and response.usage:
            if hasattr(response.usage, 'total_tokens'):
                self.token_usage += response.usage.total_tokens
            elif isinstance(response.usage, dict) and 'total_tokens' in response.usage:
                self.token_usage += response.usage['total_tokens']
        
        return response

    def generate_event_code(self) -> str:
        # Collect all unique Event classes
        event_code = self.event_imports +"\n\n"
        unique_event_classes = set(self.generated_event_classes)
        event_code += '\n\n'.join(unique_event_classes)
        return event_code 
            
    def verify_code(self, code: str, description: str, actions: Dict[str, List[Dict]], events: Dict[int, Dict]) -> Dict[str, Any]:
        prompt = f'''Please review the following combined Python code for any syntax errors, logical errors, or potential issues. The code is intended to implement a Multi-Agent system as described below:

**Description:**
\"\"\"{description}\"\"\"

**NOTE:**
- The code includes agent classes and event classes that interact with each other.
- Necessary imports and base classes are assumed to be present and correct. Ignore errors about provided imports and base classes.
- The `generate_reaction` method is inherited from the `GeneralAgent` base class, and it is already implemented to generate responses based on the language model. You do **NOT** need to re-implement this method during your review. Ignore erros about `generate_reaction`.
- The `generate_reaction` method will return a dictionary as per the instructions provided.
- The `profile_id` is a property in base class GeneralAgent and is already defined. The use of 'self.profile_id' is correct.
- The `env` is a property in base class GeneralAgent and is already defined. The use of 'self.env' is correct . 
- Focus on the correctness and logic of the code, especially the interactions between agents and events.
- Ignore erros about indentation. 
- Ensure that event usages are consistent across agents and event definitions.
- All decisions must use generate_reaction()
- Instructions for generate_reaction must explicitly request target_ids (can be single ID or a list)
- target_ids should be derived from the generate_reaction() response
- Code must handle both single ID and list responses (convert single IDs to a list with one element)
- Observation context included when needed
- Action conditions must be properly implemented in handlers, with appropriate checks at the beginning of handler methods

**Actions:**
{json.dumps(actions, indent=4)}

**Events:**
{json.dumps(events, indent=4)}

Please check if the code correctly implements the agents, actions, and events as per the above specifications. Return your analysis in JSON format with the following structure:

{{
  "class_issues": {{
    "AgentClassName1": [
      {{
        "type": "syntax"/"logical"/"other",
        "description": "Issue description"
      }},
      ...
    ],
    "AgentClassName2": [
      ...
    ],
    "EventClasses": [
      ...
    ]
  }}
}}

If a class has no issues, include it with an empty list.

```python
{code}
```'''
        parser = JsonBlockParser() 
        verification_response = self.call_llm(prompt+"\n"+parser.format_instruction) 
        try: 
            res = parser.parse(verification_response) 
            issues = res.parsed["class_issues"]
            
            # Store verification issues for statistics tracking
            self.verification_issues.append(issues)
            
            return issues
        except json.JSONDecodeError: 
            self.verification_issues = {"error": "Failed to parse verification response"}
            raise ValueError("LLM response is not valid JSON.")

    def fix_code(self, agent_code_dict: Dict[str, str], event_code: str, verification_results: Dict[str, List[Dict[str, Any]]], actions: Dict[str, List[Dict]], events: Dict[int, Dict]) -> Tuple[Dict[str, str], str]:
        fixed_agent_code_dict = agent_code_dict.copy()
        fixed_event_code = event_code

        # First, check if event code needs fixing
        event_issues = verification_results.get("EventClasses", [])
        if event_issues:
            # Prepare issues description for prompt
            issues_description = "\n".join([f"- {issue['type']}: {issue['description']}" for issue in event_issues])

            # Fix event code
            prompt = f'''Please fix the following Python code for the event classes based on the system requirements and issues listed.

System Context:
- Events are used to represent messages, state changes and triggers in the system
- Each event class should follow the event schema defined in the events dictionary
- Events may have attributes that agents can access when handling them

Requirements:
1. Each event must match its schema in the events dictionary
2. Event classes must inherit from the base Event class
3. Event attributes must be properly typed
4. Event classes must implement required methods
5. Do not add event not in Events schema
6. Do not change existed imports
7. MUST use `from onesim.events import Event` to import Event class

Events schema:
{json.dumps(events, indent=4)}

Issues:
{issues_description}

Original Event Code:
```python
{event_code}
```
Provide the corrected code. Use triple quotes (""") for multi-line strings. Do not include any additional explanations or comments. Respond with the code in a markdown fenced code block as follows:
```python
Your FixedEventCode
```'''
            fixed_event_code_response = self.call_llm(prompt).text
            #fixed_event_code_response=event_code
            # Extract code from response
            code_start = fixed_event_code_response.find("```python")
            code_end = fixed_event_code_response.find("```", code_start + 1)
            if code_start != -1 and code_end != -1:
                fixed_event_code = fixed_event_code_response[code_start + len("```python"):code_end].strip()
            else:
                raise ValueError("LLM response does not contain valid Python code for events.")

        # Now fix agent codes
        for agent_name in agent_code_dict.keys():
            agent_issues = verification_results.get(agent_name, [])
            if agent_issues:
                # Prepare issues description for prompt
                issues_description = "\n".join([f"- {issue['type']}: {issue['description']}" for issue in agent_issues])

                # Get the original agent code
                code = agent_code_dict[agent_name]

                # Get this agent's actions list to find conditions
                agent_actions = actions.get(agent_name, [])
                actions_with_conditions = [action for action in agent_actions if action.get('condition') is not None]
                conditions_info = ""
                if actions_with_conditions:
                    conditions_info = "Action Conditions:\n"
                    for action in actions_with_conditions:
                        conditions_info += f"- {action['name']}: {action.get('condition')}\n"

                # Fix agent code
                prompt = f'''Please fix the following Python code for the agent class '{agent_name}' based on actions and events information and the issues listed. The agent class '{agent_name}' should implement actions as handler for events.


Events:
{json.dumps(events, indent=4)}

{conditions_info}                
Issues:
{issues_description}

Handler Implementation Guide:
1. Action Conditions
- Each action handler should check its condition (if specified) at the beginning
- If the condition is not met, the handler should return an empty list
- Only continue with handler logic if the condition is met
Example:
```python
async def some_action(self, event: Event) -> List[Event]:
    # Check condition first if applicable
    if condition_specified and not condition_met:
        return []  # Return empty list if condition not met
    
    # Continue with handler logic only if condition is met or no condition specified
    # ...
```

2. Decision Making
- Use generate_reaction() for all decisions
- Include clear instruction for target_ids
- Provide observation context
Example:
```python
response = await self.generate_reaction(
    instruction="Based on current context, determine target_ids...",
    observation="Current state: ..."
)
```

Example of Agent Handler Implementation:
```python
{self.sample_handler_code}
```

Event Classes Available:
```python
{fixed_event_code}
```

Original Agent Code:
```python
{code}
```
Provide the corrected code. Use triple quotes (""") for multi-line strings. Do not include any additional explanations or comments. Respond with the code in a markdown fenced code block as follows:
```python
Your FixedCode
```
'''
                fixed_code_response = self.call_llm(prompt).text
                # Extract code from response
                code_start = fixed_code_response.find("```python")
                code_end = fixed_code_response.find("```", code_start + 1)
                if code_start != -1 and code_end != -1:
                    fixed_code = fixed_code_response[code_start + len("```python"):code_end].strip()
                    fixed_agent_code_dict[agent_name] = fixed_code
                else:
                    raise ValueError(f"LLM response does not contain valid Python code for agent '{agent_name}'.")
            else:
                # No issues with this agent code; keep it unchanged
                fixed_agent_code_dict[agent_name] = agent_code_dict[agent_name]

        return fixed_agent_code_dict, fixed_event_code
    
    def parse_agent_code(self, code: str) -> Dict[str, str]:
        """Parse agent code into dict of function_name -> code."""
        # Extract imports
        imports = []
        lines = code.split('\n')
        for line in lines:
            if line.startswith(('from', 'import')):
                imports.append(line)

        # Split into functions using regex
        function_pattern = r"async def (\w+)\([^)]*\).*?(?=async def|\Z)"
        functions = {}
        for match in re.finditer(function_pattern, code, re.DOTALL):
            func_name = match.group(1)
            func_code = match.group(0)
            functions[func_name] = func_code.strip()

        return {'imports': '\n'.join(imports), 'functions': functions}

    def parse_event_code(self, code: str) -> Dict[str, str]:
        """Parse event code into dict of event_name -> code."""
        # Extract imports
        imports = []
        lines = code.split('\n')
        for line in lines:
            if line.startswith(('from', 'import')):
                imports.append(line)

        # Split into event classes using regex
        class_pattern = r"class (\w+)\(Event\):.*?(?=class|\Z)"
        events = {}
        for match in re.finditer(class_pattern, code, re.DOTALL):
            event_name = match.group(1)
            event_code = match.group(0)
            events[event_name] = event_code.strip()

        return {'imports': '\n'.join(imports), 'events': events}

    def restructure_code(
        self,
        agent_code_dict: Dict[str, str],
        event_code: str,
        actions: Dict[str, List[Dict]],
        events: Dict[int, Dict],
    ) -> Dict:
        """Restructure code into new format mapped by IDs."""

        # Initialize result structure
        result = {'agents': {}, 'events': {'imports': '', 'definitions': {}}}

        # Parse event code
        parsed_events = self.parse_event_code(event_code)
        result['events']['imports'] = parsed_events['imports']

        # Map event classes to event IDs
        for event_id, event_info in events.items():
            event_name = event_info['event_name']
            if event_name in parsed_events['events']:
                result['events']['definitions'][event_id] = {
                    'code': parsed_events['events'][event_name],
                    'metadata': event_info,
                }

        # Parse and map agent code
        for agent_type, code in agent_code_dict.items():
            parsed_agent = self.parse_agent_code(code)

            agent_actions = actions.get(agent_type, [])
            agent_functions = {}

            # Map functions to action IDs
            for action in agent_actions:
                action_name = action['name']
                if action_name in parsed_agent['functions']:
                    agent_functions[action['id']] = {
                        'code': parsed_agent['functions'][action_name],
                        'metadata': action,
                    }

            result['agents'][agent_type] = {
                'imports': parsed_agent['imports'],
                'handlers': agent_functions,
            }

        return result

    def save_code_to_files(
        self,
        agent_code_dict: Dict[str, str],
        event_code: str,
        env_path: str,
        actions: Dict[str, List[Dict]],
        events: Dict[int, Dict],
    ) -> None:

        restructured = self.restructure_code(
            agent_code_dict, event_code, actions, events
        )
        output_path = os.path.join(env_path, "code_structure.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(restructured, f, indent=2)

        # Save agent code
        for agent_name, code in agent_code_dict.items():
            filename = os.path.join(env_path, agent_name + ".py")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Saved agent code to {filename}")

        # Save event code
        filename = os.path.join(env_path, "events.py")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(event_code)
        logger.info(f"Saved event code to {filename}")
