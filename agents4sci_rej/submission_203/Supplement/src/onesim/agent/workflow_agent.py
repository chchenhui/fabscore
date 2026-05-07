from typing import Optional, Any,Tuple,List
from loguru import logger
from onesim.models.core.message import Message
from .base import AgentBase
import json
from onesim.models import JsonBlockParser
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import random

class WorkflowAgent(AgentBase):
    """An agent that extracts agent interactions from a natural language description
    and generates a topology graph where nodes are agent actions (handlers) and
    edges are events (messages) with from and to handlers and event information.
    
    Extended to support:
    - Multiple parallel start points (StartEvents)
    - Data retrieval from env.get_data instead of StartEvent payload
    - Semantic terminal events
    """

    def __init__(
        self,
        model_config_name: str,
        sys_prompt: str = '',
        **kwargs: Any,
    ) -> None:
        super().__init__(
            sys_prompt=sys_prompt or (
                "You are an AI assistant specialized in analyzing event-driven Multi-Agent systems. "
                "Your task is to identify agent types, their actions, and the events (messages) "
                "that flow between these actions, including from and to handlers and event information."
            ),
            model_config_name=model_config_name,
        )
        self.actions = {}
        self.events = {}
        self.start_points = []
        self.degree = {}
        if kwargs:
            logger.warning(
                f"Unused keyword arguments are provided: {kwargs}",
            )

    def extract_workflow(self, description: str, agent_types: list,last_workflow:str=None, issues: Optional[list] = None,suggestions: Optional[list] = None) -> dict:
            """Extracts agents, actions, and events from the description.

            Arguments:
                description (`str`): The natural language description of the event.
                agent_types (`list`): A list of agent type names.
                issues (`Optional[list]`): Issues from previous validation to refine the prompt.

            Returns:
                A dictionary containing agents, actions, and events.
            """
            # Prepare the issues text if any
            self.actions={}
            self.events={}
            issues_text = ""
            if issues and last_workflow:
                last_result=f"Previous Generated Workflow:\n{last_workflow}"
                feedback=""
                for issue,suggestion in zip(issues,suggestions):
                    feedback+=f"- {issue}:\nSuggestion:{suggestion}\n"
                issues_text = last_result+f"\nPlease address the following issues identified in the previous generated workflow:\n"+feedback
                
            default_guidelines = [
                "Focus on capturing the primary workflow interactions",
                "Ensure actions and events reflect the core process described",
                "Make reasonable inferences when workflow details are implicit",
                "Prioritize clear, logical agent interactions"
            ]
            
            # Use provided guidelines or default
            guidelines = default_guidelines
        
            # Construct the prompt
            prompt = (
                f"Workflow Extraction Task\n\n"
                f"Description:\n"
                f"\"\"\"\n{description}\n\"\"\"\n\n"
                f"Agent Types:\n"
                f"{', '.join(agent_types)}\n\n"
                
                f"Objective: Extract the detailed workflow, including agents, their actions, and inter-agent events.\n\n"
                
                "Agent Type Rules:\n"
                "1. Only use agent types explicitly listed above\n"
                "2. EnvAgent is a special system agent representing the environment with EXACTLY TWO predefined actions:\n"
                "   - \"start\": Used ONLY at the beginning of workflows to initiate processes\n"
                "   - \"terminate\": Used ONLY at the end of workflows to complete processes\n"
                "   - NO OTHER ACTIONS are permitted for EnvAgent\n"
                "3. All other agents must be from the provided agent types list\n"
                "4. Maintain consistent agent type references throughout the workflow\n\n"


                f"Output Format (JSON):\n"
                "{\n"
                "  \"workflow_metadata\": {\n"
                "    \"start_events\": [\n"
                "      {\n"
                "        \"event_name\": \"StartEvent\",\n"
                "        \"from_agent_type\": \"EnvAgent\",\n"
                "        \"to_agent_type\": \"TargetAgentType\",\n"
                "        \"to_action_name\": \"start_action\",\n"
                "        \"event_info\": \"Initial trigger for this branch of workflow\"\n"
                "      }\n"
                "    ]\n"
                "  },\n"
                "  \"agents\": [\n"
                "    {\n"
                "      \"agent\": \"AgentType\", // Must be from provided agent types\n"
                "      \"actions\": [\n"
                "        {\n"
                "          \"action_name\": \"action_name\",\n"
                "          \"type\": \"OR|AND|XOR\",\n"
                "          \"condition\": \"Logical prerequisites for action execution or null if no specific prerequisites\",\n"
                "          \"events\": [\n"
                "            {\n"
                "              \"event_name\": \"EventName\",\n"
                "              \"to_agent_type\": \"TargetAgentType\",\n"
                "              \"to_action_name\": \"target_action\",\n"
                "              \"event_info\": \"Detailed event context and significance\"\n"
                "            }\n"
                "          ]\n"
                "        }\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                
                f"Formatting Rules:\n"
                "- Agent Types: PascalCase (e.g., OrderProcessor)\n"
                "- Action Names: snake_case (e.g., process_order)\n"
                "- Event Names: PascalCase, descriptive (e.g., OrderSubmittedEvent)\n"
                "- StartEvents and terminal events: Clearly defined with specific characteristics\n"
                "- Condition: Factual description of required inputs or prerequisites, not decision logic\n\n"
                
                f"Interpretation Guidelines:\n"
                + "\n".join(f"- {guideline}" for guideline in guidelines) + "\n\n"
                
                f"{issues_text}\n\n"
                
                "Workflow Start and End Specification:\n"
                "1. Multiple Start Events\n"
                "   - Each represents a parallel workflow branch start\n"
                "   - All originate from EnvAgent's start action\n"
                "   - StartEvents don't carry data - use env.get_data instead\n"
                "   - All use the same event_name StartEvent\n\n"
                "2. Terminal Events\n"
                "   - Each workflow branch may have its own terminating event\n"
                "   - ALL terminal events MUST target EnvAgent's \"terminate\" action (exact name required)\n"
                "   - Keep original event names to preserve semantic meaning (e.g., OrderCompletedEvent)\n"
                "   - May carry completion status or final results\n"
                "   - EVERY action without outgoing events to other agents MUST have a terminal event\n\n"
                
                "Action Extraction Rules:\n"
                "1. Actions Must Be Meaningful Business Operations\n"
                "   - Each action should represent a complete business operation\n"
                "   - Actions should handle their own state management\n" 
                "   - Actions should directly process incoming events\n"
                "   - Avoid creating actions that only receive or forward data\n\n"

                "2. Direct Event Flow\n"
                "   - Events should flow directly to the action that processes them\n"
                "   - Use sync patterns (AND/XOR) for multiple input events\n"
                "   - Avoid intermediate 'receive' or 'relay' actions\n"
                "   - Let processing actions manage their own state\n\n"

                "3. Action State Management\n"
                "   - OR: Process each event independently\n"
                "      * Each event triggers immediate processing\n"
                "      * No state tracking needed\n\n"
                "   - AND: Track received events internally\n"
                "      * Action maintains state of received events\n"
                "      * Process when all required events arrive\n"
                "      * Example: create_story waits for both location and characters\n\n"
                "   - XOR: Handle first event only\n"
                "      * Action tracks if already processed\n"
                "      * Ignore subsequent events\n"
                "      * Example: approve_request takes first response\n\n"

                "4. Good Practice Examples:\n"
                "   ✓ Multiple inputs to single processor:\n"
                "      provide_data1 -> DataEvent1 -→ \n"
                "                                     process_data (AND) -> ResultEvent\n"
                "      provide_data2 -> DataEvent2 -↗\n\n"
                "   ✓ Exclusive choice:\n"
                "      validate1 -> ResultEvent1 -→\n" 
                "                                  handle_result (XOR) -> FinalEvent\n"
                "      validate2 -> ResultEvent2 -↗\n\n"

                "5. Bad Practice Examples:\n"
                "   ❌ Unnecessary receivers:\n"
                "      provide -> event -> receive -> store -> process\n"
                "   ✓ Better approach:\n" 
                "      provide -> event -> process (handles own state)\n\n"
                "   ❌ Split state management:\n"
                "      input1 -> receive1 -> store1 -→ combine\n"
                "      input2 -> receive2 -> store2 -↗\n"
                "   ✓ Better approach:\n"
                "      input1 -> event1 -→ process (AND, manages state internally)\n"
                "      input2 -> event2 -↗\n\n"

                "6. State Management Rules:\n"
                "   - State belongs in the action that processes it\n"
                "   - Track received events when needed (AND/XOR patterns)\n"
                "   - Clear state after processing\n"
                "   - Document state requirements in action type\n\n"

                "7. Graph Connectivity Requirements:\n"
                "   - Every action MUST either have outgoing events to other agents OR terminate the workflow\n"
                "   - There MUST be at least one path from each StartEvent to EnvAgent.terminate\n"
                "   - The entire workflow MUST form a single connected component\n"
                "   - Every processing action must produce meaningful output events or terminate\n"
                "   - All paths must eventually reach terminal state (EnvAgent.terminate)\n"
                "   - Maintain clear data flow from start to end\n\n"

                "8. Action Conditions:\n"
                "   - Each action should have a condition field describing ONLY necessary prerequisites\n"
                "   - Focus on structural requirements: what inputs/components/data must exist\n"
                "   - DO NOT include subjective decision logic (e.g., 'if price is good' or 'when demand is high')\n"
                "   - Use null if there are no specific prerequisites (any input event triggers action)\n"
                "   - Action type (OR/AND/XOR) determines how multiple inputs are handled\n\n"

                "Evaluation Criteria:\n"
                "- Accuracy of agent interactions\n"
                "- Completeness of workflow representation\n"
                "- Clarity of event flows\n"
                "- Adherence to specified naming conventions\n\n"
                
                "Important Notes:\n"
                "1. Capture the essence of the workflow\n"
                "2. Be precise in defining actions and events\n"
                "3. Ensure logical progression of interactions\n"
                "4. Explicitly represent parallel workflow branches\n"
                "5. Maintain semantic meaning in terminal events\n\n"
                "6. Only use agents from the provided agent types list\n"
            )
            
            # Format the prompt
            parser = JsonBlockParser()
            prompt_message = self.model.format(
                Message("system", self.sys_prompt, role="system"),
                Message("user", prompt+parser.format_instruction, role="user")
            )

            # Get the response from the model
            response = self.model(prompt_message)

            # Parse the LLM's JSON response
            try:
                res = parser.parse(response)
                data = res.parsed

                workflow_metadata = data.get('workflow_metadata', {})
            
                action_id = 1
                action_name_to_id = {}
                # Process StartEvents
                start_events = workflow_metadata.get('start_events', [])
                for idx, start_event in enumerate(start_events):
                    start_event_entry = {
                        'id': -idx-1,  # Negative IDs for start events
                        'from_agent_type': 'EnvAgent',
                        'from_action_name': 'start',
                        'to_agent_type': start_event['to_agent_type'],
                        'to_action_name': start_event['to_action_name'],
                        'from_action_id': 0,  # All start events come from EnvAgent's start action
                        'to_action_id': None,  
                        'event_name': start_event['event_name'],
                        'event_info': start_event.get('event_info', 'Initial trigger for workflow branch'),
                        'fields': []  # StartEvents don't carry data - data comes from env.get_data
                    }
                    self.events[-idx-1] = start_event_entry

                agents = data.get('agents', [])
                
                event_id = 1

                for agent in agents:
                    agent_type = agent['agent']
                    actions_with_id = []
                    for action in agent['actions']:
                        action_name = action['action_name']
                        if agent_type not in self.actions:
                            self.actions[agent_type]=[]

                        existed=False
                        for existed_action in self.actions[agent_type]:
                            if existed_action['name']==action_name:
                                existed=True
                                break
                        if not existed:
                            # 添加condition字段
                            action_entry = {
                                'id': action_id,
                                'name': action_name,
                                'condition': action.get('condition', None)  # 获取condition字段，如果不存在则默认为None
                            }
                            self.actions[agent_type].append(action_entry)
                            action_name_to_id[(agent_type, action_name)] = action_id
                            action_index= action_id
                            action_id += 1
                        else:
                            action_index = existed_action['id']
                        
                        # Process events associated with this action
                        for event in action.get('events', []):
                            event_entry = {
                                'id': event_id,
                                'from_agent_type': agent_type,
                                'from_action_name': action_name,
                                'to_agent_type': event['to_agent_type'],
                                'to_action_name': event['to_action_name'],
                                'from_action_id': action_index,
                                'to_action_id': None,  # Will be filled later
                                'event_name': event['event_name'],
                                'event_info': event.get('event_info', '')
                            }
                            self.events[event_id] = event_entry
                            event_id += 1
                            
                

                action_name_to_id[('EnvAgent', 'terminate')] =-1
                
                # Process any events that go to EnvAgent.terminate
                # These will keep their original event names but share the same destination
                for event in list(self.events.values()):
                    if event['to_agent_type'] == 'EnvAgent' and event['to_action_name'] == 'terminate':
                        event['event_info'] = event.get('event_info', 'Workflow branch completion')
                        event['to_action_id'] = -1   # Special ID for terminate action
                        
                # Fill in to_action_id for other events
                for event in self.events.values():
                    
                    if event['to_action_id'] is None or event['to_action_id'] == -1:  # Skip already set terminate events
                        to_agent_type = event['to_agent_type']
                        if to_agent_type == 'EnvAgent':
                            continue
                        to_action_name = event['to_action_name']
                        to_action_id = action_name_to_id.get((to_agent_type, to_action_name),None)
                        if to_action_id is None:
                            action_entry={
                                'id': action_id,
                                'name': to_action_name,
                                'condition': None  # 为自动创建的action添加默认condition为None
                            }
                            self.actions[to_agent_type].append(action_entry)
                            action_name_to_id[(to_agent_type, to_action_name)] = action_id
                            to_action_id = action_id
                            action_id+=1
                        event['to_action_id'] = to_action_id

                return data
            except json.JSONDecodeError:
                raise ValueError("LLM response is not valid JSON.")

    def validate_workflow_structural(self, data: dict) -> Tuple[bool, List[str], List[str]]:
        """
        Performs non-LLM-based structural validation on the extracted workflow.

        Arguments:
            data (`dict`): The extracted workflow data.

        Returns:
            A tuple containing:
                - Boolean indicating if validation passed.
                - List of issues found.
                - List of suggestions to fix the issues.
        """
        issues = []
        suggestions = set()  # Use a set to avoid duplicate suggestions
        passed = True

        # Create a mapping from action_id to (agent_type, action_name)
        action_id_to_details = {}
        for agent_type, actions in self.actions.items():
            for action in actions:
                action_id = action['id']
                action_name = action['name']
                action_id_to_details[action_id] = (agent_type, action_name)
        
        # Add EnvAgent's start and terminate actions
        action_id_to_details[0] = ("EnvAgent", "start")
        action_id_to_details[-1] = ("EnvAgent", "terminate")
        
        # Store validation issues for statistics tracking
        self.validation_issues = []
        
        # Compile a set of all valid action_ids
        valid_action_ids = set(action_id_to_details.keys())

        # Check that all events reference existing actions
        for event_id, event in self.events.items():
            to_action_id = event['to_action_id']
            from_action_id = event['from_action_id']
            to_agent_type, to_action_name = action_id_to_details.get(to_action_id, (None, None))
            # Validate to_action_id
            if to_action_id not in valid_action_ids:
                issues.append(
                    f"Event '{event['event_name']}' references a non-existing '{event['to_agent_type']}' action '{event['to_action_name']}'."
                )
                suggestions.add(
                    f"Ensure that the '{event['to_agent_type']}' action '{event['to_action_name']}' exists."
                )
            else:
                to_agent_type, to_action_name = action_id_to_details[to_action_id]
                # Additional check: Ensure the agent_type in the event matches the agent_type of the action
                if event['to_agent_type'] != to_agent_type:
                    issues.append(
                        f"Event '{event['event_name']}' references action '{to_action_name}' with agent_type '{to_agent_type}', which does not match the event's agent_type '{event['to_agent_type']}'."
                    )
                    suggestions.add(
                        f"Ensure that the event's to_agent_type '{event['to_agent_type']}' matches the agent_type '{to_agent_type}' of the action '{to_action_name}'."
                    )

            # Validate from_action_id
            if from_action_id not in valid_action_ids:
                from_agent_type, from_action_name = (event['from_agent_type'], event['from_action_name'])  # Initialize to avoid NameError
                issues.append(
                    f"Event '{event['event_name']}' references a non-existing from_action_action '{from_action_name}'."
                )
                suggestions.add(
                    f"Ensure that the '{from_agent_type}' action '{from_action_name}' exists."
                )
            else:
                from_agent_type, from_action_name = action_id_to_details[from_action_id]
                # Additional check: Ensure the agent_type in the event matches the agent_type of the action
                if event['from_agent_type'] != from_agent_type:
                    issues.append(
                        f"Event '{event['event_name']}' references action '{from_action_name}' with agent_type '{from_agent_type}', which does not match the event's agent_type '{event['from_agent_type']}'."
                    )
                    suggestions.add(
                        f"Ensure that the event's from_agent_type '{event['from_agent_type']}' matches the agent_type '{from_agent_type}' of the action '{from_action_name}'."
                    )

        # Build the workflow graph
        G = self.build_topology_graph()

        # Check if the graph is weakly connected
        if not nx.is_weakly_connected(G):
            passed = False
            components = list(nx.weakly_connected_components(G))
            issues.append(f"The workflow graph has {len(components)} disconnected components.")
            suggestions.add("Ensure all actions are connected through events, originating from StartEvents.")

        # Get all Start points
        start_action_ids = self.start_points

        # Compute the union of nodes reachable from all Start points
        reachable_nodes = set()
        for start_action_id in start_action_ids:
            if start_action_id not in action_id_to_details:
                issues.append(f"StartAction ID '{start_action_id}' does not correspond to any existing action.")
                suggestions.add("Ensure that all start_action_ids correspond to valid actions.")
                continue

            reachable = nx.descendants(G, start_action_id).union({start_action_id})
            reachable_nodes.update(reachable)

        # Check that all nodes are reachable from at least one Start point
        for node in G.nodes:
            if node not in reachable_nodes:
                # If node is in action_id_to_details, get its agent_type and action_name
                if node in action_id_to_details:
                    agent_type, action_name = action_id_to_details[node]
                    if agent_type=='EnvAgent':
                        continue
                    action_identifier = f"{agent_type}'s {action_name}"
                else:
                    action_identifier = f"Unknown Action '{node}'"

                issues.append(
                    f"Action '{action_identifier}' is not reachable from any StartAction."
                )
                suggestions.add("Ensure that all actions are connected through a valid sequence of events from at least one StartEvent.")

        # Check for orphan actions (no incoming or outgoing events)
        for node in G.nodes:
            if G.in_degree(node) == 0 and node != 0:  # Allow EnvAgent.start (ID: 0)
                if node in action_id_to_details:
                    agent_type, action_name = action_id_to_details[node]
                    action_identifier = f"{agent_type}'s {action_name}"
                else:
                    action_identifier = f"Unknown Action ID '{node}'"

                issues.append(f"Action '{action_identifier}' has no incoming events.")
                suggestions.add(
                    "Ensure that every action is triggered by an event from another action or a StartEvent."
                )
            if G.out_degree(node) == 0 and node != -1:  # Allow EnvAgent.terminate (ID: -1)
                if node in action_id_to_details:
                    agent_type, action_name = action_id_to_details[node]
                    action_identifier = f"{agent_type}'s {action_name}"
                else:
                    action_identifier = f"Unknown Action ID '{node}'"

                issues.append(f"Action '{action_identifier}' has no outgoing events.")
                suggestions.add(
                    "Ensure that every action leads to another action or a terminal event."
                )

        # Determine if validation passed
        passed = not issues

        # Before returning, store issues and suggestions in validation_issues attribute
        for issue in issues:
            self.validation_issues.append({
                "issue": issue,
                "suggestions": list(suggestions)
            })
        
        return passed, issues, list(suggestions)







    def validate_workflow_extraction(self, description: str, agent_types: list, extracted_info: dict) -> dict:
        """Validates the extracted information against the description.

        Arguments:
            description (`str`): The natural language description.
            agent_types (`list`): The list of agent types.
            extracted_info (`dict`): The extracted information to validate.

        Returns:
            A dictionary with the verification result and any issues found.
        """

        structural_passed, structural_issues, structural_suggestions = self.validate_workflow_structural(extracted_info)

        if not structural_passed:
            return {
                "verification": "Fail",
                "issues": structural_issues,
                "suggestions": structural_suggestions
            }
        
        prompt =f"""
Comprehensive Information Extraction Validation Task

Description:
```
{description}
```

Agent Types:
```
{agent_types}
```

Extracted Workflow Information:
```
{json.dumps(extracted_info, indent=2)}
```

Objective: 
- Validate the workflow provides a complete and functional implementation of business requirements
- Ensure the technical implementation is sound and follows system architecture principles
- Note: The description represents business requirements, not implementation constraints

Core Validation Areas:

1. Business Requirements Coverage
   - Verify all essential business functionality is implemented
   - Confirm business process flow is logically complete
   - Check business outcomes are achievable
   - Note: Focus on business goal achievement rather than implementation details

2. Technical Implementation Soundness
   - System Components (e.g., StartEvent, EnvAgent):
     * Verify standard system components are properly configured
     * Confirm workflow initialization and termination logic
     * Ensure StartEvent leads to all subsequent actions through valid paths
     * Ensure all terminal events are reachable from StartEvent through valid paths
     * Note: These are required system infrastructure and valid by default
   
   - Graph Connectivity Requirements:
     * Every node MUST be reachable from StartEvent through a sequence of valid transitions
     * Every node MUST have at least one valid path to a termination event
     * A valid path consists of:
       - Sequence of nodes connected by valid transitions
       - Each transition must be valid for the connecting nodes
       - Path must eventually reach a termination event
     * No disconnected components or orphaned nodes allowed
     * No dead ends - all paths must lead to termination
     * Cycles are allowed only if:
       - They are part of a path from StartEvent
       - They have at least one exit path to termination
   
   - Action-Event Consistency:
     * For ALL events (including start_events and action events):
       - Referenced to_agent_type must exist in agent types
       - Referenced to_action_name must exist under the specified agent type
     * No dangling event transitions:
       - Every event must reference existing actions
       - Every referenced action must exist in the specified agent
   
   - Agent Implementation:
     * Verify business roles are properly represented
     * Confirm agent interactions support business processes
     * Note: Implementation may include support agents beyond those in description

   - Action Implementation:
     * Verify business operations are properly implemented
     * Confirm necessary technical and coordination actions are present
     * Note: Actions can include any steps needed for proper operation

   - Event Flow Implementation:
     * Verify business process sequences are properly connected
     * Confirm technical coordination events are appropriate
     * Note: Events can include any necessary coordination mechanisms

3. Implementation Quality
   - Verify workflow completeness
   - Check logical consistency
   - Confirm proper coordination between components
   - Note: Implementation details are flexible as long as they serve business goals

Verification Criteria:
- **Pass:** 
  * All business requirements are met
  * Technical implementation is complete and sound
  * Workflow is fully functional

- **Fail:**
  * Critical business requirements are not met
  * Technical implementation has fundamental flaws
  * Workflow is not fully functional
  * Any node unreachable from StartEvent
  * Any node without path to termination
  * Presence of disconnected components
  * Dead end paths that don't reach termination
  * Invalid cycles without exit paths
  * Any event references non-existent action
  * Any event references action in wrong agent type
  * Any event references invalid agent type

Output Format:
{{
  "verification": "Pass" | "Fail",
  "issues": ["Detailed issue explanation"],
  "suggestions": ["Suggestion for improvement"]
}}

Validation Guidelines:
1. Primary Focus:
   - Business requirements fulfillment
   - Technical implementation soundness
   - Overall workflow functionality

2. Implementation Freedom:
   - Description specifies WHAT to achieve
   - Implementation determines HOW to achieve it
   - Additional components and steps are valid if they serve the purpose

3. System Components:
   - Standard framework components are valid by default
   - Technical coordination mechanisms are expected
   - Implementation details can extend beyond description

4. Validation Perspective:
   - Validate against business goals rather than description text
   - Focus on functional completeness
   - Accept any valid technical approach that meets requirements
"""

        # Format the prompt
        parser = JsonBlockParser()
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.format_instruction, role="user")
        )

        # Get the response from the model
        response = self.model(prompt_message)

        # Parse the LLM's JSON response
        try:
            res = parser.parse(response)
            verification_result = res.parsed
            return verification_result
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")

    def extract_event_data_requirements(self, description: str, events: dict) -> dict:
        """
        For each extracted event, call LLM to propose what data fields are necessary.

        Arguments:
            description (`str`): The natural language description.
            events (`dict`): The events dictionary.

        Returns:
            A dictionary containing event data requirements.
        """
        # Filter out StartEvents since they don't carry data
        events_list = [{"event_name": v['event_name'], 
                       "from_agent_type":v['from_agent_type'],
                       "to_agent_type":v['to_agent_type'], 
                       "event_info":v.get('event_info','')} 
                       for k,v in events.items() 
                       if k >= 0]  # Only process events with positive IDs (non-start events)

        prompt = f"""
Event Data Requirements Extraction Task

Description:
```
{description}
```
Events Extracted:
```
{json.dumps(events_list, indent=2)}
```
Objective:
For each event, identify the data fields that should be included in the event payload. 
Provide data type and default values.

Note:
- StartEvents don't carry data (use env.get_data instead)
- Terminal events may carry completion status or results
- Avoid adding a timestamp in fields unless it is explicitly required in the description. If a timestamp is necessary, ensure it is in str format.

Output Format:
{{
  "events": [
    {{
      "event_id": "EventId",
      "event_name": "EventName",
      "fields": [
        {{
          "name":"field_name",
          "type":"data_type",
          "default_value":"default",
          "description":"Field purpose and usage"
        }}
      ]
    }}
  ]
}}
"""
        parser = JsonBlockParser()
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.format_instruction, role="user")
        )
        response = self.model(prompt_message)
        try:
            res = parser.parse(response)
            event_fields = res.parsed
            return event_fields
        except json.JSONDecodeError:
            raise ValueError("LLM response is not valid JSON.")

    def enhance_actions_with_requirements(self, description: str, agent_types: list, actions: dict, events: dict) -> dict:
        """Generate enhanced action information, including:
        - description
        - type (AND/OR/XOR for multiple inputs)
        - required_variables
        - output_updates

        Arguments:
            description (`str`): The natural language description.
            agent_types (`list`): The list of agent types.
            actions (`dict`): The actions dictionary.
            events (`dict`): The events dictionary.

        Returns:
            A dictionary containing enhanced action information.
        """
        # Prepare actions and events context
        agent_action_list = []
        for agent_type, acts in actions.items():
            agent_action_list.append({"agent":agent_type,"actions":[a["name"] for a in acts]})


        prompt = f"""
Action Requirements Enhancement Task

Description:
{description}

Agent Types:
{agent_types}

Actions extracted:
{json.dumps(agent_action_list, indent=2)}

Events extracted:
{json.dumps(events, indent=2)}

Enhanced Objective:
For each action of each agent, provide a detailed specification including:
1. A precise and descriptive explanation of the action's purpose

2. Action Type Selection:
   - OR: Multiple input events can trigger the action independently (no validation needed)
   - AND: All specified events must be received before action can be triggered
   - XOR: Only the first event is processed, subsequent events are ignored
   
   Note:
   - For simple OR cases with single event trigger, no explicit validation is needed

3. Input Requirements:
   * Identify all required variables 
   * Specify variable:
      * name
      * type  
      * context (env|agent|event)
      * description
   
   Note: 
   - Exclude StartEvent from input requirements as it carries no data
   - Actions triggered by StartEvent must source their input variables from env or agent

4. Output Updates:
   * Variables modified by the action
   * Specify:
      * name
      * type
      * context (env|agent)
      * description


Action Type Guidelines:
- Single-input actions default to OR type without validation
- Multi-input actions require explicit type selection
- Consider the logical dependency and sequencing of events

Reasoning Principles:
- Base type selection on event interaction patterns
- Reflect the real-world workflow constraints  
- Ensure logical consistency in action triggering
- The variables should focus on actual information rather than reference IDs
- Omit validation for simple OR triggers with single event
- Treat StartEvent as signal only


Output Format:
{{
  "AgentType": {{
    "action_name": {{
      "description": "Detailed action description",
      "type": "OR|AND|XOR",
      "required_variables": [
        {{
          "name": "var_name",
          "type": "data_type",
          "context": "env|agent|event",
          "description": "Description of required information"
        }}
      ],
      "output_updates": [
        {{
          "name": "var_name",
          "type": "data_type",
          "context": "env|agent",
          "description": "Description of output information"
        }}
      ]
    }}
  }}
}}
"""
        parser = JsonBlockParser()
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.format_instruction, role="user")
        )
        response = self.model(prompt_message)
        res = parser.parse(response)
        return res.parsed

    def derive_data_model_from_actions(self, actions: dict) -> dict:
        """Derive environment and Agent data models from action requirements.
        
        Arguments:
            actions (`dict`): Dictionary of actions with their requirements.
            
        Returns:
            A dictionary containing the derived data model.
        """
        env_vars = {}
        agent_vars = {}

        for agent_type, acts in actions.items():
            if agent_type not in agent_vars:
                agent_vars[agent_type] = {}
            for a in acts:
                for rv in a.get('required_variables', []):
                    if rv['context'] == 'env':
                        env_vars[rv['name']] = {"type": rv['type'], "default_value": None}
                    elif rv['context'] == 'agent':
                        agent_vars[agent_type][rv['name']] = {"type": rv['type'], "default_value": None}
                for ou in a.get('output_updates', []):
                    if ou['context'] == 'env':
                        env_vars[ou['name']] = {"type": ou['type'], "default_value": None}
                    elif ou['context'] == 'agent':
                        agent_vars[agent_type][ou['name']] = {"type": ou['type'], "default_value": None}

        # Construct final system_data_model
        system_data_model = {
            "environment": {
                "variables": [
                    {"name": k, "type": v["type"], "default_value": v["default_value"]} 
                    for k,v in env_vars.items()
                ]
            },
            "agents": {
                a_type: {
                    "variables": [
                        {"name":k, "type":v["type"], "default_value":v["default_value"]} 
                        for k,v in a_vars.items()
                    ]
                } for a_type,a_vars in agent_vars.items()
            }
        }

        return system_data_model

    def get_incoming_event(self, G: nx.DiGraph, action_id: int):
        """Find incoming event(s) for an action.
        
        Now supports:
        1. Multiple StartEvents (negative IDs)
        2. Regular events (positive IDs)
        3. Multiple input events per action (future AND/OR/XOR support)
        
        Arguments:
            G (`nx.DiGraph`): The workflow graph.
            action_id (`int`): The ID of the action.
            
        Returns:
            The incoming event or None.
        """
        preds = list(G.predecessors(action_id))
        if len(preds) == 0:
            # Check if this is a start point (has incoming StartEvent)
            for event_id, event in self.events.items():
                if event_id < 0 and event['to_action_id'] == action_id:  # StartEvent
                    return event
            return None
            
        # For regular events, return the first one for now
        # TODO: Handle AND/OR/XOR relationships for multiple input events
        for p in preds:
            edge_data = G.get_edge_data(p, action_id)
            evt_id = edge_data['event_id']
            return self.events.get(evt_id)
            
        return None
    
    def build_topology_graph(self) -> nx.DiGraph:
        """Builds a directed graph representing the workflow topology.
        
        Arguments:
            data (`dict`): The workflow data.
            
        Returns:
            A NetworkX DiGraph representing the workflow.
        """
        G = nx.DiGraph()
        
        # Add special nodes for EnvAgent
        G.add_node(0, agent='EnvAgent', action='start')  # start action node
        G.add_node(-1, agent='EnvAgent', action='terminate')  # terminate action node
        
        # Add regular action nodes
        for agent_type, acts in self.actions.items():
            for action in acts:
                action_id = action['id']
                action_name = action['name']
                G.add_node(action_id, agent=agent_type, action=action_name)

        # Process all events including StartEvents and terminal events
        for event in self.events.values():
            from_action_id = event['from_action_id']
            to_action_id = event['to_action_id']
            event_name = event['event_name']
            event_info = event.get('event_info', '')
            
            if to_action_id is not None:
                # Add edge with event information
                G.add_edge(from_action_id, to_action_id, 
                          event_id=event['id'], 
                          event_name=event_name, 
                          event_info=event_info)
            else:
                logger.warning(f"Event {event['id']} has no valid to_action_id.")

        # Find all start points (actions triggered by StartEvents)
        self.start_points = []
        for event in self.events.values():
            if event['from_agent_type'] == 'EnvAgent' and event['from_action_name'] == 'start':
                if event['to_action_id'] is not None:
                    self.start_points.append(event['to_action_id'])
        
        logger.info(f"Parallel start points: {self.start_points}")
        return G

    def validate_data_flow_with_llm(self, description: str, actions: dict, events: dict, system_data_model: dict):
        """Use LLM to validate data flow consistency.
        
        Arguments:
            description (`str`): The natural language description.
            actions (`dict`): The actions dictionary.
            events (`dict`): The events dictionary.
            system_data_model (`dict`): The system data model.
            
        Returns:
            A dictionary containing validation results.
        """
        prompt = f"""
Data Flow Validation Task

Description:
{description}

System Data Model:
{json.dumps(system_data_model, indent=2)}

Actions with Requirements:
{json.dumps(actions, indent=2)}

Events with Fields:
{json.dumps(events, indent=2)}

Objective:
Check if the data flow is consistent:
- Each action's required_variables can be fulfilled from env/agent states or event fields
- StartEvents don't carry data (use env.get_data)
- Terminal events maintain semantic meaning
- Identify any missing data, type mismatch, or logical inconsistency

Output Format:
{{
  "verification": "Pass" | "Fail",
  "issues": [],
  "suggestions": []
}}
"""
        parser = JsonBlockParser()
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.format_instruction, role="user")
        )
        response = self.model(prompt_message)
        res = parser.parse(response)
        return res.parsed

    def fix_data_flow(self, description: str, actions: dict, events: dict, system_data_model: dict, issues: list, suggestions: list) -> dict:
        """Fix data flow issues identified during validation.
        
        Arguments:
            description (`str`): The natural language description.
            actions (`dict`): The actions dictionary.
            events (`dict`): The events dictionary.
            system_data_model (`dict`): The system data model.
            issues (`list`): List of identified issues.
            suggestions (`list`): List of suggested fixes.
            
        Returns:
            A dictionary containing the fixed workflow elements.
        """
        prompt = f"""
Data Flow Repair Task

Description:
{description}

Current Extracted Workflow:
{json.dumps({"agents": actions, "events": events, "system_data_model": system_data_model}, indent=2)}

Issues found:
{json.dumps(issues, indent=2)}

Suggestions:
{json.dumps(suggestions, indent=2)}

Objective:
- Propose fixes to ensure that each action's required variables are available
- Add missing fields, variables, or correct data types
- Maintain StartEvent and terminal event constraints

Output Format:
{{
  "fixed_agents": {{ ... }},
  "fixed_events": {{ ... }},
  "fixed_system_data_model": {{ ... }}
}}
"""
        parser = JsonBlockParser()
        prompt_message = self.model.format(
            Message("system", self.sys_prompt, role="system"),
            Message("user", prompt+parser.format_instruction, role="user")
        )
        response = self.model(prompt_message)
        
        try:
            res = parser.parse(response)
            fixed_data = res.parsed
            return fixed_data
        except json.JSONDecodeError:
            return {
                "fixed_agents": actions,
                "fixed_events": events,
                "fixed_system_data_model": system_data_model
            }

    def generate_workflow(self, description: str, agent_types: list):
        """Main workflow generation method that orchestrates the entire process.
        
        Arguments:
            description (`str`): The natural language description.
            agent_types (`list`): The list of agent types.
            
        Returns:
            A tuple containing actions, events, system data model and topology graph.
        """
        # 1. Basic workflow extraction and validation
        max_attempts = 3
        attempt = 0
        verification_passed = False
        issues = None
        last_workflow = None
        suggestions = None
        
        while attempt < max_attempts and not verification_passed:
            data = self.extract_workflow(description, agent_types, last_workflow, issues, suggestions)
            last_workflow=data
            verification_result = self.validate_workflow_extraction(description, agent_types, data)
            if verification_result.get('verification') == 'Pass':
                verification_passed = True
            else:
                issues = verification_result.get('issues')
                suggestions = verification_result.get('suggestions')
                logger.warning(f"Validation failed on attempt {attempt + 1}: {issues}")
                attempt += 1

        if not verification_passed:
            raise Exception("Failed to extract valid workflow after multiple attempts.")

        logger.info(f"Data extracted successfully: {data}")

                # 4. Extract event data requirements
        event_data_reqs = self.extract_event_data_requirements(description, self.events)
        if 'events' in event_data_reqs:
            for evt in event_data_reqs['events']:
                for e_id, e in self.events.items():
                    if e['event_name'] == evt['event_name']:
                        e['fields'] = evt['fields']

        # 2. Generate action requirements
        action_requirements = self.enhance_actions_with_requirements(description, agent_types, self.actions, self.events)
        # Update actions with generated requirements
        for agent_type, acts in self.actions.items():
            for a in acts:
                req = action_requirements.get(agent_type, {}).get(a['name'], {})
                a.update(req)
            

        # 3. Derive system data model
        system_data_model = self.derive_data_model_from_actions(self.actions)

        # 5. Build topology graph
        G = self.build_topology_graph()

        # 6. Data flow validation and fixes
        max_data_flow_attempts = 3
        data_flow_attempt = 0
        while data_flow_attempt < max_data_flow_attempts:
            data_flow_result = self.validate_data_flow_with_llm(description, self.actions, self.events, system_data_model)
            if data_flow_result["verification"] == "Pass":
                logger.info("Data flow validation passed.")
                break
            else:
                logger.warning("Data flow validation failed. Attempting to fix...")
                fixed_data = self.fix_data_flow(description, self.actions, self.events, system_data_model, 
                                              data_flow_result["issues"], data_flow_result["suggestions"])

                self.actions = fixed_data.get("fixed_agents", self.actions)
                self.events = fixed_data.get("fixed_events", self.events)
                system_data_model = fixed_data.get("fixed_system_data_model", system_data_model)

            data_flow_attempt += 1

        if data_flow_attempt == max_data_flow_attempts and data_flow_result["verification"] != "Pass":
            raise Exception("Failed to achieve a consistent data flow after multiple attempts.")

        return self.actions, self.events, system_data_model, G
    
    def visualize_interactive_graph(self, G: nx.DiGraph, filename="workflow.html"):
        """Visualizes the topology graph interactively using pyvis.

        Arguments:
            G (`nx.DiGraph`): The graph to visualize.
            filename (`str`): The HTML file to save the visualization.
        """
        net = Network(height="750px", width="100%", directed=True)
        net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=150, spring_strength=0.05, damping=0.09)

        # Generate colors for each Agent
        agent_colors = {}
        for node_id, node_data in G.nodes(data=True):
            agent_type = node_data.get("agent", "UnknownAgent")
            if agent_type not in agent_colors:
                agent_colors[agent_type] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            node_color = agent_colors[agent_type]
            node_label = node_data.get('action', '')
            node_title = f"Agent: {agent_type}\nAction: {node_data.get('action', '')}"

            net.add_node(
                node_id,
                label=node_label,
                title=node_title,
                color=node_color,
                size=25,
                borderWidth=2
            )

        # Add edges with labels
        for source_id, target_id, edge_data in G.edges(data=True):
            edge_label = edge_data.get("event_name", "")
            edge_title = f"Event: {edge_data.get('event_name', '')}\nInfo: {edge_data.get('event_info', '')}"

            net.add_edge(
                source_id,
                target_id,
                label=edge_label,
                title=edge_title,
                color="gray",
                arrows="to"
            )

        # Customize the graph's appearance and interaction options
        net.set_options("""
        var options = {
        "nodes": {
            "font": {
                "size": 16,
                "face": "Arial",
                "color": "black",
                "background": "rgba(255,255,255,0.8)"
            },
            "scaling": {
                "min": 15,
                "max": 40
            },
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "color": {
                "highlight": {
                    "border": "#2B7CE9",
                    "background": "#D2E5FF"
                }
            },
            "shape": "dot"
        },
        "edges": {
            "color": {
                "color": "gray",
                "highlight": "blue",
                "hover": "blue"
            },
            "width": 1.5,
            "arrows": {
                "to": {
                    "enabled": true,
                    "scaleFactor": 1.5
                }
            },
            "font": {
                "size": 14,
                "face": "Arial",
                "align": "horizontal",
                "color": "darkred",
                "strokeWidth": 1,
                "strokeColor": "#ffffff",
                "multi": true
            },
            "smooth": {
                "type": "dynamic"
            }
        },
        "physics": {
            "enabled": true,
            "stabilization": {
                "iterations": 300
            },
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.2,
                "springLength": 150,
                "springConstant": 0.05,
                "damping": 0.09
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": false
        },
        "layout": {
            "improvedLayout": true
        }
        }
        """)

        # Save the HTML file
        net.save_graph(filename)
        logger.info(f"The interactive graph has been saved to {filename}. Please open it in a web browser to view.")