#!/usr/bin/env python3
"""
Experiment Runner for Agent Society v1
Enhanced with Comprehensive Testing Support!

Manages the complete experiment lifecycle:
- Setup with testing infrastructure
- Agent society simulation with testing
- Result collection and analysis
"""

import os
import shutil
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import re

from src.azure_client import AzureOpenAIClient
from src.agent_v1 import Agent
from src.tools_v1 import ToolRegistryV1
from src.experiment_visualizer import ExperimentVisualizer
from src.complexity_analyzer import TCIAnalyzer
from src.evolutionary_algorithm import EvolutionaryAlgorithm
import src.boids_rules as boids_rules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ExperimentRunner:
    """
    Experiment Runner v1 - Enhanced with Testing Support
    
    Features:
    - Self-contained experiment directories
    - Comprehensive testing infrastructure
    - Agent society simulation with testing phases
    - Detailed result analysis and reporting
    """
    
    def __init__(self, 
                 experiment_name: str, 
                 num_agents: int, 
                 max_rounds: int, 
                 shared_meta_prompt: str,
                 agent_specializations: List[str] = None,
                 boids_enabled: bool = True,
                 boids_k_neighbors: int = 2,
                 boids_sep_threshold: float = 0.45,
                 # NEW: Individual rule controls (like self_reflection_enabled)
                 boids_separation_enabled: bool = True,
                 boids_alignment_enabled: bool = True,
                 boids_cohesion_enabled: bool = True,
                 evolution_enabled: bool = False,
                 evolution_frequency: int = 5,
                 evolution_selection_rate: float = 0.2,
                 self_reflection_enabled: bool = False):
        
        self.experiment_name = experiment_name
        self.num_agents = num_agents
        self.max_rounds = max_rounds
        self.shared_meta_prompt = shared_meta_prompt
        self.agent_specializations = agent_specializations or []
        self.boids_enabled = boids_enabled
        self.boids_k_neighbors = boids_k_neighbors if boids_enabled else 0
        self.boids_sep_threshold = boids_sep_threshold if boids_enabled else 0
        self.boids_separation_enabled = boids_separation_enabled
        self.boids_alignment_enabled = boids_alignment_enabled
        self.boids_cohesion_enabled = boids_cohesion_enabled
        self.self_reflection_enabled = self_reflection_enabled
        
        # Evolution parameters
        self.evolution_enabled = evolution_enabled
        self.evolution_frequency = evolution_frequency
        self.evolution_selection_rate = evolution_selection_rate
        
        # Experiment paths - EVERYTHING goes inside the experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = f"experiments/{experiment_name}_{timestamp}"
        self.shared_tools_dir = os.path.join(self.experiment_dir, "shared_tools")
        self.personal_tools_dir = os.path.join(self.experiment_dir, "personal_tools")
        self.template_dir = "shared_tools_template"  # Source template stays at root
        
        # Results and logs
        self.results_file = os.path.join(self.experiment_dir, "results.json")
        self.summary_file = os.path.join(self.experiment_dir, "summary.txt")
        self.metadata_file = os.path.join(self.experiment_dir, "experiment_metadata.json")
        
        # Initialize components
        self.azure_client = None
        self.tool_registry = None
        self.agents = []
        self.round_results = []
        self.complexity_over_rounds = []  # Track average TCI per round
        self.center_history = [] # Track the global summary over rounds
        self.visualizer = ExperimentVisualizer()
        # TCI-Lite v4: No weights needed, uses absolute 10-point scale
        self.tci_analyzer = TCIAnalyzer()
        
        # Initialize evolutionary algorithm if enabled
        self.evolutionary_algorithm = None
        if self.evolution_enabled:
            self.evolutionary_algorithm = EvolutionaryAlgorithm(
                selection_rate=self.evolution_selection_rate,
                mutation_rate=0.3,
                crossover_rate=0.5,
                min_population_size=max(3, num_agents // 2)
            )

        logger.info(f"🧪 Experiment initialized: {self.experiment_name}")
        logger.info(f"📁 Experiment directory: {self.experiment_dir}")
    
    def _initialize_azure_client(self) -> bool:
        """Initialize Azure OpenAI client."""
        try:
            self.azure_client = AzureOpenAIClient()
            logger.info("✅ Azure OpenAI client initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Azure client initialization failed: {e}")
            return False
    
    def _initialize_shared_tools(self) -> bool:
        """Initialize shared tools from template."""
        try:
            # Create experiment directory
            os.makedirs(self.experiment_dir, exist_ok=True)
            
            # Initialize tool registry with experiment-specific paths
            self.tool_registry = ToolRegistryV1(
                shared_tools_dir=self.shared_tools_dir,
                personal_tools_base_dir=self.personal_tools_dir
            )
            
            logger.info("✅ Shared tools initialized from template")
            return True
            
        except Exception as e:
            logger.error(f"❌ Shared tools initialization failed: {e}")
            return False
    
    def _create_personal_tool_directories(self):
        """Create personal tool directories for each agent."""
        try:
            os.makedirs(self.personal_tools_dir, exist_ok=True)
            
            for i in range(1, self.num_agents + 1):
                agent_id = f"Agent_{i:02d}"
                agent_dir = os.path.join(self.personal_tools_dir, agent_id)
                
                # Create agent directory structure with testing support
                os.makedirs(agent_dir, exist_ok=True)
                os.makedirs(os.path.join(agent_dir, "_tests"), exist_ok=True)
                os.makedirs(os.path.join(agent_dir, "_testResults"), exist_ok=True)
                
                # Initialize empty index
                index_file = os.path.join(agent_dir, "index.json")
                if not os.path.exists(index_file):
                    with open(index_file, 'w') as f:
                        json.dump({"tools": {}}, f, indent=2)
                
                logger.info(f"📁 Created agent directory: {agent_id}")
            
            logger.info(f"✅ Personal tool directories created for {self.num_agents} agents")
            
        except Exception as e:
            logger.error(f"❌ Personal tool directories creation failed: {e}")
            raise
    
    def _initialize_agents(self):
        """Initialize all agents with their specializations."""
        try:
            for i in range(1, self.num_agents + 1):
                agent_id = f"Agent_{i:02d}"
                
                # Get specialization if available
                if i <= len(self.agent_specializations):
                    specialization = self.agent_specializations[i-1]
                else:
                    specialization = f"Focus on building practical tools for general use."
                
                # Create agent
                agent = Agent(
                    agent_id=agent_id,
                    azure_client=self.azure_client,
                    shared_tool_registry=self.tool_registry,
                    meta_prompt=self.shared_meta_prompt,
                    envs_available=["python", "file_system", "data_processing"],
                    specific_prompt=specialization,
                    personal_tool_base_dir=self.personal_tools_dir  # Use experiment-specific path
                )
                
                self.agents.append(agent)
                logger.info(f"🤖 Agent {agent_id} initialized with specialization: {specialization}")
            
            logger.info(f"✅ All {len(self.agents)} agents initialized")
            
        except Exception as e:
            logger.error(f"❌ Agent initialization failed: {e}")
            raise
    
    def _get_neighbors(self, agent_index: int) -> List[Agent]:
        """
        Determines neighbors based on a k-regular ring topology.
        """
        if self.num_agents <= 1:
            return []
        
        neighbors = []
        for i in range(1, self.boids_k_neighbors + 1):
            # Right neighbor
            right_idx = (agent_index + i) % self.num_agents
            neighbors.append(self.agents[right_idx])
            # Left neighbor
            left_idx = (agent_index - i + self.num_agents) % self.num_agents
            neighbors.append(self.agents[left_idx])
            
        return list(set(neighbors)) # Return unique neighbors

    def _run_single_round(self, round_num: int) -> Dict[str, Any]:
        """
        Run a single round of the agent society simulation.
        Supports both Boids-driven and legacy reflection-driven modes.
        """
        mode = "Boids Rules" if self.boids_enabled else "Global Reflection"
        logger.info(f"\n🔄 Starting Round {round_num} (Mode: {mode})")
        logger.info("=" * 50)
        self.visualizer.show_round_header(round_num, self.max_rounds)
        
        # Get the global summary from the previous round for Cohesion
        last_global_summary = self.center_history[-1] if self.center_history else ""
        
        round_actions_for_summary = [] # Collect data for the end-of-round summary
        
        round_results = {
            "round": round_num,
            "timestamp": datetime.now().isoformat(),
            "agent_actions": [],
            "tools_created": 0,
            "tests_created": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_tools_in_system": 0,
            "boids_metrics": [] if self.boids_enabled else None
        }
        
        for i, agent in enumerate(self.agents):
            try:
                # --- 1. Observation & Reflection ---
                self.visualizer.show_phase_header(f"{agent.agent_id}'s Turn: Observation & Reflection", "🔍")
                
                if self.boids_enabled:
                    # Boids v2.0 Mode: Semantic, rule-guided reflection
                    neighbors = self._get_neighbors(i)
                    
                    neighbor_tools_meta = []
                    for neighbor_agent in neighbors:
                        neighbor_tools_meta.extend(neighbor_agent.self_built_tools.values())

                    # 1. Prepare Boids prompt components (tunable like self-reflection)
                    alignment_prompt = ""
                    if self.boids_alignment_enabled:
                        alignment_prompt = boids_rules.prepare_alignment_prompt(neighbor_tools_meta, round_num, self.shared_tools_dir)

                    separation_prompt = ""
                    if self.boids_separation_enabled:
                        separation_prompt = boids_rules.prepare_separation_prompt(
                            neighbor_tools_meta, 
                            self.shared_tools_dir, 
                            similarity_threshold=self.boids_sep_threshold  # Fix the bug!
                        )

                    cohesion_prompt = ""
                    if self.boids_cohesion_enabled:
                        cohesion_prompt = boids_rules.prepare_cohesion_prompt(last_global_summary)
                    
                    # Add self-reflection if enabled
                    self_reflection_prompt = ""
                    if self.self_reflection_enabled:
                        self_reflection_prompt = boids_rules.prepare_self_reflection_prompt(agent.reflection_history)

                    # 2. Assemble the final prompt, ensuring the meta_prompt is central
                    system_prompt = f"""You are Agent {agent.agent_id}, a specialist in a collaborative tool-building society.

**MISSION OBJECTIVE:**
{self.shared_meta_prompt}

**METHODOLOGY: BOIDS RULES**
Your strategy is guided by Boids rules. You must reflect on your local neighborhood and the global ecosystem trend to decide what tool to build next. This means you must:
1.  Adopt successful design principles from your neighbors (Alignment).
2.  Find a unique functional niche and avoid redundancy (Separation).
3.  Contribute to the collective goal identified by the society (Cohesion).

Your task is to propose a tool that fulfills the MISSION OBJECTIVE while adhering to the BOIDS RULES methodology."""
                    
                    user_prompt = "\n\n".join(filter(None, [
                        self_reflection_prompt,
                        alignment_prompt,
                        separation_prompt,
                        cohesion_prompt,
                        "[YOUR STRATEGATEGIC REFLECTION]",
                        "Based on the Mission Objective and the Boids Rules data above, what is a specific, high-impact tool you can build now? Describe your idea and reasoning."
                    ]))
                    
                    reflection = agent.reflect(system_prompt, user_prompt)

                else:
                    # Legacy Mode: Reconstruct the original global reflection prompt for backward compatibility
                    observation = agent.observe()
                    
                    package_info = ""
                    if agent.environment_manager:
                        package_info = f"\n\n{agent.environment_manager.get_package_summary_for_agent()}"
            
                    system_prompt = f"""You are Agent {agent.agent_id} in a tool-building ecosystem. Your primary goal is to increase the collective capability of the agent society.

META CONTEXT: {self.shared_meta_prompt}

ECOSYSTEM GOAL: Create a robust and powerful tool library. Prioritize creating "deep" tools that solve complex problems by composing and chaining together existing tools.

AVAILABLE ENVIRONMENTS: {', '.join(agent.envs_available)}{package_info}

Reflect on the current tool ecosystem and think strategically about what to build next."""

                    if agent.specific_prompt:
                        system_prompt += f"\n\nSPECIFIC GUIDANCE: {agent.specific_prompt}"

                    def format_test_status(test_status_dict, title):
                        if not test_status_dict: return f"{title}: None available"
                        lines = [f"{title}:"]
                        for tool_name, status in test_status_dict.items():
                            lines.append(f"  • {tool_name}: {status.get('test_summary', 'No test info')}")
                        return "\n".join(lines)

                    neighbor_test_summary = []
                    for neighbor_id, tools in observation.get('neighbor_test_status', {}).items():
                        if tools:
                            tool_summaries = [f"{name}: {status.get('test_summary', 'No info')}" for name, status in tools.items()]
                            neighbor_test_summary.append(f"  {neighbor_id}: {'; '.join(tool_summaries[:3])}")
                    
                    user_prompt = f"""CURRENT OBSERVATION:

=== ECOSYSTEM SNAPSHOT ({len(observation.get('all_visible_tools', []))}) ===
{format_test_status(observation.get('shared_test_status', {}), "Shared Foundational Tools")}

{chr(10).join(neighbor_test_summary) if neighbor_test_summary else "No tools built by neighbors yet."}

{format_test_status(observation.get('my_test_status', {}), "My Built Tools")}

=== STRATEGIC REFLECTION ===
Reflect on:
1. What is the most significant capability missing from the ecosystem right now to achieve our goal?
2. How can I combine existing tools to create a new, more powerful tool?
3. What specific, high-impact tool should I create next that directly serves our main mission?"""
                    
                    reflection = agent.reflect(system_prompt, user_prompt)


                self.visualizer.show_agent_reflection(agent.agent_id, reflection, {}) # Pass empty observation for now

                # Log the detailed reflection context
                reflection_entry = {
                    "is_boids_reflection": self.boids_enabled,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "reflection": reflection,
                    "timestamp": datetime.now().isoformat()
                }
                agent.reflection_history.append(reflection_entry)

                # --- 2. Build Tools ---
                self.visualizer.show_phase_header(f"{agent.agent_id}'s Turn: Tool Building", "🔨")
                build_result = agent.build_tools(reflection, round_num)
                
                self.visualizer.show_tool_creation(agent.agent_id, build_result.get("tool_info", {}), build_result["success"])

                if build_result["success"]:
                    round_results["tools_created"] += 1
                    logger.info(f"   ✅ {agent.agent_id}: Built tool successfully")

                    # Add tool info to this round's summary data
                    tool_info = build_result.get("tool_info", {})
                    round_actions_for_summary.append({
                        "agent_id": agent.agent_id,
                        "action": "build_tool",
                        "tool_name": tool_info.get("tool_name"),
                        "description": tool_info.get("tool_design", "")
                    })

                    # --- 3. Build Tests ---
                    logger.info(f"   🧪 Building tests for {agent.agent_id}'s new tool...")
                    test_result = agent.build_tests(build_result["tool_info"])
                    
                    if test_result["success"]:
                        round_results["tests_created"] += 1
                        test_results = test_result.get("test_results", {})
                        tool_name = build_result["tool_info"].get("tool_name", "unknown")
                        self.visualizer.show_test_execution(agent.agent_id, tool_name, test_results)
                        
                        if test_results.get("all_passed"):
                            round_results["tests_passed"] += 1
                            self._promote_tool_to_shared(agent, tool_name)
                        else:
                            round_results["tests_failed"] += 1
                else:
                    logger.warning(f"   ⚠️  {agent.agent_id}: Tool building failed")
                    
                # Record agent action
                agent_action = {
                    "agent_id": agent.agent_id,
                    "tool_build_success": build_result["success"],
                    "test_build_success": test_result.get("success", False) if build_result["success"] else False,
                    "test_passed": test_result.get("test_results", {}).get("all_passed", False) if build_result["success"] else False,
                    "tools_built_so_far": len(agent.self_built_tools)
                }
                round_results["agent_actions"].append(agent_action)

            except Exception as e:
                logger.error(f"   ❌ {agent.agent_id} turn failed: {e}", exc_info=True)
        
        # After all agents have acted, generate the global summary for THIS round
        if self.boids_enabled:
            current_global_summary = self._summarize_global_activity(round_num, round_actions_for_summary)
            self.center_history.append(current_global_summary)
            logger.info(f"🌍 Global Summary for Round {round_num}: {current_global_summary}")
            
            # Update adoption counts at the very end of the round
            self._update_adoption_counts()
        
        # Calculate and Record System Complexity
        self._calculate_and_record_system_complexity(round_num)

        # Get total tools in system
        all_tools = self.tool_registry.get_all_tools()
        round_results["total_tools_in_system"] = len(all_tools)
        
        self.visualizer.show_round_summary(round_num, round_results)
        
        logger.info(f"\n📊 Round {round_num} Summary:")
        logger.info(f"   Tools created: {round_results['tools_created']}")
        logger.info(f"   Tests created: {round_results['tests_created']}")
        logger.info(f"   Tests passed: {round_results['tests_passed']}")
        logger.info(f"   Total tools in system: {round_results['total_tools_in_system']}")
        
        return round_results
    
    def _update_adoption_counts(self):
        """
        Scans all shared tools at the end of a round and updates their adoption counts.
        """
        logger.info("📊 Updating tool adoption counts...")
        all_tools_meta = self.tool_registry.get_all_tools()
        tool_names = list(all_tools_meta.keys())
        
        # Reset current counts to avoid double-counting
        for tool_name in tool_names:
            if "adoption_count" in all_tools_meta[tool_name]:
                all_tools_meta[tool_name]["adoption_count"] = 0

        # Regex to find context.call_tool('tool_name', ...)
        call_pattern = re.compile(r"context\.call_tool\(['\"](.*?)['\"],")

        # Iterate through each tool and scan its code for calls to other tools
        for tool_name, metadata in all_tools_meta.items():
            tool_path = os.path.join(self.shared_tools_dir, metadata['file'])
            if os.path.exists(tool_path):
                try:
                    with open(tool_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                        found_calls = call_pattern.findall(code)
                        for called_tool in found_calls:
                            if called_tool in all_tools_meta:
                                if "adoption_count" not in all_tools_meta[called_tool]:
                                    all_tools_meta[called_tool]["adoption_count"] = 0
                                all_tools_meta[called_tool]["adoption_count"] += 1
                except Exception as e:
                    logger.error(f"Error reading or parsing {tool_path}: {e}")

        # Save the updated metadata back to the shared index
        shared_index_file = os.path.join(self.shared_tools_dir, "index.json")
        if os.path.exists(shared_index_file):
            try:
                with open(shared_index_file, 'r') as f:
                    shared_index_data = json.load(f)
                
                shared_index_data["tools"] = all_tools_meta
                
                with open(shared_index_file, 'w') as f:
                    json.dump(shared_index_data, f, indent=2)
                logger.info("✅ Tool adoption counts updated successfully.")
            except Exception as e:
                logger.error(f"❌ Failed to save updated adoption counts to {shared_index_file}: {e}")

    def _summarize_global_activity(self, round_num: int, round_actions: List[Dict]) -> str:
        """
        Uses an LLM to act as a "senior architect" summarizing the collective activity of a round.
        """
        if not round_actions:
            return "No significant tool-building activity occurred in this round."

        # Prepare a summary of the actions for the LLM
        action_details = []
        for action in round_actions:
            action_details.append(
                f"- Agent: {action['agent_id']}\n"
                f"  Tool Name: {action['tool_name']}\n"
                f"  Description: {action['description']}"
            )
        
        action_summary_text = "\n\n".join(action_details)

        system_prompt = "You are a senior architect observing an AI agent society. Your task is to synthesize the agents' activities in the last round into a concise, one-paragraph summary. Identify the emerging collective goal or trend and suggest the most logical next step for the group."
        user_prompt = f"""Here are the tools built by all agents in Round {round_num}:

{action_summary_text}

Based on these activities, what is the 'center of gravity' for the ecosystem right now? What trend is emerging, and what is the clear missing piece or next step for the society as a whole? Provide a brief, one-paragraph summary."""

        try:
            summary = self.azure_client.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6
            )
            return summary
        except Exception as e:
            logger.error(f"❌ Failed to generate global summary: {e}")
            return "Error summarizing the round's activity."
    
    def _calculate_and_record_system_complexity(self, round_num: int):
        """Calculate the average TCI of all tools in the system at the end of a round."""
        total_tci = 0
        total_code_complexity = 0
        total_interface_complexity = 0
        total_compositional_complexity = 0
        tool_count = 0
        
        # Collect complexity data from all agent tools
        for agent in self.agents:
            for tool_name, tool_metadata in agent.self_built_tools.items():
                complexity_data = tool_metadata.get("complexity", {})
                if complexity_data and complexity_data.get("tci_score", 0) > 0:
                    total_tci += complexity_data.get("tci_score", 0)
                    total_code_complexity += complexity_data.get("code_complexity", 0)
                    total_interface_complexity += complexity_data.get("interface_complexity", 0)
                    total_compositional_complexity += complexity_data.get("compositional_complexity", 0)
                    tool_count += 1
        
        # Also check shared tools if they exist
        shared_index_file = os.path.join(self.shared_tools_dir, "index.json")
        if os.path.exists(shared_index_file):
            shared_index_data = self._load_index_json(shared_index_file)
            for tool_name, tool_metadata in shared_index_data.get("tools", {}).items():
                complexity_data = tool_metadata.get("complexity", {})
                if complexity_data and complexity_data.get("tci_score", 0) > 0:
                    total_tci += complexity_data.get("tci_score", 0)
                    total_code_complexity += complexity_data.get("code_complexity", 0)
                    total_interface_complexity += complexity_data.get("interface_complexity", 0)
                    total_compositional_complexity += complexity_data.get("compositional_complexity", 0)
                    tool_count += 1
        
        average_tci = total_tci / tool_count if tool_count > 0 else 0
        avg_code = total_code_complexity / tool_count if tool_count > 0 else 0
        avg_interface = total_interface_complexity / tool_count if tool_count > 0 else 0
        avg_compositional = total_compositional_complexity / tool_count if tool_count > 0 else 0

        self.complexity_over_rounds.append({
            "round": round_num,
            "average_tci": average_tci,
            "avg_code_complexity": avg_code,
            "avg_interface_complexity": avg_interface,
            "avg_compositional_complexity": avg_compositional,
            "tool_count": tool_count
        })
        logger.info(f"   📈 System Complexity: Avg TCI = {average_tci:.2f} across {tool_count} tools.")

    def _promote_tool_to_shared(self, agent, tool_name: str):
        """
        Copies a successful and tested tool from an agent's personal directory
        to the experiment's shared tools directory for all agents to use.
        """
        logger.info(f"   🤝 Promoting successful tool '{tool_name}' from {agent.agent_id} to shared library...")
        try:
            # 1. Define source and destination paths
            personal_index_file = os.path.join(agent.personal_tool_dir, "index.json")
            shared_index_file = os.path.join(self.shared_tools_dir, "index.json")

            src_tool_file = os.path.join(agent.personal_tool_dir, f"{tool_name}.py")
            dest_tool_file = os.path.join(self.shared_tools_dir, f"{tool_name}.py")

            src_test_file = os.path.join(agent.personal_tests_dir, f"{tool_name}_test.py")
            dest_test_dir = os.path.join(self.shared_tools_dir, "_tests")
            os.makedirs(dest_test_dir, exist_ok=True)
            dest_test_file = os.path.join(dest_test_dir, f"{tool_name}_test.py")

            src_results_file = os.path.join(agent.personal_test_results_dir, f"{tool_name}_results.json")
            dest_results_dir = os.path.join(self.shared_tools_dir, "_testResults")
            os.makedirs(dest_results_dir, exist_ok=True)
            dest_results_file = os.path.join(dest_results_dir, f"{tool_name}_results.json")

            # 2. Copy the tool, test, and results files
            shutil.copy2(src_tool_file, dest_tool_file)
            if os.path.exists(src_test_file):
                shutil.copy2(src_test_file, dest_test_file)
            if os.path.exists(src_results_file):
                shutil.copy2(src_results_file, dest_results_file)
            
            # 3. Load metadata from the agent's personal index
            personal_index_data = self._load_index_json(personal_index_file)
            tool_metadata = personal_index_data.get("tools", {}).get(tool_name)

            if tool_metadata:
                # 4. Add the tool's metadata to the shared index
                shared_index_data = self._load_index_json(shared_index_file)
                
                # Ensure paths in metadata are relative to the shared directory
                tool_metadata["file"] = f"{tool_name}.py"
                if os.path.exists(src_test_file):
                    tool_metadata["has_test"] = True
                    tool_metadata["test_file"] = f"_tests/{tool_name}_test.py"
                if os.path.exists(src_results_file):
                    tool_metadata["test_results_file"] = f"_testResults/{tool_name}_results.json"
                
                shared_index_data["tools"][tool_name] = tool_metadata
                self._save_index_json(shared_index_file, shared_index_data)
                logger.info(f"   ✅ Promoted '{tool_name}' to the shared tool index.")
            else:
                logger.warning(f"   ⚠️  Could not find metadata for '{tool_name}' in personal index during promotion.")

        except Exception as e:
            logger.error(f"   ❌ Failed to promote tool '{tool_name}': {e}")
            
    def _load_index_json(self, index_file: str) -> Dict[str, Any]:
        """DRY: Load index JSON with error handling."""
        if os.path.exists(index_file):
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"   ⚠️  Error loading {index_file}: {e}")
                return {"tools": {}}
        return {"tools": {}}
    
    def _save_index_json(self, index_file: str, index_data: Dict[str, Any]) -> bool:
        """DRY: Save index JSON with error handling."""
        try:
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2)
            return True
        except Exception as e:
            logger.warning(f"   ⚠️  Error saving {index_file}: {e}")
            return False

    def run_experiment(self):
        """Run the complete experiment with testing support."""
        logger.info(f"🚀 Starting Experiment: {self.experiment_name}")
        logger.info("=" * 60)
        
        # Show beautiful experiment header
        self.visualizer.show_experiment_header(self.experiment_name, self.num_agents, self.max_rounds)
        
        try:
            # 1. Initialize Azure client
            if not self._initialize_azure_client():
                return False
            
            # 2. Initialize shared tools
            if not self._initialize_shared_tools():
                return False
            
            # 3. Create personal tool directories
            self._create_personal_tool_directories()
            
            # 4. Initialize agents
            self._initialize_agents()
            
            # 5. Run simulation rounds
            logger.info(f"\n🔄 Running {self.max_rounds} rounds of agent society simulation")
            
            for round_num in range(1, self.max_rounds + 1):
                round_result = self._run_single_round(round_num)
                self.round_results.append(round_result)
                
                # Evolution step (if enabled and time for evolution)
                if (self.evolution_enabled and 
                    self.evolutionary_algorithm and 
                    round_num % self.evolution_frequency == 0 and 
                    round_num < self.max_rounds):  # Don't evolve on the last round
                    
                    logger.info(f"\n🧬 Evolution triggered at round {round_num}")
                    self.agents = self.evolutionary_algorithm.evolve_population_with_discovery(
                        agents=self.agents,
                        tci_analyzer=self.tci_analyzer,
                        azure_client=self.azure_client,
                        shared_tool_registry=self.tool_registry,
                        meta_prompt=self.shared_meta_prompt,
                        envs_available=["python", "file_system", "data_processing"],
                        personal_tool_base_dir=self.personal_tools_dir
                    )
                    
                    # Update experiment metadata after evolution
                    self.num_agents = len(self.agents)
                    logger.info(f"🧬 Population evolved: {self.num_agents} agents for next rounds")
                
                # Brief pause between rounds
                import time
                time.sleep(1)
            
            # 6. Save results and reflection histories
            self._save_experiment_results()
            self._save_all_reflection_histories()
            
            # Show beautiful experiment summary
            final_stats = self._collect_final_statistics()
            self.visualizer.show_experiment_summary(final_stats, self.experiment_dir)
            
            # Show evolution summary if enabled
            if self.evolution_enabled and self.evolutionary_algorithm:
                evolution_summary = self.evolutionary_algorithm.get_evolution_summary()
                if evolution_summary.get("generations", 0) > 0:
                    logger.info(f"\n🧬 Evolution Summary:")
                    logger.info(f"   Generations: {evolution_summary['generations']}")
                    logger.info(f"   Final avg complexity: {evolution_summary['latest_avg_complexity']:.2f}")
                    logger.info(f"   Final max complexity: {evolution_summary['latest_max_complexity']:.2f}")
                    logger.info(f"   Complexity improvement: {evolution_summary['complexity_improvement']:.2f}")
                    logger.info(f"   Total agents eliminated: {evolution_summary['total_agents_eliminated']}")
                    logger.info(f"   Total agents created: {evolution_summary['total_agents_created']}")
            
            # Save visualization log
            self.visualizer.save_visualization_log(self.experiment_dir)
            
            logger.info(f"\n✅ Experiment completed successfully!")
            logger.info(f"📁 Results saved in: {self.experiment_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Experiment failed: {e}")
            return False
    
    def _save_experiment_results(self):
        """Save comprehensive experiment results with testing analysis."""
        
        # Collect final statistics
        final_stats = self._collect_final_statistics()
        
        # Save detailed results
        results_data = {
            "experiment_metadata": {
                "name": self.experiment_name,
                "num_agents": self.num_agents,
                "max_rounds": self.max_rounds,
                "shared_meta_prompt": self.shared_meta_prompt,
                "agent_specializations": self.agent_specializations,
                "experiment_dir": self.experiment_dir,
                "timestamp": datetime.now().isoformat(),
                "boids_enabled": self.boids_enabled,
                "evolution_enabled": self.evolution_enabled,
                "evolution_frequency": self.evolution_frequency if self.evolution_enabled else None
            },
            "round_results": self.round_results,
            "complexity_over_rounds": self.complexity_over_rounds,  # Add breakdown data
            "final_statistics": final_stats,
            "agent_summaries": self._get_agent_summaries(),
            "center_history": self.center_history,
            "evolution_summary": self.evolutionary_algorithm.get_evolution_summary() if (self.evolution_enabled and self.evolutionary_algorithm) else None
        }
        
        # Save JSON results
        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save human-readable summary
        self._save_human_readable_summary(final_stats)
        
        # Save experiment metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(results_data["experiment_metadata"], f, indent=2)
        
        logger.info(f"💾 Results saved:")
        logger.info(f"   📄 Detailed results: {self.results_file}")
        logger.info(f"   📄 Summary: {self.summary_file}")
        logger.info(f"   📄 Metadata: {self.metadata_file}")
    
    def _collect_final_statistics(self) -> Dict[str, Any]:
        """Collect final experiment statistics with testing metrics."""
        
        # Get all tools with test status
        tools_with_tests = self.tool_registry.get_all_tools_with_test_status()
        
        stats = {
            "total_rounds": len(self.round_results),
            "total_tools_created": sum(r["tools_created"] for r in self.round_results),
            "total_tests_created": sum(r["tests_created"] for r in self.round_results),
            "total_tests_passed": sum(r["tests_passed"] for r in self.round_results),
            "total_tests_failed": sum(r["tests_failed"] for r in self.round_results),
            "final_tools_in_system": len(tools_with_tests),
            "testing_coverage": {
                "tools_with_tests": len([t for t in tools_with_tests.values() if t["has_test"]]),
                "tools_with_results": len([t for t in tools_with_tests.values() if t["has_results"]]),
                "tools_passed_tests": len([t for t in tools_with_tests.values() if t["test_passed"]]),
                "tools_failed_tests": len([t for t in tools_with_tests.values() if t["has_results"] and not t["test_passed"]])
            },
            "agent_productivity": {
                agent.agent_id: {
                    "tools_built": len(agent.self_built_tools),
                    "tests_built": len(agent.self_built_tests),
                    "reflections": len(agent.reflection_history),
                    "test_results": len(agent.test_results_history)
                }
                for agent in self.agents
            }
        }
        
        # Calculate rates
        if stats["total_tests_created"] > 0:
            stats["test_pass_rate"] = stats["total_tests_passed"] / stats["total_tests_created"]
        else:
            stats["test_pass_rate"] = 0.0
        
        if stats["final_tools_in_system"] > 0:
            stats["testing_coverage_rate"] = stats["testing_coverage"]["tools_with_tests"] / stats["final_tools_in_system"]
        else:
            stats["testing_coverage_rate"] = 0.0
        
        return stats
    
    def _get_agent_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed summaries for each agent."""
        summaries = {}
        
        for agent in self.agents:
            summaries[agent.agent_id] = {
                "specialization": agent.specific_prompt,
                "tools_built": agent.self_built_tools,
                "tests_built": agent.self_built_tests,
                "reflection_count": len(agent.reflection_history),
                "test_results_count": len(agent.test_results_history),
                "personal_tool_dir": agent.personal_tool_dir,
                "latest_reflection": agent.reflection_history[-1]["reflection"] if agent.reflection_history else None
            }
        
        return summaries
    
    def _save_human_readable_summary(self, final_stats: Dict[str, Any]):
        """Save a human-readable experiment summary."""
        
        with open(self.summary_file, 'w') as f:
            f.write(f"🧪 EXPERIMENT SUMMARY: {self.experiment_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"📊 OVERVIEW:\n")
            f.write(f"   Experiment: {self.experiment_name}\n")
            f.write(f"   Agents: {self.num_agents}\n")
            f.write(f"   Rounds: {self.max_rounds}\n")
            f.write(f"   Directory: {self.experiment_dir}\n")
            f.write(f"   Timestamp: {datetime.now().isoformat()}\n\n")
            
            f.write(f"🔧 TOOL CREATION RESULTS:\n")
            f.write(f"   Total tools created: {final_stats['total_tools_created']}\n")
            f.write(f"   Final tools in system: {final_stats['final_tools_in_system']}\n")
            f.write(f"   Tools per round: {final_stats['total_tools_created'] / self.max_rounds:.1f}\n\n")
            
            f.write(f"🧪 TESTING RESULTS:\n")
            f.write(f"   Total tests created: {final_stats['total_tests_created']}\n")
            f.write(f"   Tests passed: {final_stats['total_tests_passed']}\n")
            f.write(f"   Tests failed: {final_stats['total_tests_failed']}\n")
            f.write(f"   Test pass rate: {final_stats['test_pass_rate']:.2%}\n")
            f.write(f"   Testing coverage: {final_stats['testing_coverage_rate']:.2%}\n\n")
            
            f.write(f"🤖 AGENT PRODUCTIVITY:\n")
            for agent_id, productivity in final_stats['agent_productivity'].items():
                f.write(f"   {agent_id}:\n")
                f.write(f"     Tools: {productivity['tools_built']}\n")
                f.write(f"     Tests: {productivity['tests_built']}\n")
                f.write(f"     Reflections: {productivity['reflections']}\n")
                f.write(f"     Test Results: {productivity['test_results']}\n")
            
            f.write(f"\n📁 EXPERIMENT FILES:\n")
            f.write(f"   Shared tools: {self.shared_tools_dir}/\n")
            f.write(f"   Personal tools: {self.personal_tools_dir}/\n")
            f.write(f"   Results: {self.results_file}\n")
            f.write(f"   Metadata: {self.metadata_file}\n")
    
    def _save_all_reflection_histories(self):
        """Save reflection histories for all agents."""
        logger.info("💭 Saving agent reflection histories...")
        
        for agent in self.agents:
            try:
                success = agent.save_reflection_history()
                if success:
                    logger.info(f"   ✅ {agent.agent_id}: {len(agent.reflection_history)} reflections saved")
                else:
                    logger.warning(f"   ⚠️  {agent.agent_id}: Failed to save reflections")
            except Exception as e:
                logger.error(f"   ❌ {agent.agent_id}: Error saving reflections: {e}")
        
        logger.info("💭 All reflection histories processing complete")


def main():
    """Run a sample experiment with testing support."""
    
    print("🧪 Starting Enhanced Agent Society Experiment with Testing!")
    print("=" * 70)
    
    # Define agent specializations
    specializations = [
        "Focus on building SORTING and ORDERING algorithms. Create tools for data organization.",
        "Focus on building MATHEMATICAL and CALCULATION tools. Create tools for numerical operations.",
        "Focus on building TEXT PROCESSING and STRING manipulation tools. Create tools for text analysis."
    ]# optional 
    
    # Create experiment runner with tunable boids rules
    runner = ExperimentRunner(
        experiment_name="boids_testing_demo",
        num_agents=3,
        max_rounds=3,
        shared_meta_prompt="You are in a collaborative tool-building ecosystem. Follow the Boids rules to create high-quality, specialized, and collaborative tools.",
        agent_specializations=specializations,
        boids_enabled=True,
        boids_k_neighbors=2,
        boids_sep_threshold=0.45,
        # NEW: Individual rule controls (like self_reflection_enabled)
        boids_separation_enabled=True,   # Enable separation rule
        boids_alignment_enabled=True,    # Enable alignment rule  
        boids_cohesion_enabled=True,     # Enable cohesion rule
        evolution_enabled=False,  # Set to True to enable evolution
        evolution_frequency=5,    # Evolve every 5 rounds
        evolution_selection_rate=0.2,  # Remove bottom 20%
        self_reflection_enabled=True # Enable self-reflection
    )
    
    # For testing different configurations, you can easily switch:
    # - Only separation: boids_separation_enabled=True, others=False
    # - Only alignment: boids_alignment_enabled=True, others=False  
    # - Only cohesion: boids_cohesion_enabled=True, others=False
    # - No boids rules: boids_enabled=False
    
    # Run the experiment
    success = runner.run_experiment()
    
    if success:
        print(f"\n✅ Experiment completed successfully!")
        print(f"📁 Check results in: {runner.experiment_dir}")
        
        # Show directory structure
        print(f"\n📂 Experiment Directory Structure:")
        for root, dirs, files in os.walk(runner.experiment_dir):
            level = root.replace(runner.experiment_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print(f"\n❌ Experiment failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 