"""
Evolutionary Algorithm Module for Agent Tool Evolution
Minimal implementation with prompt-based mutation and crossover.
"""

import random
import statistics
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import os


class EvolutionaryAlgorithm:
    """
    Minimal evolutionary algorithm for agent tool evolution.
    
    Features:
    - Selection: Remove bottom 20% agents by tool complexity
    - Mutation: Prompt-based variation of agent specializations
    - Crossover: Prompt-based combination of successful agent traits
    """
    
    def __init__(self, 
                 selection_rate: float = 0.2,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.5,
                 min_population_size: int = 3):
        """
        Initialize evolutionary algorithm parameters.
        
        Args:
            selection_rate: Fraction of agents to remove (bottom performers)
            mutation_rate: Probability of mutation for new agents
            crossover_rate: Probability of crossover vs pure mutation
            min_population_size: Minimum agents to maintain
        """
        self.selection_rate = selection_rate
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.min_population_size = min_population_size
        self.generation = 0
        self.evolution_history = []
    
    def evolve_population(self, 
                         agents: List['Agent'], 
                         tci_analyzer,
                         azure_client,
                         shared_tool_registry,
                         meta_prompt: str,
                         envs_available: List[str],
                         personal_tool_base_dir: str) -> List['Agent']:
        """
        Evolve the agent population through selection, crossover, and mutation.
        
        Args:
            agents: Current agent population
            tci_analyzer: Tool complexity analyzer
            azure_client: Azure OpenAI client for prompt generation
            shared_tool_registry: Shared tool registry
            meta_prompt: Base meta prompt
            envs_available: Available environments
            personal_tool_base_dir: Base directory for personal tools
            
        Returns:
            Evolved agent population
        """
        if len(agents) < self.min_population_size:
            print(f"🧬 Population too small ({len(agents)}) - skipping evolution")
            return agents
        
        self.generation += 1
        print(f"\n🧬 Evolution Generation {self.generation}")
        print("=" * 50)
        
        # Step 1: Evaluate and rank agents by tool complexity
        ranked_agents = self._evaluate_and_rank_agents(agents, tci_analyzer)
        
        # Step 2: Selection - remove bottom 20%
        survivors = self._selection(ranked_agents)
        
        # Step 3: Generate new agents to replace removed ones
        new_agents = self._generate_new_agents(
            survivors, 
            len(agents) - len(survivors),
            azure_client,
            shared_tool_registry,
            meta_prompt,
            envs_available,
            personal_tool_base_dir
        )
        
        # Step 4: Combine survivors and new agents
        evolved_population = survivors + new_agents
        
        # Record evolution history
        self._record_evolution(ranked_agents, survivors, new_agents)
        
        print(f"🧬 Evolution complete: {len(survivors)} survivors + {len(new_agents)} new agents")
        return evolved_population
    
    def _evaluate_and_rank_agents(self, agents: List['Agent'], tci_analyzer) -> List[Tuple['Agent', float]]:
        """
        Evaluate agents by their tool complexity and rank them.
        
        Returns:
            List of (agent, avg_complexity_score) tuples, sorted by score (descending)
        """
        agent_scores = []
        
        for agent in agents:
            if not agent.self_built_tools:
                # Agent has no tools - assign minimum score
                agent_scores.append((agent, 0.0))
                continue
            
            # Calculate average TCI score for agent's tools
            tool_complexities = []
            for tool_name in agent.self_built_tools:
                try:
                    tool_file = f"{agent.personal_tool_dir}/{tool_name}.py"
                    if os.path.exists(tool_file):
                        with open(tool_file, 'r') as f:
                            tool_code = f.read()
                        
                        tci_result = tci_analyzer.analyze_tool_complexity(tool_code, tool_name)
                        if tci_result and 'tci_score' in tci_result:
                            tool_complexities.append(tci_result['tci_score'])
                except Exception as e:
                    print(f"⚠️  Error analyzing {tool_name}: {e}")
                    continue
            
            avg_complexity = statistics.mean(tool_complexities) if tool_complexities else 0.0
            agent_scores.append((agent, avg_complexity))
        
        # Sort by complexity score (descending - highest first)
        ranked_agents = sorted(agent_scores, key=lambda x: x[1], reverse=True)
        
        print(f"🏆 Agent Rankings:")
        for i, (agent, score) in enumerate(ranked_agents):
            tools_count = len(agent.self_built_tools)
            print(f"   {i+1}. {agent.agent_id}: {score:.2f} avg TCI ({tools_count} tools)")
        
        return ranked_agents
    
    def _selection(self, ranked_agents: List[Tuple['Agent', float]]) -> List['Agent']:
        """
        Selection: Keep top performers, remove bottom 20%.
        
        Returns:
            List of surviving agents
        """
        num_to_remove = max(1, int(len(ranked_agents) * self.selection_rate))
        num_survivors = len(ranked_agents) - num_to_remove
        
        # Ensure we don't go below minimum population
        num_survivors = max(num_survivors, self.min_population_size)
        num_to_remove = len(ranked_agents) - num_survivors
        
        survivors = [agent for agent, score in ranked_agents[:num_survivors]]
        eliminated = [agent for agent, score in ranked_agents[num_survivors:]]
        
        print(f"🔥 Selection: Keeping {len(survivors)} survivors, eliminating {len(eliminated)}")
        if eliminated:
            eliminated_ids = [agent.agent_id for agent in eliminated]
            print(f"   Eliminated: {eliminated_ids}")
        
        return survivors
    
    def _generate_new_agents(self, 
                           survivors: List['Agent'],
                           num_new_agents: int,
                           azure_client,
                           shared_tool_registry,
                           meta_prompt: str,
                           envs_available: List[str],
                           personal_tool_base_dir: str) -> List['Agent']:
        """
        Generate new agents through crossover and mutation.
        
        Returns:
            List of new agents
        """
        if num_new_agents <= 0:
            return []
        
        new_agents = []
        
        for i in range(num_new_agents):
            if random.random() < self.crossover_rate and len(survivors) >= 2:
                # Crossover: Combine traits from two successful agents
                new_agent = self._crossover_agents(
                    survivors, i, azure_client, shared_tool_registry, 
                    meta_prompt, envs_available, personal_tool_base_dir
                )
            else:
                # Mutation: Vary a successful agent's traits
                new_agent = self._mutate_agent(
                    survivors, i, azure_client, shared_tool_registry,
                    meta_prompt, envs_available, personal_tool_base_dir
                )
            
            new_agents.append(new_agent)
        
        return new_agents
    
    def _crossover_agents(self, 
                         survivors: List['Agent'],
                         agent_index: int,
                         azure_client,
                         shared_tool_registry,
                         meta_prompt: str,
                         envs_available: List[str],
                         personal_tool_base_dir: str) -> 'Agent':
        """
        Create new agent by crossing over traits from two successful agents.
        """
        # Select two random successful agents as parents
        parent1, parent2 = random.sample(survivors, 2)
        
        # Generate crossover prompt
        system_prompt = """You are creating a new AI agent by combining the best traits of two successful agents. 
Create a specific_prompt that combines their strengths and specializations in a novel way."""
        
        user_prompt = f"""Parent Agent 1 ({parent1.agent_id}):
Specialization: {parent1.specific_prompt or "General tool building"}
Tools built: {len(parent1.self_built_tools)}

Parent Agent 2 ({parent2.agent_id}):
Specialization: {parent2.specific_prompt or "General tool building"}  
Tools built: {len(parent2.self_built_tools)}

Create a new agent specialization that combines the best aspects of both parents.
Output ONLY the specific_prompt text (2-3 sentences max)."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            new_specialization = azure_client.chat(messages, temperature=0.7)
        except Exception as e:
            print(f"⚠️  Crossover prompt generation failed: {e}")
            new_specialization = f"Hybrid agent combining {parent1.agent_id} and {parent2.agent_id} approaches"
        
        # Create new agent
        from src.agent_v1 import Agent
        new_agent_id = f"EvoAgent_Gen{self.generation}_{agent_index+1}_Cross"
        
        new_agent = Agent(
            agent_id=new_agent_id,
            azure_client=azure_client,
            shared_tool_registry=shared_tool_registry,
            meta_prompt=meta_prompt,
            envs_available=envs_available,
            specific_prompt=new_specialization,
            personal_tool_base_dir=personal_tool_base_dir
        )
        
        print(f"🧬 Crossover: {new_agent_id} from {parent1.agent_id} + {parent2.agent_id}")
        print(f"   New specialization: {new_specialization[:100]}...")
        
        return new_agent
    
    def _mutate_agent(self,
                     survivors: List['Agent'],
                     agent_index: int,
                     azure_client,
                     shared_tool_registry,
                     meta_prompt: str,
                     envs_available: List[str],
                     personal_tool_base_dir: str) -> 'Agent':
        """
        Create new agent by mutating a successful agent's traits.
        """
        # Select random successful agent as parent
        parent = random.choice(survivors)
        
        # Generate mutation prompt
        system_prompt = """You are creating a new AI agent by mutating/varying the specialization of a successful agent.
Create a related but distinct specialization that explores a different aspect or approach."""
        
        user_prompt = f"""Parent Agent ({parent.agent_id}):
Current specialization: {parent.specific_prompt or "General tool building"}
Tools built: {len(parent.self_built_tools)}

Create a mutation of this specialization - keep it related but explore a different angle, technique, or domain.
Output ONLY the specific_prompt text (2-3 sentences max)."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            new_specialization = azure_client.chat(messages, temperature=0.8)  # Higher temp for more variation
        except Exception as e:
            print(f"⚠️  Mutation prompt generation failed: {e}")
            new_specialization = f"Mutated variant of {parent.agent_id} specialization"
        
        # Create new agent
        from src.agent_v1 import Agent
        new_agent_id = f"EvoAgent_Gen{self.generation}_{agent_index+1}_Mut"
        
        new_agent = Agent(
            agent_id=new_agent_id,
            azure_client=azure_client,
            shared_tool_registry=shared_tool_registry,
            meta_prompt=meta_prompt,
            envs_available=envs_available,
            specific_prompt=new_specialization,
            personal_tool_base_dir=personal_tool_base_dir
        )
        
        print(f"🧬 Mutation: {new_agent_id} from {parent.agent_id}")
        print(f"   New specialization: {new_specialization[:100]}...")
        
        return new_agent
    
    def _record_evolution(self, 
                         ranked_agents: List[Tuple['Agent', float]],
                         survivors: List['Agent'],
                         new_agents: List['Agent']):
        """Record evolution statistics for analysis."""
        evolution_record = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "population_size": len(ranked_agents),
            "survivors": len(survivors),
            "new_agents": len(new_agents),
            "agent_scores": [(agent.agent_id, score) for agent, score in ranked_agents],
            "eliminated": [agent.agent_id for agent, score in ranked_agents[len(survivors):]],
            "avg_complexity": statistics.mean([score for _, score in ranked_agents]) if ranked_agents else 0.0,
            "max_complexity": max([score for _, score in ranked_agents]) if ranked_agents else 0.0,
            "min_complexity": min([score for _, score in ranked_agents]) if ranked_agents else 0.0
        }
        
        self.evolution_history.append(evolution_record)
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of evolutionary progress."""
        if not self.evolution_history:
            return {"generations": 0, "no_evolution": True}
        
        latest = self.evolution_history[-1]
        initial = self.evolution_history[0]
        
        return {
            "generations": self.generation,
            "latest_avg_complexity": latest["avg_complexity"],
            "latest_max_complexity": latest["max_complexity"],
            "complexity_improvement": latest["avg_complexity"] - initial["avg_complexity"],
            "total_agents_eliminated": sum(len(record["eliminated"]) for record in self.evolution_history),
            "total_agents_created": sum(record["new_agents"] for record in self.evolution_history),
            "evolution_history": self.evolution_history
        }
    
    def evolve_population_with_discovery(self, 
                                       agents: List['Agent'], 
                                       tci_analyzer,
                                       azure_client,
                                       shared_tool_registry,
                                       meta_prompt: str,
                                       envs_available: List[str],
                                       personal_tool_base_dir: str) -> List['Agent']:
        """
        Enhanced evolution with dynamic specialization discovery.
        
        This method:
        1. Discovers/updates specializations from actual behavior before evolution
        2. Then performs standard crossover/mutation on discovered specializations
        
        Args:
            agents: Current agent population
            tci_analyzer: Tool complexity analyzer
            azure_client: Azure OpenAI client for prompt generation
            shared_tool_registry: Shared tool registry
            meta_prompt: Base meta prompt (collaborative context)
            envs_available: Available environments
            personal_tool_base_dir: Base directory for personal tools
            
        Returns:
            Evolved agent population with updated specializations
        """
        if len(agents) < self.min_population_size:
            print(f"🧬 Population too small ({len(agents)}) - skipping evolution")
            return agents
        
        self.generation += 1
        print(f"\n🧬 Evolution Generation {self.generation} (with Discovery)")
        print("=" * 60)
        
        # Step 0: Discover/Update specializations from behavior
        print(f"🔍 Phase 1: Discovering emergent specializations...")
        self._update_specializations_from_behavior(agents, azure_client, meta_prompt)
        
        # Step 1: Evaluate and rank agents by tool complexity
        ranked_agents = self._evaluate_and_rank_agents(agents, tci_analyzer)
        
        # Step 2: Selection - remove bottom 20%
        survivors = self._selection(ranked_agents)
        
        # Step 3: Generate new agents to replace removed ones
        new_agents = self._generate_new_agents(
            survivors, 
            len(agents) - len(survivors),
            azure_client,
            shared_tool_registry,
            meta_prompt,
            envs_available,
            personal_tool_base_dir
        )
        
        # Step 4: Combine survivors and new agents
        evolved_population = survivors + new_agents
        
        # Record evolution history
        self._record_evolution(ranked_agents, survivors, new_agents)
        
        print(f"🧬 Evolution complete: {len(survivors)} survivors + {len(new_agents)} new agents")
        return evolved_population
    
    def _update_specializations_from_behavior(self, agents: List['Agent'], azure_client, meta_prompt: str):
        """
        Analyze each agent's behavior and update their specific_prompt accordingly.
        This creates true emergent specialization based on what agents actually do.
        """
        print(f"🔍 Analyzing {len(agents)} agents for emergent specializations...")
        
        specialization_updates = []
        
        for agent in agents:
            try:
                # Skip if agent has no history or tools to analyze
                if not agent.reflection_history and not agent.self_built_tools:
                    print(f"🔍 {agent.agent_id}: No behavioral data - keeping current specialization")
                    continue
                
                # Discover specialization from behavior
                discovered_spec = self._discover_specialization_from_behavior(agent, azure_client, meta_prompt)
                
                # Record the update
                old_spec = agent.specific_prompt or "Generic"
                agent.specific_prompt = discovered_spec
                
                specialization_updates.append({
                    "agent_id": agent.agent_id,
                    "old_specialization": old_spec,
                    "new_specialization": discovered_spec,
                    "tools_count": len(agent.self_built_tools),
                    "reflections_count": len(agent.reflection_history)
                })
                
                print(f"🔍 {agent.agent_id}: '{old_spec[:40]}...' → '{discovered_spec[:40]}...'")
                
            except Exception as e:
                print(f"⚠️  Failed to update specialization for {agent.agent_id}: {e}")
                continue
        
        print(f"🔍 Specialization discovery complete: {len(specialization_updates)} agents updated")
    
    def _discover_specialization_from_behavior(self, agent: 'Agent', azure_client, meta_prompt: str) -> str:
        """
        Analyze an agent's reflection history and created tools to discover its emergent specialization.
        """
        # Gather behavioral data
        tools_created = list(agent.self_built_tools.keys())
        recent_reflections = agent.reflection_history[-5:] if len(agent.reflection_history) > 5 else agent.reflection_history
        
        # Get tool descriptions with complexity for deeper analysis
        tool_descriptions = []
        for tool_name in tools_created[:3]:  # Analyze up to 3 most recent tools
            if tool_name in agent.self_built_tools:
                tool_meta = agent.self_built_tools[tool_name]
                desc = tool_meta.get('description', '')[:150]  # First 150 chars
                complexity = tool_meta.get('complexity', {}).get('tci_score', 0)
                tool_descriptions.append(f"• {tool_name} (TCI: {complexity:.1f}): {desc}")
        
        system_prompt = f"""You are analyzing an AI agent's behavioral patterns to discover and refine its emergent specialization within a collaborative multi-agent ecosystem.

COLLABORATIVE CONTEXT:
The meta-prompt for all agents is: "{meta_prompt}"

Your task is to analyze this specific agent's behavioral data and update its specific_prompt, which acts as the 'gene' that will guide the agent's future decisions and tool creation. This specialization should complement the broader collaborative goal while leveraging the agent's discovered strengths.

The specific_prompt you generate will be used in crossover and mutation during evolution, so it must capture the agent's core behavioral essence."""

        user_prompt = f"""AGENT BEHAVIORAL ANALYSIS: {agent.agent_id}

=== CURRENT GENETIC CODE ===
Current specific_prompt: "{agent.specific_prompt or 'None specified'}"

=== BEHAVIORAL EVIDENCE ===
Tools Created ({len(tools_created)} total):
{chr(10).join(tool_descriptions) if tool_descriptions else "No tools created yet"}

Recent Reflection History:
{chr(10).join(f"• {reflection}" for reflection in recent_reflections) if recent_reflections else "No reflections yet"}

=== ANALYSIS FRAMEWORK ===
Please analyze this agent along these dimensions:

1. CORE COMPETENCIES: What is this agent demonstrably good at?
   - What types of tools does it create successfully?
   - What problem-solving approaches does it favor?
   - What technical domains does it gravitate toward?

2. UNIQUE PERSPECTIVE: What is this agent's distinctive viewpoint?
   - How does it approach problems differently from others?
   - What unique angles or methodologies does it bring?
   - What patterns emerge in its reflection style?

3. COLLABORATIVE ROLE: What is its unique sub-role in the ecosystem?
   - How does it contribute to the meta-prompt objective?
   - What niche does it fill that others might not?
   - How does it complement other agents' work?

4. EVOLUTIONARY POTENTIAL: What other possibilities exist for this agent?
   - What adjacent domains could it explore?
   - What untapped capabilities does its behavior suggest?
   - What directions could mutation/crossover take it?

=== OUTPUT REQUIREMENTS ===
Based on this analysis, generate an updated specific_prompt that:
- Captures the agent's emergent behavioral essence
- Defines its specialized role within the collaborative ecosystem  
- Provides clear guidance for future tool creation decisions
- Serves as effective genetic material for evolution (crossover/mutation)
- Maintains connection to the broader meta-prompt objective

Format: 2-3 sentences that will become the agent's new specific_prompt.

UPDATED SPECIFIC_PROMPT:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            discovered_specialization = azure_client.chat(messages, temperature=0.6)
            return discovered_specialization.strip()
        except Exception as e:
            print(f"⚠️  LLM call failed for specialization discovery: {e}")
            return self._fallback_specialization(agent)
    
    def _fallback_specialization(self, agent: 'Agent') -> str:
        """
        Generate fallback specialization when LLM call fails.
        """
        tools = list(agent.self_built_tools.keys())
        if tools:
            # Create specialization based on actual tools created
            return f"Specialize in {', '.join(tools[:2])}-related tool development and expand into adjacent problem domains with emphasis on practical utility."
        elif agent.specific_prompt and agent.specific_prompt != "General tool building":
            # Keep existing specialization if it exists
            return agent.specific_prompt
        else:
            # Generic fallback
            return "Focus on practical tool development with emphasis on discovering unique problem-solving approaches and contributing to collaborative objectives." 