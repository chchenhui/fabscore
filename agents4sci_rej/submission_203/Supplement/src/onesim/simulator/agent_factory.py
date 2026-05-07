from typing import Dict, Any, List, Optional
import csv
import json
import random
import os
import copy
import importlib.util
from dataclasses import asdict
from loguru import logger
from onesim.models.core.message import Message
from onesim.models import ModelManager
from onesim.events import get_event_bus
from onesim.profile import AgentProfile, AgentSchema
from onesim.relationship import RelationshipManager
from onesim.memory import MemoryStrategy, AgentContext
from onesim.planning import PlanningBase
from tqdm import tqdm
from onesim.distribution.node import get_node, NodeRole
from onesim.distribution.agent_allocator import AgentAllocator
from onesim.distribution.master import MasterNode
import asyncio
from onesim.config import SimulatorConfig, AgentConfig, AgentMemoryConfig

class AgentFactory:
    def __init__(self, simulator_config: SimulatorConfig, model_config_name: str, env_path: str, agent_config: Optional[AgentConfig] = None) -> None:
        self.simulator_config = simulator_config
        self.agent_config = agent_config
        self.model_config_name = model_config_name
        self.env_path = env_path
        self.initialize()

    def initialize(self):
        """Initialize the factory with basic settings and models"""
        model_manager = ModelManager.get_instance()
        self.model = model_manager.get_model(config_name=self.model_config_name)

        # Initialize agent storage as dictionaries
        self.all_agents = {}
        self.agent_index = 1

        # Load action definitions
        with open(
            os.path.join(self.env_path, "actions.json"), "r", encoding='utf-8'
        ) as f:
            actions = json.load(f)

        # Initialize empty dictionaries for each agent type
        self.all_agents = {action_type: {} for action_type in actions}
        self.profile_id2agent = {}

        # Load schema definitions
        self.schemas = self._load_schemas()

        # Track tiles per agent type
        self.tiles_needed = 0
        self.global_base_count = 0

    def _load_schemas(self) -> Dict[str, AgentSchema]:
        """Load AgentSchema for all agent types"""
        schemas = {}

        # Get agent profiles from agent config if available
        agent_profiles = {}
        if self.agent_config and hasattr(self.agent_config, 'profile'):
            agent_profiles = self.agent_config.profile

        for agent_type in self.all_agents.keys():
            schema_path = None

            # Try to get schema path from the agent config
            if agent_type in agent_profiles:
                profile_config = agent_profiles[agent_type]
                schema_path = profile_config.get("schema_path")
                if schema_path:
                    schema_path = os.path.join(self.env_path, "profile", "schema", schema_path)
                else:
                    schema_path = os.path.join(self.env_path, "profile", "schema", f"{agent_type}.json")

            # Fallback to default path if not found
            if not schema_path:
                schema_path = os.path.join(self.env_path, f"profile/schema/{agent_type}.json")

            # Try to load the schema
            try:
                with open(schema_path, "r", encoding='utf-8') as f:
                    schema = json.load(f)
                schemas[agent_type] = AgentSchema(schema)
                logger.info(f"Loaded schema for {agent_type} from {schema_path}")
            except FileNotFoundError:
                logger.warning(f"Schema file not found: {schema_path}. Skipping {agent_type}.")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in schema file: {schema_path}. Skipping {agent_type}.")
            except Exception as e:
                logger.error(f"Error loading schema for {agent_type}: {e}. Skipping.")

        return schemas

    def load_agent_module_from_file(self, agent_type: str):
        """Load agent class from file"""
        module_path = os.path.join(self.env_path, "code", f"{agent_type}.py")
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"Agent module file not found: {module_path}")

        env_name = self.env_path.split(os.sep)[-1]
        package_name = f"envs.{env_name}.code"
        module_name = f"{package_name}.{agent_type}"

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, agent_type): # Ensure class name matches the expected agent type
            raise AttributeError(f"Agent class '{agent_type}' not found in {module_path}")

        return getattr(module, agent_type)

    def prepare_profiles(self) -> Dict[str, List[AgentProfile]]:
        """Prepare all needed profiles for agents"""
        # Get agent profiles from agent config
        agent_profiles = {}
        if self.agent_config and hasattr(self.agent_config, 'profile'):
            agent_profiles = self.agent_config.profile

        base_profiles_per_type = {}
        counts_per_type = {}
        all_profiles_per_type = {}

        # 1. Collect base profiles for each agent type
        for agent_type in self.all_agents.keys():
            count = 5  # Default count
            profile_file_path = None

            # Try to get profile from agent config
            if agent_type in agent_profiles:
                profile_config = agent_profiles[agent_type]
                count = profile_config.get("count", 5)
                profile_path = profile_config.get("profile_path")
                if profile_path:
                    profile_file_path = os.path.join(self.env_path, "profile", "data", profile_path)

            # Fallback to default path if not found
            if not profile_file_path:
                profile_file_path = os.path.join(self.env_path, "profile","data",f"{agent_type}.json")

            schema = self.schemas.get(agent_type)
            if not schema:
                logger.warning(f"No schema found for {agent_type}. Skipping profile generation.")
                continue

            if os.path.exists(profile_file_path):
                base_profiles = self.load_profiles(
                    agent_type, schema, profile_file_path, count, self.agent_index
                )
            else:
                base_profiles = self.generate_profiles(
                    agent_type, schema, self.model, count,
                    profile_file_path,
                    self.agent_index
                )

            if not base_profiles:
                logger.warning(f"No base profiles for {agent_type}. Skipping.")
                continue

            self.agent_index += len(base_profiles)
            base_profiles_per_type[agent_type] = base_profiles
            counts_per_type[agent_type] = count

        # 2. Calculate needed tiles (maximum of tiles needed per agent type)
        tiles_needed = 0
        for agent_type, count in counts_per_type.items():
            base_count = len(base_profiles_per_type[agent_type])
            # Tiles needed for this agent type
            tile_for_type = (count + base_count - 1) // base_count
            tiles_needed = max(tiles_needed, tile_for_type)

        self.tiles_needed = tiles_needed

        # 3. Calculate total base profile count for relationship offsets
        self.global_base_count = sum(len(p) for p in base_profiles_per_type.values())

        # 4. Generate all profiles by tiling base profiles
        all_profiles_per_type = {}
        for agent_type in base_profiles_per_type.keys():
            all_profiles_per_type[agent_type] = []
            for profile in base_profiles_per_type[agent_type]:
                all_profiles_per_type[agent_type].append(profile)
                self.profile_id2agent[profile.get_agent_profile_id()] = profile

        for tile_index in range(tiles_needed):
            for agent_type, base_profiles in base_profiles_per_type.items():
                needed_count = counts_per_type[agent_type]
                for profile in base_profiles:
                    # Stop if we have enough profiles for this agent type
                    if len(all_profiles_per_type[agent_type]) >= needed_count:
                        break

                    # Deep copy the profile data
                    profile_data = copy.deepcopy(profile.get_profile(include_private=True))
                    new_profile = AgentProfile(agent_type, self.schemas[agent_type], profile_data=profile_data)

                    # Set unique profile ID
                    if 'id' not in profile_data:
                        new_profile.set_agent_profile_id(self.agent_index)
                        self.agent_index += 1
                    else:
                        new_profile.set_agent_profile_id(profile_data['id']+"_"+str(tile_index))

                    all_profiles_per_type[agent_type].append(new_profile)
                    self.profile_id2agent[new_profile.get_agent_profile_id()] = new_profile

        return all_profiles_per_type

    def create_agent_configs(self, profiles_per_type: Dict[str, List[AgentProfile]], 
                       relationships_by_agent: Dict[str, List[Dict]] = None) -> List[Dict]:
        """Create agent configurations based on profiles with relationships included"""
        all_agent_configs = []

        # Standard system prompt for all agents
        system_prompt = (
            "You are an intelligent agent. Based on your Profile, Memory, and current Observation, "
            "respond to each Instruction in a way that aligns with your defined identity. "
            "Ensure each response reflects your background, past interactions, and current context."
        )

        # Get memory config from agent_config
        memory_config = None
        if self.agent_config and hasattr(self.agent_config, 'memory') and self.agent_config.memory:
            memory_config = asdict(self.agent_config.memory)

        # Get planning config from agent_config
        planning_config = None
        if self.agent_config and hasattr(self.agent_config, 'planning'):
            planning_config = self.agent_config.planning

        # Create configs for each agent type and profile
        for agent_type, profiles_list in profiles_per_type.items():
            for profile in profiles_list:
                # agent_name = profile.get_profile().get("name", agent_type)
                agent_id = profile.get_agent_profile_id()
                # Get relationships for this agent
                agent_relationships = relationships_by_agent.get(agent_id, []) if relationships_by_agent else []

                # Collect agent configuration with relationships
                agent_config = {
                    "env": self.env_path,
                    "type": agent_type,
                    "schema": self.schemas[agent_type].to_dict(),
                    "id": agent_id,
                    # "name": agent_name,
                    "profile_data": profile.get_profile(include_private=True),
                    "sys_prompt": system_prompt,
                    "model_config_name": self.model_config_name,
                    "memory_config": memory_config,
                    "planning_config": planning_config,
                    "relationships": agent_relationships
                }
                all_agent_configs.append(agent_config)
                self.profile_id2agent[agent_id] = agent_config

        return all_agent_configs

    def create_local_agents(self, agent_configs: List[Dict]) -> None:
        """Create local agent instances with pre-configured relationships"""
        # Get memory config from agent_config
        memory_config = None
        if self.agent_config and hasattr(self.agent_config, 'memory'):
            memory_config = self.agent_config.memory
        planning_config = None
        if self.agent_config and hasattr(self.agent_config, 'planning'):
            planning_config = self.agent_config.planning

        # Group configurations by agent type
        configs_by_type = {}
        for config in agent_configs:
            agent_type = config["type"]
            if agent_type not in configs_by_type:
                configs_by_type[agent_type] = []
            configs_by_type[agent_type].append(config)

        # Create agents for each type
        for agent_type, configs in configs_by_type.items():
            # Load the agent class
            AgentClass = self.load_agent_module_from_file(agent_type)

            # Create each agent instance
            for config in tqdm(configs):
                agent_id = config["id"]

                # Create profile instance from saved data
                profile = AgentProfile(
                    agent_type, 
                    self.schemas[agent_type], 
                    profile_data=config["profile_data"]
                )
                profile.set_agent_profile_id(agent_id)

                # Create relationship manager and add relationships
                rm = RelationshipManager(profile_id=agent_id)

                # Add pre-loaded relationships if present
                if "relationships" in config:
                    for relationship in config["relationships"]:
                        rm.add_relationship(
                            target_id=relationship["target_id"],
                            description=relationship["description"],
                            target_info=relationship["target_info"]
                        )

                # Create memory instance
                memory_instance=None
                if memory_config:
                    memory_instance = self.load_memory(memory_config, self.model_config_name)

                # Create planning instance if configured
                planning_instance = None
                if planning_config:
                    planning_instance = self.load_planning(planning_config, 
                                                           self.model_config_name, 
                                                           config["sys_prompt"])

                # Create agent instance
                agent = AgentClass(
                    # name=config["name"],
                    profile=profile,
                    sys_prompt=config["sys_prompt"],
                    model_config_name=self.model_config_name,
                    #model_config_name="chat_load_balancer",
                    memory=memory_instance,
                    planning=planning_instance,
                    event_bus_queue=get_event_bus().queue,
                    relationship_manager=rm
                )

                # Store the agent in our dictionaries
                if agent_type not in self.all_agents:
                    self.all_agents[agent_type] = {}
                self.all_agents[agent_type][agent_id] = agent
                self.profile_id2agent[agent_id] = agent

    async def create_distributed_agents(self, agent_configs: List[Dict]) -> None:
        """Create distributed agent instances on worker nodes"""
        node = get_node()
        if not isinstance(node, MasterNode):
            logger.error("Node is not a MasterNode instance")
            return

        # Create allocator
        allocator = AgentAllocator(node)

        # Batch allocate agents to workers
        allocations = await allocator.allocate_agents_batch(agent_configs)
        if not allocations:
            logger.error("Failed to allocate agents to workers")
            return

        # Create agents on workers

        success = await allocator.create_agents_on_workers(allocations, agent_configs)
        if success:
            logger.info(f"Successfully created {len(agent_configs)} agents on workers")
        else:
            logger.error("Failed to create some or all agents on workers")

        # Update local dictionaries with proxy references
        for config in agent_configs:
            agent_type = config["type"]
            agent_id = config["id"]

            # In distributed mode, we store the config as a placeholder
            if agent_type not in self.all_agents:
                self.all_agents[agent_type] = {}
            self.all_agents[agent_type][agent_id] = config

    def load_relationship_data(self, relationship_file: str) -> Dict[str, List[Dict]]:
        """Load relationships from CSV and organize by source agent"""
        relationships_by_agent = {}

        if not os.path.exists(relationship_file):
            logger.info("No relationship file provided. Agents will start with empty relationships.")
            return relationships_by_agent

        # Load base relationships from CSV
        base_relationships = []
        with open(relationship_file, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                base_relationships.append(row)

        # Skip if no base profiles to work with
        if self.global_base_count == 0:
            logger.warning("global_base_count = 0, no base profiles to offset. Skipping relationships.")
            return relationships_by_agent

        logger.info(f"Loading relationships for {self.tiles_needed} tiles, each tile size = {self.global_base_count} profiles.")

        # Process relationships for all tiles
        for tile_index in range(self.tiles_needed):
            for rel in base_relationships:
                # CSV base IDs (typically 0-based)
                source_base_id = str(rel["source_id"])
                target_base_id = str(rel["target_id"])

                # Add offset and convert to 1-based system IDs
                source_id = source_base_id + "_" + str(tile_index) if tile_index > 0 else source_base_id
                target_id = target_base_id + "_" + str(tile_index) if tile_index > 0 else target_base_id

                description = rel.get("relationship_type", "Unknown")
                direction = rel.get("direction", "unidirectional")

                # Get target agent profile info
                target_agent = self.profile_id2agent.get(target_id,None)
                if not target_agent:
                    # logger.warning(f"Relationship skipped: target agent {target_id} not found.")
                    continue

                # Get target agent profile info
                if isinstance(target_agent, dict):  # For distributed mode
                    target_info = target_agent["profile_data"]
                else:  # For local mode
                    target_info = target_agent.get_profile(include_private=False)

                # Add relationship to source agent's relationships
                if source_id not in relationships_by_agent:
                    relationships_by_agent[source_id] = []

                relationships_by_agent[source_id].append({
                    "target_id": target_id,
                    "description": description,
                    "target_info": target_info
                })

                # If bidirectional, add the reverse relationship
                if direction == "bidirectional":
                    # Get source agent profile info
                    source_agent = self.profile_id2agent.get(source_id,None)
                    if not source_agent:
                        # logger.warning(f"Reverse relationship skipped: source agent {source_id} not found.")
                        continue

                    # Get source agent profile info
                    if isinstance(source_agent, dict):  # For distributed mode
                        source_info = source_agent["profile_data"]
                    else:  # For local mode
                        source_info = source_agent.get_profile(include_private=False)

                    # Add relationship to target agent's relationships
                    if target_id not in relationships_by_agent:
                        relationships_by_agent[target_id] = []

                    relationships_by_agent[target_id].append({
                        "target_id": source_id,
                        "description": description,
                        "target_info": source_info
                    })

        logger.info(f"Loaded relationship data for {len(relationships_by_agent)} agents.")
        return relationships_by_agent

    async def create_agents(self) -> Dict[str, Dict[str, Any]]:
        """Main method to create all agents"""
        # Check if we're running in distributed mode
        node = get_node()
        is_master = node and node.role == NodeRole.MASTER

        # 1. Prepare all profiles
        logger.info("Step 1: Preparing agent profiles")
        profiles_per_type = self.prepare_profiles()

        # 2. Load relationship data
        logger.info("Step 2: Loading relationship data")

        # Get from config if available
        if self.agent_config and hasattr(self.agent_config, 'relationship_path'):
            relationship_path = os.path.join(self.env_path, 'profile', 'data', self.agent_config.relationship_path)
        else:
            relationship_path = os.path.join(self.env_path, 'profile', 'data', 'Relationship.csv')

        relationships_by_agent = self.load_relationship_data(relationship_path)

        # 3. Create agent configurations with relationships included
        logger.info("Step 3: Creating agent configurations")
        agent_configs = self.create_agent_configs(profiles_per_type, relationships_by_agent)

        # 4. Create agent instances (local or distributed)
        logger.info("Step 4: Creating agent instances")
        if is_master:
            # In master mode, create distributed agents and wait for completion
            logger.info(f"Creating {len(agent_configs)} agents in distributed mode")

            await self.create_distributed_agents(agent_configs)
        else:
            # In local mode, create agents directly
            logger.info(f"Creating {len(agent_configs)} agents in local mode")
            self.create_local_agents(agent_configs)

        return self.all_agents

    def load_planning(self, planning_config, model_config_name: str, sys_prompt: str) -> PlanningBase:
        """Create planning strategy instance"""
        if not planning_config:
            return None
        try:
            # Load planning module and class
            planning_module = importlib.import_module("onesim.planning")
            PlanningClass = getattr(planning_module, planning_config)

            # Initialize planning instance
            planning_instance = PlanningClass(model_config_name, sys_prompt)
            return planning_instance
        except ImportError as e:
            logger.error(f"Failed to import module: {e}")
        except AttributeError as e:
            logger.error(f"Planning class {planning_config} not found: {e}")

        return None

    def load_memory(self, memory_config: AgentMemoryConfig, model_config_name: str) -> MemoryStrategy:
        """Create memory strategy instance"""
        if not memory_config:
            return None
            # logger.warning("No memory configuration provided. Using default memory strategy.")
            # # Create a minimal default memory config
            # memory_config = AgentMemoryConfig(strategy="ListStrategy")

        strategy_class_name = memory_config.strategy
        try:
            # Load memory module and class
            memory_module = importlib.import_module("onesim.memory")
            MemoryClass = getattr(memory_module, strategy_class_name)

            # Initialize memory instance - convert dataclass to dict if needed
            if hasattr(memory_config, "__dict__"):
                # Convert dataclass to dictionary
                memory_config_dict = {
                    "strategy": memory_config.strategy,
                    "storages": memory_config.storages,
                    "metric_weights": memory_config.metric_weights,
                    "transfer_conditions": memory_config.transfer_conditions,
                    "operations": memory_config.operations,
                    "metrics": memory_config.metrics
                }
                memory_instance = MemoryClass(memory_config_dict, model_config_name=model_config_name)
            else:
                # Already a dictionary
                memory_instance = MemoryClass(memory_config, model_config_name=model_config_name)

            return memory_instance
        except ImportError as e:
            logger.error(f"Failed to import module: {e}")
        except AttributeError as e:
            logger.error(f"Class {strategy_class_name} not found in onesim.memory: {e}")
        except Exception as e:
            logger.error(f"An error occurred during memory initialization: {e}")
        # Fallback to a simple memory strategy
        try:
            memory_module = importlib.import_module("onesim.memory")
            ListStrategy = getattr(memory_module, "ListStrategy")
            # Create minimal config for ListStrategy with basic operations
            fallback_config = {
                "strategy": "ListStrategy",
                "operations": {
                    "add": {"class": "AddMemoryOperation"},
                    "retrieve": {"class": "RetrieveMemoryOperation"},
                    "remove": {"class": "RemoveMemoryOperation"},
                },
            }

            logger.warning(
                f"Falling back to ListStrategy with config: {fallback_config}"
            )
            return ListStrategy(fallback_config, model_config_name=model_config_name)
        except Exception as e:
            logger.error(f"Failed to create fallback memory strategy: {e}")
            return None

    @staticmethod
    def generate_profiles(agent_type: str, schema: AgentSchema, model, num_profiles: int, 
                         output_path: str, index: int = 0) -> List[AgentProfile]:
        """Generate agent profiles using LLM"""
        profiles = []
        existing_profiles_set = set()
        all_profiles_data = []

        # Read existing profiles to ensure uniqueness
        profile_file_exists = False

        try:
            with open(output_path, mode='r', encoding='utf-8') as jsonfile:
                all_profiles_data = json.load(jsonfile)
                profile_file_exists = True

                for existing_profile in all_profiles_data:
                    profile_json = json.dumps(existing_profile, ensure_ascii=False, sort_keys=True)
                    existing_profiles_set.add(profile_json)

                # 如果已经存在足够数量的profiles并且没有要求生成更多，直接返回空列表
                if num_profiles <= 0:
                    logger.info(f"No new profiles needed for {agent_type}, {len(all_profiles_data)} already exist.")
                    return []

                logger.info(f"Adding {num_profiles} new profiles to existing {len(all_profiles_data)} profiles for {agent_type}")

        except (json.JSONDecodeError, FileNotFoundError):
            logger.info(f"No valid existing profiles found at '{output_path}'. Will generate {num_profiles} new profiles.")

        # Always keep track of the highest id used
        max_id = index
        if profile_file_exists:
            for profile in all_profiles_data:
                if 'id' in profile:
                    profile_id = profile['id']
                    try:
                        if isinstance(profile_id, str) and profile_id.isdigit():
                            max_id = max(max_id, int(profile_id) + 1)
                        elif isinstance(profile_id, int):
                            max_id = max(max_id, profile_id + 1)
                    except Exception as e:
                        logger.warning(f"Error parsing profile ID '{profile_id}': {e}")

            # Use the highest id as our starting index
            index = max(index, max_id)
            logger.info(f"Starting profile generation with index {index}")

        attempts = 0
        max_attempts = 5
        remaining = num_profiles
        batch_size = 10

        while len(profiles) < num_profiles and attempts < max_attempts:
            try:
                # Generate profiles in batches
                generated_profiles = AgentProfile.generate_profiles(
                    model=model,
                    schema=schema,
                    agent_type=agent_type,
                    count=min(remaining, batch_size)
                )

                # Filter duplicates
                unique_generated = []
                for profile in generated_profiles:
                    profile_data = profile.get_profile(include_private=True)
                    profile_json = json.dumps(profile_data, ensure_ascii=False, sort_keys=True)
                    if profile_json not in existing_profiles_set:
                        profile.set_agent_profile_id(index)
                        unique_generated.append(profile)
                        existing_profiles_set.add(profile_json)
                        index += 1
                        if len(profiles) + len(unique_generated) >= num_profiles:
                            break

                profiles.extend(unique_generated)
                remaining = num_profiles - len(profiles)
                attempts += 1

            except ValueError as e:
                logger.warning(f"Attempt {attempts + 1} failed: {e}")
                attempts += 1

        # If we couldn't generate enough unique profiles, allow duplicates
        if len(profiles) < num_profiles:
            logger.warning("Reached maximum attempts, could not generate enough unique profiles. Using duplicate profiles.")
            try:
                additional_profiles = AgentProfile.generate_profiles(
                    model=model,
                    schema=schema,
                    agent_type=agent_type,
                    count=(num_profiles - len(profiles))
                )
                for profile in additional_profiles:
                    profile.set_agent_profile_id(index)
                    profiles.append(profile)
                    index += 1
            except ValueError as e:
                logger.error(f"Failed to generate additional profiles: {e}")
                raise ValueError(f"Unable to generate the desired number of profiles. Requested: {num_profiles}, Generated: {len(profiles)}.")

        # If no new profiles were generated, return empty list
        if not profiles:
            logger.info(f"No new profiles generated for {agent_type}")
            return []

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Add new profiles to existing ones
        profiles_data = [profile.get_profile(include_private=True) for profile in profiles]
        all_profiles_data += profiles_data

        # Write all profiles to JSON file
        with open(output_path, mode='w', encoding='utf-8') as jsonfile:
            json.dump(all_profiles_data, jsonfile, ensure_ascii=False, indent=4)

        logger.info(f"Successfully saved {len(profiles)} new profiles to '{output_path}', total {len(all_profiles_data)}")
        return profiles

    @staticmethod
    def load_profiles(agent_type: str, schema: AgentSchema, json_file: str, 
                     count: int = 1, index: int = 0) -> List[AgentProfile]:
        """Load profiles from JSON file"""
        profiles = []

        # Check if file exists
        if not os.path.exists(json_file):
            logger.warning(f"File '{json_file}' does not exist. Returning an empty profile list.")
            return profiles

        # Read profiles from JSON
        with open(json_file, mode='r', encoding='utf-8') as jsonfile:
            all_profile_rows = json.load(jsonfile)

        # Create profiles up to requested count
        for row in all_profile_rows:
            profile = AgentProfile(agent_type, schema, profile_data=row)
            profiles.append(profile)
            if 'id' not in row:
                profile.set_agent_profile_id(index)
            else:
                profile.set_agent_profile_id(row['id'])
            if len(profiles) == count:
                break
            index += 1

        logger.info(f"Loaded {len(profiles)} base profiles from '{json_file}'.")
        return profiles

    def add_env_relationship(self, profile_id: str):
        """Add relationship to environment agent"""
        agent = self.profile_id2agent.get(profile_id)
        if agent:
            if isinstance(agent, dict):  # For distributed mode
                # logger.info(f"Environment relationship for distributed agent {profile_id} will be set up.")
                pass
                # Implementation depends on distributed architecture
            else:  # For local mode
                agent.add_relationship("ENV", "Towards EnvAgent", {"agent_type": "EnvAgent"})
        else:
            logger.warning(f"Cannot add environment relationship. Agent with profile_id {profile_id} not found.")

    def get_agent_ids(self, agent_types: List[str] = None) -> Dict[str, List[str]]:
        """Get all agent IDs by type"""
        agent_ids = {}
        if not agent_types:
            agent_types = list(self.all_agents.keys())

        for agent_type in agent_types:
            agent_ids[agent_type] = list(self.all_agents.get(agent_type, {}).keys())

        return agent_ids

    def get_agent_profile_ids(self, agent_types: List[str] = None) -> Dict[str, List[str]]:
        """Get all agent profile IDs by type"""
        agent_profile_ids = {}
        if not agent_types:
            agent_types = list(self.all_agents.keys())

        for agent_type in agent_types:
            if agent_type in self.all_agents:
                agents = self.all_agents[agent_type]
                profile_ids = []

                for agent_id, agent in agents.items():
                    profile_ids.append(agent_id)

                agent_profile_ids[agent_type] = profile_ids
            else:
                agent_profile_ids[agent_type] = []

        return agent_profile_ids

    def get_agent_types(self) -> List[str]:
        """Get all agent types"""
        return list(self.all_agents.keys())

    def get_agent_by_profile_id(self, profile_id: str):
        """Get agent by profile ID"""
        return self.profile_id2agent.get(profile_id)
