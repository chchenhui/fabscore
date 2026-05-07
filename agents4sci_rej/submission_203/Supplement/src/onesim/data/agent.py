import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from loguru import logger

from .database import DatabaseManager


class AgentManager:
    """
    Manager for agent data. Handles operations related to storing and retrieving agent information and states.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize agent manager with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager or DatabaseManager.get_instance()
    
    async def register_agent(
        self,
        trail_id: str,
        agent_id: str,
        agent_type: str,
        name: str,
        initial_profile: Dict[str, Any],
        universe_id: str = "main",
        system_prompt: Optional[str] = None,
        model_config_name: Optional[str] = None,
        memory_config: Optional[Dict[str, Any]] = None,
        planning_config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register an agent in the database.
        
        Args:
            trail_id: ID of the trail
            agent_id: ID of the agent
            agent_type: Type of the agent
            name: Name of the agent
            initial_profile: Initial profile data
            universe_id: Universe ID
            system_prompt: System prompt for the agent
            model_config_name: Model configuration name
            memory_config: Memory configuration
            planning_config: Planning configuration
            
        Returns:
            Success flag
        """
        # If database is disabled, just return success
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping agent registration for {agent_id}")
            return True
            
        query = """
        INSERT INTO agents (
            agent_id, trail_id, universe_id, agent_type, name,
            initial_profile, system_prompt, model_config_name,
            memory_config, planning_config
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (agent_id, trail_id, universe_id) 
        DO UPDATE SET 
            agent_type = $4,
            name = $5,
            initial_profile = $6,
            system_prompt = $7,
            model_config_name = $8,
            memory_config = $9,
            planning_config = $10,
            updated_at = CURRENT_TIMESTAMP
        RETURNING agent_id
        """
        
        try:
            result = await self.db.execute(
                query, 
                agent_id, 
                trail_id, 
                universe_id,
                agent_type,
                name,
                json.dumps(initial_profile),
                system_prompt,
                model_config_name,
                json.dumps(memory_config) if memory_config else None,
                json.dumps(planning_config) if planning_config else None
            )
            logger.debug(f"Registered agent {agent_id} in trail {trail_id}, universe {universe_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent: {e}")
            return False
    
    async def get_agent(
        self,
        trail_id: str,
        agent_id: str,
        universe_id: str = "main"
    ) -> Dict[str, Any]:
        """
        Get agent data by ID.
        
        Args:
            trail_id: ID of the trail
            agent_id: ID of the agent
            universe_id: Universe ID
            
        Returns:
            Agent data
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for agent {agent_id}")
            return None
            
        query = """
        SELECT * FROM agents
        WHERE trail_id = $1 AND agent_id = $2 AND universe_id = $3
        """
        
        row = await self.db.fetchrow(query, trail_id, agent_id, universe_id)
        if not row:
            return None
        
        result = dict(row)
        # Parse JSON fields
        for field in ['initial_profile', 'memory_config', 'planning_config']:
            if field in result and result[field]:
                result[field] = json.loads(result[field])
        
        return result
    
    async def list_agents(
        self,
        trail_id: str,
        universe_id: str = "main",
        agent_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all agents in a trail.
        
        Args:
            trail_id: ID of the trail
            universe_id: Universe ID
            agent_type: Optional agent type to filter by
            
        Returns:
            List of agent data
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
            
        conditions = ["trail_id = $1", "universe_id = $2"]
        params = [trail_id, universe_id]
        
        if agent_type:
            conditions.append(f"agent_type = ${len(params) + 1}")
            params.append(agent_type)
        
        query = f"""
        SELECT * FROM agents
        WHERE {' AND '.join(conditions)}
        ORDER BY name
        """
        
        rows = await self.db.fetch(query, *params)
        result = []
        
        for row in rows:
            agent_data = dict(row)
            # Parse JSON fields
            for field in ['initial_profile', 'memory_config', 'planning_config']:
                if field in agent_data and agent_data[field]:
                    agent_data[field] = json.loads(agent_data[field])
            result.append(agent_data)
        
        return result
    
    async def save_agent_state(
        self,
        trail_id: str,
        agent_id: str,
        step: int,
        profile: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, Any]] = None,
        relationships: Optional[Dict[str, Any]] = None,
        additional_state: Optional[Dict[str, Any]] = None,
        universe_id: str = "main",
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Save agent state for a specific step.
        
        Args:
            trail_id: ID of the trail
            agent_id: ID of the agent
            step: Step number
            profile: Agent profile state
            memory: Agent memory state
            relationships: Agent relationships state
            additional_state: Additional state data
            universe_id: Universe ID
            timestamp: Time of state (defaults to now)
            
        Returns:
            state_id: ID of the saved state
        """
        state_id = str(uuid.uuid4())
        
        # If database is disabled, just return the ID
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning generated state ID: {state_id}")
            return state_id
            
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check if agent exists
        agent = await self.get_agent(trail_id, agent_id, universe_id)
        if not agent:
            logger.warning(f"Agent {agent_id} not found in trail {trail_id}, universe {universe_id}")
            return None
        
        query = """
        INSERT INTO agent_states (
            state_id, trail_id, universe_id, agent_id, step, timestamp,
            profile, memory, relationships, additional_state
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (trail_id, universe_id, agent_id, step) 
        DO UPDATE SET 
            profile = $7,
            memory = $8,
            relationships = $9,
            additional_state = $10,
            timestamp = $6
        RETURNING state_id
        """
        
        try:
            result = await self.db.execute(
                query, 
                state_id, 
                trail_id, 
                universe_id, 
                agent_id, 
                step, 
                timestamp,
                json.dumps(profile) if profile else None,
                json.dumps(memory) if memory else None,
                json.dumps(relationships) if relationships else None,
                json.dumps(additional_state) if additional_state else None
            )
            logger.debug(f"Saved state for agent {agent_id} at step {step}")
            return state_id
        except Exception as e:
            logger.error(f"Failed to save agent state: {e}")
            raise
    
    async def get_agent_state(
        self,
        trail_id: str,
        agent_id: str,
        step: int,
        universe_id: str = "main"
    ) -> Dict[str, Any]:
        """
        Get agent state for a specific step.
        
        Args:
            trail_id: ID of the trail
            agent_id: ID of the agent
            step: Step number
            universe_id: Universe ID
            
        Returns:
            Agent state data
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for agent {agent_id} state at step {step}")
            return None
            
        query = """
        SELECT * FROM agent_states
        WHERE trail_id = $1 AND universe_id = $2 AND agent_id = $3 AND step = $4
        """
        
        row = await self.db.fetchrow(query, trail_id, universe_id, agent_id, step)
        if not row:
            return None
        
        result = dict(row)
        # Parse JSON fields
        for field in ['profile', 'memory', 'relationships', 'additional_state']:
            if field in result and result[field]:
                result[field] = json.loads(result[field])
        
        return result
    
    async def get_agent_states(
        self,
        trail_id: str,
        agent_id: str,
        universe_id: str = "main",
        start_step: Optional[int] = None,
        end_step: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get a range of agent states.
        
        Args:
            trail_id: ID of the trail
            agent_id: ID of the agent
            universe_id: Universe ID
            start_step: First step to include (inclusive)
            end_step: Last step to include (inclusive)
            
        Returns:
            List of agent state data
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
            
        conditions = ["trail_id = $1", "universe_id = $2", "agent_id = $3"]
        params = [trail_id, universe_id, agent_id]
        
        if start_step is not None:
            conditions.append(f"step >= ${len(params) + 1}")
            params.append(start_step)
        
        if end_step is not None:
            conditions.append(f"step <= ${len(params) + 1}")
            params.append(end_step)
        
        query = f"""
        SELECT * FROM agent_states
        WHERE {' AND '.join(conditions)}
        ORDER BY step ASC
        """
        
        rows = await self.db.fetch(query, *params)
        result = []
        
        for row in rows:
            state_data = dict(row)
            # Parse JSON fields
            for field in ['profile', 'memory', 'relationships', 'additional_state']:
                if field in state_data and state_data[field]:
                    state_data[field] = json.loads(state_data[field])
            result.append(state_data)
        
        return result
    
    async def get_latest_agent_state(
        self,
        trail_id: str,
        agent_id: str,
        universe_id: str = "main"
    ) -> Dict[str, Any]:
        """
        Get the latest agent state.
        
        Args:
            trail_id: ID of the trail
            agent_id: ID of the agent
            universe_id: Universe ID
            
        Returns:
            Latest agent state data
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for latest state of agent {agent_id}")
            return None
            
        query = """
        SELECT * FROM agent_states
        WHERE trail_id = $1 AND universe_id = $2 AND agent_id = $3
        ORDER BY step DESC
        LIMIT 1
        """
        
        row = await self.db.fetchrow(query, trail_id, universe_id, agent_id)
        if not row:
            return None
        
        result = dict(row)
        # Parse JSON fields
        for field in ['profile', 'memory', 'relationships', 'additional_state']:
            if field in result and result[field]:
                result[field] = json.loads(result[field])
        
        return result
    
    async def save_agent_decision(
        self,
        trail_id: str,
        agent_id: str,
        step: int,
        prompt: str,
        output: str,
        decision_id: Optional[str] = None,
        event_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        processing_time: Optional[float] = None,
        feedback: Optional[str] = None,
        universe_id: str = "main",
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Save agent decision for training data.
        
        Args:
            trail_id: ID of the trail
            agent_id: ID of the agent
            step: Step number
            prompt: Input prompt
            output: Output from the agent
            event_id: Optional associated event ID
            context: Context information
            processing_time: Time taken to process
            feedback: Feedback on the decision
            universe_id: Universe ID
            timestamp: Time of decision (defaults to now)
            
        Returns:
            decision_id: ID of the saved decision
        """
        # Generate a random UUID for the decision ID
        decision_id = str(uuid.uuid4()) if not decision_id else decision_id
        
        # Default timestamp to now
        if timestamp is None:
            timestamp = datetime.now()
        
        conn = None
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            # Insert decision
            cursor.execute(
                """
                INSERT INTO agent_decisions
                (decision_id, trail_id, universe_id, agent_id, step, timestamp,
                 event_id, context, prompt, output, processing_time, feedback)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING decision_id
                """,
                (
                    decision_id,
                    trail_id,
                    universe_id,
                    agent_id,
                    step,
                    timestamp,
                    event_id,
                    json.dumps(context or {}),
                    prompt,
                    output,
                    processing_time,
                    feedback
                )
            )
            
            decision_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"Saved agent decision for {agent_id} in trail {trail_id}, step {step}")
            
            return decision_id
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to save agent decision: {e}")
            raise
        finally:
            if conn:
                self.db._release_connection(conn)
    
    async def get_agent_decisions(
        self,
        trail_id: str,
        agent_id: str,
        universe_id: str = "main",
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        with_feedback_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get agent decisions for training data.
        
        Args:
            trail_id: ID of the trail
            agent_id: ID of the agent
            universe_id: Universe ID
            start_step: First step to include (inclusive)
            end_step: Last step to include (inclusive)
            with_feedback_only: If True, only return decisions with feedback
            
        Returns:
            List of decision data
        """
        conn = None
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    decision_id, trail_id, universe_id, agent_id, step,
                    timestamp, event_id, context, prompt, output,
                    processing_time, feedback, created_at
                FROM agent_decisions 
                WHERE trail_id = %s AND agent_id = %s AND universe_id = %s
            """
            params = [trail_id, agent_id, universe_id]
            
            if start_step is not None:
                query += " AND step >= %s"
                params.append(start_step)
            
            if end_step is not None:
                query += " AND step <= %s"
                params.append(end_step)
            
            if with_feedback_only:
                query += " AND feedback IS NOT NULL"
            
            query += " ORDER BY step ASC"
            
            cursor.execute(query, tuple(params))
            
            decisions = []
            for row in cursor.fetchall():
                decisions.append({
                    'decision_id': row[0],
                    'trail_id': row[1],
                    'universe_id': row[2],
                    'agent_id': row[3],
                    'step': row[4],
                    'timestamp': row[5],
                    'event_id': row[6],
                    'context': row[7],
                    'prompt': row[8],
                    'output': row[9],
                    'processing_time': row[10],
                    'feedback': row[11],
                    'created_at': row[12]
                })
            
            return decisions
        except Exception as e:
            logger.error(f"Failed to get agent decisions: {e}")
            raise
        finally:
            if conn:
                self.db._release_connection(conn)
    
    async def add_feedback_to_decision(
        self,
        decision_id: str,
        feedback: str
    ) -> bool:
        """
        Add feedback to an agent decision for reinforcement learning.
        
        Args:
            decision_id: ID of the decision
            feedback: Feedback text
            
        Returns:
            Success flag
        """
        conn = None
        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                """
                UPDATE agent_decisions SET
                feedback = %s
                WHERE decision_id = %s
                """,
                (feedback, decision_id)
            )
            
            if cursor.rowcount == 0:
                logger.warning(f"Decision not found for feedback: {decision_id}")
                return False
            
            conn.commit()
            logger.info(f"Added feedback to decision: {decision_id}")
            return True
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to add feedback to decision: {e}")
            raise
        finally:
            if conn:
                self.db._release_connection(conn)
    
    async def get_agent_relationships(
        self,
        trail_id: str,
        agent_id: str,
        step: Optional[int] = None,
        universe_id: str = 'main'
    ) -> Dict[str, Any]:
        """
        Get agent relationships
        
        Args:
            trail_id: Trail ID
            agent_id: Agent ID
            step: Simulation step (if None, get latest)
            universe_id: Universe ID
            
        Returns:
            Agent relationships or None if not found
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for agent {agent_id} relationships")
            return None
            
        if step is None:
            # Get latest state
            latest_state = await self.get_latest_agent_state(trail_id, agent_id, universe_id)
            if not latest_state:
                return None
            return latest_state.get('relationships')
        else:
            # Get state at specific step
            state = await self.get_agent_state(trail_id, agent_id, step, universe_id)
            if not state:
                return None
            return state.get('relationships')
    
    async def get_agent_memory(
        self,
        trail_id: str,
        agent_id: str,
        step: Optional[int] = None,
        universe_id: str = 'main'
    ) -> Dict[str, Any]:
        """
        Get agent memory
        
        Args:
            trail_id: Trail ID
            agent_id: Agent ID
            step: Simulation step (if None, get latest)
            universe_id: Universe ID
            
        Returns:
            Agent memory or None if not found
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for agent {agent_id} memory")
            return None
            
        if step is None:
            # Get latest state
            latest_state = await self.get_latest_agent_state(trail_id, agent_id, universe_id)
            if not latest_state:
                return None
            return latest_state.get('memory')
        else:
            # Get state at specific step
            state = await self.get_agent_state(trail_id, agent_id, step, universe_id)
            if not state:
                return None
            return state.get('memory')
    
    async def get_agent_profile(
        self,
        trail_id: str,
        agent_id: str,
        step: Optional[int] = None,
        universe_id: str = 'main'
    ) -> Dict[str, Any]:
        """
        Get agent profile
        
        Args:
            trail_id: Trail ID
            agent_id: Agent ID
            step: Simulation step (if None, get latest)
            universe_id: Universe ID
            
        Returns:
            Agent profile or None if not found
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for agent {agent_id} profile")
            return None
            
        if step is None:
            # Get latest state
            latest_state = await self.get_latest_agent_state(trail_id, agent_id, universe_id)
            if not latest_state:
                # If no state exists yet, return initial profile
                agent = await self.get_agent(trail_id, agent_id, universe_id)
                return agent.get('initial_profile') if agent else None
            return latest_state.get('profile')
        else:
            # Get state at specific step
            state = await self.get_agent_state(trail_id, agent_id, step, universe_id)
            if not state:
                # If no state exists at this step, return initial profile
                if step == 0:
                    agent = await self.get_agent(trail_id, agent_id, universe_id)
                    return agent.get('initial_profile') if agent else None
                return None
            return state.get('profile')
    
    async def delete_agent_states(
        self,
        trail_id: str,
        agent_id: Optional[str] = None,
        universe_id: Optional[str] = None,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None
    ) -> int:
        """
        Delete agent states
        
        Args:
            trail_id: Trail ID
            agent_id: Optional agent ID (if None, delete for all agents)
            universe_id: Optional universe ID (if None, delete from all universes)
            start_step: Optional starting step (inclusive)
            end_step: Optional ending step (inclusive)
            
        Returns:
            Number of deleted states
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping delete for trail {trail_id}")
            return 0
            
        conditions = ["trail_id = $1"]
        params = [trail_id]
        
        if agent_id is not None:
            conditions.append(f"agent_id = ${len(params) + 1}")
            params.append(agent_id)
        
        if universe_id is not None:
            conditions.append(f"universe_id = ${len(params) + 1}")
            params.append(universe_id)
        
        if start_step is not None:
            conditions.append(f"step >= ${len(params) + 1}")
            params.append(start_step)
        
        if end_step is not None:
            conditions.append(f"step <= ${len(params) + 1}")
            params.append(end_step)
        
        query = f"""
        DELETE FROM agent_states
        WHERE {' AND '.join(conditions)}
        RETURNING state_id
        """
        
        result = await self.db.fetch(query, *params)
        count = len(result)
        
        logger.info(f"Deleted {count} agent states for trail {trail_id}")
        return count 