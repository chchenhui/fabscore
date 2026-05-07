"""
Agent decision management for OneSim

This module provides functionality to store and retrieve agent decision data,
which is useful for training data collection and analysis.
"""

import uuid
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from loguru import logger

from onesim.data.database import DatabaseManager
from onesim.utils.common import convert_sql_data
class DecisionManager:
    """
    Manager for agent decision data. Handles operations related to storing and retrieving
    agent decisions for training and analysis.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize decision manager with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager or DatabaseManager.get_instance()
    
    async def record_decision(self, 
                             trail_id: str,
                             agent_id: str,
                             step: int,
                             prompt: str,
                             output: str,
                             decision_id: Optional[str] = None,
                             agent_type: Optional[str] = None,
                             action: Optional[str] = None,
                             event_id: Optional[str] = None,
                             context: Optional[Dict[str, Any]] = None,
                             processing_time: Optional[float] = None,
                             universe_id: str = 'main',
                             timestamp: Optional[datetime] = None,
                             rating: Optional[float] = None,
                             feedback: Optional[str] = None,
                             reason: Optional[str] = None) -> str:
        """
        Record an agent decision
        
        Args:
            trail_id: Trail ID
            agent_id: Agent ID
            step: Step number
            prompt: The prompt sent to the agent
            output: The output/response from the agent
            event_id: Optional ID of the event that triggered this decision
            context: Optional context information
            processing_time: Optional processing time in seconds
            universe_id: Universe ID (for parallel universes)
            timestamp: Optional timestamp (defaults to current time)
            rating: Optional rating (float)
            feedback: Optional feedback
            reason: Optional reason
            agent_type: Optional type of the agent
            action: Optional action taken by the agent
            
        Returns:
            decision_id: UUID of the recorded decision
        """
        decision_id = str(uuid.uuid4()) if not decision_id else decision_id
        
        # If database is disabled, just return the ID
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning generated decision ID: {decision_id}")
            return decision_id
            
        if timestamp is None:
            timestamp = datetime.now()
        
        # Insert decision record
        query = """
        INSERT INTO agent_decisions (
            decision_id, trail_id, universe_id, agent_id, step, timestamp,
            event_id, context, prompt, output, processing_time, feedback,
            rating, reason, agent_type, action
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
        RETURNING decision_id
        """
        
        try:
            await self.db.execute(
                query, 
                decision_id, 
                trail_id, 
                universe_id, 
                agent_id,
                agent_type, 
                step,
                timestamp,
                event_id,
                json.dumps(context) if context else None,
                prompt,
                output,
                processing_time,
                action,
                feedback,
                rating,
                reason,
                agent_type,
                action
            )
            logger.debug(f"Recorded decision for agent {agent_id} in trail {trail_id}, step {step}")
            return decision_id
        except Exception as e:
            logger.error(f"Failed to record agent decision: {e}")
            raise
    
    async def get_decision(self, decision_id: str) -> Dict[str, Any]:
        """
        Get a decision by ID
        
        Args:
            decision_id: Decision ID
            
        Returns:
            Decision data
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for decision {decision_id}")
            return None
        
        query = "SELECT * FROM agent_decisions WHERE decision_id = $1"
        row = await self.db.fetchrow(query, decision_id)
        
        if not row:
            return None
        
        result = dict(row)
        # Parse JSON fields
        if result.get('context'):
            result['context'] = json.loads(result['context'])
        
        return result
    
    async def get_agent_decisions(self, 
                                 trail_id: str,
                                 agent_id: str,
                                 universe_id: str = 'main',
                                 start_step: Optional[int] = None,
                                 end_step: Optional[int] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get decisions for a specific agent
        
        Args:
            trail_id: Trail ID
            agent_id: Agent ID
            universe_id: Universe ID
            start_step: Optional start step
            end_step: Optional end step
            limit: Maximum number of decisions to return
            
        Returns:
            List of decision data
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
        
        conditions = ["trail_id = $1", "agent_id = $2", "universe_id = $3"]
        params = [trail_id, agent_id, universe_id]
        
        if start_step is not None:
            conditions.append(f"step >= ${len(params) + 1}")
            params.append(start_step)
        
        if end_step is not None:
            conditions.append(f"step <= ${len(params) + 1}")
            params.append(end_step)
        
        query = f"""
        SELECT decision_id, trail_id, universe_id, agent_id, agent_type, step, timestamp,
               event_id, context, prompt, output, processing_time, action, feedback,
               rating, reason, created_at
        FROM agent_decisions
        WHERE {' AND '.join(conditions)}
        ORDER BY step ASC
        LIMIT {limit}
        """
        
        rows = await self.db.fetch(query, *params)
        
        result = []
        for row in rows:
            decision_data = dict(row)
            # Parse JSON fields
            if decision_data.get('context'):
                decision_data['context'] = json.loads(decision_data['context'])
            result.append(decision_data)
        
        return result
    
    async def get_decisions_by_event(self, 
                                    trail_id: str,
                                    event_id: str,
                                    universe_id: str = 'main') -> List[Dict[str, Any]]:
        """
        Get all decisions related to a specific event
        
        Args:
            trail_id: Trail ID
            event_id: Event ID
            universe_id: Universe ID
            
        Returns:
            List of decision data
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
        
        query = """
        SELECT * FROM agent_decisions
        WHERE trail_id = $1 AND universe_id = $2 AND event_id = $3
        ORDER BY timestamp ASC
        """
        
        rows = await self.db.fetch(query, trail_id, universe_id, event_id)
        
        result = []
        for row in rows:
            decision_data = dict(row)
            # Parse JSON fields
            if decision_data.get('context'):
                decision_data['context'] = json.loads(decision_data['context'])
            result.append(decision_data)
        
        return result
    
    async def add_feedback(self, decision_id: str, feedback: str) -> bool:
        """
        Add feedback to a decision (useful for training data)
        
        Args:
            decision_id: Decision ID
            feedback: Feedback text
            
        Returns:
            Success flag
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping feedback for decision {decision_id}")
            return True
        
        query = """
        UPDATE agent_decisions
        SET feedback = $2
        WHERE decision_id = $1
        RETURNING decision_id
        """
        
        result = await self.db.fetchval(query, decision_id, feedback)
        success = result is not None
        
        if success:
            logger.debug(f"Added feedback to decision {decision_id}")
        else:
            logger.warning(f"Decision {decision_id} not found")
        
        return success
    
    async def add_rating(self, decision_id: str, rating: float) -> bool:
        """
        Add rating to a decision
        
        Args:
            decision_id: Decision ID
            rating: Rating value (float)
            
        Returns:
            Success flag
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping rating for decision {decision_id}")
            return True
        
        query = """
        UPDATE agent_decisions
        SET rating = $2
        WHERE decision_id = $1
        RETURNING decision_id
        """
        
        result = await self.db.fetchval(query, decision_id, rating)
        success = result is not None
        
        if success:
            logger.debug(f"Added rating to decision {decision_id}")
        else:
            logger.warning(f"Decision {decision_id} not found")
        
        return success
    
    async def add_reason(self, decision_id: str, reason: str) -> bool:
        """
        Add reason to a decision
        
        Args:
            decision_id: Decision ID
            reason: Reason text
            
        Returns:
            Success flag
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping reason for decision {decision_id}")
            return True
        
        query = """
        UPDATE agent_decisions
        SET reason = $2
        WHERE decision_id = $1
        RETURNING decision_id
        """
        
        result = await self.db.fetchval(query, decision_id, reason)
        success = result is not None
        
        if success:
            logger.debug(f"Added reason to decision {decision_id}")
        else:
            logger.warning(f"Decision {decision_id} not found")
        
        return success
    
    async def export_training_data(self, 
                                  trail_id: Optional[str] = None,
                                  agent_id: Optional[str] = None,
                                  format: str = 'json',
                                  include_context: bool = True,
                                  limit: int = 1000) -> str:
        """
        Export decision data in a format suitable for training
        
        Args:
            trail_id: Optional trail ID to filter by
            agent_id: Optional agent ID to filter by
            format: Output format ('jsonl', 'json', or 'csv')
            include_context: Whether to include context in the export
            limit: Maximum number of records to export
            
        Returns:
            String containing the exported data
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty export")
            return "" if format == 'csv' else "[]"
        
        conditions = []
        params = []
        
        if trail_id is not None:
            conditions.append(f"trail_id = ${len(params) + 1}")
            params.append(trail_id)
        
        if agent_id is not None:
            conditions.append(f"agent_id = ${len(params) + 1}")
            params.append(agent_id)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        query = f"""
        SELECT decision_id, trail_id, universe_id, agent_id, agent_type, step, timestamp,
               event_id, context, prompt, output, processing_time, action, feedback,
               rating, reason, created_at
        FROM agent_decisions
        {where_clause}
        ORDER BY timestamp ASC
        LIMIT {limit}
        """
        
        rows = await self.db.fetch(query, *params)
        
        if format == 'jsonl':
            result = []
            for row in rows:
                data = dict(row)
                
                # Convert datetime objects to strings
                for key, value in data.items():
                    if isinstance(value, datetime):
                        data[key] = value.isoformat()
                
                # Parse JSON context if needed
                if include_context and 'context' in data and data['context']:
                    data['context'] = json.loads(data['context'])
                elif not include_context:
                    data.pop('context', None)
                
                result.append(json.dumps(data))
            
            return '\n'.join(result)
        
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            fieldnames = ['decision_id', 'trail_id', 'universe_id', 'agent_id', 'agent_type',
                         'step', 'timestamp', 'prompt', 'output', 'action', 'feedback',
                         'rating', 'reason']
            
            if include_context:
                fieldnames.append('context')
            
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in rows:
                data = dict(row)
                
                # Convert datetime objects to strings
                for key, value in data.items():
                    if isinstance(value, datetime):
                        data[key] = value.isoformat()
                
                # Parse JSON context if needed
                if include_context and 'context' in data and data['context']:
                    data['context'] = json.dumps(json.loads(data['context']))
                
                # Filter to only the fields we want
                filtered_data = {k: v for k, v in data.items() if k in fieldnames}
                writer.writerow(filtered_data)
            
            return output.getvalue()
        
        elif format == 'json':
            result = []
            for row in rows:
                data = dict(row)
                data=convert_sql_data(data)
                result.append(data)
            return json.dumps(result)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def get_decision_statistics(self, 
                                     trail_id: Optional[str] = None,
                                     agent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about agent decisions
        
        Args:
            trail_id: Optional trail ID to filter by
            agent_id: Optional agent ID to filter by
            
        Returns:
            Dictionary with statistics
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty statistics")
            return {
                "total_count": 0,
                "avg_processing_time": 0,
                "agents": {},
                "steps": {}
            }
        
        conditions = []
        params = []
        
        if trail_id is not None:
            conditions.append(f"trail_id = ${len(params) + 1}")
            params.append(trail_id)
        
        if agent_id is not None:
            conditions.append(f"agent_id = ${len(params) + 1}")
            params.append(agent_id)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        # Get total count
        count_query = f"""
        SELECT COUNT(*) FROM agent_decisions
        {where_clause}
        """
        total_count = await self.db.fetchval(count_query, *params)
        
        # Get average processing time
        time_query = f"""
        SELECT AVG(processing_time) FROM agent_decisions
        {where_clause}
        """
        avg_processing_time = await self.db.fetchval(time_query, *params)
        
        # Get counts by agent
        agent_query = f"""
        SELECT agent_id, COUNT(*) as count
        FROM agent_decisions
        {where_clause}
        GROUP BY agent_id
        ORDER BY count DESC
        """
        agent_rows = await self.db.fetch(agent_query, *params)
        agents = {row['agent_id']: row['count'] for row in agent_rows}
        
        # Get counts by step
        step_query = f"""
        SELECT step, COUNT(*) as count
        FROM agent_decisions
        {where_clause}
        GROUP BY step
        ORDER BY step ASC
        """
        step_rows = await self.db.fetch(step_query, *params)
        steps = {row['step']: row['count'] for row in step_rows}
        
        return {
            "total_count": total_count,
            "avg_processing_time": avg_processing_time,
            "agents": agents,
            "steps": steps
        }