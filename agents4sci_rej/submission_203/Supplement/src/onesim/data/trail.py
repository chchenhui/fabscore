"""
Trail management for OneSim
"""

import uuid
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
from loguru import logger

from onesim.data.database import DatabaseManager

class TrailStatus(str, Enum):
    """Trail status enum"""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"

class TrailManager:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """Initialize trail manager"""
        self.db = db_manager or DatabaseManager.get_instance()
    
    async def create_trail(self, 
                          scenario_id: str, 
                          name: str, 
                          description: Optional[str] = None,
                          config: Optional[Dict[str, Any]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new trail
        
        Args:
            scenario_id: ID of the scenario this trail belongs to
            name: Trail name
            description: Trail description
            config: Configuration for this trail run
            metadata: Additional metadata
            
        Returns:
            trail_id: UUID of the created trail
        """
        trail_id = str(uuid.uuid4())
        
        # If database is disabled, just return the ID
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning generated trail ID: {trail_id}")
            return trail_id
        
        # Check if scenario exists
        scenario_query = "SELECT scenario_id FROM scenarios WHERE scenario_id = $1"
        scenario = await self.db.fetchval(scenario_query, scenario_id)
        
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} does not exist")
        
        # Insert trail record
        query = """
        INSERT INTO trails (
            trail_id, scenario_id, name, description, 
            status, config, metadata
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING trail_id
        """
        
        try:
            await self.db.execute(
                query, 
                trail_id, 
                scenario_id, 
                name, 
                description,
                TrailStatus.CREATED.value,
                json.dumps(config) if config else None,
                json.dumps(metadata) if metadata else None
            )
            logger.info(f"Created trail '{name}' with ID {trail_id}")
            return trail_id
        except Exception as e:
            logger.error(f"Failed to create trail: {e}")
            raise
    
    async def get_trail(self, trail_id: str) -> Dict[str, Any]:
        """Get trail by ID"""
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for trail {trail_id}")
            return None
        
        query = "SELECT * FROM trails WHERE trail_id = $1"
        row = await self.db.fetchrow(query, trail_id)
        
        if not row:
            return None
        
        return dict(row)
    
    async def list_trails(self, 
                         scenario_id: Optional[str] = None, 
                         status: Optional[Union[TrailStatus, str]] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        List trails, optionally filtered by scenario and status
        
        Args:
            scenario_id: Optional scenario ID to filter by
            status: Optional status to filter by
            limit: Maximum number of trails to return
            
        Returns:
            List of trail records
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
        
        conditions = []
        params = []
        
        if scenario_id:
            conditions.append(f"scenario_id = ${len(params) + 1}")
            params.append(scenario_id)
        
        if status:
            if isinstance(status, TrailStatus):
                status_value = status.value
            else:
                status_value = status
            conditions.append(f"status = ${len(params) + 1}")
            params.append(status_value)
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        query = f"""
        SELECT * FROM trails
        {where_clause}
        ORDER BY created_at DESC
        LIMIT {limit}
        """
        
        rows = await self.db.fetch(query, *params)
        return [dict(row) for row in rows]
    
    async def update_trail_status(self, trail_id: str, status: Union[TrailStatus, str]) -> bool:
        """
        Update trail status
        
        Args:
            trail_id: Trail ID
            status: New status
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping update for trail {trail_id}")
            return True
        
        if isinstance(status, TrailStatus):
            status_value = status.value
        else:
            status_value = status
        
        # Update status and timestamps based on the new status
        updates = ["status = $2", "updated_at = CURRENT_TIMESTAMP"]
        params = [trail_id, status_value]
        
        # Set start_time if transitioning to RUNNING
        if status_value == TrailStatus.RUNNING.value:
            updates.append("start_time = CURRENT_TIMESTAMP")
        
        # Set end_time if transitioning to a terminal state
        if status_value in [TrailStatus.COMPLETED.value, TrailStatus.FAILED.value, TrailStatus.ABORTED.value]:
            updates.append("end_time = CURRENT_TIMESTAMP")
        
        query = f"""
        UPDATE trails
        SET {', '.join(updates)}
        WHERE trail_id = $1
        RETURNING trail_id
        """
        
        result = await self.db.fetchval(query, *params)
        success = result is not None
        
        if success:
            logger.info(f"Updated trail {trail_id} status to {status_value}")
        else:
            logger.warning(f"Failed to update trail {trail_id} status")
        
        return success
    
    async def increment_step(self, trail_id: str) -> int:
        """
        Increment the step counter for a trail
        
        Args:
            trail_id: Trail ID
            
        Returns:
            New step count
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning 0 for trail {trail_id}")
            return 0
        
        query = """
        UPDATE trails
        SET step_count = step_count + 1, updated_at = CURRENT_TIMESTAMP
        WHERE trail_id = $1
        RETURNING step_count
        """
        
        return await self.db.fetchval(query, trail_id)
    
    async def update_trail_metadata(self, 
                                  trail_id: str, 
                                  metadata: Dict[str, Any],
                                  merge: bool = True) -> bool:
        """
        Update trail metadata
        
        Args:
            trail_id: Trail ID
            metadata: New metadata
            merge: If True, merge with existing metadata; if False, replace
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping update for trail {trail_id}")
            return True
        
        if merge:
            # Get existing metadata
            query = "SELECT metadata FROM trails WHERE trail_id = $1"
            existing_metadata = await self.db.fetchval(query, trail_id)
            
            if existing_metadata:
                # Parse and merge
                try:
                    existing_dict = json.loads(existing_metadata)
                    existing_dict.update(metadata)
                    metadata = existing_dict
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse existing metadata for trail {trail_id}")
        
        # Update metadata
        query = """
        UPDATE trails
        SET metadata = $2, updated_at = CURRENT_TIMESTAMP
        WHERE trail_id = $1
        RETURNING trail_id
        """
        
        result = await self.db.fetchval(query, trail_id, json.dumps(metadata))
        return result is not None
    
    async def delete_trail(self, trail_id: str) -> bool:
        """
        Delete a trail and all associated data
        
        Args:
            trail_id: Trail ID
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping delete for trail {trail_id}")
            return True
        
        # Use a transaction to ensure all related data is deleted
        pool = await self.db.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                # Delete related data first (due to foreign key constraints)
                await conn.execute("DELETE FROM agent_decisions WHERE trail_id = $1", trail_id)
                await conn.execute("DELETE FROM events WHERE trail_id = $1", trail_id)
                await conn.execute("DELETE FROM agent_states WHERE trail_id = $1", trail_id)
                await conn.execute("DELETE FROM agents WHERE trail_id = $1", trail_id)
                await conn.execute("DELETE FROM environment_states WHERE trail_id = $1", trail_id)
                
                # Finally delete the trail
                result = await conn.fetchval(
                    "DELETE FROM trails WHERE trail_id = $1 RETURNING trail_id", 
                    trail_id
                )
        
        success = result is not None
        if success:
            logger.info(f"Deleted trail {trail_id} and all associated data")
        else:
            logger.warning(f"Trail {trail_id} not found")
        
        return success 