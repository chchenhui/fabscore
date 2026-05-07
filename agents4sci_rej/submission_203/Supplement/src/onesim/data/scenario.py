"""
Scenario management for OneSim
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from loguru import logger
import asyncpg

from onesim.data.database import DatabaseManager


class ScenarioManager:
    """
    Manager for scenario data. Handles operations related to storing and retrieving scenarios.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize scenario manager with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager or DatabaseManager.get_instance()
    
    async def create_scenario(
        self,
        name: str,
        folder_path: str,
        description: str = "",
        tags: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new scenario in the database.
        
        Args:
            name: Name of the scenario
            folder_path: Path to the scenario folder
            description: Description of the scenario
            tags: Tags for the scenario
            metadata: Additional metadata
            
        Returns:
            scenario_id: ID of the created scenario
        """
        # If database is disabled, just return the ID
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning scenario ID: {name}")
            return name
            
        # Normalize the folder path
        folder_path = os.path.abspath(folder_path) if folder_path else ""
        
        # Make sure the folder exists
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                logger.info(f"Created scenario folder: {folder_path}")
            except Exception as e:
                logger.error(f"Failed to create scenario folder: {e}")
                raise
        
        # Generate a scenario ID based on the name and timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        scenario_id = f"{name.lower().replace(' ', '_')}_{timestamp}"
        
        # Check if scenario already exists
        existing = await self.get_scenario(scenario_id)
        if existing:
            raise ValueError(f"Scenario with ID {scenario_id} already exists")
        
        # Insert scenario record
        query = """
        INSERT INTO scenarios (scenario_id, name, description, folder_path, tags, metadata)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING scenario_id
        """
        
        try:
            await self.db.execute(
                query, 
                scenario_id, 
                name, 
                description, 
                folder_path,
                json.dumps(tags) if tags else None,
                json.dumps(metadata) if metadata else None
            )
            logger.info(f"Created scenario '{name}' with ID {scenario_id}")
            return scenario_id
        except Exception as e:
            logger.error(f"Failed to create scenario: {e}")
            raise
    
    async def get_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """
        Get scenario data by ID.
        
        Args:
            scenario_id: ID of the scenario
            
        Returns:
            Scenario data
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for scenario {scenario_id}")
            return None
        
        query = "SELECT * FROM scenarios WHERE scenario_id = $1"
        row = await self.db.fetchrow(query, scenario_id)
        
        if not row:
            return None
        
        return dict(row)
    
    async def get_scenario_by_name(self, name: str, exact_match: bool = True) -> List[Dict[str, Any]]:
        """
        Get scenario data by name.
        
        Args:
            name: Name of the scenario to search for
            exact_match: If True, searches for exact name match; if False, uses partial match
            
        Returns:
            List of matching scenario data
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning empty list for scenario name {name}")
            return []
        
        if exact_match:
            query = "SELECT * FROM scenarios WHERE name = $1 ORDER BY updated_at DESC"
            rows = await self.db.fetch(query, name)
        else:
            # Use ILIKE for case-insensitive partial match
            query = "SELECT * FROM scenarios WHERE name ILIKE $1 ORDER BY updated_at DESC"
            rows = await self.db.fetch(query, f"%{name}%")
        
        if not rows:
            logger.debug(f"No scenarios found with name: {name}")
            return []
        
        return [dict(row) for row in rows]
    
    async def list_scenarios(self, tag: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all scenarios, optionally filtered by tag.
        
        Args:
            tag: Optional tag to filter by
            limit: Maximum number of scenarios to return
            
        Returns:
            List of scenario data
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
        
        if tag:
            # Filter by tag
            tag_conditions = []
            for tag in tag.split():
                tag_conditions.append(f"tags ? '{tag}'")
            
            query = f"""
            SELECT * FROM scenarios 
            WHERE {' AND '.join(tag_conditions)}
            ORDER BY updated_at DESC
            LIMIT $1
            """
            rows = await self.db.fetch(query, limit)
        else:
            # Get all scenarios
            query = "SELECT * FROM scenarios ORDER BY updated_at DESC LIMIT $1"
            rows = await self.db.fetch(query, limit)
        
        return [dict(row) for row in rows]
    
    async def update_scenario(
        self,
        scenario_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        folder_path: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update scenario data.
        
        Args:
            scenario_id: ID of the scenario
            name: New name for the scenario
            description: New description
            folder_path: New folder path
            tags: New tags
            metadata: New metadata
            
        Returns:
            Success flag
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping update for scenario {scenario_id}")
            return True
        
        # Get current scenario data
        current = await self.get_scenario(scenario_id)
        if not current:
            logger.warning(f"Cannot update non-existent scenario: {scenario_id}")
            return False
        
        # Update with new values if provided
        if name is not None:
            current['name'] = name
        if description is not None:
            current['description'] = description
        if folder_path is not None:
            current['folder_path'] = os.path.abspath(folder_path)
        if tags is not None:
            current['tags'] = tags
        if metadata is not None:
            current['metadata'] = metadata
        
        # Build update query dynamically based on provided fields
        update_parts = []
        params = [scenario_id]
        param_idx = 2  # Start from $2
        
        if name is not None:
            update_parts.append(f"name = ${param_idx}")
            params.append(name)
            param_idx += 1
        
        if description is not None:
            update_parts.append(f"description = ${param_idx}")
            params.append(description)
            param_idx += 1
        
        if folder_path is not None:
            update_parts.append(f"folder_path = ${param_idx}")
            params.append(folder_path)
            param_idx += 1
        
        if tags is not None:
            update_parts.append(f"tags = ${param_idx}")
            params.append(json.dumps(tags))
            param_idx += 1
        
        if metadata is not None:
            update_parts.append(f"metadata = ${param_idx}")
            params.append(json.dumps(metadata))
            param_idx += 1
        
        # Always update the updated_at timestamp
        update_parts.append("updated_at = CURRENT_TIMESTAMP")
        
        if not update_parts:
            logger.warning("No fields to update for scenario")
            return False
        
        query = f"""
        UPDATE scenarios 
        SET {', '.join(update_parts)}
        WHERE scenario_id = $1
        RETURNING scenario_id
        """
        
        result = await self.db.fetchval(query, *params)
        success = result is not None
        
        if success:
            logger.info(f"Updated scenario {scenario_id}")
        else:
            logger.warning(f"Scenario {scenario_id} not found")
        
        return success
    
    async def delete_scenario(self, scenario_id: str, delete_folder: bool = False) -> bool:
        """
        Delete scenario from database. Optionally delete the folder.
        
        Args:
            scenario_id: ID of the scenario
            delete_folder: If True, also delete the scenario folder
            
        Returns:
            Success flag
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping delete for scenario {scenario_id}")
            return True
        
        # Get scenario folder path if deleting folder
        folder_path = None
        if delete_folder:
            scenario = await self.get_scenario(scenario_id)
            if scenario:
                folder_path = scenario['folder_path']
        
        # First check if there are any trails for this scenario
        check_query = "SELECT COUNT(*) FROM trails WHERE scenario_id = $1"
        count = await self.db.fetchval(check_query, scenario_id)
        
        if count > 0:
            logger.warning(f"Cannot delete scenario {scenario_id} with {count} trails")
            return False
        
        # Use a transaction to ensure all related data is deleted
        pool = await self.db.get_pool()
        if not pool:
            return False
            
        async with pool.acquire() as conn:
            async with conn.transaction():
                if delete_folder and folder_path and os.path.exists(folder_path):
                    try:
                        import shutil
                        shutil.rmtree(folder_path)
                        logger.info(f"Deleted scenario folder: {folder_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete scenario folder: {e}")
                
                # Delete the scenario
                result = await conn.fetchval(
                    "DELETE FROM scenarios WHERE scenario_id = $1 RETURNING scenario_id", 
                    scenario_id
                )
        
        success = result is not None
        if success:
            logger.info(f"Deleted scenario {scenario_id}")
        else:
            logger.warning(f"Scenario {scenario_id} not found")
        
        return success 