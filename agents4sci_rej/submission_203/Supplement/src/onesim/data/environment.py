import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from loguru import logger

from .database import DatabaseManager


class EnvironmentStateManager:
    """
    Manager for environment state data. Handles operations related to storing and retrieving environment states.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize environment state manager with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager or DatabaseManager.get_instance()
    
    async def save_state(self, 
                        trail_id: str, 
                        step: int, 
                        state: Dict[str, Any],
                        universe_id: str = 'main',
                        timestamp: Optional[datetime] = None) -> str:
        """
        Save environment state
        
        Args:
            trail_id: Trail ID
            step: Simulation step
            state: Environment state data
            universe_id: Universe ID (for parallel universes)
            timestamp: Timestamp (defaults to current time)
            
        Returns:
            state_id: UUID of the saved state
        """
        state_id = str(uuid.uuid4())
        
        # If database is disabled, just return the ID
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning generated state ID: {state_id}")
            return state_id
            
        if timestamp is None:
            timestamp = datetime.now()
        
        query = """
        INSERT INTO environment_states (
            state_id, trail_id, universe_id, step, timestamp, state
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (trail_id, universe_id, step) 
        DO UPDATE SET 
            state = $6,
            timestamp = $5,
            updated_at = CURRENT_TIMESTAMP
        RETURNING state_id
        """
        
        try:
            result = await self.db.execute(
                query, 
                state_id, 
                trail_id, 
                universe_id, 
                step, 
                timestamp,
                json.dumps(state)
            )
            logger.debug(f"Saved environment state for trail {trail_id}, step {step}, universe {universe_id}")
            return state_id
        except Exception as e:
            logger.error(f"Failed to save environment state: {e}")
            raise
    
    async def get_state(self, 
                       trail_id: str, 
                       step: int,
                       universe_id: str = 'main') -> Dict[str, Any]:
        """
        Get environment state
        
        Args:
            trail_id: Trail ID
            step: Simulation step
            universe_id: Universe ID
            
        Returns:
            Environment state data or None if not found
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for state at step {step}")
            return None
            
        query = """
        SELECT * FROM environment_states
        WHERE trail_id = $1 AND universe_id = $2 AND step = $3
        """
        
        row = await self.db.fetchrow(query, trail_id, universe_id, step)
        if not row:
            return None
        
        result = dict(row)
        # Parse JSON state
        if 'state' in result and result['state']:
            result['state'] = json.loads(result['state'])
        
        return result
    
    async def get_state_by_id(self, state_id: str) -> Dict[str, Any]:
        """Get environment state by ID"""
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for state {state_id}")
            return None
            
        query = "SELECT * FROM environment_states WHERE state_id = $1"
        row = await self.db.fetchrow(query, state_id)
        
        if not row:
            return None
        
        result = dict(row)
        # Parse JSON state
        if 'state' in result and result['state']:
            result['state'] = json.loads(result['state'])
        
        return result
    
    async def get_latest_state(self, 
                              trail_id: str,
                              universe_id: str = 'main') -> Dict[str, Any]:
        """
        Get the latest environment state for a trail
        
        Args:
            trail_id: Trail ID
            universe_id: Universe ID
            
        Returns:
            Latest environment state or None if not found
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for latest state")
            return None
            
        query = """
        SELECT * FROM environment_states
        WHERE trail_id = $1 AND universe_id = $2
        ORDER BY step DESC
        LIMIT 1
        """
        
        row = await self.db.fetchrow(query, trail_id, universe_id)
        if not row:
            return None
        
        result = dict(row)
        # Parse JSON state
        if 'state' in result and result['state']:
            result['state'] = json.loads(result['state'])
        
        return result
    
    async def list_states(self, 
                         trail_id: str,
                         universe_id: str = 'main',
                         start_step: Optional[int] = None,
                         end_step: Optional[int] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        List environment states for a trail
        
        Args:
            trail_id: Trail ID
            universe_id: Universe ID
            start_step: Optional starting step (inclusive)
            end_step: Optional ending step (inclusive)
            limit: Maximum number of states to return
            
        Returns:
            List of environment states
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
            
        conditions = ["trail_id = $1", "universe_id = $2"]
        params = [trail_id, universe_id]
        
        if start_step is not None:
            conditions.append(f"step >= ${len(params) + 1}")
            params.append(start_step)
        
        if end_step is not None:
            conditions.append(f"step <= ${len(params) + 1}")
            params.append(end_step)
        
        query = f"""
        SELECT * FROM environment_states
        WHERE {' AND '.join(conditions)}
        ORDER BY step ASC
        LIMIT {limit}
        """
        
        rows = await self.db.fetch(query, *params)
        result = []
        
        for row in rows:
            state_data = dict(row)
            # Parse JSON state
            if 'state' in state_data and state_data['state']:
                state_data['state'] = json.loads(state_data['state'])
            result.append(state_data)
        
        return result
    
    async def compare_states(self, 
                            state_id1: str, 
                            state_id2: str) -> Dict[str, Any]:
        """
        Compare two environment states
        
        Args:
            state_id1: First state ID
            state_id2: Second state ID
            
        Returns:
            Dictionary with comparison results
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty comparison")
            return {"differences": [], "similarity": 0}
            
        # Get both states
        state1 = await self.get_state_by_id(state_id1)
        state2 = await self.get_state_by_id(state_id2)
        
        if not state1 or not state2:
            raise ValueError("One or both states not found")
        
        # Extract the actual state data
        state1_data = state1['state']
        state2_data = state2['state']
        
        # Perform comparison
        differences = []
        
        # Flatten nested dictionaries for comparison
        def flatten_dict(d, parent_key=''):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat1 = flatten_dict(state1_data)
        flat2 = flatten_dict(state2_data)
        
        # Find keys in both dictionaries
        all_keys = set(flat1.keys()) | set(flat2.keys())
        
        # Compare values
        for key in all_keys:
            if key not in flat1:
                differences.append({
                    "key": key,
                    "type": "added",
                    "value": flat2[key]
                })
            elif key not in flat2:
                differences.append({
                    "key": key,
                    "type": "removed",
                    "value": flat1[key]
                })
            elif flat1[key] != flat2[key]:
                differences.append({
                    "key": key,
                    "type": "changed",
                    "old_value": flat1[key],
                    "new_value": flat2[key]
                })
        
        # Calculate similarity (percentage of unchanged keys)
        unchanged_count = len(all_keys) - len(differences)
        similarity = unchanged_count / len(all_keys) if all_keys else 1.0
        
        return {
            "state1": {
                "id": state_id1,
                "trail_id": state1["trail_id"],
                "step": state1["step"],
                "universe_id": state1["universe_id"],
                "timestamp": state1["timestamp"].isoformat() if isinstance(state1["timestamp"], datetime) else state1["timestamp"]
            },
            "state2": {
                "id": state_id2,
                "trail_id": state2["trail_id"],
                "step": state2["step"],
                "universe_id": state2["universe_id"],
                "timestamp": state2["timestamp"].isoformat() if isinstance(state2["timestamp"], datetime) else state2["timestamp"]
            },
            "differences": differences,
            "similarity": similarity,
            "unchanged_count": unchanged_count,
            "total_keys": len(all_keys)
        }
    
    async def delete_states(self, 
                           trail_id: str,
                           universe_id: Optional[str] = None,
                           start_step: Optional[int] = None,
                           end_step: Optional[int] = None) -> int:
        """
        Delete environment states
        
        Args:
            trail_id: Trail ID
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
        DELETE FROM environment_states
        WHERE {' AND '.join(conditions)}
        RETURNING state_id
        """
        
        result = await self.db.fetch(query, *params)
        count = len(result)
        
        logger.info(f"Deleted {count} environment states for trail {trail_id}")
        return count 