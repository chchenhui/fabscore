import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from loguru import logger

from .database import DatabaseManager


class EventManager:
    """
    Manager for event data. Handles operations related to storing and retrieving events.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize event manager with database manager.
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager or DatabaseManager.get_instance()
    
    async def create_event(self, 
                          trail_id: str,
                          event_id: str,                 
                          step: int,                     
                          event_type: str,               
                          source_id: str,                
                          payload: Dict[str, Any],           
                          timestamp: float,              
                          target_id: Optional[str] = None,
                          source_type: Optional[str] = None, 
                          target_type: Optional[str] = None, 
                          priority: int = 0,
                          universe_id: str = 'main') -> str:
        """
        Create a new event. This method is designed to be compatible with unpacking 
        the dictionary from Event.to_dict() along with trail_id and potentially step.
        
        Args:
            trail_id: Trail ID (passed explicitly)
            event_id: Event ID (from event_data)
            step: Simulation step (from event_data or explicit)
            event_type: Type of event (from event_data["event_type"])
            source_id: ID of source entity (from event_data["source_id"])
            payload: Event payload/data (from event_data["payload"])
            timestamp: Event timestamp as a float (from event_data["timestamp"])
            target_id: ID of target entity (from event_data["target_id"], optional)
            source_type: Type of source entity. If None, attempts to get from data['source_type'].
            target_type: Type of target entity. If None, attempts to get from data['target_type'].
            priority: Event priority
            universe_id: Universe ID
            
        Returns:
            event_id: The event_id of the created (or processed) event
        """
        # Use the provided event_id directly
        
        # If database is disabled, just return the ID
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning provided event ID: {event_id}")
            return event_id
            
        # Convert float timestamp to datetime object for DB
        dt_timestamp = datetime.fromtimestamp(timestamp)
        
        # Determine final source_type and target_type
        final_source_type = source_type if source_type is not None else payload.get('source_type')
        final_target_type = target_type if target_type is not None else payload.get('target_type')

        query = """
        INSERT INTO events (
            event_id, trail_id, universe_id, step, timestamp,
            event_type, source_type, source_id, target_type, target_id,
            priority, payload, processed
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        RETURNING event_id
        """
        
        try:
            await self.db.execute(
                query, 
                event_id,          # Use provided event_id
                trail_id, 
                universe_id, 
                step, 
                dt_timestamp,      # Use converted datetime timestamp
                event_type,        # Directly from parameter
                final_source_type, # Use determined source_type
                source_id,         # Directly from parameter
                final_target_type, # Use determined target_type
                target_id,         # Directly from parameter
                priority,
                json.dumps(payload),  # Serialize the 'data' dictionary as payload
                False              # not processed yet
            )
            logger.debug(f"Created event {event_type} from {source_id} at step {step} with ID {event_id}")
            return event_id
        except Exception as e:
            logger.error(f"Failed to create event: {e}")
            raise
    
    async def get_event(self, event_id: str) -> Dict[str, Any]:
        """Get event by ID"""
        if not self.db.enabled:
            logger.debug(f"Database disabled, returning None for event {event_id}")
            return None
            
        query = "SELECT * FROM events WHERE event_id = $1"
        row = await self.db.fetchrow(query, event_id)
        
        if not row:
            return None
        
        result = dict(row)
        # Parse JSON payload
        if 'payload' in result and result['payload']:
            result['payload'] = json.loads(result['payload'])
        if 'processing_result' in result and result['processing_result']:
            result['processing_result'] = json.loads(result['processing_result'])
        
        return result
    
    async def list_events(self, 
                         trail_id: str,
                         universe_id: str = 'main',
                         step: Optional[int] = None,
                         event_type: Optional[str] = None,
                         source_id: Optional[str] = None,
                         target_id: Optional[str] = None,
                         processed: Optional[bool] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        List events
        
        Args:
            trail_id: Trail ID
            universe_id: Universe ID
            step: Optional step to filter by
            event_type: Optional event type to filter by
            source_id: Optional source ID to filter by
            target_id: Optional target ID to filter by
            processed: Optional processed status to filter by
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
            
        conditions = ["trail_id = $1", "universe_id = $2"]
        params = [trail_id, universe_id]
        
        if step is not None:
            conditions.append(f"step = ${len(params) + 1}")
            params.append(step)
        
        if event_type is not None:
            conditions.append(f"event_type = ${len(params) + 1}")
            params.append(event_type)
        
        if source_id is not None:
            conditions.append(f"source_id = ${len(params) + 1}")
            params.append(source_id)
        
        if target_id is not None:
            conditions.append(f"target_id = ${len(params) + 1}")
            params.append(target_id)
        
        if processed is not None:
            conditions.append(f"processed = ${len(params) + 1}")
            params.append(processed)
        
        query = f"""
        SELECT * FROM events
        WHERE {' AND '.join(conditions)}
        ORDER BY priority DESC, timestamp ASC
        LIMIT {limit}
        """
        
        rows = await self.db.fetch(query, *params)
        result = []
        
        for row in rows:
            event_data = dict(row)
            # Parse JSON fields
            if 'payload' in event_data and event_data['payload']:
                event_data['payload'] = json.loads(event_data['payload'])
            if 'processing_result' in event_data and event_data['processing_result']:
                event_data['processing_result'] = json.loads(event_data['processing_result'])
            result.append(event_data)
        
        return result
    
    async def get_unprocessed_events(self, 
                                    trail_id: str,
                                    universe_id: str = 'main',
                                    limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get unprocessed events ordered by priority
        
        Args:
            trail_id: Trail ID
            universe_id: Universe ID
            limit: Maximum number of events to return
            
        Returns:
            List of unprocessed events
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
            
        query = """
        SELECT * FROM events
        WHERE trail_id = $1 AND universe_id = $2 AND processed = FALSE
        ORDER BY priority DESC, timestamp ASC
        LIMIT $3
        """
        
        rows = await self.db.fetch(query, trail_id, universe_id, limit)
        result = []
        
        for row in rows:
            event_data = dict(row)
            # Parse JSON fields
            if 'payload' in event_data and event_data['payload']:
                event_data['payload'] = json.loads(event_data['payload'])
            if 'processing_result' in event_data and event_data['processing_result']:
                event_data['processing_result'] = json.loads(event_data['processing_result'])
            result.append(event_data)
        
        return result
    
    async def mark_event_processed(self, 
                                  event_id: str, 
                                  processing_result: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark an event as processed
        
        Args:
            event_id: Event ID
            processing_result: Optional result of processing
            
        Returns:
            True if successful, False otherwise
        """
        if not self.db.enabled:
            logger.debug(f"Database disabled, skipping update for event {event_id}")
            return True
            
        query = """
        UPDATE events
        SET processed = TRUE, processing_result = $2
        WHERE event_id = $1
        RETURNING event_id
        """
        
        result = await self.db.fetchval(
            query, 
            event_id, 
            json.dumps(processing_result) if processing_result else None
        )
        
        success = result is not None
        if success:
            logger.debug(f"Marked event {event_id} as processed")
        else:
            logger.warning(f"Event {event_id} not found")
        
        return success
    
    async def get_events_for_agent(self, 
                                  trail_id: str,
                                  agent_id: str,
                                  universe_id: str = 'main',
                                  processed: Optional[bool] = None,
                                  limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get events where the agent is either the source or target
        
        Args:
            trail_id: Trail ID
            agent_id: Agent ID
            universe_id: Universe ID
            processed: Optional processed status to filter by
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
            
        conditions = [
            "trail_id = $1", 
            "universe_id = $2", 
            "(source_id = $3 OR target_id = $3)"
        ]
        params = [trail_id, universe_id, agent_id]
        
        if processed is not None:
            conditions.append(f"processed = ${len(params) + 1}")
            params.append(processed)
        
        query = f"""
        SELECT * FROM events
        WHERE {' AND '.join(conditions)}
        ORDER BY step DESC, priority DESC, timestamp ASC
        LIMIT {limit}
        """
        
        rows = await self.db.fetch(query, *params)
        result = []
        
        for row in rows:
            event_data = dict(row)
            # Parse JSON fields
            if 'payload' in event_data and event_data['payload']:
                event_data['payload'] = json.loads(event_data['payload'])
            if 'processing_result' in event_data and event_data['processing_result']:
                event_data['processing_result'] = json.loads(event_data['processing_result'])
            result.append(event_data)
        
        return result
    
    async def get_events_by_type(self, 
                                trail_id: str,
                                event_type: str,
                                universe_id: str = 'main',
                                start_step: Optional[int] = None,
                                end_step: Optional[int] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get events of a specific type
        
        Args:
            trail_id: Trail ID
            event_type: Event type
            universe_id: Universe ID
            start_step: Optional starting step (inclusive)
            end_step: Optional ending step (inclusive)
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        if not self.db.enabled:
            logger.debug("Database disabled, returning empty list")
            return []
            
        conditions = ["trail_id = $1", "universe_id = $2", "event_type = $3"]
        params = [trail_id, universe_id, event_type]
        
        if start_step is not None:
            conditions.append(f"step >= ${len(params) + 1}")
            params.append(start_step)
        
        if end_step is not None:
            conditions.append(f"step <= ${len(params) + 1}")
            params.append(end_step)
        
        query = f"""
        SELECT * FROM events
        WHERE {' AND '.join(conditions)}
        ORDER BY step ASC, priority DESC, timestamp ASC
        LIMIT {limit}
        """
        
        rows = await self.db.fetch(query, *params)
        result = []
        
        for row in rows:
            event_data = dict(row)
            # Parse JSON fields
            if 'payload' in event_data and event_data['payload']:
                event_data['payload'] = json.loads(event_data['payload'])
            if 'processing_result' in event_data and event_data['processing_result']:
                event_data['processing_result'] = json.loads(event_data['processing_result'])
            result.append(event_data)
        
        return result
    
    async def delete_events(self, 
                           trail_id: str,
                           universe_id: Optional[str] = None,
                           start_step: Optional[int] = None,
                           end_step: Optional[int] = None,
                           event_type: Optional[str] = None) -> int:
        """
        Delete events
        
        Args:
            trail_id: Trail ID
            universe_id: Optional universe ID (if None, delete from all universes)
            start_step: Optional starting step (inclusive)
            end_step: Optional ending step (inclusive)
            event_type: Optional event type to filter by
            
        Returns:
            Number of deleted events
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
        
        if event_type is not None:
            conditions.append(f"event_type = ${len(params) + 1}")
            params.append(event_type)
        
        query = f"""
        DELETE FROM events
        WHERE {' AND '.join(conditions)}
        RETURNING event_id
        """
        
        result = await self.db.fetch(query, *params)
        count = len(result)
        
        logger.info(f"Deleted {count} events for trail {trail_id}")
        return count 