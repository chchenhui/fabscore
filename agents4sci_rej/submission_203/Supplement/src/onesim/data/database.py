"""
Database connection and schema management for OneSim
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional
from loguru import logger

class DatabaseManager:
    _instance = None
    _initialized = False
    _pool = None
    _lock = asyncio.Lock()
    _enabled = False
    
    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None):
        """Get singleton instance of DatabaseManager"""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize database manager with configuration"""
        if DatabaseManager._instance is not None:
            raise RuntimeError("DatabaseManager is a singleton. Use get_instance() instead.")
        
        self.config = config or {
            "host": "localhost",
            "port": 5432,
            "dbname": "onesim",
            "user": "postgres",
            "password": ""
        }
        
        # Check if data module is enabled
        self._enabled = config.get("enabled", False) if config else False
        
        # Store the instance
        DatabaseManager._instance = self
    
    @property
    def enabled(self):
        """Check if database is enabled"""
        return self._enabled
    
    async def get_pool(self):
        """Get or create connection pool"""
        if not self._enabled:
            logger.debug("Database operations are disabled")
            return None
            
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    try:
                        # Import asyncpg only if database is enabled
                        import asyncpg
                        self._pool = await asyncpg.create_pool(
                            host=self.config["host"],
                            port=self.config["port"],
                            database=self.config["dbname"],
                            user=self.config["user"],
                            password=self.config["password"],
                            min_size=5,
                            max_size=20
                        )
                        logger.info(f"Connected to database {self.config['dbname']} at {self.config['host']}:{self.config['port']}")
                    except ImportError:
                        logger.warning("asyncpg package not installed. Database operations will be disabled.")
                        self._enabled = False
                        return None
                    except Exception as e:
                        print("password", self.config["password"])
                        logger.error(f"Failed to connect to database: {e}")
                        self._enabled = False
                        return None
        return self._pool
    
    async def execute(self, query: str, *args, timeout: float = 10):
        """Execute a query"""
        if not self._enabled:
            logger.debug(f"Database disabled, skipping query: {query}")
            return None
            
        pool = await self.get_pool()
        if not pool:
            return None
            
        async with pool.acquire() as conn:
            return await conn.execute(query, *args, timeout=timeout)
    
    async def fetch(self, query: str, *args, timeout: float = 10):
        """Fetch results from a query"""
        if not self._enabled:
            logger.debug(f"Database disabled, skipping query: {query}")
            return []
            
        pool = await self.get_pool()
        if not pool:
            return []
            
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args, timeout=timeout)
    
    async def fetchrow(self, query: str, *args, timeout: float = 10):
        """Fetch a single row from a query"""
        if not self._enabled:
            logger.debug(f"Database disabled, skipping query: {query}")
            return None
            
        pool = await self.get_pool()
        if not pool:
            return None
            
        async with pool.acquire() as conn:
            return await conn.fetchrow(query, *args, timeout=timeout)
    
    async def fetchval(self, query: str, *args, timeout: float = 10):
        """Fetch a single value from a query"""
        if not self._enabled:
            logger.debug(f"Database disabled, skipping query: {query}")
            return None
            
        pool = await self.get_pool()
        if not pool:
            return None
            
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args, timeout=timeout)
    
    async def transaction(self):
        """Get a connection and start a transaction"""
        if not self._enabled:
            logger.debug("Database disabled, skipping transaction")
            return None
            
        pool = await self.get_pool()
        if not pool:
            return None
            
        return await pool.acquire()
    
    def initialize_schema(self):
        """Initialize database schema"""
        if not self._enabled:
            logger.info("Database disabled, skipping schema initialization")
            return
            
        asyncio.create_task(self._initialize_schema())
    
    async def _initialize_schema(self):
        """Create database schema if it doesn't exist"""
        if not self._enabled:
            return
            
        # Define schema SQL
        schema_sql = """
        -- Scenarios table
        CREATE TABLE IF NOT EXISTS scenarios (
            scenario_id VARCHAR(255) PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            folder_path VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            tags JSONB,
            metadata JSONB
        );

        -- Trails table
        CREATE TABLE IF NOT EXISTS trails (
            trail_id UUID PRIMARY KEY,
            scenario_id VARCHAR(255) REFERENCES scenarios(scenario_id),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            status VARCHAR(20) NOT NULL DEFAULT 'CREATED',
            step_count INTEGER DEFAULT 0,
            config JSONB,
            metadata JSONB
        );

        -- Environment states table
        CREATE TABLE IF NOT EXISTS environment_states (
            state_id UUID PRIMARY KEY,
            trail_id UUID REFERENCES trails(trail_id),
            universe_id VARCHAR(50) DEFAULT 'main',
            step INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            state JSONB NOT NULL,
            UNIQUE(trail_id, universe_id, step)
        );

        CREATE INDEX IF NOT EXISTS idx_env_states_trail_universe_step 
        ON environment_states(trail_id, universe_id, step);

        -- Agents table
        CREATE TABLE IF NOT EXISTS agents (
            agent_id VARCHAR(255),
            trail_id UUID REFERENCES trails(trail_id),
            universe_id VARCHAR(50) DEFAULT 'main',
            agent_type VARCHAR(100) NOT NULL,
            name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            initial_profile JSONB NOT NULL,
            system_prompt TEXT,
            model_config_name VARCHAR(100),
            memory_config JSONB,
            planning_config JSONB,
            PRIMARY KEY (agent_id, trail_id, universe_id)
        );

        -- Agent states table
        CREATE TABLE IF NOT EXISTS agent_states (
            state_id UUID PRIMARY KEY,
            trail_id UUID REFERENCES trails(trail_id),
            universe_id VARCHAR(50) DEFAULT 'main',
            agent_id VARCHAR(255) NOT NULL,
            step INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            profile JSONB,
            memory JSONB,
            relationships JSONB,
            additional_state JSONB,
            UNIQUE(trail_id, universe_id, agent_id, step),
            FOREIGN KEY (agent_id, trail_id, universe_id) 
                REFERENCES agents(agent_id, trail_id, universe_id)
        );

        CREATE INDEX IF NOT EXISTS idx_agent_states_trail_universe_step 
        ON agent_states(trail_id, universe_id, step);

        -- Events table
        CREATE TABLE IF NOT EXISTS events (
            event_id UUID PRIMARY KEY,
            trail_id UUID REFERENCES trails(trail_id),
            universe_id VARCHAR(50) DEFAULT 'main',
            step INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            event_type VARCHAR(100) NOT NULL,
            source_type VARCHAR(20) NOT NULL,
            source_id VARCHAR(255) NOT NULL,
            target_type VARCHAR(20),
            target_id VARCHAR(255),
            priority INTEGER DEFAULT 0,
            payload JSONB NOT NULL,
            processed BOOLEAN DEFAULT FALSE,
            processing_result JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_events_trail_universe_step 
        ON events(trail_id, universe_id, step);
        
        CREATE INDEX IF NOT EXISTS idx_events_trail_source 
        ON events(trail_id, source_id);
        
        CREATE INDEX IF NOT EXISTS idx_events_trail_target 
        ON events(trail_id, target_id);

        -- Agent decisions table
        CREATE TABLE IF NOT EXISTS agent_decisions (
            decision_id UUID PRIMARY KEY,
            trail_id UUID REFERENCES trails(trail_id),
            universe_id VARCHAR(50) DEFAULT 'main',
            agent_id VARCHAR(255) NOT NULL,
            agent_type VARCHAR(100),
            step INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            event_id UUID REFERENCES events(event_id),
            context JSONB,
            prompt TEXT NOT NULL,
            output TEXT NOT NULL,
            processing_time FLOAT,
            action VARCHAR(100),
            feedback TEXT,
            rating FLOAT,
            reason TEXT,
            agent_type VARCHAR(100),
            action VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (agent_id, trail_id, universe_id) 
                REFERENCES agents(agent_id, trail_id, universe_id)
        );

        CREATE INDEX IF NOT EXISTS idx_decisions_trail_agent 
        ON agent_decisions(trail_id, agent_id);
        """
        
        try:
            # Import asyncpg only if database is enabled
            import asyncpg
            pool = await self.get_pool()
            if not pool:
                return
                
            async with pool.acquire() as conn:
                await conn.execute(schema_sql)
            logger.info("Database schema initialized successfully")
            self._initialized = True
        except ImportError:
            logger.warning("asyncpg package not installed. Database schema initialization skipped.")
            self._enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")
            self._enabled = False 