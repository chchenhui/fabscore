import asyncio
import time
from typing import Optional, Dict, Protocol
from loguru import logger
from onesim.distribution import grpc_impl
from onesim.distribution.node import NodeRole
from onesim.distribution.node import get_node

# Define a common Lock Protocol
class LockProtocol(Protocol):
    async def acquire(self) -> bool: ...
    async def release(self) -> bool: ...
    async def __aenter__(self): ...
    async def __aexit__(self, exc_type, exc_val, exc_tb): ...

# Keep the original DistributedLock implementation (implements LockProtocol)
class DistributedLock: # No need to explicitly inherit Protocol if methods match
    def __init__(self, lock_id: str, timeout: float = 30.0):
        self.lock_id = lock_id
        self.timeout = timeout
        self.node = get_node()
        self._lock_holder: Optional[str] = None
        self._lock_expiry: Optional[float] = None
        # This local lock protects internal state, not the distributed resource itself
        self._internal_state_lock = asyncio.Lock()

    async def acquire(self) -> bool:
        # Use internal lock to protect access to _lock_holder/_lock_expiry
        async with self._internal_state_lock:
            current_time = time.time()

            # Check if this node already holds the lock locally
            if self._lock_holder == self.node.node_id:
                if self._lock_expiry and current_time < self._lock_expiry:
                    # Already hold a valid lock
                    return True
                else:
                    # Our lock expired, clear local state
                    self._lock_holder = None
                    self._lock_expiry = None
                    # Note: We might still hold the lock on the master if timeout logic differs
                    # Consider attempting release here if expired? Or rely on master cleanup.

            # Attempt to acquire lock via master if master address exists
            if not self.node.master_address or not self.node.master_port:
                 logger.error(f"Cannot acquire distributed lock {self.lock_id}: Master address/port not configured.")
                 return False

            try:
                success = await grpc_impl.acquire_lock(
                    self.node.master_address,
                    self.node.master_port,
                    self.lock_id,
                    self.node.node_id,
                    self.timeout
                )

                if success:
                    self._lock_holder = self.node.node_id
                    self._lock_expiry = current_time + self.timeout
                    logger.debug(f"Node {self.node.node_id} acquired lock {self.lock_id}")
                    return True
                else:
                    logger.debug(f"Node {self.node.node_id} failed to acquire lock {self.lock_id}")
                    return False

            except Exception as e:
                logger.error(f"Error acquiring lock {self.lock_id} via gRPC: {e}")
                return False

    async def release(self) -> bool:
        # Use internal lock to protect access to _lock_holder/_lock_expiry
        async with self._internal_state_lock:
            if self._lock_holder != self.node.node_id:
                # Trying to release a lock we don't hold locally
                logger.warning(f"Node {self.node.node_id} attempted to release lock {self.lock_id} it doesn't appear to hold locally")
                # Should we still attempt release on master? Maybe.
                # Let's return False for now, assuming local state is source of truth.
                # Or potentially just proceed to attempt release anyway. For safety, let's try release:
                # pass # Fall through to attempt release on master

            # Attempt to release lock via master if master address exists
            if not self.node.master_address or not self.node.master_port:
                 logger.error(f"Cannot release distributed lock {self.lock_id}: Master address/port not configured.")
                 # Clear local state anyway?
                 self._lock_holder = None
                 self._lock_expiry = None
                 return False # Indicate failure due to config

            release_attempted = False
            try:
                # Only release if we think we hold it or attempt cleanup
                if self._lock_holder == self.node.node_id:
                     release_attempted = True
                     success = await grpc_impl.release_lock(
                        self.node.master_address,
                        self.node.master_port,
                        self.lock_id,
                        self.node.node_id
                     )
                     if success:
                         logger.debug(f"Node {self.node.node_id} released lock {self.lock_id}")
                         # Clear local state ONLY on successful release confirmation
                         self._lock_holder = None
                         self._lock_expiry = None
                         return True
                     else:
                         logger.debug(f"Node {self.node.node_id} failed to release lock {self.lock_id} on master")
                         # Don't clear local state? Or clear anyway? Let's clear for now.
                         self._lock_holder = None
                         self._lock_expiry = None
                         return False
                else:
                     # We don't think we hold it locally, maybe don't try releasing?
                     # Or maybe try anyway for cleanup? Let's skip for now.
                     logger.warning(f"Node {self.node.node_id} skipped releasing lock {self.lock_id} as it wasn't held locally.")
                     return False


            except Exception as e:
                logger.error(f"Error releasing lock {self.lock_id} via gRPC: {e}")
                # Clear local state on error?
                if release_attempted:
                     self._lock_holder = None
                     self._lock_expiry = None
                return False


    async def __aenter__(self):
        # Keep trying to acquire the lock (could add retry logic/timeout here)
        retry_delay = 0.1
        max_attempts = 50 # ~5 seconds total wait
        attempts = 0
        while attempts < max_attempts:
            if await self.acquire():
                return self
            attempts += 1
            await asyncio.sleep(retry_delay)
            # Optional: Implement exponential backoff
            # retry_delay *= 1.5
        raise RuntimeError(f"Failed to acquire lock {self.lock_id} after multiple attempts")


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


# New Local Lock implementation using asyncio.Lock
class LocalLock: # Implements LockProtocol implicitly
    def __init__(self, lock_id: str):
        self.lock_id = lock_id # For potential debugging/logging
        self._lock = asyncio.Lock()
        # logger.debug(f"Creating local lock: {self.lock_id}") # Optional debug log

    async def acquire(self) -> bool:
        # logger.debug(f"Acquiring local lock: {self.lock_id}")
        await self._lock.acquire()
        # logger.debug(f"Acquired local lock: {self.lock_id}")
        return True # asyncio.Lock.acquire() blocks until acquired

    async def release(self) -> bool:
        # logger.debug(f"Releasing local lock: {self.lock_id}")
        try:
            self._lock.release()
            # logger.debug(f"Released local lock: {self.lock_id}")
            return True
        except RuntimeError as e:
            # Attempted to release an unlocked lock
            logger.warning(f"Attempted to release non-acquired local lock {self.lock_id}: {e}")
            return False


    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()


# Global lock registries
_distributed_lock_registry: Dict[str, DistributedLock] = {}
_local_lock_registry: Dict[str, LocalLock] = {}
_registry_lock = asyncio.Lock() # Lock for accessing registries

async def get_lock(lock_id: str, timeout: float = 30.0) -> LockProtocol:
    """
    Get or create a lock instance appropriate for the current execution mode.

    Args:
        lock_id: Unique identifier for the lock
        timeout: Lock timeout in seconds (only relevant for DistributedLock)

    Returns:
        LockProtocol instance (either DistributedLock or LocalLock)
    """
    node = get_node()
    # Determine mode: Check if master address is configured (simplest check)

    is_distributed_mode = bool(node.role != NodeRole.SINGLE)

    async with _registry_lock:
        if is_distributed_mode:
            if lock_id not in _distributed_lock_registry:
                _distributed_lock_registry[lock_id] = DistributedLock(lock_id, timeout)
            return _distributed_lock_registry[lock_id]
        else:
            if lock_id not in _local_lock_registry:
                _local_lock_registry[lock_id] = LocalLock(lock_id)
            return _local_lock_registry[lock_id]
