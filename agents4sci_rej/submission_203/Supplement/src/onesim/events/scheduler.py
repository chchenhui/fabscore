import asyncio
from typing import List, Dict, Any, Optional

class Scheduler:
    def __init__(self, event_bus):
        self.tasks: List[asyncio.Task] = []  # Save all scheduled tasks
        self.event_bus = event_bus
        self._running = False
        self._done_tasks: List[asyncio.Task] = []
        self._paused_tasks: Dict[asyncio.Task, bool] = {}  # Track paused state of tasks
        self._task_intervals: Dict[asyncio.Task, float] = {}  # Store intervals for tasks
        self._task_events: Dict[asyncio.Task, Any] = {}  # Store events for tasks
        self._task_counts: Dict[asyncio.Task, Dict[str, int]] = {}  # Store max_count and current_count for tasks

    def schedule_task(self, interval: float, event: Any, max_count: Optional[int] = None) -> asyncio.Task:
        """
        Schedule a new task with the given interval and event.
        
        Args:
            interval: Time interval between events in seconds
            event: The event to dispatch
            max_count: Maximum number of times to execute the event (None for infinite)
        
        Returns:
            The created task for future reference
        """
        task = asyncio.create_task(self._trigger_event(interval, event, max_count))
        self.tasks.append(task)
        self._paused_tasks[task] = False
        self._task_intervals[task] = interval
        self._task_events[task] = event
        self._task_counts[task] = {
            'max_count': max_count,
            'current_count': 0
        }
        return task

    async def run(self):
        """Start running all scheduled tasks."""
        self._running = True
        scheduled_tasks = []
        for task in self.tasks:
            scheduled_tasks.append(task)
        await asyncio.gather(*scheduled_tasks)

    async def stop(self):
        """Stop all tasks and clean up."""
        self._running = False
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        
        # Clear all task collections
        self.tasks.clear()
        self._paused_tasks.clear()
        self._task_intervals.clear()
        self._task_events.clear()
        self._task_counts.clear()

    async def pause(self, task: asyncio.Task = None):
        """
        Pause specific task or all tasks if no task is specified.
        """
        if task:
            if task in self._paused_tasks:
                self._paused_tasks[task] = True
        else:
            # Pause all tasks
            for t in self._paused_tasks:
                self._paused_tasks[t] = True

    async def resume(self, task: asyncio.Task = None):
        """
        Resume specific task or all tasks if no task is specified.
        """
        if task:
            if task in self._paused_tasks:
                self._paused_tasks[task] = False
        else:
            # Resume all tasks
            for t in self._paused_tasks:
                self._paused_tasks[t] = False

    async def _trigger_event(self, interval: float, event: Any, max_count: Optional[int] = None):
        """
        Internal method to handle event triggering with pause support and execution count.
        """
        task = asyncio.current_task()
        count = 0

        while self._running:
            # Check if this specific task is paused
            if self._paused_tasks.get(task, False):
                await asyncio.sleep(0.1)  # Small sleep to prevent busy waiting
                continue
            
            # Check if we've reached the maximum execution count
            if max_count is not None and count >= max_count:
                # Update task counts before completing
                if task in self._task_counts:
                    self._task_counts[task]['current_count'] = count
                return
                
            await self.event_bus.dispatch_event(event)
            count += 1
            
            # Update current count in task_counts
            if task in self._task_counts:
                self._task_counts[task]['current_count'] = count
                if self._task_counts[task]['max_count'] is not None and count >= self._task_counts[task]['max_count']:
                    self._done_tasks.append(task)
            await asyncio.sleep(interval)

    def is_done(self):
        return len(self._done_tasks) == len(self.tasks)

    def get_task_info(self, task: asyncio.Task) -> Dict[str, Any]:
        """
        Get information about a specific task.
        """
        counts = self._task_counts.get(task, {'max_count': None, 'current_count': 0})
        return {
            'interval': self._task_intervals.get(task),
            'event': self._task_events.get(task),
            'paused': self._paused_tasks.get(task, False),
            'running': not task.done() if task else False,
            'max_count': counts['max_count'],
            'current_count': counts['current_count']
        }

    def get_remaining_executions(self, task: asyncio.Task) -> Optional[int]:
        """
        Get the number of remaining executions for a task.
        Returns None if the task has infinite executions.
        """
        if task not in self._task_counts:
            return None
            
        counts = self._task_counts[task]
        max_count = counts['max_count']
        current_count = counts['current_count']
        
        if max_count is None:
            return None
            
        remaining = max_count - current_count
        return max(0, remaining)
    