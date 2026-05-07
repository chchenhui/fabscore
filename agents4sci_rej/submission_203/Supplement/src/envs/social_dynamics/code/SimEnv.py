from onesim.simulator import BasicSimEnv
from onesim.events import Event
from .events import StartEvent

class SimEnv(BasicSimEnv):
    async def _create_start_event(self, target_id: str) -> Event:
        # Extract the source agent ID from self.data
        source_id = self.data.get('source_agent_id', 'ENV')
        # Create and return a StartEvent instance
        return StartEvent(from_agent_id=source_id, to_agent_id=target_id)