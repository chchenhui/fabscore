# OneSim Data Storage Module

This module provides database storage capabilities for OneSim, enabling persistent storage of:

- **Scenarios**: Information about simulation scenarios, including configuration and metadata
- **Trails**: Execution instances of scenarios
- **Environment States**: Snapshots of the environment at each step
- **Agents**: Information about agents in the simulation
- **Agent States**: Agent state snapshots at each step
- **Events**: All events occurring during simulation
- **Agent Decisions**: Records of agent decision-making for training data collection

## Setup

### 1. Install PostgreSQL

You'll need a PostgreSQL database. You can install it locally or use a remote database.

On Debian/Ubuntu:

```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

On Mac with Homebrew:

```bash
brew install postgresql
```

### 2. Install Python Dependencies

```bash
pip install -e .
```

### 3. Configure OneSim

Enable the data module in your OneSim configuration:

```python
config = {
    # ... other config options ...
    "data": {
        "enabled": True,
        "database": {
            "host": "localhost",
            "port": 5432,
            "dbname": "onesim",
            "user": "postgres",
            "password": "your_password"
        }
    }
}

# Initialize OneSim with this config
import onesim
onesim.init(config_path=None, **config)
```

You can also set these options in a YAML or JSON configuration file.

### 4. Create the Database

```bash
createdb onesim
```

## Usage Examples

### Working with Scenarios

```python
from onesim.data import ScenarioManager

# Create a scenario manager
scenario_mgr = ScenarioManager()

# Create a new scenario
scenario_id = scenario_mgr.create_scenario(
    name="Urban Planning Simulation",
    folder_path="./scenarios/urban_planning",
    description="Simulation of urban development policies",
    tags={"domain": "urban_planning", "version": "1.0"}
)

# List all scenarios
scenarios = scenario_mgr.list_scenarios()
```

### Working with Trails

```python
from onesim.data import TrailManager, TrailStatus

# Create a trail manager
trail_mgr = TrailManager()

# Create a new trail run of a scenario
trail_id = trail_mgr.create_trail(
    scenario_id=scenario_id,
    name="Urban Planning Run 1",
    description="Testing impact of zoning changes",
    config={"num_agents": 100, "steps": 1000}
)

# Start the trail
trail_mgr.start_trail(trail_id)

# ... run simulation ...

# Complete the trail
trail_mgr.complete_trail(trail_id, step_count=1000)
```

### Storing Environment States

```python
from onesim.data import EnvironmentStateManager

# Create environment state manager
env_mgr = EnvironmentStateManager()

# Save environment state for a specific step
env_mgr.save_environment_state(
    trail_id=trail_id,
    step=10,
    state={"population": 1000, "resources": {"water": 100, "food": 200}}
)

# Get environment state
state = env_mgr.get_environment_state(trail_id=trail_id, step=10)
```

### Working with Agents and Agent States

```python
from onesim.data import AgentManager

# Create agent manager
agent_mgr = AgentManager()

# Register an agent
agent_mgr.register_agent(
    trail_id=trail_id,
    agent_id="agent_001",
    agent_type="citizen",
    name="John Doe",
    initial_profile={"age": 35, "occupation": "teacher"}
)

# Save agent state
agent_mgr.save_agent_state(
    trail_id=trail_id,
    agent_id="agent_001",
    step=10,
    profile={"age": 35, "occupation": "teacher", "mood": "happy"},
    memory={"recent_events": ["went_to_work", "bought_groceries"]},
    relationships={"agent_002": {"type": "friend", "strength": 0.8}}
)
```

### Working with Events

```python
from onesim.data import EventManager

# Create event manager
event_mgr = EventManager()

# Save an event
event_id = event_mgr.save_event(
    trail_id=trail_id,
    step=10,
    event_type="interaction",
    source_type="agent",
    source_id="agent_001",
    target_type="agent",
    target_id="agent_002",
    payload={"interaction_type": "greeting", "content": "Hello!"}
)

# Mark event as processed
event_mgr.mark_event_processed(
    event_id=event_id,
    processing_result={"success": True, "response": "Hello back!"}
)
```

## Data Export

The data storage module makes it easy to export data for analysis:

- **Training data**: Collect agent decisions and their outcomes
- **Visualization**: Export state histories for visualization tools
- **Analysis**: Perform SQL queries on the database for complex analytics

## Schema Diagram

The database schema reflects the relationships between:
- Scenarios (1) -> Trails (many)
- Trails (1) -> Environment States (many)
- Trails (1) -> Agents (many)
- Agents (1) -> Agent States (many)
- Trails (1) -> Events (many)
- Events (1) -> Agent Decisions (1) 