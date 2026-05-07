# Memory Module

The Memory module provides a flexible and extensible memory management system for intelligent agents. It supports various storage types, strategies, and operations to adapt to different application scenarios.

## Architecture

The Memory module employs a modular design, primarily comprising the following components:

### Core Components

- **MemoryItem**: The basic unit of memory, containing an agent ID, content, a unique ID, a timestamp, and attributes.
- **MemoryManager**: The entry point that loads configurations and initializes strategy, storages, and metrics.
- **MemoryStrategy**: An abstract base class defining the interface for memory management. Concrete strategies implement specific memory management logic.

### Storage

- **MemoryStorage**: Abstract base class for memory storage with methods for add, query, delete, etc.
  - **ListMemoryStorage**: List-based storage with configurable capacity and eviction policies.
  - **VectorMemoryStorage**: Vector-based semantic storage using embeddings with similarity search capabilities.

### Operations

- **MemoryOperation**: Abstract base class for operations with an `async execute` method.
  - **AddMemoryOperation**: Adds a memory item to storage.
  - **RetrieveMemoryOperation**: Retrieves memory items based on a query.
  - **RemoveMemoryOperation**: Removes memory items from storage.
  - **ReflectMemoryOperation**: Generates insights from short-term memories and adds them to long-term storage.
  - **MergeOperation**: Merges memories within storage.
  - **ForgetOperation**: Removes memories based on criteria.

### Metrics

- **MemoryMetric**: Abstract base class for metrics with an `async calculate` method.
  - **ImportanceMetric**: Calculates memory importance, often using LLMs.
  - **RecencyMetric**: Calculates a score based on memory age.
  - **RelevanceMetric**: Calculates memory relevance to a query.

## Usage Examples

### Creating and Configuring MemoryManager

```python
from onesim.memory import MemoryManager

memory_manager = MemoryManager("config.json")

if hasattr(memory_manager.strategy, 'set_agent_context'):
    memory_manager.strategy.set_agent_context(agent_context)
```

### Adding Memory

```python
from onesim.memory import MemoryItem

memory_item = MemoryItem(agent_id="agent1", content="This is an important memory.")

await memory_manager.strategy.add(memory_item)
```

### Retrieving Memory

```python
memories = await memory_manager.strategy.retrieve(query="What are recent important events?", top_k=5)

for memory in memories:
    print(f"Content: {memory.content}, Score: {memory.attributes.get('score')}")
```

### Triggering Reflection

```python
if hasattr(memory_manager.strategy, 'reflect'):
    await memory_manager.strategy.reflect()
```

## Extensions

The Memory module is designed to be extensible:

1. Create new storage types by inheriting from `MemoryStorage` and implementing its abstract methods.
2. Implement custom strategies by inheriting from `MemoryStrategy` and implementing its abstract methods.
3. Add new operations by inheriting from `MemoryOperation` and implementing the `async execute` method.
4. Define new metrics by inheriting from `MemoryMetric` and implementing the `async calculate` method.

Remember to update your configuration to use any new custom components.

## Best Practices

1. **Asynchronous Usage**: Utilize `async`/`await` when calling asynchronous methods.
2. **Error Handling**: Implement appropriate error handling for memory operations.
3. **Performance Considerations**: Use batch operations for large memory collections.
4. **Configuration Tuning**: Configure storage capacities, eviction policies, and metrics weights to suit your application.

## Dependencies

- Python 3.7+
- `loguru` (for logging)
- `numpy` (for vector operations)
- `faiss` (for vector indexing and retrieval)
- `asyncio` (built-in library for asynchronous programming)

## Memory Strategy Pattern

The Memory module uses the Strategy pattern to enable different memory management approaches.

### Core Strategy Interface

**`MemoryStrategy`** (in `strategy/strategy.py`) defines key methods:

- `_initialize_components(config)`: Loads storages, operations, metrics.
- `set_agent_context(agent_context)`: Sets contextual information for the agent.
- `async execute(operation_name, *args, **kwargs)`: Executes a registered operation.
- `async retrieve(query, top_k)`: Abstract method for retrieving memories.
- `async add(memory_item)`: Abstract method for adding memory items.
- `select_storage(memory_item)`: Selects appropriate storage for a memory item.

### Concrete Strategies

**`ListStrategy`** (in `strategy/list_strategy.py`) implements a simple list-based memory strategy.

**`ShortLongStrategy`** (in `strategy/short_long_strategy.py`) implements dual-store memory management with short-term and long-term storage. It provides automatic reflection triggers based on memory importance.

### Using Memory Strategies

```python
await memory_manager.strategy.add(memory_item)
memories = await memory_manager.strategy.retrieve("search query", top_k=3)
await memory_manager.strategy.reflect()
```

### Adding Custom Strategies

```python
from onesim.memory.strategy import MemoryStrategy
from onesim.memory import MemoryItem

class MyCustomStrategy(MemoryStrategy):
    def __init__(self, config, model_config_name=None):
        super().__init__(config, model_config_name)

    async def add(self, memory_item: MemoryItem):
        pass

    async def retrieve(self, query, top_k=5):
        pass
```

## Operations (Command Pattern)

Memory operations follow the Command pattern, encapsulating actions as objects. These operations are executed through the strategy's `execute` method:

```python
await memory_manager.strategy.execute('add_op', storage_name='long_term_storage', memory_item=memory_item)

await memory_manager.strategy.execute('reflect_op')
```

Operation names used with `execute` depend on how they're configured. `ShortLongStrategy` pre-registers `ReflectMemoryOperation` as `'reflect'`.