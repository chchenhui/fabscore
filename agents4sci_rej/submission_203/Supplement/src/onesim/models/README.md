# Models Package

This package provides a unified interface for interacting with various language models, handling responses, and managing configurations.

## Directory Structure

- `core/`: Core classes and interfaces
  - `message.py`: Message data structures
  - `model_base.py`: Base model wrapper interface
  - `model_response.py`: Standard response format
  - `model_manager.py`: Configuration and model registry

- `providers/`: Model-specific implementations
  - `openai_chat.py`: OpenAI chat model wrapper

- `parsers/`: Response parsing utilities
  - `json_parsers.py`: JSON and structured data parsers

- `utils/`: Helper utilities for models

## Usage

### Basic Example

```python
from onesim.models import get_model, SystemMessage, UserMessage

# Get a configured model
model = get_model("my_model_config")

# Create messages
system_msg = SystemMessage("You are a helpful assistant.")
user_msg = UserMessage("Tell me about language models.")

# Generate a response
response = await model.acall(model.format(system_msg, user_msg))

# Get the response text
print(response.text)
```

### Configuration

Models are configured via the `ModelManager` which can load configurations from a JSON file:

```python
from onesim.models import get_model_manager

manager = get_model_manager()
manager.load_model_configs("path/to/config.json")
```

Configuration format:

```json
[
  {
    "config_name": "my_model_config",
    "model_type": "openai_chat",
    "model_name": "gpt-4",
    "api_key": "your-api-key"
  }
]
```

### Working with Structured Responses

Use parsers to extract structured data from model responses:

```python
from onesim.models.parsers.json_parsers import JsonDictParser

parser = JsonDictParser(required_keys=["name", "description"])
instruction = parser.format_instruction

# Include instruction in prompt
user_msg = UserMessage(f"Generate JSON data. {instruction}")

# Generate and parse
response = await model.acall(model.format(user_msg))
parsed_response = parser.parse(response)

print(parsed_response.parsed)  # Structured data
```

## Load Balancing

OneSim provides load balancing for LLM and embedding models, distributing requests across multiple models for better performance and availability.

### Configuring Load Balancers

```python
from onesim.models import configure_load_balancer, get_model, get_embedding_model

# Configure LLM load balancer
configure_load_balancer(
    model_configs=["gpt-4", "gpt-3.5-turbo"],  # Models to balance between
    strategy="round_robin",                    # Strategy: "round_robin" or "random"
    config_name="load_balancer",               # Configuration name
    model_type="llm"                           # Model type: "llm" or "embedding"
)

# Configure embedding load balancer
configure_load_balancer(
    model_configs=["text-embedding-ada-002", "text-embedding-3-small"],
    strategy="round_robin",
    config_name="embedding_load_balancer",
    model_type="embedding"
)

# Get load balancers - defaults to respective type's load balancer if configured
llm_model = get_model()                        # Returns LLM load balancer
embedding_model = get_embedding_model()        # Returns embedding load balancer

# Or get specific load balancers by name
llm_model = get_model("load_balancer")
embedding_model = get_model("embedding_load_balancer")

# Use load balancer like any model
response = await llm_model.acall(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, please introduce yourself."}
])

# Use embedding load balancer
embeddings = await embedding_model.acall(texts=["This is a text"])
```

### Complete Load Balancing Example

```python
import asyncio
from onesim.models import (
    configure_load_balancer, get_model, get_model_manager, 
    SystemMessage, UserMessage
)

async def main():
    # Load model configurations
    manager = get_model_manager()
    manager.load_model_configs({
        "config_name": "gpt-4",
        "model_type": "openai_chat",
        "model_name": "gpt-4",
        "api_key": "your_api_key"
    })
    
    manager.load_model_configs({
        "config_name": "gpt-3.5-turbo",
        "model_type": "openai_chat",
        "model_name": "gpt-3.5-turbo",
        "api_key": "your_api_key"
    })
    
    # Configure load balancer
    configure_load_balancer(
        model_configs=["gpt-4", "gpt-3.5-turbo"],
        strategy="round_robin"
    )
    
    # Get load balancer
    load_balancer = get_model()
    
    # Prepare messages
    messages = [
        SystemMessage("You are a helpful assistant."),
        UserMessage("Hello, please introduce yourself.")
    ]
    
    # Send requests via load balancer
    formatted_messages = load_balancer.format(messages)
    response = await load_balancer.acall(formatted_messages)
    
    print(f"Response: {response.content}")
    print(f"Model used: {response.model_info.get('model_name', 'unknown')}")
    
    # Second request, will use next model
    response = await load_balancer.acall(formatted_messages)
    print(f"Second response model: {response.model_info.get('model_name', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Synchronous and Asynchronous Calls

OneSim supports both synchronous and asynchronous model calls for different use cases.

### Synchronous Calls

```python
from onesim.models import get_model, SystemMessage, UserMessage

model = get_model("gpt-4")

system_msg = SystemMessage("You are a helpful assistant.")
user_msg = UserMessage("Briefly explain the difference between synchronous and asynchronous programming.")

formatted_messages = model.format(system_msg, user_msg)
response = model(formatted_messages)  # Direct call using __call__

print(response.text)
```

### Asynchronous Calls

```python
import asyncio
from onesim.models import get_model, SystemMessage, UserMessage

async def generate_response():
    model = get_model("gpt-4")
    
    system_msg = SystemMessage("You are a helpful assistant.")
    user_msg = UserMessage("Briefly explain the difference between synchronous and asynchronous programming.")
    
    formatted_messages = model.format(system_msg, user_msg)
    response = await model.acall(formatted_messages)
    
    return response.text

response_text = asyncio.run(generate_response())
print(response_text)
```

### Creating Custom Model Adapters

When implementing custom model adapters, provide both synchronous and asynchronous interfaces:

```python
class MyModelAdapter(ModelAdapterBase):
    def __call__(self, *args, **kwargs):
        # Synchronous API call
        return self._process_request(*args, **kwargs)
    
    async def acall(self, *args, **kwargs):
        # Asynchronous API call
        if hasattr(self, 'async_client'):
            return await self._process_request_async(*args, **kwargs)
        else:
            # Run synchronous API in a thread
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self._process_request(*args, **kwargs)
            )
    
    def _process_request(self, *args, **kwargs):
        # Shared processing logic - synchronous version
        pass
```

