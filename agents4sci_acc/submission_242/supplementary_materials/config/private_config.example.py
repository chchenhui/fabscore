"""
Private Configuration Example for SimulEval++

Copy this file to 'private_config.py' and fill in your actual credentials.
The private_config.py file is git-ignored for security.

Choose ONE of the configurations below based on your setup.
"""

# Option 1: Standard OpenAI API Configuration
OPENAI_CONFIG = {
    "api_key": "sk-your-openai-api-key-here",
    "model": "gpt-4o-mini",
    "provider": "openai"
}

# Option 2: Other Proxy Configuration (for organizations with custom endpoints)
OTHER_CONFIG = {
    "api_key": "your-other-api-key-here",
    "model": "gpt-4o-mini",
    "base_url": "https://your-organization.example.com/llm-proxy/v1",
    "provider": "openai"
}

# Option 3: Custom Configuration
# You can also define your own configuration for other providers
CUSTOM_CONFIG = {
    "api_key": "your-api-key",
    "model": "gpt-4o-mini",
    "base_url": "https://your-custom-endpoint.com/v1",  # Optional
    "provider": "custom",
    "temperature": 0.7,
    "max_retries": 3
}
