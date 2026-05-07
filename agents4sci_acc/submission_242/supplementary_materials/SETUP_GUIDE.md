# SimulEval++ Setup Guide

## Quick Setup

SimulEval++ supports multiple LLM providers through a flexible configuration system.

### Option 1: Environment Variables (Recommended)

**For Standard OpenAI:**
```bash
export OPENAI_API_KEY='sk-your-openai-key-here'
python simuleval_core.py
```

**For Custom LLM Proxy:**
```bash
export OTHER_API_KEY='your-proxy-api-key'
python simuleval_core.py
```

### Option 2: Configuration File

1. Copy the example configuration:
```bash
cp config/private_config.example.py config/private_config.py
```

2. Edit `config/private_config.py` with your credentials:
```python
# For OpenAI
OPENAI_CONFIG = {
    "api_key": "sk-your-actual-key-here",
    "model": "gpt-4o-mini",
    "provider": "openai"
}

# OR for custom proxy (organization-specific)
OTHER_CONFIG = {
    "api_key": "your-actual-proxy-key",
    "model": "gpt-4o-mini", 
    "base_url": "https://your-internal-proxy.company.com/v1",
    "provider": "openai"
}
```

3. Run the simulation:
```bash
python simuleval_core.py
```

## Testing Your Setup

Before running the full simulation, test your connection:

```bash
python test_llm_connection.py
```

This will verify your API configuration is working correctly.

## Configuration Options

### API Configuration
- **OPENAI_CONFIG**: Uses standard OpenAI API with gpt-4o-mini
- **OTHER_CONFIG**: Uses custom proxy endpoint with gpt-4o-mini
- **Environment variables**: Override file-based config

## Priority Order

The system checks for configuration in this order:
1. OTHER_API_KEY environment variable
2. OPENAI_API_KEY environment variable  
3. OTHER_CONFIG in private_config.py
4. OPENAI_CONFIG in private_config.py

## Security Notes

- The `config/private_config.py` file is automatically git-ignored
- Never commit API keys to version control
- Use environment variables in production environments
