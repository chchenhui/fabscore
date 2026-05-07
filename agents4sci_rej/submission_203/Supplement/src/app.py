import os
import asyncio
import uvicorn
from loguru import logger

# Import from our restructured app
from backend.main import app
from backend.utils.model_management import load_default_config

# Initialize global configuration
def init():
    # Load default configuration
    default_config = load_default_config()
    logger.info("Application initialized")
    return default_config

# Initialize on startup
default_config = init()

def start_server():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

# Run the application
if __name__ == "__main__":
    start_server()