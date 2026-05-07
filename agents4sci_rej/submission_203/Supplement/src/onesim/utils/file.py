from typing import Dict, Any
import asyncio
import uuid
import os
import time
import datetime
import json
from loguru import logger


def load_json(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return the contents as a dictionary"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {file_path}: {e}")
        return {}