# -*- coding: utf-8 -*-
""" Base class for Agent """

from __future__ import annotations
from types import GeneratorType
from typing import Optional, Generator, Tuple
from typing import Sequence
from typing import Union
from typing import Any
import json
import uuid
from loguru import logger

from onesim.models.core.model_manager import ModelManager


class AgentBase:
    """Base class for all agents.

    All agents should inherit from this class and implement the `reply`
    function.
    """

    _version: int = 1

    def __init__(
        self,
        sys_prompt: Optional[str] = None,
        model_config_name: str = None,
    ) -> None:
        r"""Initialize an agent from the given arguments.

        Args:
            name (`str`):
                The name of the agent.
            sys_prompt (`Optional[str]`):
                The system prompt of the agent, which can be passed by args
                or hard-coded in the agent.
            model_config_name (`str`, defaults to None):
                The name of the model config, which is used to load model from
                configuration.
        """
        self.sys_prompt = sys_prompt
        
        # Use a different name for the internal attribute
        self._agent_id = self.generate_agent_id()

        model_manager = ModelManager.get_instance()
        if model_config_name is not None:
            
            self.model = model_manager.get_model(
                config_name=model_config_name,
            )
        else:
            self.model = model_manager.get_model()

    @classmethod
    def generate_agent_id(cls) -> str:
        """Generate the agent_id of this agent instance"""
        # TODO: change cls.__name__ into a global unique agent_type
        return uuid.uuid4().hex

    def __str__(self) -> str:
        serialized_fields = {
            "type": self.__class__.__name__,
            "sys_prompt": self.sys_prompt,
            "agent_id": self.agent_id,
        }
        if hasattr(self, "model"):
            serialized_fields["model"] = {
                "model_type": self.model.model_type,
                "config_name": self.model.config_name,
            }
        return json.dumps(serialized_fields, ensure_ascii=False)
    
    @property
    def agent_id(self) -> str:
        """The unique id of this agent.

        Returns:
            str: agent_id
        """
        return self._agent_id

    @agent_id.setter
    def agent_id(self, agent_id: str) -> None:
        """Set the unique id of this agent."""
        self._agent_id = agent_id