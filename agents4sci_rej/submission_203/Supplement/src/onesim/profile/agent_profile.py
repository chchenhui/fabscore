from abc import ABC, abstractmethod
from typing import Iterable, Sequence, Optional, Any, Union, Dict, List
from loguru import logger
from .profile import ProfileBase
import json
import csv
import os
import random
from onesim.models.core.message import Message
from onesim.models.parsers import JsonBlockParser
from .profile import AgentSchema
from datetime import datetime

class AgentProfile(ProfileBase):
    """Class that handles public and private profiles based on configuration."""

    def __init__(self,agent_type:str, schema: AgentSchema =None, profile_data: Optional[Dict[str, Any]] = None):
        """
        Initialize profile either from a config file or directly from provided profile data.
        """
        super().__init__(schema)
        # self.agent_type = agent_type
        # self._public_fields = {}
        # self._private_fields = {}
        #self._public_fields['agent_type']=agent_type
        self._public_fields['agent_type']=agent_type
        self._load_profile_data(schema,profile_data)


    def _load_profile_data(self, schema:AgentSchema, profile_data: Dict[str, Any]) -> None:
        """Load profile data into public and private fields based on privacy settings."""
        for key, value in schema.schema.items():
            if profile_data is not None and key in profile_data:
                value=profile_data[key]
            else:
                value = value.get("default", None)
            if self.schema.is_private(key):
                self._private_fields[key] = value
            else:
                self._public_fields[key] = value
        for key, value in profile_data.items():
            if key not in self.schema.schema:
                self._public_fields[key] = value

    def generate_profile(self, model):
        """Generate the complete profile using the LLM or default values."""
        prompt = self._build_prompt()
        if model is not None and len(prompt) > 0:
            max_retries = 3
            retry_count = 0
            success = False
            
            while not success and retry_count < max_retries:
                try:
                    prompt = model.format(Message("user", prompt, role="user"))
                    response = model(prompt)
                    parser = JsonBlockParser()
                    res = parser.parse(response)
                    generated_profile = res.parsed
                    
                    # Validate the generated profile
                    if not generated_profile or not isinstance(generated_profile, dict):
                        logger.warning(f"Retry {retry_count+1}/{max_retries}: Generated profile is empty or not a dictionary")
                        retry_count += 1
                        continue
                    
                    # Apply the generated profile values
                    for key, value in generated_profile.items():
                        if key in self._public_fields:
                            self._public_fields[key] = value
                        elif key in self._private_fields:
                            self._private_fields[key] = value
                    
                    success = True
                    
                except (json.JSONDecodeError, ValueError, AttributeError) as e:
                    retry_count += 1
                    logger.warning(f"Retry {retry_count}/{max_retries}: Failed to generate profile - {str(e)}")
                    
            if not success:
                logger.error(f"Failed to generate profile after {max_retries} attempts.")
                raise ValueError("LLM failed to generate a valid profile after multiple attempts.")
                
        # Continue with default/random sampling regardless of LLM success
        for key, settings in self.schema.schema.items():
            sampling_method = settings.get("sampling", "default")
            private = settings.get("private", False)
            if sampling_method == "random":
                value = self._random_sampling(key, settings)
            elif sampling_method == "default":
                value = settings.get("default", None)
            else:
                continue
            if private:
                self._private_fields[key] = value
            else:
                self._public_fields[key] = value

    @staticmethod
    def _random_sampling(key: str, settings: Dict[str, Any]):
        """Randomly sample a value based on the field settings."""
        field_type = settings.get("type", "str")
        try:
            if field_type == "int":
                min_val, max_val = settings.get("range", [0, 100])
                return random.randint(min_val, max_val)
            elif field_type == "float":
                min_val, max_val = settings.get("range", [0.0, 1.0])
                return random.uniform(min_val, max_val)
            elif field_type == "str":
                choices = settings.get("choices", ["Default String"])
                return random.choice(choices)
            elif field_type == "list":
                choices = settings.get("choices", [])
                sample_size = settings.get("sample_size", [1, len(choices)])
                k = sample_size if isinstance(sample_size, int) else random.randint(sample_size[0], min(sample_size[1], len(choices)))
                return random.sample(choices, k)
            return settings.get("default", None)
        except ValueError:
            logger.error(f"Error getting field '{key}' with settings: {settings}")
            return None

    def _build_prompt(self) -> str:
        """Build the prompt to send to the LLM for profile generation."""
        fields_to_generate = [
            {
                "field_name": key,
                "type": settings.get("type", "str"),
                "description": settings.get("description", "")
            }
            for key, settings in self.schema.schema.items() if settings.get("sampling") == "llm"
        ]

        if not fields_to_generate:
            return ""
        return (
            f"Please generate a complete personal profile in JSON format for {self.agent_type} based on the following field requirements :\n"
            f"{json.dumps(fields_to_generate, ensure_ascii=False, indent=4)}\n"
            "Ensure that the generated JSON format is correct and contains all specified fields."
        )

    @staticmethod
    def generate_profiles(agent_type:str,schema: AgentSchema,model, count: int = 1) -> List['AgentProfile']:
        """
        Generate multiple profiles using the LLM or default values.

        Args:
            model: The LLM model to use for generating profiles.
            count (int): The number of profiles to generate.

        Returns:
            List[AgentProfile]: A list of generated AgentProfile instances.
        """
        prompt = AgentProfile._build_bulk_prompt(agent_type,schema,count=count)
        generated_profiles = []

        if model is not None:
            if len(prompt) > 0:
                max_retries = 3
                retry_count = 0
                parsed_profiles = None
                
                while parsed_profiles is None and retry_count < max_retries:
                    try:
                        formatted_prompt = model.format(Message("user", prompt, role="user"))
                        response = model(formatted_prompt)
                        parser = JsonBlockParser()
                        parsed_profiles = parser.parse(response)
                        parsed_profiles = parsed_profiles.parsed
                        
                        # Validate the profiles
                        if not parsed_profiles or not isinstance(parsed_profiles, list) or len(parsed_profiles) != count:
                            logger.warning(f"Retry {retry_count+1}/{max_retries}: Expected {count} profiles, but got {len(parsed_profiles) if parsed_profiles and isinstance(parsed_profiles, list) else 'invalid data'}")
                            parsed_profiles = None
                            retry_count += 1
                            continue
                            
                    except (json.JSONDecodeError, ValueError, AttributeError) as e:
                        retry_count += 1
                        logger.warning(f"Retry {retry_count}/{max_retries}: Failed to generate profiles - {str(e)}")
                
                if parsed_profiles is None:
                    logger.error(f"Failed to generate profiles after {max_retries} attempts. Using empty profiles.")
                    parsed_profiles = [{} for _ in range(count)]
            else:
                parsed_profiles = [{} for _ in range(count)]

            for profile_data in parsed_profiles:
                profile = AgentProfile(agent_type=agent_type, schema=schema, profile_data=profile_data)
                # Fill in remaining fields based on schema
                for key, settings in schema.schema.items():
                    if key in profile_data:
                        continue  # Already set from LLM
                    sampling_method = settings.get("sampling", "default")
                    private = settings.get("private", False)
                    if sampling_method == "random":
                        value = AgentProfile._random_sampling(key, settings)
                    elif sampling_method == "default":
                        value = settings.get("default", None)
                    else:
                        continue
                    if private:
                        profile._private_fields[key] = value
                    else:
                        profile._public_fields[key] = value
                generated_profiles.append(profile)
           
        return generated_profiles

    @staticmethod
    def _build_bulk_prompt(agent_type:str ,schema:AgentSchema, count: int) -> str:
        """Build the prompt to send to the LLM for bulk profile generation."""
        fields_to_generate = [
            {
                "field_name": key,
                "type": settings.get("type", "str"),
                "description": settings.get("description", "")
            }
            for key, settings in schema.schema.items() if settings.get("sampling") == "llm"
        ]

        if not fields_to_generate:
            return ""
        return (
            f"Please generate {count} complete personal profiles in JSON array format for {agent_type} based on the following field requirements:\n"
            f"{json.dumps(fields_to_generate, ensure_ascii=False, indent=4)}\n"
            "Ensure that each profile in the JSON array is correctly formatted and contains all specified fields."
        )

    def get_profile(self, include_private: bool = False) -> Dict[str, Any]:
        """Return the profile as a dictionary."""
        return {**self._public_fields, **self._private_fields} if include_private else self._public_fields

    def get_profile_str(self, include_private: bool = False) -> str:
        """Return a string representation of the profile in JSON format."""
        profile_data = self.get_profile(include_private=include_private)
        def json_serializable(obj):
            """处理无法被JSON直接序列化的对象类型"""
            try:
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif hasattr(obj, "__dict__"):
                    return obj.__dict__
                elif hasattr(obj, "__str__"):
                    return str(obj)
                else:
                    return repr(obj)
            except Exception:
                # 如果上述所有尝试都失败，则放弃该值
                return None
        
        # 递归地处理嵌套字典和列表中的不可序列化值
        def clean_data(data):
            if isinstance(data, dict):
                return {k: clean_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_data(item) for item in data]
            else:
                try:
                    # 测试值是否可以被json序列化
                    json.dumps(data)
                    return data
                except (TypeError, OverflowError):
                    return json_serializable(data)
        
        cleaned_data = clean_data(profile_data)
        return json.dumps(cleaned_data, ensure_ascii=False, indent=4)[:4096]

    def update_field(self, key: str, value: Any, to_private: Optional[bool] = None):
        """Update a specific field in the profile."""
        if key in self._public_fields or key in self._private_fields:
            if to_private is not None:
                self.move_field(key, to_private)
            if key in self._public_fields:
                self._public_fields[key] = value
            elif key in self._private_fields:
                self._private_fields[key] = value
        else:
            raise KeyError(f"Field '{key}' not found in the profile.")

    def move_field(self, key: str, to_private: bool = True):
        """Move a field between public and private profiles."""
        if to_private:
            if key in self._public_fields:
                self._private_fields[key] = self._public_fields.pop(key)
        else:
            if key in self._private_fields:
                self._public_fields[key] = self._private_fields.pop(key)

    
    def get_agent_type(self) -> str:
        """Return the agent type."""
        return self.agent_type

    def get_agent_profile_id(self) -> str:
        """Return the agent profile id."""
        return str(self._private_fields.get("id", None))
    
    def set_agent_profile_id(self, id):
        """Set the agent profile id."""
        self._private_fields["id"] = str(id)

    @property
    def agent_type(self) -> str:
        """Return the agent type."""
        return self.get_data("agent_type")

    @property
    def agent_profile_id(self) -> str:
        """Return the agent profile id."""
        return str(self._private_fields.get("id", None))
    
    @agent_profile_id.setter
    def agent_profile_id(self, id: str):
        """Set the agent profile id."""
        self._private_fields["id"] = id




    def get_data(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a field value from the profile with optional default value.
        Supports dot notation for accessing nested fields (e.g., "user.name").
        Also supports list access by index (e.g., "users.0.name").
        
        Args:
            key (str): The key of the field to retrieve. Can use dot notation for nested fields.
            default (Optional[Any], optional): Default value to return if the key is not found
                                        or the value is None. Defaults to None.
        
        Returns:
            Any: The value of the field if found and not None, otherwise the default value.
        """
        # 首先根据第一个键从公共或私有字段获取初始值
        parts = key.split('.')
        current_key = parts[0]
        
        # 获取初始值
        if current_key in self._public_fields:
            value = self._public_fields[current_key]
        elif current_key in self._private_fields:
            value = self._private_fields[current_key]
        else:
            return default
        
        # 处理值为None的情况
        if value is None:
            return default
        
        # 循环处理剩余的路径部分
        for part in parts[1:]:
            if isinstance(value, dict):
                # 字典访问
                if part in value:
                    value = value[part]
                else:
                    return default  # 键不存在
            else:
                # 无法继续遍历
                return default
            
            # 在每一步检查值是否为None
            if value is None:
                return default
        
        return value

    def update_data(self, key: str, data: Any) -> bool:
        """
        Update a field value in the profile.
        
        Args:
            key (str): The key of the field to update.
            data (Any): The new value to set.
        
        Returns:
            bool: True if the update was successful, False otherwise.
        """
        if hasattr(self, '_private_fields') and key in self._private_fields:
            self._private_fields[key] = data
            return True
        
        if hasattr(self, '_public_fields'):
            self._public_fields[key] = data
            return True
        
        if not hasattr(self, '_public_fields'):
            super().__setattr__('_public_fields', {})
        self._public_fields[key] = data
        return True


    def update_if(self, key, condition_func, update_func):
        """条件更新
        Args:
            key: 要更新的键
            condition_func: 判断条件的函数,接收当前值返回bool
            update_func: 更新函数,接收当前值返回新值
        """
        with self.lock:
            current_value = self.get_data(key)
            if condition_func(current_value):
                new_value = update_func(current_value)
                self.update_data(key, new_value)
                return True, new_value
            return False, current_value

    # def __getattr__(self, name: str) -> Any:
    #     """
    #     Provide dynamic property-like access to profile fields.
    #     Falls back to get_data() with no default value for compatibility.
        
    #     Args:
    #         name (str): Name of the field to access
            
    #     Returns:
    #         Any: Value of the requested field
            
    #     Raises:
    #         AttributeError: If the field doesn't exist or value is None
    #     """
    #     value = self.get_data(name)
    #     # if value is None:
    #     #     raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")
    #     return value

    def __setattr__(self, name: str, value: Any):
        """
        Provide dynamic property-like setting of profile fields.
        Falls back to update_data() for field updates.
        
        Args:
            name (str): Name of the field to set
            value (Any): Value to set
        """
        # Allow setting of special attributes which should not go into profile fields.
        if name in ['schema', '_public_fields', '_private_fields', 'lock']:
            super().__setattr__(name, value)
            return

        # For all other attributes, use update_data to handle them as profile fields.
        self.update_data(name, value)