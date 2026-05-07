"""Model evaluators for different API providers."""

import os
from typing import Optional, Dict, Any
import time
from .base_evaluator import BaseEvaluator


class OpenAIEvaluator(BaseEvaluator):
    """Evaluator for OpenAI models."""
    
    def __init__(self, model_name: str = "gpt-4", verbose: bool = False, 
                 api_key: Optional[str] = None, **kwargs):
        from .prompt_configs import PromptType
        prompt_type = kwargs.pop("prompt_type", PromptType.STEP_BY_STEP_BOXED)
        super().__init__(model_name, verbose, prompt_type)
        
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=api_key or os.getenv("OPENAI_API_KEY")
            )
        except ImportError:
            raise ImportError("openai package is required for OpenAI models")
        
        self.generation_kwargs = kwargs
    
    def generate_response(self, problem: str) -> str:
        """Generate response using OpenAI API."""
        user_prompt = self.prompt_config["user_template"].format(problem=problem)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompt_config["system_message"]},
                    {"role": "user", "content": user_prompt}
                ],
                **self.generation_kwargs
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return f"Error: {str(e)}"


class AnthropicEvaluator(BaseEvaluator):
    """Evaluator for Anthropic Claude models."""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", verbose: bool = False,
                 api_key: Optional[str] = None, **kwargs):
        from .prompt_configs import PromptType
        prompt_type = kwargs.pop("prompt_type", PromptType.STEP_BY_STEP_BOXED)
        super().__init__(model_name, verbose, prompt_type)
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(
                api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
            )
        except ImportError:
            raise ImportError("anthropic package is required for Claude models")
        
        self.generation_kwargs = kwargs
    
    def generate_response(self, problem: str) -> str:
        """Generate response using Anthropic API."""
        user_prompt = self.prompt_config["user_template"].format(problem=problem)
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                **self.generation_kwargs
            )
            
            return response.content[0].text.strip()
        
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return f"Error: {str(e)}"


class TogetherEvaluator(BaseEvaluator):
    """Evaluator for Together AI models."""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct-Turbo", 
                 verbose: bool = False, api_key: Optional[str] = None, **kwargs):
        from .prompt_configs import PromptType
        prompt_type = kwargs.pop("prompt_type", PromptType.STEP_BY_STEP_BOXED)
        super().__init__(model_name, verbose, prompt_type)
        
        try:
            import together
            self.client = together.Together(
                api_key=api_key or os.getenv("TOGETHER_API_KEY")
            )
        except ImportError:
            raise ImportError("together package is required for Together AI models")
        
        self.generation_kwargs = kwargs
    
    def generate_response(self, problem: str) -> str:
        """Generate response using Together AI API."""
        user_prompt = self.prompt_config["user_template"].format(problem=problem)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.prompt_config["system_message"]},
                    {"role": "user", "content": user_prompt}
                ],
                **self.generation_kwargs
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            self.logger.error(f"Together AI API error: {e}")
            return f"Error: {str(e)}"


class HuggingFaceEvaluator(BaseEvaluator):
    """Evaluator for Hugging Face models (local inference)."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", verbose: bool = False,
                 device: str = "auto", **kwargs):
        from .prompt_configs import PromptType
        prompt_type = kwargs.pop("prompt_type", PromptType.STEP_BY_STEP_BOXED)
        super().__init__(model_name, verbose, prompt_type)
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.device = device
            self.logger.info(f"Loading model {model_name} on {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with optimized settings for Qwen models
            model_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
                "trust_remote_code": True,  # Required for some Qwen models
            }
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            if device != "cuda" or "device_map" not in model_kwargs:
                self.model = self.model.to(device)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
        except ImportError:
            raise ImportError("transformers and torch packages are required for Hugging Face models")
        
        # Set pad_token_id and merge with provided kwargs
        self.generation_kwargs = {
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
    
    def generate_response(self, problem: str) -> str:
        """Generate response using local Hugging Face model."""
        user_prompt = self.prompt_config["user_template"].format(problem=problem)
        
        try:
            import torch  # Import torch here to fix the scope issue
            
            # Handle Qwen3 models with proper chat template and non-thinking mode
            if "Qwen3" in self.model_name or "qwen3" in self.model_name.lower():
                messages = [{"role": "user", "content": user_prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False  # Disable thinking mode for direct arithmetic answers
                )
                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            else:
                # Standard tokenization for non-Qwen3 models
                inputs = self.tokenizer(user_prompt, return_tensors="pt", padding=True, truncation=True)
                if hasattr(self.model, 'device') and self.model.device.type != 'meta':
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                elif self.device != "auto":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_kwargs
                )
            
            # Decode only the new tokens (exclude the input)
            if "Qwen3" in self.model_name or "qwen3" in self.model_name.lower():
                output_ids = outputs[0][len(inputs.input_ids[0]):].tolist()
                response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            else:
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], 
                    skip_special_tokens=True
                )
            
            return response.strip()
        
        except Exception as e:
            self.logger.error(f"Hugging Face model error: {e}")
            return f"Error: {str(e)}"


def create_evaluator(provider: str, model_name: str, **kwargs) -> BaseEvaluator:
    """Factory function to create evaluators."""
    provider = provider.lower()
    
    if provider == "openai":
        return OpenAIEvaluator(model_name, **kwargs)
    elif provider == "anthropic" or provider == "claude":
        return AnthropicEvaluator(model_name, **kwargs)
    elif provider == "together":
        return TogetherEvaluator(model_name, **kwargs)
    elif provider == "huggingface" or provider == "hf":
        return HuggingFaceEvaluator(model_name, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. "
                        f"Supported providers: openai, anthropic, together, huggingface")