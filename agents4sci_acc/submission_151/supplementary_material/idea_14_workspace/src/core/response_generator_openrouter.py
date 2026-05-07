import requests
import logging
import time
import yaml
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenRouterResponseGenerator:
    def __init__(self, api_key: str, config_path: str = "configs/project_config.yaml"):
        """
        Initializes the OpenRouter response generator.
        
        Args:
            api_key (str): OpenRouter API key
            config_path (str): Path to project config file
        """
        self.api_key = api_key
        
        # Load config
        self.config = self._load_config(config_path)
        self.base_url = self.config.get('openrouter', {}).get('api_base_url', 'https://openrouter.ai/api/v1')
        self.model_mappings = self.config.get('openrouter', {}).get('model_mappings', {})
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/alignment-research",
            "X-Title": "AI Alignment Research - Semantic Entropy"
        }
        logging.info(f"OpenRouter response generator initialized with {len(self.model_mappings)} model mappings")
    
    def _load_config(self, config_path: str) -> dict:
        """Load project configuration"""
        # Try multiple possible paths for config
        possible_paths = [
            config_path,
            f"/{config_path}",  # Absolute path
            f"/root/{config_path}",  # Modal container path
            "/configs/project_config.yaml",  # Modal mounted path
            "../configs/project_config.yaml",  # Relative path
            "/mnt/storage/configs/project_config.yaml"  # Volume path
        ]
        
        for path in possible_paths:
            try:
                with open(path, 'r') as file:
                    config = yaml.safe_load(file)
                    logging.info(f"Successfully loaded config from {path}")
                    return config
            except Exception:
                continue
                
        logging.warning(f"Could not load config from any of {possible_paths}")
        return {}
    
    def get_openrouter_model_name(self, original_model: str) -> str:
        """Map original model names to OpenRouter model names"""
        mapped_model = self.model_mappings.get(original_model, original_model)
        if mapped_model != original_model:
            logging.info(f"Mapped {original_model} -> {mapped_model}")
        return mapped_model
    
    def _make_request(self, model_name: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
        """Make a single API request to OpenRouter"""
        payload = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logging.error(f"Unexpected API response format: {result}")
                return ""
                
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return ""
        except Exception as e:
            logging.error(f"Error processing response: {e}")
            return ""
    
    def generate_responses(self, prompt: str, model_name: str, n: int, temperature: float, top_p: float, max_new_tokens: int) -> List[str]:
        """
        Generates N responses for a given prompt using OpenRouter API.

        Args:
            prompt (str): The input prompt for the model.
            model_name (str): The OpenRouter model identifier (e.g., 'meta-llama/llama-3.1-70b-instruct')
            n (int): The number of responses to generate.
            temperature (float): The value used to control randomness in sampling.
            top_p (float): The cumulative probability for nucleus sampling.
            max_new_tokens (int): The maximum number of tokens to generate.

        Returns:
            List[str]: A list of N generated responses.
        """
        # Map model name using config
        openrouter_model = self.get_openrouter_model_name(model_name)
        
        logging.info(f"Generating {n} responses for model {openrouter_model}")
        logging.info(f"Prompt: {prompt[:100]}...")
        
        responses = []
        for i in range(n):
            logging.info(f"Generating response {i+1}/{n}")
            
            response = self._make_request(
                model_name=openrouter_model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens
            )
            
            if response:
                responses.append(response)
                logging.info(f"Response {i+1} generated successfully ({len(response)} chars)")
                logging.info(f"Response {i+1} content: {response[:200]}{'...' if len(response) > 200 else ''}")
            else:
                logging.warning(f"Failed to generate response {i+1}")
                responses.append("")  # Add empty response to maintain count
            
            # Small delay to be respectful to API limits
            if i < n - 1:
                time.sleep(0.5)
        
        logging.info(f"Generated {len([r for r in responses if r])} successful responses out of {n}")
        return responses

# Model mappings are now loaded from project_config.yaml

if __name__ == '__main__':
    # Example usage
    import os
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("Please set OPENROUTER_API_KEY environment variable")
        exit(1)
    
    generator = OpenRouterResponseGenerator(api_key)
    
    prompt = "What is artificial intelligence?"
    model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"  # Will be mapped to OpenRouter format
    
    responses = generator.generate_responses(
        prompt=prompt,
        model_name=model, 
        n=3,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=100
    )
    
    print(f"\nPrompt: {prompt}")
    for i, response in enumerate(responses):
        print(f"\nResponse {i+1}: {response}")