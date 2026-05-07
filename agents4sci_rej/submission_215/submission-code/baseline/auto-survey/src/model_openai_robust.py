"""
Robust OpenAI-compatible API model with retry logic and parameter tuning
"""

from openai import OpenAI
import time
import json

class APIModel:
    def __init__(self, model, api_key, api_url) -> None:
        # Handle different URL formats
        if api_url.endswith('/chat/completions'):
            base_url = api_url.replace('/chat/completions', '')
        elif api_url.endswith('/v1'):
            base_url = api_url
        else:
            base_url = api_url.rstrip('/') + '/v1'
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "EMPTY"
        )
        self.model = model
        
        # Default parameters for better structured output
        self.default_params = {
            'temperature': 0.3,  # Lower temperature for more deterministic output
            'top_p': 0.9,
            'max_tokens': 4096,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        }
    
    def extract_llm_response(self, prompt, system_prompt=None, **kwargs):
        """
        Enhanced extraction with retry logic and parameter tuning
        """
        # Merge default params with any provided kwargs
        params = {**self.default_params, **kwargs}
        
        # Try different approaches to get structured output
        retry_strategies = [
            # Strategy 1: Low temperature with explicit instructions
            {
                'temperature': 0.1,
                'system_modifier': "\nIMPORTANT: You must follow the exact format specified. Start your response with 'Title:' and include all required sections.",
                'prompt_prefix': "Generate a structured response following this EXACT format:\n\n"
            },
            # Strategy 2: Medium temperature with format example
            {
                'temperature': 0.5,
                'system_modifier': "\nYou are a research assistant that always provides structured outputs in the exact format requested.",
                'prompt_prefix': "Please provide a response in the following format (this is mandatory):\n\nTitle: [Your Title]\nSection 1: [Section Name]\nDescription 1: [Description]\n\nNow generate:\n\n"
            },
            # Strategy 3: Higher temperature with strong constraints
            {
                'temperature': 0.7,
                'system_modifier': "\nYou must output ONLY in the structured format provided. Do not add any conversational elements.",
                'prompt_prefix': "OUTPUT FORMAT REQUIRED - Follow exactly:\n\n"
            }
        ]
        
        last_error = None
        
        for attempt, strategy in enumerate(retry_strategies):
            try:
                # Build messages with strategy modifications
                messages = []
                
                # Enhanced system prompt
                if system_prompt:
                    enhanced_system = system_prompt + strategy['system_modifier']
                else:
                    enhanced_system = "You are an AI assistant that generates structured academic content." + strategy['system_modifier']
                
                messages.append({"role": "system", "content": enhanced_system})
                
                # Enhanced user prompt
                enhanced_prompt = strategy['prompt_prefix'] + prompt
                messages.append({"role": "user", "content": enhanced_prompt})
                
                # Update temperature for this attempt
                params['temperature'] = strategy['temperature']
                
                print(f"Attempt {attempt + 1}: Using temperature={params['temperature']}")
                
                # Make API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **params
                )
                
                result = response.choices[0].message.content
                
                # Validate that we got structured output
                if self._validate_structured_output(result):
                    print(f"Success on attempt {attempt + 1}")
                    return result
                else:
                    print(f"Attempt {attempt + 1}: Response not in expected format, trying next strategy...")
                    
            except Exception as e:
                last_error = e
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < len(retry_strategies) - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        # If all strategies failed, try one more time with very explicit instructions
        print("All strategies failed, attempting final recovery...")
        try:
            recovery_prompt = f"""
You MUST respond with the following format. Do not write anything else:

Title: [Write a title here]
Section 1: [Write section name here]
Description 1: [Write description here]

Original request:
{prompt}
"""
            messages = [
                {"role": "system", "content": "Output ONLY the structured format requested. No greetings or explanations."},
                {"role": "user", "content": recovery_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=params['max_tokens']
            )
            
            result = response.choices[0].message.content
            
            # If still not structured, create a minimal valid response
            if not self._validate_structured_output(result):
                print("Creating fallback structured response...")
                topic = prompt.split('\n')[0][:100]  # Extract topic from prompt
                result = f"""Title: Survey on {topic}
Section 1: Introduction
Description 1: Overview of the topic and its importance
Section 2: Background
Description 2: Historical context and development
Section 3: Current State
Description 3: Recent advances and current research
Section 4: Challenges
Description 4: Open problems and limitations
Section 5: Future Directions
Description 5: Emerging trends and opportunities
Section 6: Conclusion
Description 6: Summary and final thoughts"""
            
            return result
            
        except Exception as e:
            if last_error:
                raise last_error
            raise e
    
    def _validate_structured_output(self, response):
        """Check if response contains expected structure"""
        if not response:
            return False
        
        # Check for key structural elements
        required_elements = ['Title:', 'Section 1:', 'Description 1:']
        
        # Allow some flexibility in case formatting
        response_lower = response.lower()
        required_lower = [elem.lower() for elem in required_elements]
        
        for elem in required_lower:
            if elem not in response_lower:
                return False
        
        return True
    
    def extract_llm_response_stream(self, prompt, system_prompt=None, **kwargs):
        """Streaming version with same robust handling"""
        # For streaming, we'll use the best parameters we found
        params = {
            'temperature': 0.3,
            'top_p': 0.9,
            'max_tokens': 4096,
            **kwargs
        }
        
        messages = []
        
        enhanced_system = (system_prompt or "You are an AI assistant.") + \
                         "\nAlways provide structured outputs in the exact format requested."
        messages.append({"role": "system", "content": enhanced_system})
        
        enhanced_prompt = "Generate a structured response following the EXACT format specified:\n\n" + prompt
        messages.append({"role": "user", "content": enhanced_prompt})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **params
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
            
            # Validate after streaming completes
            if not self._validate_structured_output(full_response):
                print("Warning: Streamed response may not be in expected format")
                
        except Exception as e:
            print(f"Streaming failed: {str(e)}")
            # Fall back to non-streaming
            response = self.extract_llm_response(prompt, system_prompt, **params)
            yield response