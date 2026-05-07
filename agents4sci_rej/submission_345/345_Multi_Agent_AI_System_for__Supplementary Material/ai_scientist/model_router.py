"""
Multi-LLM Model Router - Following agent_llmusage.md routing policy exactly
Implements provider-agnostic interface with intelligent task routing
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

# Load environment config
from dotenv import load_dotenv
load_dotenv()

class TaskType(Enum):
    """Task types for intelligent routing"""
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    COMPLEX_REASONING = "complex_reasoning"
    BULK_PARSING = "bulk_parsing" 
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    OBJECTIVE_REVIEW = "objective_review"
    LONG_CONTEXT = "long_context"
    MULTIMODAL = "multimodal"

@dataclass
class UsageStats:
    """Track API usage and costs"""
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_cents: float
    timestamp: datetime
    task_type: str

@dataclass
class ModelResponse:
    """Standardized response from any model"""
    content: str
    model_used: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_cents: float
    citations: Optional[List[str]] = None
    confidence_score: Optional[float] = None

class ModelRouter:
    """
    Intelligent router following agent_llmusage.md policy:
    
    - Hypothesis/complex reasoning → GPT-5
    - Bulk parse/format/routing → DeepSeek V3.1  
    - Objective review → Perplexity Sonar
    - Long context → Gemini 2.5 Pro or Claude Sonnet 4
    """
    
    def __init__(self):
        self.usage_stats: List[UsageStats] = []
        self.daily_spend_cents = 0
        self.last_reset = datetime.now().date()
        
        # Load config from environment
        self.config = self._load_config()
        
        # Initialize providers
        self.providers = self._initialize_providers()
        
        print(f"[MODEL ROUTER] Initialized with providers: {list(self.providers.keys())}")
        print(f"[PRIMARY] {self.config['primary_provider']} | [SEARCH] {self.config['search_provider']}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from .env file"""
        return {
            'primary_provider': os.getenv('PRIMARY_PROVIDER', 'openai'),
            'search_provider': os.getenv('SEARCH_PROVIDER', 'openai'),
            'enable_perplexity_review': os.getenv('ENABLE_PERPLEXITY_REVIEW', 'false').lower() == 'true',
            'daily_budget_cents': int(os.getenv('DAILY_BUDGET_CENTS', '300')),
            'max_tokens_gpt5': int(os.getenv('MAX_TOKENS_GPT5', '6000')),
            'max_tokens_deepseek': int(os.getenv('MAX_TOKENS_DEEPSEEK', '4000')),
            'max_tokens_review': int(os.getenv('MAX_TOKENS_REVIEW', '3000')),
            'enable_usage_logs': os.getenv('ENABLE_USAGE_LOGS', 'true').lower() == 'true',
            'fallback_on_error': os.getenv('FALLBACK_ON_PROVIDER_ERROR', 'true').lower() == 'true'
        }
    
    def _initialize_providers(self) -> Dict[str, Any]:
        """Initialize all configured providers"""
        providers = {}
        
        # OpenAI (GPT-5 for complex reasoning)
        if os.getenv('OPENAI_API_KEY'):
            try:
                import openai
                providers['openai'] = {
                    'client': openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')),
                    'model': os.getenv('OPENAI_MODEL', 'gpt-5-2025-08-07'),  # GPT-5 model
                    'use_web_search': os.getenv('OPENAI_USE_WEB_SEARCH', 'false').lower() == 'true'
                }
                print(f"[PROVIDER] OpenAI initialized: {providers['openai']['model']}")
            except ImportError:
                print(f"[WARNING] OpenAI library not available")
        
        # DeepSeek (bulk processing)
        if os.getenv('DEEPSEEK_API_KEY'):
            try:
                import openai  # DeepSeek uses OpenAI-compatible API
                providers['deepseek'] = {
                    'client': openai.OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
                    ),
                    'model': os.getenv('DEEPSEEK_MODEL', 'deepseek-chat'),
                    'enable_cache': os.getenv('DEEPSEEK_ENABLE_CACHE', 'true').lower() == 'true'
                }
                print(f"[PROVIDER] DeepSeek initialized: {providers['deepseek']['model']}")
            except ImportError:
                print(f"[WARNING] DeepSeek client not available")
        
        # Perplexity (objective review with citations)
        if os.getenv('PERPLEXITY_API_KEY'):
            try:
                import openai  # Perplexity also uses OpenAI-compatible API
                providers['perplexity'] = {
                    'client': openai.OpenAI(
                        api_key=os.getenv('PERPLEXITY_API_KEY'),
                        base_url=os.getenv('PERPLEXITY_BASE_URL', 'https://api.perplexity.ai')
                    ),
                    'model': os.getenv('PERPLEXITY_MODEL', 'sonar-reasoning-pro')
                }
                print(f"[PROVIDER] Perplexity initialized: {providers['perplexity']['model']}")
            except ImportError:
                print(f"[WARNING] Perplexity client not available")
        
        # Anthropic (long context fallback)
        if os.getenv('ANTHROPIC_API_KEY'):
            try:
                import anthropic
                providers['anthropic'] = {
                    'client': anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')),
                    'model': os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')
                }
                print(f"[PROVIDER] Anthropic initialized: {providers['anthropic']['model']}")
            except ImportError:
                print(f"[WARNING] Anthropic library not available")
        
        # Google (very long context)
        if os.getenv('GOOGLE_API_KEY'):
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
                providers['google'] = {
                    'client': genai,
                    'model': os.getenv('GEMINI_MODEL', 'gemini-1.5-pro')  # fallback to available model
                }
                print(f"[PROVIDER] Google initialized: {providers['google']['model']}")
            except ImportError:
                print(f"[WARNING] Google AI library not available")
        
        return providers
    
    def route_task(self, task_type: TaskType, context_length: Optional[int] = None) -> str:
        """
        Route task to appropriate model following agent_llmusage.md policy
        """
        
        # Check daily budget
        if self._check_daily_budget():
            return self._get_fallback_provider()
        
        # Apply routing policy exactly as specified
        if task_type in [TaskType.HYPOTHESIS_GENERATION, TaskType.COMPLEX_REASONING]:
            # "Hypothesis / design / complex reasoning → GPT-5"
            return 'openai' if 'openai' in self.providers else self._get_fallback_provider()
        
        elif task_type in [TaskType.BULK_PARSING, TaskType.CLASSIFICATION, TaskType.SUMMARIZATION]:
            # "Bulk parse/format/routing/summarize → DeepSeek V3.1" 
            return 'deepseek' if 'deepseek' in self.providers else self._get_fallback_provider()
        
        elif task_type == TaskType.OBJECTIVE_REVIEW:
            # "Objective review w/ citations & score → Perplexity Sonar"
            if self.config['enable_perplexity_review'] and 'perplexity' in self.providers:
                return 'perplexity'
            else:
                return self._get_fallback_provider()
        
        elif task_type == TaskType.LONG_CONTEXT or (context_length and context_length > 200000):
            # "Very long ctx → Claude Sonnet 4 1M (Gemini fallback but unreliable on free tier)"
            if 'anthropic' in self.providers:
                return 'anthropic'  # Claude 1M context - more reliable
            elif 'google' in self.providers:
                return 'google'     # Gemini 2M context but unreliable on free tier
            else:
                return self._get_fallback_provider()
        
        else:
            # Default to primary provider
            return self.config['primary_provider'] if self.config['primary_provider'] in self.providers else self._get_fallback_provider()
    
    def generate(self, prompt: str, task_type: TaskType, system_prompt: Optional[str] = None, 
                max_tokens: Optional[int] = None, temperature: float = 0.7,
                context_length: Optional[int] = None, response_format: Optional[Dict[str, str]] = None) -> ModelResponse:
        """
        Generate response using appropriate model based on task type
        """
        
        # Route to appropriate provider
        provider = self.route_task(task_type, context_length)
        
        try:
            return self._call_provider(provider, prompt, system_prompt, max_tokens, temperature, task_type, response_format)
        except Exception as e:
            error_str = str(e).lower()
            print(f"[ERROR] {provider} failed: {e}")
            
            # Special case: Google quota exceeded → Claude for long context
            if provider == 'google' and task_type == TaskType.LONG_CONTEXT and ('quota' in error_str or 'limit' in error_str):
                if 'anthropic' in self.providers:
                    print(f"[GOOGLE QUOTA] Falling back to Claude for long context")
                    return self._call_provider('anthropic', prompt, system_prompt, max_tokens, temperature, task_type, response_format)
            
            # General fallback if enabled
            if self.config['fallback_on_error']:
                fallback_provider = self._get_fallback_provider(exclude=provider)
                print(f"[FALLBACK] Using {fallback_provider}")
                return self._call_provider(fallback_provider, prompt, system_prompt, max_tokens, temperature, task_type, response_format)
            else:
                raise e
    
    def _call_provider(self, provider: str, prompt: str, system_prompt: Optional[str],
                      max_tokens: Optional[int], temperature: float, task_type: TaskType, 
                      response_format: Optional[Dict[str, str]] = None) -> ModelResponse:
        """Call specific provider"""
        
        if provider not in self.providers:
            raise ValueError(f"Provider {provider} not available")
        
        start_time = time.time()
        
        if provider == 'openai':
            return self._call_openai(prompt, system_prompt, max_tokens, temperature, task_type, response_format)
        elif provider == 'deepseek':
            return self._call_deepseek(prompt, system_prompt, max_tokens, temperature, task_type)
        elif provider == 'perplexity':
            return self._call_perplexity(prompt, system_prompt, max_tokens, temperature, task_type)
        elif provider == 'anthropic':
            return self._call_anthropic(prompt, system_prompt, max_tokens, temperature, task_type)
        elif provider == 'google':
            return self._call_google(prompt, system_prompt, max_tokens, temperature, task_type)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _call_openai(self, prompt: str, system_prompt: Optional[str], max_tokens: Optional[int], 
                    temperature: float, task_type: TaskType, response_format: Optional[Dict[str, str]] = None) -> ModelResponse:
        """Call OpenAI GPT-5 (gpt-5-2025-08-07) with optional web search"""
        
        client = self.providers['openai']['client']
        model = self.providers['openai']['model']
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Use web search tool if enabled and appropriate
        tools = None
        if (self.providers['openai']['use_web_search'] and 
            task_type in [TaskType.HYPOTHESIS_GENERATION, TaskType.COMPLEX_REASONING]):
            tools = [{"type": "web_search"}] if hasattr(client, 'web_search') else None
        
        # GPT-5 only supports temperature=1.0
        gpt5_temperature = 1.0 if "gpt-5" in model else temperature
        
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": messages,
            "max_completion_tokens": max_tokens or self.config['max_tokens_gpt5'],
            "temperature": gpt5_temperature
        }
        
        # Add structured output format if requested
        if response_format:
            api_params["response_format"] = response_format
            
        # Add tools if available
        if tools:
            api_params["tools"] = tools
            
        response = client.chat.completions.create(**api_params)
        
        content = response.choices[0].message.content
        if content is None:
            print(f"[DEBUG] OpenAI returned None content. Response: {response}")
            print(f"[DEBUG] Choices: {response.choices}")
            print(f"[DEBUG] Message: {response.choices[0].message}")
            content = ""  # Fallback to empty string
        
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # OpenAI pricing: $1.25/M input, $10/M output
        cost_cents = (input_tokens * 0.125 + output_tokens * 1.0) / 1000
        
        self._log_usage('openai', model, input_tokens, output_tokens, cost_cents, task_type.value)
        
        return ModelResponse(
            content=content,
            model_used=model,
            provider='openai',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_cents=cost_cents
        )
    
    def _call_deepseek(self, prompt: str, system_prompt: Optional[str], max_tokens: Optional[int],
                      temperature: float, task_type: TaskType) -> ModelResponse:
        """Call DeepSeek for bulk processing with context caching"""
        
        print(f"[DEEPSEEK] Starting DeepSeek API call...")
        print(f"[DEEPSEEK] Task type: {task_type.value}")
        print(f"[DEEPSEEK] Prompt length: {len(prompt)}")
        print(f"[DEEPSEEK] System prompt length: {len(system_prompt) if system_prompt else 0}")
        
        client = self.providers['deepseek']['client']
        model = self.providers['deepseek']['model']
        
        print(f"[DEEPSEEK] Using model: {model}")
        print(f"[DEEPSEEK] Max tokens: {max_tokens or self.config['max_tokens_deepseek']}")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        print(f"[DEEPSEEK] Messages prepared: {len(messages)} messages")
        print(f"[DEEPSEEK] About to call client.chat.completions.create...")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens or self.config['max_tokens_deepseek'],
                temperature=temperature
            )
            
            print(f"[DEEPSEEK] API call successful!")
            print(f"[DEEPSEEK] Response received, extracting content...")
            
            content = response.choices[0].message.content
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            
            print(f"[DEEPSEEK] Content extracted: {len(content)} chars")
            print(f"[DEEPSEEK] Tokens: {input_tokens} input, {output_tokens} output")
            
            # DeepSeek pricing: $0.56/M input, $1.68/M output (cache miss)
            cost_cents = (input_tokens * 0.056 + output_tokens * 0.168) / 100
            
            self._log_usage('deepseek', model, input_tokens, output_tokens, cost_cents, task_type.value)
            
            print(f"[DEEPSEEK] Returning ModelResponse...")
            
            return ModelResponse(
                content=content,
                model_used=model,
                provider='deepseek',
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_cents=cost_cents
            )
            
        except Exception as e:
            print(f"[DEEPSEEK] API call failed: {e}")
            print(f"[DEEPSEEK] Error type: {type(e).__name__}")
            raise e
    
    def _call_perplexity(self, prompt: str, system_prompt: Optional[str], max_tokens: Optional[int],
                        temperature: float, task_type: TaskType) -> ModelResponse:
        """Call Perplexity for objective review with citations"""
        
        client = self.providers['perplexity']['client']
        model = self.providers['perplexity']['model']
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens or self.config['max_tokens_review'],
            temperature=temperature
        )
        
        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        # Perplexity Sonar Reasoning Pro: $2/M input, $8/M output  
        cost_cents = (input_tokens * 0.2 + output_tokens * 0.8) / 100
        
        # Extract citations if present
        citations = self._extract_citations(content)
        
        self._log_usage('perplexity', model, input_tokens, output_tokens, cost_cents, task_type.value)
        
        return ModelResponse(
            content=content,
            model_used=model,
            provider='perplexity',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_cents=cost_cents,
            citations=citations
        )
    
    def _call_anthropic(self, prompt: str, system_prompt: Optional[str], max_tokens: Optional[int],
                       temperature: float, task_type: TaskType) -> ModelResponse:
        """Call Anthropic Claude for long context"""
        
        client = self.providers['anthropic']['client']
        model = self.providers['anthropic']['model']
        
        messages = [{"role": "user", "content": prompt}]
        
        response = client.messages.create(
            model=model,
            system=system_prompt or "",
            messages=messages,
            max_tokens=max_tokens or 4000,
            temperature=temperature
        )
        
        content = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        # Claude Sonnet 4: $3/M input, $15/M output (standard context)
        cost_cents = (input_tokens * 0.3 + output_tokens * 1.5) / 100
        
        self._log_usage('anthropic', model, input_tokens, output_tokens, cost_cents, task_type.value)
        
        return ModelResponse(
            content=content,
            model_used=model,
            provider='anthropic',
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_cents=cost_cents
        )
    
    def _call_google(self, prompt: str, system_prompt: Optional[str], max_tokens: Optional[int],
                    temperature: float, task_type: TaskType) -> ModelResponse:
        """Call Google Gemini for very long context"""
        
        genai = self.providers['google']['client']
        model_name = self.providers['google']['model']
        
        model = genai.GenerativeModel(model_name)
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens or 4000,
                temperature=temperature
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        )
        
        # Handle different response states
        if response.candidates and response.candidates[0].content.parts:
            content = response.text
        elif hasattr(response, 'prompt_feedback') and response.prompt_feedback:
            raise ValueError(f"Gemini prompt blocked: {response.prompt_feedback}")
        else:
            # Fallback content if response is blocked or empty
            content = "Response blocked by content policy. Unable to generate content for this query."
        
        # Approximate token usage (Google doesn't always provide exact counts)
        input_tokens = len(full_prompt.split()) * 1.3  # rough estimate
        output_tokens = len(content.split()) * 1.3
        
        # Gemini 2.5 Pro: $0.625/M input, $5/M output (≤200K context)
        cost_cents = (input_tokens * 0.0625 + output_tokens * 0.5) / 100
        
        self._log_usage('google', model_name, int(input_tokens), int(output_tokens), cost_cents, task_type.value)
        
        return ModelResponse(
            content=content,
            model_used=model_name,
            provider='google',
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            cost_cents=cost_cents
        )
    
    def _get_fallback_provider(self, exclude: Optional[str] = None) -> str:
        """Get fallback provider"""
        available = [p for p in self.providers.keys() if p != exclude]
        
        if not available:
            raise RuntimeError("No providers available")
        
        # Prefer primary provider if available
        if self.config['primary_provider'] in available:
            return self.config['primary_provider']
        
        # Otherwise return first available
        return available[0]
    
    def _check_daily_budget(self) -> bool:
        """Check if daily budget exceeded"""
        today = datetime.now().date()
        
        # Reset daily counter if new day
        if today > self.last_reset:
            self.daily_spend_cents = 0
            self.last_reset = today
        
        return self.daily_spend_cents >= self.config['daily_budget_cents']
    
    def _log_usage(self, provider: str, model: str, input_tokens: int, output_tokens: int,
                  cost_cents: float, task_type: str):
        """Log usage statistics"""
        
        self.daily_spend_cents += cost_cents
        
        if self.config['enable_usage_logs']:
            usage = UsageStats(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_cents=cost_cents,
                timestamp=datetime.now(),
                task_type=task_type
            )
            self.usage_stats.append(usage)
            
            print(f"[USAGE] {provider} | {task_type} | ${cost_cents/100:.4f} | {input_tokens}->{output_tokens} tokens")
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from Perplexity response"""
        # Simple citation extraction - could be enhanced
        citations = []
        lines = content.split('\n')
        for line in lines:
            if line.strip().startswith('[') and ']' in line:
                citations.append(line.strip())
        return citations
    
    def get_usage_report(self) -> Dict[str, Any]:
        """Get comprehensive usage report"""
        
        total_cost = sum(stat.cost_cents for stat in self.usage_stats) if self.usage_stats else 0
        total_calls = len(self.usage_stats)
        
        by_provider = {}
        for stat in self.usage_stats:
            if stat.provider not in by_provider:
                by_provider[stat.provider] = {"calls": 0, "cost_cents": 0}
            by_provider[stat.provider]["calls"] += 1
            by_provider[stat.provider]["cost_cents"] += stat.cost_cents
        
        return {
            "total_cost_cents": total_cost,
            "total_calls": total_calls,
            "daily_spend_cents": self.daily_spend_cents,
            "budget_remaining_cents": self.config['daily_budget_cents'] - self.daily_spend_cents,
            "by_provider": by_provider
        }


# Singleton instance
_router_instance = None

def get_router() -> ModelRouter:
    """Get global router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = ModelRouter()
    return _router_instance


# Demo function
def demo_model_router():
    """Demonstrate model routing following agent_llmusage.md policy"""
    
    router = get_router()
    
    print("=== MODEL ROUTER DEMO ===")
    
    # Test routing decisions
    test_tasks = [
        (TaskType.HYPOTHESIS_GENERATION, "Generate research hypothesis about drug forecasting"),
        (TaskType.BULK_PARSING, "Parse this list of pharmaceutical queries"),
        (TaskType.OBJECTIVE_REVIEW, "Review this analysis for accuracy and provide citations"),
        (TaskType.LONG_CONTEXT, "Analyze this 500K token clinical document", 500000)
    ]
    
    for task_type, description, *args in test_tasks:
        context_length = args[0] if args else None
        provider = router.route_task(task_type, context_length)
        print(f"Task: {task_type.value}")
        print(f"  Description: {description}")
        print(f"  Routed to: {provider}")
        print()
    
    # Show usage report
    report = router.get_usage_report()
    print("Usage Report:")
    print(f"  Daily spend: ${report['daily_spend_cents']/100:.4f}")
    print(f"  Budget remaining: ${report['budget_remaining_cents']/100:.4f}")


if __name__ == "__main__":
    demo_model_router()