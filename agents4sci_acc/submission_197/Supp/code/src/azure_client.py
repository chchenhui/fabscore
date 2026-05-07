"""
Minimal Azure OpenAI client wrapper.
Just the essentials: init, chat(), chat_to_json(), structured_output()
"""
import os
import json
from typing import Optional, Dict, Any, List, TypeVar, Type
from openai import AzureOpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

# Type variable for structured output
T = TypeVar('T', bound=BaseModel)


class AzureOpenAIClient:
    """Minimal wrapper for Azure OpenAI API."""
    
    def __init__(self):
        """Initialize with environment variables."""
        self.client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 30000) -> str:
        """
        Simple chat completion.
        
        Args:
            messages: List of {"role": "user/system/assistant", "content": "..."}
            temperature: 0.0-1.0 randomness
            max_tokens: Max response length
            
        Returns:
            String response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Chat error: {e}")
            return f"Error: {str(e)}"
    
    def chat_to_json(self, messages: List[Dict[str, str]], temperature: float = 0.1, max_tokens: int = 30000) -> Dict[str, Any]:
        """
        Chat completion with JSON mode - forces structured output.
        
        Args:
            messages: List of messages (system message should specify JSON format)
            temperature: Lower for more consistent JSON
            max_tokens: Max response length
            
        Returns:
            Parsed JSON dict, or error dict if parsing fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}  # Force JSON mode
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {"error": "invalid_json", "raw_content": content}
        except Exception as e:
            print(f"Chat JSON error: {e}")
            return {"error": "api_error", "message": str(e)}
    
    def structured_output(self, messages: List[Dict[str, str]], response_model: Type[T], 
                         temperature: float = 0.1, max_tokens: int = 30000) -> T:
        """
        Chat completion with structured Pydantic output - guaranteed type safety.
        
        Args:
            messages: List of messages 
            response_model: Pydantic model class for the expected response
            temperature: Lower for more consistent results
            max_tokens: Max response length
            
        Returns:
            Parsed Pydantic model instance
        """
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.deployment_name,
                messages=messages,
                response_format=response_model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.parsed
            
        except Exception as e:
            print(f"Structured output error: {e}")
            # Return a default instance with error info
            if hasattr(response_model, 'model_validate'):
                try:
                    # Try to create a minimal valid instance
                    return response_model.model_validate({"error": str(e)})
                except:
                    # If that fails, let the exception bubble up
                    raise e
            else:
                raise e


# Pydantic models for common boids patterns
class AgentDecision(BaseModel):
    """Standard agent decision structure."""
    action: str  # build_tool, use_tool, rest
    tool_type: Optional[str] = None  # data, logic, utility, code
    confidence: float  # 0.0-1.0
    reason: str  # brief explanation
    target_tool: Optional[str] = None  # for use_tool actions

class ToolCreation(BaseModel):
    """Tool creation specification."""
    name: str
    tool_type: str  # data, logic, utility, code
    description: str
    dependencies: List[str] = []
    code_outline: str  # what the tool should do

class NetworkAnalysis(BaseModel):
    """Agent network analysis."""
    agent_id: str
    specialization: Optional[str] = None
    energy: int
    tool_count: int
    collaboration_score: float  # 0.0-1.0


def main():
    """Test the Azure client with examples."""
    
    print("🧪 Testing Minimal Azure OpenAI Client")
    print("=" * 50)
    
    # Check if environment variables are set
    required_vars = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        print("   Set them in .env file or environment")
        return
    
    client = AzureOpenAIClient()
    
    # Test 1: Simple chat
    print("\n🗨️  Test 1: Simple Chat")
    print("-" * 30)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is 2+2? Answer in one sentence."}
    ]
    
    response = client.chat(messages)
    print(f"Response: {response}")
    
    # Test 2: JSON mode (legacy)
    print("\n📝 Test 2: JSON Mode (Legacy)")
    print("-" * 30)
    
    json_messages = [
        {"role": "system", "content": """You must respond with valid JSON in this exact format:
{
    "action": "build_tool" or "use_tool" or "rest",
    "tool_type": "data" or "logic" or "utility" or "code",
    "confidence": 0.0 to 1.0,
    "reason": "brief explanation"
}"""},
        {"role": "user", "content": "I want to create a tool that processes text data. What should I do?"}
    ]
    
    json_response = client.chat_to_json(json_messages)
    print(f"JSON Response: {json.dumps(json_response, indent=2)}")
    
    # Test 3: Structured output (NEW!)
    print("\n🎯 Test 3: Structured Output (Pydantic)")
    print("-" * 30)
    
    structured_messages = [
        {"role": "system", "content": "You are a boids agent making decisions. Analyze the situation and decide what to do."},
        {"role": "user", "content": "I see my neighbors building data processing tools. I have low energy. What should I do?"}
    ]
    
    try:
        structured_response = client.structured_output(structured_messages, AgentDecision)
        print(f"Structured Response:")
        print(f"  Action: {structured_response.action}")
        print(f"  Tool Type: {structured_response.tool_type}")
        print(f"  Confidence: {structured_response.confidence}")
        print(f"  Reason: {structured_response.reason}")
        print(f"  Target Tool: {structured_response.target_tool}")
    except Exception as e:
        print(f"Structured output failed: {e}")
    
    # Test 4: Tool creation specification
    print("\n🔧 Test 4: Tool Creation Spec")
    print("-" * 30)
    
    tool_messages = [
        {"role": "system", "content": "You are designing a new computational tool. Specify exactly what it should do."},
        {"role": "user", "content": "Create a tool that can analyze text sentiment and extract key phrases."}
    ]
    
    try:
        tool_spec = client.structured_output(tool_messages, ToolCreation)
        print(f"Tool Specification:")
        print(f"  Name: {tool_spec.name}")
        print(f"  Type: {tool_spec.tool_type}")
        print(f"  Description: {tool_spec.description}")
        print(f"  Dependencies: {tool_spec.dependencies}")
        print(f"  Code Outline: {tool_spec.code_outline}")
    except Exception as e:
        print(f"Tool creation spec failed: {e}")
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main() 