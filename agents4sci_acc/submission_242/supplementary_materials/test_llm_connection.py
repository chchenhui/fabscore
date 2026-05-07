#!/usr/bin/env python3
"""
Test LLM API connection for SimulEval++

This script tests the connection to your configured LLM provider before running the full simulation.
Supports both standard OpenAI API and UMAMI proxy.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from config.llm_config import LLMConfigManager, create_openai_client

def test_llm_connection():
    """Test connection to configured LLM provider"""
    
    try:
        # Get configuration
        config = LLMConfigManager.get_config()
        print(f"🔄 Testing {config.provider} connection...")
        print(f"   Model: {config.model}")
        if config.base_url:
            print(f"   Endpoint: {config.base_url}")
        
        # Create client and test
        client = create_openai_client(config)
        
        completion = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Say '{config.provider.upper()} connection successful!' if you can read this."}
            ],
            max_tokens=50
        )
        
        response = completion.choices[0].message.content.strip()
        print(f"✅ Connection successful!")
        print(f"📝 Response: {response}")
        print(f"🔢 Tokens used: {completion.usage.total_tokens}")
        
        return True
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📋 Setup Instructions:")
        print("1. Set OPENAI_API_KEY environment variable, or")
        print("2. Set OTHER_API_KEY environment variable, or") 
        print("3. Copy config/private_config.example.py to config/private_config.py and configure")
        return False
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def main():
    """Run connection test"""
    print("🚀 LLM Connection Test for SimulEval++")
    print("=" * 50)
    
    success = test_llm_connection()
    
    if success:
        print("\n🎉 Ready to run SimulEval++!")
        print("Run: python simuleval_core.py")
    else:
        print("\n❌ Please fix connection issues before running SimulEval++")
    
    return success

if __name__ == "__main__":
    main()
