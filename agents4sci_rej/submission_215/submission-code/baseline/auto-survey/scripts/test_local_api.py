#!/usr/bin/env python3
"""
Test script to verify local OpenAI-compatible API is working
"""

import requests
import json
import sys

def test_local_api(api_url="http://localhost:8000/v1/chat/completions", 
                   model="meta-llama/Llama-3.1-8B-Instruct"):
    """
    Test if local OpenAI-compatible API is accessible and working
    """
    print(f"Testing API: {api_url}")
    print(f"Model: {model}")
    print("-" * 40)
    
    # Test payload
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Say 'API is working' if you can read this."
            }
        ],
        "temperature": 0.7,
        "max_tokens": 50
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy"  # vLLM doesn't need real key
    }
    
    try:
        response = requests.post(api_url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print("✓ API is working!")
            print(f"Response: {content}")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API. Is the server running?")
        print(f"  Make sure vLLM is running at: {api_url}")
        return False
    except requests.exceptions.Timeout:
        print("✗ Request timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Allow custom URL from command line
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1/chat/completions"
    model = sys.argv[2] if len(sys.argv) > 2 else "meta-llama/Llama-3.1-8B-Instruct"
    
    if test_local_api(api_url, model):
        print("\n✓ Your local API is ready for AutoSurvey!")
        print("Run: python run_local_survey.py")
    else:
        print("\n✗ Please start your local OpenAI-compatible server first")
        print("Example with vLLM:")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model meta-llama/Llama-3.1-8B-Instruct \\")
        print("    --port 8000")