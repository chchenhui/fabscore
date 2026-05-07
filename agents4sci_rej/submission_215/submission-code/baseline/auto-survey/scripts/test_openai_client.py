#!/usr/bin/env python3
"""Test OpenAI client with vLLM"""

from openai import OpenAI

# Test with vLLM
client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY"  # any non-empty string works for vLLM
)

print("Testing OpenAI client with vLLM...")
print("-" * 40)

try:
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'vLLM is working!' if you can read this."}
        ],
        temperature=0.2,
    )
    
    print("✓ Success!")
    print(f"Response: {resp.choices[0].message.content}")
    
    # Test the APIModel wrapper
    print("\n" + "-" * 40)
    print("Testing APIModel wrapper...")
    
    import sys
    sys.path.insert(0, '.')
    from src.model_openai import APIModel
    
    model = APIModel(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key="EMPTY",
        api_url="http://127.0.0.1:8000/v1"
    )
    
    response = model.chat("Say 'APIModel is working!' if you can read this.", temperature=0.2)
    print(f"✓ APIModel response: {response}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()