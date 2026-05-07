import argparse
import requests
import json
import time as tm
import os



# API endpoints configuration
api_details = {
    "gemini-1.5-pro-002": {
        "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-002:generateContent",
        "headers": {
            "Content-Type": "application/json"
        }
    },
    "claude-3-5-sonnet-20240620": {
        "endpoint": "https://api.anthropic.com/v1/messages",
        "headers": {
            "Content-Type": "application/json"
        }
    }
}


def call_anthropic_api(model, message, max_tokens=2000):
    details = api_details["claude-3-5-sonnet-20240620"]

    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is required for Claude models")

    url = details['endpoint']

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "user", "content": message}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)

    # Raise an exception if the request failed
    response.raise_for_status()
    data = response.json()

    print("DATA: ", data["content"][0]["text"])

    return data["content"][0]["text"]



def call_gemini_api(system_prompt, query):
    """Handles the specific case of calling the Gemini API."""
    details = api_details["gemini-1.5-pro-002"]

    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required for Gemini models")

    # Prepare the payload according to the expected structure of the Gemini API
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": query
                    }
                ]
            }
        ]
    }

    # Build the full URL with the API key
    url = f"{details['endpoint']}?key={api_key}"

    # Make the API request
    response = requests.post(url, headers=details["headers"], json=payload)
    response.raise_for_status()
    data = response.json()

    # Extract and return the generated text (response format as shown in the example)
    return data["candidates"][0]["content"]["parts"][0]["text"]




def call_llm_api(model, system_prompt, query):
    # Special handling for Gemini API due to different request/response format
    if model == "gemini-1.5-pro-002":
        return call_gemini_api(system_prompt, query)

    if model == "claude-3-5-sonnet-20240620":
        return call_anthropic_api(model, query)

    # Retrieve model details
    details = api_details.get(model)
    if not details:
        raise ValueError(f"Model '{model}' is not supported.")

    # Prepare API request payload
    payload = {
        "system_prompt": system_prompt,
        "prompt": query,
        "max_tokens": 2000,  # Adjust according to the API specifications
        "temperature": 0.7  # Adjust according to the API specifications
    }

    # Make the API request for models other than Gemini
    response = requests.post(details["endpoint"], headers=details["headers"], json=payload)
    response.raise_for_status()
    data = response.json()

    # Extract and return the generated text (response field may vary by API)
    return data.get("text", data)  # Change "text" key according to API response format


def main():
    parser = argparse.ArgumentParser(description="Call different LLM APIs using JSON input.")
    parser.add_argument("--model", choices=api_details.keys(), help="The name of the model to use.")
    parser.add_argument("--input_file", type=str, help="Path to the JSON input file.")
    parser.add_argument("--output_file", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()

    # Read the input JSON file
    with open(args.input_file, 'r') as f:
        test_cases = json.load(f)

    results = []

    # Loop through each test case and call the LLM API
    for test_case in test_cases:
        name = test_case["name"]
        system_prompt = test_case["system_prompt"]
        query = test_case["query"]
        print(query)

        try:
            # Measure start time
            start_time = tm.time()

            # Call the LLM API
            result_text = call_llm_api(args.model, system_prompt, query)

            print(result_text)

            # Measure end time and calculate duration
            end_time = tm.time()
            duration = end_time - start_time

            results.append({
                "test_case": {
                    "name": name,
                    "system_prompt": system_prompt,
                    "query": query
                },
                "results": [
                    {
                        "approach": "zero-shot",
                        "result": result_text,
                        "time": duration,
                        "status": "success"
                    }
                ]
            })
        except Exception as e:
            results.append({
                "test_case": {
                    "name": name,
                    "system_prompt": system_prompt,
                    "query": query
                },
                "results": [
                    {
                        "approach": "zero-shot",
                        "result": str(e),
                        "time": None,
                        "status": "error"
                    }
                ]
            })
            print("Sleeping...")
            tm.sleep(5)

    # Save the results to the output JSON file
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
