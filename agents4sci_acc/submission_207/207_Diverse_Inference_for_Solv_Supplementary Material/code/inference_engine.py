from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizer

from openai import OpenAI
import os
import json
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
# Calculate pricing
pricing_rates = {
    'o1-mini': {'input': 3.00, 'output': 12.00},
    'gpt-4o': {'input': 5.00, 'output': 15.00},
    'gpt-3.5-turbo': {'input': 1.50, 'output': 2.00},
    # Add other models and their rates as needed
}

def calculate_pricing(pricing_rates, model_name, usage):
  # Obtain usage
  # usage = dict(dict(completion).get('usage'))
  prompt_tokens = usage['prompt_tokens']
  completion_tokens = usage['completion_tokens']
  total_tokens = usage['total_tokens']

  # Calculate costs
  model_rates = pricing_rates[model_name]
  input_cost = (prompt_tokens / 1_000_000) * model_rates['input']
  output_cost = (completion_tokens / 1_000_000) * model_rates['output']
  total_cost = input_cost + output_cost

  print(f"Input Cost: ${input_cost:.6f}")
  print(f"Output Cost: ${output_cost:.6f}")
  print(f"Total Cost: ${total_cost:.6f}")

# Append Result to json

def append_to_json(prompt, test_input, completion_data, idx, file_path):
    """
    Append a dictionary with 'prompt' and 'completion_data' to a JSON file.

    Args:
        prompt (str): The prompt data to include.
        completion_data (str): The completion data to include.
        file_path (str): The path to the JSON file.
    """
    # Ensure the JSON data is appended properly
    if os.path.exists(file_path):
        # File exists; load its content
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)
                if not isinstance(existing_data, list):
                    raise ValueError("JSON file does not contain a list.")
            except json.JSONDecodeError:
                existing_data = []
    else:
        # File does not exist; start with an empty list
        existing_data = []

    # Append the new dictionary
    new_entry = {"idx": idx, "prompt": prompt, "test_input": test_input, "completion_data": completion_data}
    existing_data.append(new_entry)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=2)

    print(f"Data appended successfully to {file_path}")



# import re

# def convert_matrix_string(input_string):
#     """
#     Extract the last matrix from the input string and append '#'.
#     Handles multiple matrices or descriptive text.
#     """
#     # Regular expression to match all matrices in the input
#     matches = re.findall(r"\[\[.*?\]\]", input_string, re.DOTALL)
#     if matches:
#         # Get the last matrix
#         last_matrix = matches[-1]
#         # Append '#' to the last matrix and return it in a list
#         return [last_matrix + "#"]
#     else:
#         raise ValueError("No matrix content found in the input string")

import re

def convert_matrix_string(input_string):
    """
    Extract the last matrix from the input string and append '#'.
    Handles multiple matrices or descriptive text.
    """
    # Updated regular expression to correctly handle Python code blocks with matrices
    matches = re.findall(r"\[\s*\[.*?\]\s*\]", input_string, re.DOTALL)
    if matches:
        # Get the last matrix
        last_matrix = matches[-1]
        # Append '#' to the last matrix and return it in a list
        return [last_matrix + "#"]
    else:
        raise ValueError("No matrix content found in the input string")



def process_requests_api(
    selected_test_prompts_start_idx: int, 
    selected_test_prompts_end_idx: int, 
    inputs_to_remember,
    model_name: str,
    result_file_path: str, 
    temperature: float, 
    test_prompts: List[Tuple[str, str]]
) -> Dict[str, List[str]]:
    """Continuously process a list of prompts and handle the outputs."""
    all_outputs_gpt: Dict[str, List[str]] = {}


    ## Option 1: Add API request using o1
    #### Difficulty on format of output
    all_outputs_gpt = {}
    test_prompts = test_prompts[selected_test_prompts_start_idx: selected_test_prompts_end_idx]
    for _ in range(len(test_prompts)):
        if test_prompts:
            prompt, idx = test_prompts.pop(0)
            request_id = idx
            find_start = prompt.find("<|begin_of_text|>") + len("<|begin_of_text|>")
            prompt = prompt[find_start:]
            
            # GPT model LLM prompting
            completion = client.chat.completions.create(model=model_name, messages=[{"role": "user", "content": prompt}], temperature = temperature)
            usage_data = dict(dict(completion).get('usage'))
            print(model_name, "cost:")
            calculate_pricing(pricing_rates, model_name, usage_data)
            gpt_response = completion.choices[0].message.content
            completion_data_str = completion.model_dump_json(indent=2)
            completion_data = json.loads(completion_data_str) 
            test_input = inputs_to_remember[idx]
            append_to_json(prompt, test_input, completion_data, idx, result_file_path)
            print(f"{model_name} prompt =", prompt)
            print(f"{model_name} response =", gpt_response)


            all_outputs_gpt[str(request_id)] = convert_matrix_string(gpt_response)
            all_outputs_gpt[str(request_id)]

    print('all_outputs_gpt', all_outputs_gpt)
    print('len(all_outputs_gpt)', len(all_outputs_gpt))
    return all_outputs_gpt