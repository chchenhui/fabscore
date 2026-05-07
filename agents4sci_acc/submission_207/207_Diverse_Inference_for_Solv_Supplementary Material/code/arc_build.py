import os
import json

def create_optillm_input(task_prompts_dir, output_json_path):
    """
    Reads task prompt files from the specified directory and creates a JSON file for OptiLLM input.

    Args:
        task_prompts_dir (str): The directory containing the task prompt .txt files.
        output_json_path (str): The path to save the resulting JSON file.
    """
    optillm_input = []

    # List all task prompt files in the directory
    prompt_files = [file for file in os.listdir(task_prompts_dir) if file.endswith('_prompt.txt')]

    for prompt_file in prompt_files:
        task_id = os.path.splitext(prompt_file)[0].replace('_prompt', '')
        prompt_path = os.path.join(task_prompts_dir, prompt_file)

        with open(prompt_path, 'r') as f:
            prompt_content = f.read()

        dic = {
            'name': f'ARC Task {task_id}',
            'system_prompt': "",  # Can be populated with specific system-level instructions if needed
            'query': prompt_content
        }

        optillm_input.append(dic)

    with open(output_json_path, 'w') as f:
        json.dump(optillm_input, f, indent=4)

    print(f"Successfully saved OptiLLM input JSON to {output_json_path}")

# Example usage
task_prompts_directory = os.path.join(os.getcwd(), "task_prompts")
output_json = os.path.join(os.getcwd(), "optillm-arc-input.json")
create_optillm_input(task_prompts_directory, output_json)
