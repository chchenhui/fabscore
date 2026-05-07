import os
import json

def process_json_files_to_txt(input_dir):
    """
    Reads JSON files from the specified directory and writes their content in a specified format to separate .txt files
    under a new folder in the current working directory.

    Args:
        input_dir (str): The directory containing the JSON files.
    """
    # Create a new folder under the current working directory
    output_dir = os.path.join(os.getcwd(), "task_prompts")
    os.makedirs(output_dir, exist_ok=True)

    # List all JSON files in the directory
    json_files = [file for file in os.listdir(input_dir) if file.endswith('.json')]

    for json_file in json_files:
        json_path = os.path.join(input_dir, json_file)
        task_id = os.path.splitext(json_file)[0]
        output_txt_path = os.path.join(output_dir, f"{task_id}_prompt.txt")

        with open(json_path, 'r') as f:
            data = json.load(f)

        with open(output_txt_path, 'w') as txt_file:
            # Add the introductory string
            txt_file.write("Find the common rule that maps an input grid to an output grid, given the examples below.\n\n")

            # Add examples from training data
            for idx, example in enumerate(data.get('train', [])):
                txt_file.write(f"Example {idx + 1}:\n\n")

                # Input grid
                txt_file.write("Input:\n")
                for row in example['input']:
                    txt_file.write(" ".join(map(str, row)) + "\n")
                txt_file.write("\n")

                # Output grid
                txt_file.write("Output:\n")
                for row in example['output']:
                    txt_file.write(" ".join(map(str, row)) + "\n")
                txt_file.write("\n")

            # Add the test input grid
            test_example = data.get('test', [])[0]
            txt_file.write("\nBelow is a test input grid. Predict the corresponding output grid by applying the rule you found. Your final answer should just be the text output grid itself.\n\n")
            txt_file.write("Input:\n")
            for row in test_example['input']:
                txt_file.write(" ".join(map(str, row)) + "\n")

    print(f"Data has been successfully written to individual task prompt files in: {output_dir}")

# Example usage
input_directory = "ARC-AGI-master/data/evaluation"
process_json_files_to_txt(input_directory)
