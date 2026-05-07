
import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse

# --- 1. Configuration ---
class Config:
    # Model and Paths
    MODEL_NAME = "../../Qwen3-1.7B"  # Adjusted path relative to the script location
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEST_THEMES_FILE = "../compare_experiments/test_themes_qwen1.7b-50.json" # Assumes file is in the same directory
    OUTPUT_FILE = "results_baseline_scipip.json" # Specific output file for this baseline
    
    # Generation Parameters
    MAX_NEW_TOKENS = 4096  # Max length for the generated proposal
    TEMPERATURE = 0.7
    TOP_P = 0.9

# --- 2. Core Generation Function ---

system_prompt = "Now you are a researcher in the field of AI with innovative and pioneering abilities. You are good at transforming a brief scientific idea into a concrete algorithm."

def run_cot_generation(theme_object: dict, model, tokenizer, device) -> str:
    """
    Generates a single candidate for a theme using a Zero-Shot 
    Chain-of-Thought prompt.
    """
    theme = theme_object['theme']
    elaboration = theme_object['elaboration']
    
    # Zero-Shot Chain-of-Thought Prompt
    prompt = f"""# Task Description: You are an AI researcher conducting studies in a specific domain. Someone has provided you with a brief scientific idea, and your task is to transform it into a detailed, feasible, and concrete algorithm. If necessary, you may incorporate formulas to elaborate on the algorithm in the Latex format. I will give you an example. The example begins with “# Example 1” and includes a Brief Scientific Idea and its corresponding Detailed Scientific Idea. Then, your task starts with “# Your Task”, containing “Your Brief Scientific Idea”. Your job is to expand Your Brief Scientific Idea into a Detailed Scientific Idea by referring to Example 1. Note that the ideas in Example 1 are unrelated to your idea, so the key focus should be on the relationship between the Brief Scientific Idea and the Detailed Scientific Idea. You should directly start with your response and do not start with a section title like “## Detailed Scientific Idea”.
# Example 1
## Example Brief Scientific Idea {{Example brief idea}}
## Example Detailed Scientific Idea {{Example detailed idea}}
# Your Task
## Your Research Background {elaboration}
## Your Brief Scientific Idea
{theme}
"""
    
    messages = [ {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}]
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_prompt], return_tensors="pt").to(device)

    print(f"Generating a single scipip candidate for theme: '{theme[:80]}...'")
    
    # Simple, single generation call
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=Config.MAX_NEW_TOKENS,
        do_sample=True,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=1
    )
    
    # Decode the output
    output_text = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
    
    print("Generation complete.")
    return output_text

# --- 3. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN scipip prompting on test themes')
    parser.add_argument('--modelpath', type=str, required=True, help='Path of LLM model weights, e.g., ../../Qwen3-8B')
    parser.add_argument('--dbpath', type=str, required=True, help='Path of databases, e.g., ./Qwen3-db')
    parser.add_argument('--testfile', type=str, required=True, help='Test themes filepath, e.g., ./test_set/test_themes_qwen1.7b-50.json')
    parser.add_argument('--outfile', type=str, required=True, help='Path to save the result, e.g., ./result/results_cgmcts.json')
    args = parser.parse_args()

    cfg = Config()
    cfg.MODEL_NAME = args.modelpath
    cfg.TEST_THEMES_FILE = args.testfile
    cfg.OUTPUT_FILE = args.outfile

    os.makedirs(os.path.dirname(args.outfile), exists_ok=True)

    
    # Load Model and Tokenizer
    print(f"Loading model: {cfg.MODEL_NAME} on {cfg.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        dtype="auto",
        device_map="auto"
    ).eval()
    print("Model loaded successfully.")
    
    # Load Test Data
    print(f"Loading test themes from: {cfg.TEST_THEMES_FILE}")
    try:
        with open(cfg.TEST_THEMES_FILE, 'r', encoding='utf-8') as f:
            test_themes = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test themes file not found at {cfg.TEST_THEMES_FILE}. Please ensure the file exists.")
        exit()
        
    all_results = []
    
    print(f"\n--- Starting SIMPLE Baseline Experiment: Scipip (1-of-1) ---")
    
    for i, theme_obj in enumerate(test_themes):
        print(f"\n--- Processing Theme {i+1}/{len(test_themes)} (ID: {theme_obj.get('id', 'N/A')}) ---")
        
        # Run the generation
        output = run_cot_generation(theme_obj, model, tokenizer, model.device)
        
        # Store the result
        result = {
            "id": theme_obj.get('id'),
            "theme": theme_obj.get('theme'),
            "elaboration": theme_obj.get('elaboration'),
            "baseline_method": "scipip",
            "output": output
        }
        all_results.append(result)
        
        # Save incrementally to avoid data loss on long runs
        print(f"Saving intermediate results to {cfg.OUTPUT_FILE}...")
        with open(cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, indent=2, ensure_ascii=False)
            
    print("\n--- Experiment finished successfully! ---")
    print(f"All results saved to {cfg.OUTPUT_FILE}")
