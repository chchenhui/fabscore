import json
import os
import re
import time
from typing import List, Dict

# Assuming the local Qwen model interface is in a file accessible via path
# For this example, we will simulate the interface based on cg_mcts_qwen.py
# In a real scenario, you would import it: from some.path.to.llm import LLMInterface
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

import argparse

# --- Start: LLM Interface (adapted from cg_mcts_qwen.py) ---
class LLMInterface:
    """A wrapper for a local Qwen model to handle generation."""
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        print(f"Loading model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        ).eval()
        self.device = self.model.device
        print("Model loaded successfully.")

    def generate(self, messages: List[Dict], max_length: int = 2048, temperature: float = 0.7) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            top_p=0.9,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        return response

def extract_json_between_markers(text: str) -> Dict:
    """Extracts a JSON object from a string marked by ```json ... ```."""
    try:
        match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        print(f"Warning: Could not extract JSON from text.\nText was: {text}")
        return None
    return None
# --- End: LLM Interface ---


# --- Core Logic for Idea Generation (with original prompts) ---

# This is the original prompt from aiscientist_generate_ideas.py, with file-based placeholders removed.
idea_first_prompt_template = """Come up with the next impactful and creative idea for research experiments and directions based on the seed idea provided below.

Seed Idea:
'''
{seed_idea}
'''

Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

Respond in the following format:

THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and motivations for the idea. Detail your high-level plan, necessary design choices and ideal outcomes of the experiments.

In <JSON>, provide the new idea in JSON format with the following fields:
- "Name": A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A title for the idea, will be used for the report writing.
- "Experiment": An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ...
- "Elaboration": "A single, ** detailed ** paragraph explaining the core research idea. This should elaborate on the connection you found, outline the designed approach."
- "Interestingness": A rating from 1 to 10 (lowest to highest).
- "Feasibility": A rating from 1 to 10 (lowest to highest).
- "Novelty": A rating from 1 to 10 (lowest to highest).

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt_template = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


def generate_novel_idea_with_reflection(
    theme: str,
    elaboration: str,
    llm: LLMInterface,
    num_reflections: int = 3,
) -> Dict:
    """
    Generates a single novel idea using the original iterative reflection logic.
    """
    seed_idea = f"Theme: {theme}\nElaboration: {elaboration}"
    
    # System prompt can be minimal as the main instructions are in the user prompt
    system_prompt = "You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field."
    
    # Initial Generation
    print(f"  Generating initial idea for theme: '{theme[:80]}...' ")
    initial_prompt = idea_first_prompt_template.format(
        seed_idea=seed_idea,
        num_reflections=num_reflections
    )
    
    msg_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt}
    ]
    
    response_text = llm.generate(msg_history)
    msg_history.append({"role": "assistant", "content": response_text})

    json_output = extract_json_between_markers(response_text)
    if not json_output:
        print("  Failed to generate initial idea. Skipping.")
        return {"error": "Failed to parse initial LLM output", "raw_output": response_text}

    # Iterative Reflection
    for i in range(num_reflections - 1):
        if "I am done" in response_text:
            print(f"  Idea generation converged after {i + 1} iterations.")
            break

        current_round = i + 2
        print(f"  Reflecting on idea... (Round {current_round}/{num_reflections})")
        
        reflection_prompt = idea_reflection_prompt_template.format(
            current_round=current_round,
            num_reflections=num_reflections
        )
        
        msg_history.append({"role": "user", "content": reflection_prompt})
        response_text = llm.generate(msg_history)
        msg_history.append({"role": "assistant", "content": response_text})
        
        refined_json = extract_json_between_markers(response_text)
        if refined_json:
            json_output = refined_json
        else:
            print(f"  Warning: Failed to parse refinement in round {current_round}. Keeping previous version.")

    print("  Final idea generated.")
    return json_output


# --- Main Experiment Runner ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN AI Scientist prompting on test themes')
    parser.add_argument('--modelpath', type=str, required=True, help='Path of LLM model weights, e.g., ../../Qwen3-8B')
    parser.add_argument('--dbpath', type=str, required=True, help='Path of databases, e.g., ./Qwen3-db')
    parser.add_argument('--testfile', type=str, required=True, help='Test themes filepath, e.g., ./test_set/test_themes_qwen1.7b-50.json')
    parser.add_argument('--outfile', type=str, required=True, help='Path to save the result, e.g., ./result/results_cgmcts.json')
    args = parser.parse_args()

    # --- Configuration ---
    MODEL_PATH = args.modelpath
    TEST_THEMES_FILE = args.testfile
    OUTPUT_FILE = args.outfile

    os.makedirs(os.path.dirname(args.outfile), exists_ok=True)

    
    # --- Setup ---
    if not os.path.exists(MODEL_PATH):
        print(f"FATAL: Model not found at '{MODEL_PATH}'.")
        print("Please update the MODEL_PATH variable in the script to the correct location.")
        exit(1)

    llm = LLMInterface(model_name=MODEL_PATH)
    
    if not os.path.exists(TEST_THEMES_FILE):
        print(f"FATAL: Test themes file not found at '{TEST_THEMES_FILE}'.")
        exit(1)
        
    print(f"Loading test themes from: {TEST_THEMES_FILE}")
    with open(TEST_THEMES_FILE, 'r', encoding='utf-8') as f:
        test_themes = json.load(f)
        
    all_results = []
    
    print(f"\n--- Starting AI Scientist Idea Generation Experiment (with original logic) ---")
    
    # --- Execution Loop ---
    for i, theme_obj in enumerate(test_themes):
        print(f"\n--- Processing Theme {i+1}/{len(test_themes)} (ID: {theme_obj.get('id', 'N/A')}) ---")
        
        start_time = time.time()
        
        novel_idea = generate_novel_idea_with_reflection(
            theme=theme_obj['theme'],
            elaboration=theme_obj['elaboration'],
            llm=llm,
            num_reflections=5 # Using original value
        )
        
        end_time = time.time()
        print(f"  Finished in {end_time - start_time:.2f} seconds.")
        
        # --- Store Result ---
        result = {
            "id": theme_obj.get('id'),
            "original_theme": theme_obj.get('theme'),
            "original_elaboration": theme_obj.get('elaboration'),
            "method": "ai_scientist_iterative_original_prompt",
            "generated_idea": novel_idea
        }
        all_results.append(result)
        
        # --- Save Intermediate Results ---
        print(f"  Saving intermediate results to {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
                json.dump(all_results, f_out, indent=2, ensure_ascii=False)
        except IOError as e:
            print(f"  Error saving results: {e}")
            
    print(f"\n--- AI Scientist Experiment finished successfully! ---")
    print(f"All results saved to {OUTPUT_FILE}")

