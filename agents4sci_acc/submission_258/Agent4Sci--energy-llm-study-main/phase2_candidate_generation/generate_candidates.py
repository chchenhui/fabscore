import os
import shutil
import argparse
import random
import numpy as np
import ollama
import re
import ast

# --- Configuration ---
MODEL_ID = "codellama:70b-instruct"
ROOT_SEED = 42

# --- Helper Functions ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def extract_tagged_code(full_output: str) -> str:
    """
    Finds a code block within <python_code> tags and validates its syntax.
    """
    pattern = re.compile(r"<python_code>(.*?)</python_code>", re.DOTALL)
    match = pattern.search(full_output)

    if not match:
        return ""

    code_to_validate = match.group(1).strip()
    
    try:
        ast.parse(code_to_validate)
        return code_to_validate
    except (SyntaxError, IndentationError):
        return ""

def call_llm(client, prompt, seed, temperature, top_p):
    print(f"  -> Trial generation (seed={seed}, temp={temperature})")
    try:
        response = client.generate(
            model=MODEL_ID,
            prompt=prompt,  # The prompt already contains [INST] ... [/INST]
            options={'seed': seed, 'temperature': temperature, 'top_p': top_p, 'num_predict': 512}
        )
        return extract_tagged_code(response.get('response', ''))
    except Exception as e:
        print(f"  -> ERROR: LLM call failed: {e}")
        return ""

def main():
    parser = argparse.ArgumentParser(description="Generate multiple code candidates for each benchmark task.")
    parser.add_argument("--num-candidates", type=int, default=20,
                        help="Number of candidates to generate per task.")
    args = parser.parse_args()

    print(f"--- Candidate Generation ---")
    print(f"Target: {args.num_candidates} per task | Root seed={ROOT_SEED}")

    try:
        client = ollama.Client()
        client.list()
        print("✅ Connected to Ollama server.")
    except Exception as e:
        print("❌ ERROR: Failed to connect to Ollama. Ensure the Ollama app is running.")
        return

    benchmarks_dir = "benchmarks"
    output_dir = f"generated_candidates-{MODEL_ID}-prompt_v5"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    generation_seed_counter = ROOT_SEED
    task_dirs = sorted([d for d in os.listdir(benchmarks_dir) if os.path.isdir(os.path.join(benchmarks_dir, d))])

    for task_id in task_dirs:
        prompt_file = os.path.join(benchmarks_dir, task_id, "prompt.txt")
        if not os.path.exists(prompt_file):
            continue

        print(f"\n📂 Task: {task_id}")
        with open(prompt_file, 'r') as f:
            prompt = f.read().strip()

        task_output_dir = os.path.join(output_dir, task_id)
        os.makedirs(task_output_dir)

        successful_candidates = 0
        attempts = 0
        while successful_candidates < args.num_candidates and attempts < args.num_candidates * 2:
            set_seed(generation_seed_counter)
            temperature = 0.8
            top_p = 0.95

            generated_code = call_llm(client, f"[INST] {prompt} [/INST]",
                                      generation_seed_counter, temperature, top_p)
            
            if generated_code:
                filename = os.path.join(task_output_dir, f"candidate_{successful_candidates}.py")
                with open(filename, 'w') as f:
                    f.write(generated_code)
                print(f"  ✅ Saved candidate_{successful_candidates}.py")
                successful_candidates += 1

            generation_seed_counter += 1
            attempts += 1
        
        if successful_candidates < args.num_candidates:
            print(f"  ⚠️ Only {successful_candidates}/{args.num_candidates} valid candidates generated.")

    print("\n🎉 Candidate generation complete.")

if __name__ == "__main__":
    main()
