
import torch
import json
import os
import re
from collections import deque
from typing import Callable, List, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse

# --- 1. Configuration ---
class Config:
    # Model and Paths
    MODEL_NAME = "../../Qwen3-1.7B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEST_THEMES_FILE = "test_themes_qwen1.7b-50.json"
    OUTPUT_FILE = "Qwen3-1.7B_results/results_baseline_tot.json"
    
    # ToT Hyperparameters
    N_STEPS = 3  # Number of thought/generation steps
    BREADTH_LIMIT = 3 # Number of best states to keep at each step (for BFS)
    N_EVALS = 1 # Number of times to ask the LLM to evaluate a state
    
    # Generation Parameters for underlying LLM calls
    MAX_NEW_TOKENS = 1024 # Tokens for each thought/step, not the whole text
    TEMPERATURE = 0.7
    TOP_P = 0.9

# --- 2. ToT Framework Classes (Adapted for local transformers model) ---

class TreeNode:
    """A node in the Tree of Thoughts."""
    def __init__(self, state: str, thought: str, value: float = 0.0):
        self.state = state      # The full text of the proposal up to this point
        self.thought = thought  # The specific thought/paragraph that led to this state
        self.value = value      # The evaluated score of the state
        self.children: List[TreeNode] = []

class TreeOfThoughts:
    """Adapted Tree of Thoughts class for local model execution."""
    def __init__(
            self, 
            model: AutoModelForCausalLM,
            tokenizer: AutoTokenizer,
            input_seq: str,
            n_steps: int,
            get_thought_gen_prompt: Callable,
            get_state_eval_prompt: Callable,
            heuristic_calculator: Callable,
            n_evals: int,
            breadth_limit: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.input_seq = input_seq
        self.root = TreeNode(state='', thought='')
        self.n_steps = n_steps
        self.get_thought_gen_prompt = get_thought_gen_prompt
        self.get_state_eval_prompt = get_state_eval_prompt
        self.heuristic_calculator = heuristic_calculator
        self.n_evals = n_evals
        self.breadth_limit = breadth_limit

    def chat_completions(
            self,
            prompt: str,
            n: int = 1,
    ) -> List[str]:
        """Custom chat completion function using the local transformers model."""
        messages = [{'role': "user", 'content': prompt}]
        text_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            do_sample=True,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=n
        )
        
        return self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    def thought_generator(self, state: str) -> List[str]:
        prompt = self.get_thought_gen_prompt(self.input_seq, state)
        # We use a 'propose' strategy: one call returns multiple thoughts
        response = self.chat_completions(prompt, n=1)[0]
        # Thoughts are separated by a specific delimiter
        thoughts = [t.strip() for t in response.split('---') if t.strip()]
        return thoughts

    def state_evaluator(self, state: str) -> float:
        prompt = self.get_state_eval_prompt(self.input_seq, state)
        state_evals = self.chat_completions(prompt, n=self.n_evals)
        value = self.heuristic_calculator(state, state_evals)
        return value

    def bfs(self, verbose: bool = True) -> str:
        """Performs a Breadth-First Search on the thought tree."""
        queue = deque()
        queue.append(self.root)

        for step in range(1, self.n_steps + 1):
            if verbose: print(f"\nStep {step}/{self.n_steps}...")
            
            level_nodes = list(queue)
            queue.clear()

            for node in level_nodes:
                thoughts = self.thought_generator(state=node.state)
                for thought in thoughts:
                    updated_state = (node.state + "\n\n" + thought).strip()
                    child = TreeNode(state=updated_state, thought=thought)
                    node.children.append(child)
                    queue.append(child)
            
            if not queue: break

            if verbose: print(f"  Generated {len(queue)} new states. Evaluating...")
            for i, node in enumerate(queue):
                node.value = self.state_evaluator(state=node.state)
                if verbose: print(f"    State {i+1}/{len(queue)} evaluated with score: {node.value:.2f}")

            # Pruning
            sorted_nodes = sorted(queue, key=lambda n: n.value, reverse=True)
            limit = 1 if step == self.n_steps else self.breadth_limit
            top_nodes = sorted_nodes[:limit]
            
            queue = deque(top_nodes)
            if verbose: print(f"  Pruned to {len(queue)} best states.")

        if not queue:
            return "ToT search failed to produce a result."
        
        return queue[0].state

# --- 3. Custom Callables for Research Proposal Generation ---

def get_thought_gen_prompt(input_seq: str, state: str) -> str:
    """Creates a prompt to generate the next thoughts/paragraphs."""
    if not state:
        return f"""You are a research scientist brainstorming a proposal.
Initial Idea: {input_seq}

Based on this, generate 3 distinct and promising opening paragraphs for the proposal. Each paragraph should explore a slightly different angle or focus.
IMPORTANT: Present each paragraph separated by '---'.

Paragraph 1:
"""
    else:
        return f"""You are a research scientist continuing a proposal draft.
Initial Idea: {input_seq}

Proposal so far:
---
{state}
---

Based on the proposal so far, generate 3 distinct and logical next paragraphs to continue the proposal. Each should build upon the existing text in a unique way.
IMPORTANT: Present each paragraph separated by '---'.

Next Paragraph 1:
"""

def get_state_eval_prompt(input_seq: str, state: str) -> str:
    """Creates a prompt for an LLM to evaluate the quality of a proposal draft."""
    return f"""You are a strict, expert peer reviewer.
The original research theme is: {input_seq}

Here is a partial research proposal draft:
---
{state}
---

Evaluate this draft on a scale of 1 to 10 based on its potential to become a high-impact paper. Consider its novelty, clarity, and scientific feasibility.
Your response MUST be a single integer from 1 to 10, with 10 being the best. Do not add any other text.

Score:"""

def heuristic_calculator(state: str, state_evals: List[str]) -> float:
    """Parses the numeric score from the LLM's evaluation response."""
    total_score = 0.0
    count = 0
    for eval_str in state_evals:
        match = re.search(r'\b(\d+)\b', eval_str)
        if match:
            score = int(match.group(1))
            if 1 <= score <= 10:
                total_score += score
                count += 1
    return total_score / count if count > 0 else 0.0

# --- 4. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN ToT on test themes')
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
    
    print(f"Loading model: {cfg.MODEL_NAME} on {cfg.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.MODEL_NAME,
        torch_dtype="auto", 
        device_map="auto"
    ).eval()
    print("Model loaded successfully.")
    
    print(f"Loading test themes from: {cfg.TEST_THEMES_FILE}")
    try:
        with open(cfg.TEST_THEMES_FILE, 'r', encoding='utf-8') as f:
            test_themes = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test themes file not found at {cfg.TEST_THEMES_FILE}.")
        exit()
        
    all_results = []
    print(f"\n--- Starting Baseline Experiment: Tree of Thoughts (BFS) ---")
    
    for i, theme_obj in enumerate(test_themes):
        print(f"\n--- Processing Theme {i+1}/{len(test_themes)} (ID: {theme_obj.get('id', 'N/A')}) ---")
        
        input_sequence = f"Theme: {theme_obj['theme']}\nElaboration: {theme_obj['elaboration']}"
        
        tot_search = TreeOfThoughts(
            model=model,
            tokenizer=tokenizer,
            input_seq=input_sequence,
            n_steps=cfg.N_STEPS,
            get_thought_gen_prompt=get_thought_gen_prompt,
            get_state_eval_prompt=get_state_eval_prompt,
            heuristic_calculator=heuristic_calculator,
            n_evals=cfg.N_EVALS,
            breadth_limit=cfg.BREADTH_LIMIT,
        )
        
        final_proposal = tot_search.bfs(verbose=True)
        
        result = {
            "id": theme_obj.get('id'),
            "theme": theme_obj.get('theme'),
            "elaboration": theme_obj.get('elaboration'),
            "baseline_method": "tree_of_thoughts_bfs",
            "output": final_proposal
        }
        all_results.append(result)
        
        print(f"Saving intermediate results to {cfg.OUTPUT_FILE}...")
        with open(cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, indent=2, ensure_ascii=False)
            
    print("\n--- ToT Experiment finished successfully! ---")
    print(f"All results saved to {cfg.OUTPUT_FILE}")
