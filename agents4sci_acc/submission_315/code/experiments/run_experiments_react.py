
import torch
import json
import os
import re
import faiss
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse

# --- 1. Configuration ---
class Config:
    MODEL_NAME = "../../Qwen3-1.7B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TEST_THEMES_FILE = "test_themes_qwen1.7b-50.json"
    OUTPUT_FILE = "Qwen3-1.7B_results/results_baseline_react.json"
    
    # Knowledge Base files for Search tool
    FAISS_INDEX_PATH = "../database/Qwen3-1.7B-db/faiss_index_Qwen3-1.7B.bin"
    METADATA_PATH = "../database/Qwen3-1.7B-db/paper_metadata.json"
    
    # ReAct Hyperparameters
    MAX_STEPS = 2  # Max number of Thought-Action-Observation cycles
    
    # Generation Parameters
    MAX_NEW_TOKENS_REACT =  1024 # For generating thought-action pairs
    MAX_NEW_TOKENS_WRITE = 4096 # For generating actual proposal text
    TEMPERATURE = 0.7
    TOP_P = 0.9

# --- 2. ReAct Tools ---

def search_tool(query: str, faiss_index, metadata, llm_interface) -> str:
    """Performs a search in the knowledge base and returns a summary."""
    print(f"  [Tool] Searching for: '{query}'")
    try:
        # Use the same vectorization as the knowledge base
        query_vector = llm_interface["get_vector"](query)
        faiss.normalize_L2(query_vector.reshape(1, -1))
        
        _, I = faiss_index.search(query_vector.reshape(1, -1), k=3)
        
        results = []
        for idx in I[0]:
            if 0 <= idx < len(metadata):
                results.append(f"- {metadata[idx]['title']}: {metadata[idx]['abstract'][:200]}...")
        
        if not results:
            return "No relevant documents found."
        return "Found relevant documents:\n" + "\n".join(results)
    except Exception as e:
        return f"Search failed: {e}"

# --- 3. ReAct Core Logic ---

def get_react_prompt_template() -> str:
    """Returns the main ReAct prompt with few-shot example."""
    return """You are a research scientist assistant. Your goal is to write a detailed research proposal based on an initial theme. You operate in a loop of Thought, Action, Observation.
At each step, you must first think about your plan, then choose ONE of the following actions:
- Search[topic]: Search for a specific topic in the internal knowledge base to get more information.
- Write[paragraph_plan]: Write the next section of the proposal based on your plan.
- Finish[]: Conclude the process when the proposal is complete.

Here is an example:
---
Initial Idea: Theme: Using Graph Neural Networks (GNNs) for protein-protein interaction (PPI) prediction. Elaboration: Current methods are slow and not interpretable.

Thought: The idea is good, but generic. I need to find a specific, novel GNN architecture to propose. I will search for limitations of current GNNs in this area.
Action: Search[GNN limitations for protein interaction]

Observation: Found documents mentioning scalability issues and problems with dynamic graphs.

Thought: The key challenges are scalability and dynamic interactions. I can propose a new model using a temporal GNN architecture. I will now write the introduction and methods section based on this plan.
Action: Write[Write the introduction explaining the problem and the proposed temporal GNN model. Then, detail the model architecture in the methods section.]

Observation: Paragraphs successfully written.

Thought: The proposal has an introduction and methods. Now I need to describe the experiments and expected outcomes.
Action: Write[Write the 'Experiments' section, describing datasets, metrics, and baselines. Then write the 'Expected Outcomes' section.]

Observation: Paragraphs successfully written.

Thought: The proposal is complete with all core sections. I will now finish.
Action: Finish[]
---

Now, begin with the following task:
Initial Idea: {initial_idea}
"""

def parse_action(text: str) -> tuple[str, str]:
    """Parses the action and its argument from the LLM's output."""
    match = re.search(r"Action:\s*(\w+)\[(.*?)\]", text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

def run_react_generation(theme_obj: dict, model, tokenizer, llm_interface, faiss_index, metadata) -> str:
    prompt_template = get_react_prompt_template()
    initial_idea = f"Theme: {theme_obj['theme']}\nElaboration: {theme_obj['elaboration']}"
    
    prompt_history = prompt_template.format(initial_idea=initial_idea)
    final_proposal = ""
    
    for step in range(Config.MAX_STEPS):
        print(f"\n--- ReAct Step {step + 1}/{Config.MAX_STEPS} ---")
        
        # 1. REASON (Generate Thought and Action)
        messages = [{"role": "user", "content": prompt_history}]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)
        
        generated_ids = model.generate(
            model_inputs.input_ids, attention_mask=model_inputs.attention_mask,
            max_new_tokens=Config.MAX_NEW_TOKENS_REACT, temperature=0.2, top_p=Config.TOP_P,
            pad_token_id=tokenizer.eos_token_id, num_return_sequences=1
        )
        response_text = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        
        thought_match = re.search(r"Thought:\s*(.*?)\s*Action:", response_text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        action_type, action_arg = parse_action(response_text)
        
        print(f"  Thought: {thought}")
        print(f"  Action: {action_type}[{action_arg}]")
        prompt_history += f"\n\nThought: {thought}\nAction: {action_type}[{action_arg}]"

        # 2. ACT (Execute action)
        if action_type is None:
            observation = "Invalid action format. Please use Thought: ... Action: ... format."
        elif action_type == "Search":
            observation = search_tool(action_arg, faiss_index, metadata, llm_interface)
        elif action_type == "Write":
            write_prompt = f"You are a research scientist. Based on the following plan, please write the specified section of the research proposal.\n\nPlan: {action_arg}\n\nProposal Section:"
            write_messages = [{"role": "user", "content": write_prompt}]
            write_text_prompt = tokenizer.apply_chat_template(write_messages, tokenize=False, add_generation_prompt=True)
            write_inputs = tokenizer([write_text_prompt], return_tensors="pt").to(model.device)
            
            write_ids = model.generate(
                write_inputs.input_ids, attention_mask=write_inputs.attention_mask,
                max_new_tokens=Config.MAX_NEW_TOKENS_WRITE, temperature=Config.TEMPERATURE, top_p=Config.TOP_P,
                pad_token_id=tokenizer.eos_token_id, num_return_sequences=1
            )
            written_text = tokenizer.batch_decode(write_ids[:, write_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
            final_proposal += "\n\n" + written_text.strip()
            observation = "Paragraphs successfully written."
        elif action_type == "Finish":
            print("  [Action] Finish action received. Ending loop.")
            return final_proposal.strip()
        else:
            observation = f"Unknown action type: {action_type}."

        print(f"  Observation: {observation}")
        prompt_history += f"\n\nObservation: {observation}"

    return final_proposal.strip()

# --- 4. Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN ReAct on test themes')
    parser.add_argument('--modelpath', type=str, required=True, help='Path of LLM model weights, e.g., ../../Qwen3-8B')
    parser.add_argument('--dbpath', type=str, required=True, help='Path of databases, e.g., ./Qwen3-db')
    parser.add_argument('--testfile', type=str, required=True, help='Test themes filepath, e.g., ./test_set/test_themes_qwen1.7b-50.json')
    parser.add_argument('--outfile', type=str, required=True, help='Path to save the result, e.g., ./result/results_cgmcts.json')
    args = parser.parse_args()

    cfg = Config()
    cfg.MODEL_NAME = args.modelpath
    cfg.TEST_THEMES_FILE = args.testfile
    cfg.OUTPUT_FILE = args.outfile
    cfg.FAISS_INDEX_PATH = os.path.join(args.dbpath, "faiss_index.bin")
    cfg.METADATA_PATH = os.path.join(args.dbpath, "paper_metadata.json")

    os.makedirs(os.path.dirname(args.outfile), exists_ok=True)

    
    print(f"Loading model: {cfg.MODEL_NAME} on {cfg.DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(cfg.MODEL_NAME, torch_dtype="auto", device_map="auto").eval()
    
    print(f"Loading FAISS index from: {cfg.FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(cfg.FAISS_INDEX_PATH)
    print(f"Loading metadata from: {cfg.METADATA_PATH}")
    with open(cfg.METADATA_PATH, 'r', encoding='utf-8') as f: metadata = json.load(f)

    # Create an interface for helper functions to pass to tools
    def get_vector(text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            outputs = model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1][0, -1, :].cpu().to(torch.float32).numpy()
    llm_interface = {"get_vector": get_vector}

    print(f"Loading test themes from: {cfg.TEST_THEMES_FILE}")
    with open(cfg.TEST_THEMES_FILE, 'r', encoding='utf-8') as f: test_themes = json.load(f)
        
    all_results = []
    print(f"\n--- Starting Baseline Experiment: ReAct ---")
    
    for i, theme_obj in enumerate(test_themes):
        print(f"\n--- Processing Theme {i+1}/{len(test_themes)} (ID: {theme_obj.get('id', 'N/A')}) ---")
        
        final_output = run_react_generation(theme_obj, model, tokenizer, llm_interface, faiss_index, metadata)
        
        result = {
            "id": theme_obj.get('id'),
            "theme": theme_obj.get('theme'),
            "elaboration": theme_obj.get('elaboration'),
            "baseline_method": "react",
            "output": final_output
        }
        all_results.append(result)
        
        print(f"Saving intermediate results to {cfg.OUTPUT_FILE}...")
        with open(cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, indent=2, ensure_ascii=False)
            
    print("\n--- ReAct Experiment finished successfully! ---")
    print(f"All results saved to {cfg.OUTPUT_FILE}")
