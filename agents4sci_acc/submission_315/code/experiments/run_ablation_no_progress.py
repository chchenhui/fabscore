
import torch
import json
import os
import sys
import faiss
import time

# --- 1. Setup Environment ---
#script_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
#sys.path.append(parent_dir)

import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from magellan.cg_mcts_qwen_noter import Config as MCTS_Config, LLMInterface, CG_MCTS

# --- 2. Experiment Configuration ---
class ExpConfig:
    TEST_THEMES_FILE = "../compare_experiments/test_themes_qwen1.7b-50.json" # Assumes file is in the same directory
    FAISS_INDEX_PATH = "../database/Qwen3-1.7B-db/faiss_index_Qwen3-1.7B.bin"

    OUTPUT_FILE = "results_ablation_no_progress.json"

# --- 3. Core Execution Function ---
def run_cgmcts_search(theme_obj: dict, llm_interface: LLMInterface, novelty_db, mcts_config: MCTS_Config) -> str:
    theme = theme_obj['theme']
    elaboration = theme_obj['elaboration']
    init_narrative = f"{theme}\n\n{elaboration}"
    
    print(f"Initializing CG_MCTS for theme: '{theme[:80]}...'")
    mcts = CG_MCTS(
        llm_interface=llm_interface,
        novelty_db=novelty_db,
        config=mcts_config,
        theme=theme,
        init_narrative=init_narrative
    )

    print("Starting MCTS search...")
    start_time = time.time()
    mcts.search()
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.2f} seconds.")

    final_narrative = mcts.get_best_sequence(debug=False)
    return final_narrative

# --- 4. Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RUN Magellan on test themes')
    parser.add_argument('--modelpath', type=str, required=True, help='Path of LLM model weights, e.g., ../../Qwen3-8B')
    parser.add_argument('--dbpath', type=str, required=True, help='Path of databases, e.g., ./Qwen3-db')
    parser.add_argument('--testfile', type=str, required=True, help='Test themes filepath, e.g., ./test_set/test_themes_qwen1.7b-50.json')
    parser.add_argument('--outfile', type=str, required=True, help='Path to save the result, e.g., ./result/results_cgmcts.json')
    args = parser.parse_args()


    exp_cfg = ExpConfig()
    mcts_cfg = MCTS_Config()

    exp_cfg.TEST_THEMES_FILE = args.testfile
    exp_cfg.OUTPUT_FILE = args.outfile
    exp_cfg.FAISS_INDEX_PATH = os.path.join(args.dbpath, "faiss_index.bin")

    mcts_cfg.MODEL_NAME = args.modelpath

    os.makedirs(os.path.dirname(args.outfile), exists_ok=True)
    
    
    # --- ABLATION MODIFICATION ---
    print("\n*** RUNNING ABLATION: NO PROGRESS REWARD ***")
    mcts_cfg.W_PROG = 0.0
    mcts_cfg.MIN_PROGRESS_THRESHOLD = 0.0 # Also disable the hard pruning
    mcts_cfg.NUM_ITERATIONS = 10
    # ---------------------------

    #model_path_from_parent = mcts_cfg.MODEL_NAME
    #absolute_model_path = os.path.abspath(os.path.join(parent_dir, model_path_from_parent))

    llm = LLMInterface(model_name=mcts_cfg.MODEL_NAME, device=mcts_cfg.DEVICE)
    
    print(f"Loading FAISS index from: {exp_cfg.FAISS_INDEX_PATH}")
    novelty_db = faiss.read_index(exp_cfg.FAISS_INDEX_PATH)
    
    print(f"Loading test themes from: {exp_cfg.TEST_THEMES_FILE}")
    with open(exp_cfg.TEST_THEMES_FILE, 'r', encoding='utf-8') as f:
        test_themes = json.load(f) # Process only the first 20 themes
        
    all_results = []
    
    print(f"\n--- Starting Ablation Experiment: No Progress Reward ---")
    
    for i, theme_obj in enumerate(test_themes):
        print(f"\n--- Processing Theme {i+1}/{len(test_themes)} (ID: {theme_obj.get('id', 'N/A')}) ---")
        
        final_output = run_cgmcts_search(theme_obj, llm, novelty_db, mcts_cfg)
        
        result = {
            "id": theme_obj.get('id'),
            "theme": theme_obj.get('theme'),
            "elaboration": theme_obj.get('elaboration'),
            "method": "ablation_no_progress",
            "output": final_output
        }
        all_results.append(result)
        
        print(f"Saving intermediate results to {exp_cfg.OUTPUT_FILE}...")
        with open(exp_cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, indent=2, ensure_ascii=False)
            
    print("\n--- Ablation experiment finished successfully! ---")
    print(f"All results saved to {exp_cfg.OUTPUT_FILE}")
