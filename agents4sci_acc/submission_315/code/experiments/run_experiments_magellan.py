
import torch
import json
import os
import sys
import faiss
import time

import argparse

# Import the necessary components from your algorithm's code
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from magellan.cg_mcts_qwen import Config as MCTS_Config, LLMInterface, CG_MCTS

# --- 2. Experiment Configuration ---
class ExpConfig:
    TEST_THEMES_FILE = "test_themes_qwen1.7b-50.json" # Assumes file is in the same directory
    OUTPUT_FILE = "results_cgmcts.json"
    
    # Paths to knowledge base, relative to this script's location
    FAISS_INDEX_PATH = "../database/Qwen3-1.7B-db/faiss_index_Qwen3-1.7B.bin"
    
    # The MCTS config is imported, but we can confirm model path is correct relative to parent
    # MCTS_Config.MODEL_NAME is '../Qwen3-0.6B', which is correct from the parent dir perspective.

# --- 3. Core Execution Function ---

def run_cgmcts_search(theme_obj: dict, llm_interface: LLMInterface, novelty_db, mcts_config: MCTS_Config) -> str:
    """
    Initializes and runs the CG_MCTS search for a given theme.
    """
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

    # Get the final result without the debug trace for clean output
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
    
    # Correctly resolve the model path relative to the parent directory where cg_mcts_qwen.py is
    #model_path_from_parent = mcts_cfg.MODEL_NAME
    #absolute_model_path = os.path.abspath(os.path.join(parent_dir, model_path_from_parent))

    # Load resources
    # Note: We instantiate LLMInterface from the imported module
    llm = LLMInterface(model_name=mcts_cfg.MODEL_NAME, device=mcts_cfg.DEVICE)
    
    print(f"Loading FAISS index from: {exp_cfg.FAISS_INDEX_PATH}")
    novelty_db = faiss.read_index(exp_cfg.FAISS_INDEX_PATH)
    
    print(f"Loading test themes from: {exp_cfg.TEST_THEMES_FILE}")
    with open(exp_cfg.TEST_THEMES_FILE, 'r', encoding='utf-8') as f:
        test_themes = json.load(f)
        
    all_results = []
    
    print(f"\n--- Starting Main Algorithm Experiment: CG-MCTS ---")
    
    for i, theme_obj in enumerate(test_themes):
        print(f"\n--- Processing Theme {i+1}/{len(test_themes)} (ID: {theme_obj.get('id', 'N/A')}) ---")
        
        final_output = run_cgmcts_search(theme_obj, llm, novelty_db, mcts_cfg)
        
        result = {
            "id": theme_obj.get('id'),
            "theme": theme_obj.get('theme'),
            "elaboration": theme_obj.get('elaboration'),
            "method": "cg_mcts",
            "output": final_output
        }
        all_results.append(result)
        
        print(f"Saving intermediate results to {exp_cfg.OUTPUT_FILE}...")
        with open(exp_cfg.OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            json.dump(all_results, f_out, indent=2, ensure_ascii=False)
            
    print("\n--- CG-MCTS Experiment finished successfully! ---")
    print(f"All results saved to {exp_cfg.OUTPUT_FILE}")
