import json
import random
import numpy as np
import os
import faiss
import pickle

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from magellan.cg_mcts_qwen import LLMInterface, AutomatedThemeGenerator, Config

import argparse

# Constants
POOL_SIZE = 200
SAMPLE_SIZE = 50

def main(output_dir="test_set", model_path="../../Qwen3-0.6B", database_path='./database'):

    # Initialize LLMInterface (assuming a default model or configuration)
    # You might need to adjust the model_name or other parameters based on your actual setup
    llm_interface = LLMInterface(model_path, "cuda") 

    # Load required data for AutomatedThemeGenerator
    metadata_filename = f"{database_path}/paper_metadata.json"
    vectors_filename = f"{database_path}/vectors.npy"
    faiss_index_filename = f"{database_path}/faiss_index.bin"


    try:
        vectors = np.load(vectors_filename)
        novelty_db = faiss.read_index(faiss_index_filename)
        with open(metadata_filename, "r", encoding='utf-8') as f:
            paper_metadata = json.load(f)
        
        # The user should ensure the metadata format is a list of dicts, 
        # with each dict having at least 'title' and 'abstract' keys.
        novelty_documents = [item.get('title', '') + ': ' + item.get('abstract', '') for item in paper_metadata]

        print("Successfully loaded pre-computed data.")

    except FileNotFoundError as e:
        print(f"Error: Could not find pre-computed data file: {e.filename}")
        print("Please ensure the necessary .npy, .bin, and .json files are in the same directory.")
        exit()

    # Initialize AutomatedThemeGenerator
    # Assuming Config has default values or can be initialized without specific args for this purpose
    config = Config() # You might need to pass specific config parameters if required by AutomatedThemeGenerator
    theme_generator = AutomatedThemeGenerator(llm_interface, novelty_documents, vectors, config)

    theme_pool = []
    print(f"Generating {POOL_SIZE} themes...")
    for i in range(POOL_SIZE):
        print(f"Generating theme {i+1}/{POOL_SIZE}...")
        theme, elaboration, concept_original_list = theme_generator.generate_theme()
        theme_pool.append({
            "id": f"theme_{i+1:03d}",
            "theme": theme,
            "elaboration": elaboration,
            "concept_original_list": concept_original_list
        })
    
    # Write the complete theme pool to theme_pool.json
    theme_pool_output_path = os.path.join(output_dir, "theme_pool.json")
    with open(theme_pool_output_path, 'w', encoding='utf-8') as f:
        json.dump(theme_pool, f, ensure_ascii=False, indent=4)
    print(f"Complete theme pool saved to {theme_pool_output_path}")

    # Randomly sample SAMPLE_SIZE themes
    if len(theme_pool) < SAMPLE_SIZE:
        print(f"Warning: Theme pool size ({len(theme_pool)}) is less than sample size ({SAMPLE_SIZE}). Sampling all available themes.")
        test_themes = theme_pool
    else:
        test_themes = random.sample(theme_pool, SAMPLE_SIZE)
    
    # Write the sampled test set to test_themes.json
    test_themes_output_path = os.path.join(output_dir, "test_themes.json")
    with open(test_themes_output_path, 'w', encoding='utf-8') as f:
        json.dump(test_themes, f, ensure_ascii=False, indent=4)
    print(f"Sampled test themes saved to {test_themes_output_path}")
    
    # Write the theme_generator to theme_generator.pkl
    theme_generator_output_path = os.path.join(output_dir, "theme_generator.pkl")
    with open(theme_generator_output_path, 'wb') as f:
        pickle.dump(theme_generator, f)
    print(f"Sampled test themes saved to {theme_generator_output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct testset')
    parser.add_argument('--modelpath', type=str, required=True, help='Path of LLM model weights, e.g., ../../Qwen3-8B')
    parser.add_argument('--dbpath', type=str, required=True, help='Path of databases, e.g., ./Qwen3-db')
    parser.add_argument('--outdir', type=str, required=True, help='Path to save the test set, e.g., ./test_set')
    args = parser.parse_args()


    os.makedirs(args.outdir, exist_ok=True)

    main(output_dir=args.outdir, model_path=args.model_path, database_path=args.dbpath)
