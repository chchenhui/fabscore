import json
import os
import glob
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re # Included as part of the provided code, though not directly used in vectorization
import argparse

# --- 1. Configuration ---
class Config:
    MODEL_NAME = "../../Qwen3-0.6B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_ITERATIONS = 30
    
    # MCTS Core
    EXPLORATION_CONSTANT = 1.5
    K_EXPAND = 3
    # N_ROLLOUTS is deprecated in Narrative MCTS, evaluation is direct.
    
    # LLM Generation Lengths
    EXPAND_MAX_LENGTH = 1024
    THEME_GEN_MAX_LENGTH = 1024

    # Automated Theme Generation
    NUM_CLUSTERS = 3
    
    # T-MCTS/PE Hyperparameters
    ALPHA_NOVELTY = 0.7
    W_DIR = 1.0
    W_COH = 0.5
    W_NOV = 0.3
    W_PROG = 0.2

# --- 2. Helper Functions ---
def parse_llm_json_output(response: str) -> dict | None:
    """Robustly parses JSON from LLM output that might include markdown."""
    try:
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if match:
            json_str = match.group(1)
            return json.loads(json_str)
        else:
            return json.loads(response)
    except (json.JSONDecodeError, IndexError):
        print(f"Warning: Failed to parse LLM response as JSON.")
        return None

# --- 3. LLM Interface ---
class LLMInterface:
    def __init__(self, model_name, device):
        print(f"Loading model: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype="auto",
            device_map="auto"
        ).eval()
        self.device = self.model.device
        print("Model loaded successfully.")

    def get_vector(self, text: str) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs, output_hidden_states=True)
            # Using the last hidden state of the last token as the embedding
            vector = outputs.hidden_states[-1][0, -1, :].cpu().to(torch.float32).numpy()
            del inputs, outputs
            # Clear CUDA cache if using GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            return vector

    def generate_chat_completion(self, messages: list, max_length: int, temperature: float = 0.7, thinking=True) -> str:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_length,
            do_sample=True, top_p=0.9, temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
        del text, model_inputs, generated_ids
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return response

    def get_prob_and_vector(self, text: str) -> tuple[float, np.ndarray]:
        with torch.no_grad():
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs, labels=inputs.input_ids, output_hidden_states=True)
                log_prob = -outputs.loss.item()
                vector = outputs.hidden_states[-1][0, -1, :].cpu().to(torch.float32).numpy()
                del inputs, outputs
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                return log_prob, vector
            except Exception as e:
                print(f"Error in get_prob_and_vector: {e}")
                return 0.0, None

# --- 4. Helper to build FAISS index ---
def build_faiss_index(vectors: np.ndarray):
    print("Building FAISS index...")
    dimension = vectors.shape[1]
    # Using IndexFlatIP for Inner Product similarity, common for normalized embeddings
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors) # Normalize vectors for cosine similarity with IndexFlatIP
    index.add(vectors)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

# --- 5. Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Database.')
    parser.add_argument('--modelpath', type=str, required=True, help='Path of LLM model weights, e.g., ../../Qwen3-8B')
    parser.add_argument('--inputpath', type=str, required=True, help='Path of collect paper metadata, e.g., ./data_use')
    parser.add_argument('--outdir', type=str, required=True, help='Path to save the data file, e.g., ./Qwen3-8B-db')
    args = parser.parse_args()

    cfg = Config()

    cfg.MODEL_NAME = args.modelpath
    outsave_dir = args.outdir

    os.makedirs(outsave_dir, exist_ok=True)

    
    llm_interface = LLMInterface(cfg.MODEL_NAME, cfg.DEVICE)

    data_use_dir = args.inputpath # Assuming this script is run from the 'database' directory
    json_files = glob.glob(os.path.join(data_use_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {data_use_dir}. Please ensure the path is correct and files exist.")
    else:
        print(f"Found {len(json_files)} JSON files in '{data_use_dir}'.")

        all_papers_text = []
        paper_metadata = [] # To store original paper data for retrieval later

        for file_path in json_files:
            print(f"Processing file: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    papers_in_file = json.load(f)
                    if isinstance(papers_in_file, list):
                        for paper in papers_in_file:
                            title = paper.get("title", "")
                            abstract = paper.get("abstract", "")
                            # Combine title and abstract for vectorization
                            combined_text = f"Title: {title}\nAbstract: {abstract}"
                            all_papers_text.append(combined_text)
                            paper_metadata.append(paper) # Store original paper data
                    else:
                        print(f"Warning: File {file_path} does not contain a list of papers.")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {file_path}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while reading {file_path}: {e}")

        if all_papers_text:
            print(f"\nVectorizing {len(all_papers_text)} documents...")
            # Generate vectors for all combined texts
            # Note: For very large datasets, you might want to process in batches
            vectors = np.array([llm_interface.get_vector(text) for text in all_papers_text]).astype('float32')

            # Extract a clean model name for filenames
            model_name_for_file = os.path.basename(cfg.MODEL_NAME)
            if not model_name_for_file: # Handle cases like ".." or "/"
                model_name_for_file = "default_model"

            # Save vectors to a .npy file
            vectors_filename = f"{outsave_dir}/vectors.npy"
            np.save(vectors_filename, vectors)
            print(f"Vectors saved to {vectors_filename}")

            # Build the FAISS index
            paper_knowledge_base_index = build_faiss_index(vectors)

            # Save the FAISS index
            faiss_index_filename = f"{outsave_dir}/faiss_index.bin"
            faiss.write_index(paper_knowledge_base_index, faiss_index_filename)
            print(f"FAISS index saved to {faiss_index_filename}")

            # Save the paper metadata
            metadata_filename = f"{outsave_dir}/paper_metadata.json"
            with open(metadata_filename, "w", encoding='utf-8') as f:
                json.dump(paper_metadata, f, ensure_ascii=False, indent=4)
            print(f"Paper metadata saved to {metadata_filename}")

            print("\nKnowledge base and FAISS index created successfully.")
            print("You can now use the saved files for your application.")
        else:
            print("No paper texts were extracted for vectorization.")

