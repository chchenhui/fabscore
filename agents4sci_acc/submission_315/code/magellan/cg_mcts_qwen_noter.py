import torch
import numpy as np
import faiss
import math
import time
import re
import json
import random
import os
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. Configuration ---
class Config:
    MODEL_NAME = "../../Qwen3-1.7B"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_ITERATIONS = 30
    
    # MCTS Core
    EXPLORATION_CONSTANT = 1.5
    K_EXPAND = 3
    MIN_PROGRESS_THRESHOLD = 0.05 # Minimum vector distance from parent to not be considered a repeat
    # N_ROLLOUTS is deprecated in Narrative MCTS, evaluation is direct.
    
    # LLM Generation Lengths
    EXPAND_MAX_LENGTH = 4096
    THEME_GEN_MAX_LENGTH = 1024

    # Automated Theme Generation
    NUM_CLUSTERS = 20
    
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
            vector = outputs.hidden_states[-1][0, -1, :].cpu().to(torch.float32).numpy()
            del inputs, outputs
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
        return response

    def get_prob_and_vector(self, text: str) -> tuple[float, np.ndarray]:
        with torch.no_grad():
            try:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                outputs = self.model(**inputs, labels=inputs.input_ids, output_hidden_states=True)
                log_prob = -outputs.loss.item()
                vector = outputs.hidden_states[-1][0, -1, :].cpu().to(torch.float32).numpy()
                del inputs, outputs
                return log_prob, vector
            except Exception as e:
                print(f"Error in get_prob_and_vector: {e}")
                return 0.0, None

# --- 4. Automated Theme Generator ---
class AutomatedThemeGenerator:
    def __init__(self, llm_interface: LLMInterface, documents: list, vectors: np.ndarray, config: Config):
        print("\n--- Initializing Automated Theme Generator ---")
        self.llm = llm_interface
        self.documents = documents
        self.vectors = vectors
        self.config = config
        self.labels = None
        self.centroids = None
        self._build_knowledge_map()

    def _build_knowledge_map(self):
        print(f"Phase 0: Clustering {len(self.documents)} documents into {self.config.NUM_CLUSTERS} 'Conceptual Continents'...")
        kmeans = KMeans(n_clusters=self.config.NUM_CLUSTERS, random_state=42, n_init=10)
        self.labels = kmeans.fit_predict(self.vectors)
        self.centroids = kmeans.cluster_centers_
        print("Knowledge map built.")

    def generate_theme(self) -> tuple[str, str]:
        print("\n--- Generating a new theme automatically ---")
        print("Phase 1 & 2: Sampling continents and finding concepts...")
        cluster_a_idx = random.randint(0, self.config.NUM_CLUSTERS - 1)
        distances = np.linalg.norm(self.centroids - self.centroids[cluster_a_idx], axis=1)
        sorted_indices = np.argsort(distances)
        num_clusters = self.config.NUM_CLUSTERS
        medium_dist_start = num_clusters // 3
        medium_dist_end = 2 * (num_clusters // 3)
        medium_distance_indices = sorted_indices[medium_dist_start:medium_dist_end]
        if not medium_distance_indices.any():
            medium_distance_indices = [i for i in sorted_indices if i != cluster_a_idx]
        cluster_b_idx = random.choice(medium_distance_indices)

        def get_representative_doc(cluster_idx):
            doc_indices = np.where(self.labels == cluster_idx)[0]
            return self.documents[random.choice(doc_indices)]

        concept_a = get_representative_doc(cluster_a_idx)
        concept_b = get_representative_doc(cluster_b_idx)

        print("Phase 3: Synthesizing a theme and elaboration...")
        prompt = f'''
        You are a creative scientist tasked with generating a novel research proposal by synthesizing two concepts.

        Concept 1: "{concept_a}"
        Concept 2: "{concept_b}"

        First, think step-by-step in a <think> block. Analyze both concepts. Find a plausible, insightful, and forward-looking connection. You could apply a technique from one to a problem in the other, find a shared principle, or use one as an analogy for the other.

        After your thinking process, output the result as a JSON object with two keys:
        1. "theme": A concise, high-level research theme that captures the core idea. This should be a single, memorable sentence.
        2. "elaboration": A detailed, one-paragraph explanation of the theme. This should elaborate on the connection you found, outline the potential approach, and highlight the novelty. This will serve as the introductory context for the research proposal.

        Example format:
        ```json
        {{
            "theme": "Leveraging Quantum-Inspired Tensor Networks for Explainable Large-Scale Graph Representation Learning.",
            "elaboration": "Current Graph Neural Networks (GNNs) often act as black boxes, limiting their trustworthiness in high-stakes domains. This research proposes a novel framework that adapts principles from quantum many-body physics, specifically tensor networks, to create a new class of GNNs. By representing graph structures and features as a tensor network, we can leverage efficient contraction algorithms (like DMRG) for node classification and link prediction, while the inherent structure of the network provides a direct, model-based explanation for its predictions, addressing the critical need for interpretability in complex graph data."
        }}
        ```
        '''
        print(concept_a, '\n', concept_b)
        response = self.llm.generate_chat_completion([{"role": "user", "content": prompt}], self.config.THEME_GEN_MAX_LENGTH, temperature=0.5)
        print('Generating...', response)

        parsed_json = parse_llm_json_output(response)

        if parsed_json and "theme" in parsed_json and "elaboration" in parsed_json:
            return parsed_json["theme"], parsed_json["elaboration"], [concept_a, concept_b]
        else:
            print("Warning: Failed to parse theme and elaboration. Using the raw response as the theme.")
            # Fallback for safety
            theme = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            return theme, "", [concept_a, concept_b]

# --- 5. MCTS Components (Narrative MCTS) ---
class MCTSNode:
    def __init__(self, narrative_block: str, parent=None, thoughts: str = None):
        self.narrative_block = narrative_block
        self.parent = parent
        self.thoughts = thoughts
        self.children = []
        self.N = 0
        self.Q = 0.0
        self.is_terminal = False
        self._vector = None

    def get_vector(self, llm_interface: LLMInterface) -> np.ndarray:
        if self._vector is None and self.get_full_text():
            # The vector represents the entire narrative up to this point
            self._vector = llm_interface.get_vector(self.get_full_text())
            if self._vector is not None: faiss.normalize_L2(self._vector.reshape(1, -1))
        return self._vector

    def get_full_text(self) -> str:
        path_nodes = self.get_path()
        # The root node's narrative_block is the theme, subsequent blocks are continuations.
        # We join them to form the full narrative.
        return "\n\n".join(n.narrative_block for n in path_nodes).strip()

    def get_path(self):
        path = []
        curr = self
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        return list(reversed(path))

    def get_depth(self) -> int:
        return len(self.get_path()) - 1

    @property
    def value(self):
        return self.Q / self.N if self.N > 0 else 0

    def is_leaf(self):
        return len(self.children) == 0

class CG_MCTS:
    def __init__(self, llm_interface, novelty_db, config, theme: str, init_narrative:str):
        self.llm = llm_interface
        self.novelty_db = novelty_db
        self.config = config
        self.theme = theme
        # The root node contains the initial theme as the first narrative block
        self.root = MCTSNode(init_narrative)
        self.v_target = self._initialize_target_vector()

    def _initialize_target_vector(self):
        print("\n--- Phase 1: Target Setting (Calculating v_target) ---")
        prompt = f'''
        First, think in a <think> block about how to deconstruct the scientific theme 
'{self.theme}
' into core \'problems\' and potential \'mechanisms\'. Then, output the result as a JSON object.

        Example:
        <think>The user wants to break down \'AI for drug discovery\'. The problems are the goals, like finding protein structures. The mechanisms are the AI tools used, like GNNs or Transformers.</think>
        ```json
        {{
            "problems": ["protein folding prediction", "molecule screening"],
            "mechanisms": ["graph neural networks", "reinforcement learning"]
        }}
        ```
        '''
        response = self.llm.generate_chat_completion([{"role": "user", "content": prompt}], self.config.THEME_GEN_MAX_LENGTH, temperature=0.2)
        print(response)
        components = parse_llm_json_output(response)

        if not components or not components.get("problems") or not components.get("mechanisms"):
            print("Could not deconstruct theme. Using a random target vector.")
            return np.random.rand(self.llm.model.config.hidden_size)

        problem_concept = components["problems"][0]
        mechanism_concept = components["mechanisms"][0]
        print(f"Selected Pair: Problem='{problem_concept}', Mechanism='{mechanism_concept}'")

        v_p = self.llm.get_vector(problem_concept)
        v_m = self.llm.get_vector(mechanism_concept)

        proj_v_m_on_v_p = (np.dot(v_m, v_p) / np.dot(v_p, v_p)) * v_p
        v_m_ortho = v_m - proj_v_m_on_v_p
        v_target = v_p + self.config.ALPHA_NOVELTY * v_m_ortho
        faiss.normalize_L2(v_target.reshape(1, -1))
        print("v_target calculated and normalized.")
        return v_target

    def select(self, node: MCTSNode):
        while not node.is_leaf():
            if not node.children: break
            def score_func(c):
                if c.N == 0: return float('inf')
                exploitation = c.Q / c.N
                exploration = self.config.EXPLORATION_CONSTANT * math.sqrt(math.log(node.N) / c.N)
                direction_guidance = 0
                child_vec = c.get_vector(self.llm)
                if child_vec is not None and self.v_target is not None:
                    direction_guidance = self.config.W_DIR * np.dot(child_vec, self.v_target)
                return exploitation + exploration + direction_guidance
            node = max(node.children, key=score_func)
        return node

    def expand(self, node: MCTSNode):
        if node.is_terminal: return

        print(f"  Expanding node {node.get_depth()} with principle-guided generation...")
        idea_so_far = node.get_full_text()
        
        base_prompt = f'''
        You are a world-class Principal Investigator, known for writing clear, compelling, and fundable research proposals.
        Your current task is to expand on the following research idea:
        ---
        {self.theme}
        ---
        You will now write the **next section** of this proposal.
        Based on these principles, generate a distinct, detailed, well-reasoned and deepen "next section" for the research plan. 

        To do this, you must follow these core principles of scientific writing:
        1.  Progressive Deepening: Your new section MUST logically follow from the existing text. It should deepen the idea, moving from a general concept to specific details, or from a hypothesis to a method of testing it. Do not repeat existing information; build upon it.
        
        2.  Concrete Detail: Be specific and avoid vague language. When you are describing a mechanism, explain it with sufficient details for another expert to understand (with math, if needed).

        3.  Critical Thinking: Briefly acknowledge potential challenges, limitations, or alternative approaches to your proposed section. This demonstrates foresight.
        
        First think about what is missing in the paragraph for a well-writen research plan, or which part is not detailed enough. And then finish it.
        ** Make sure the whole article is coherent and logical. **
        '''


        #To do this, you must follow these core principles of scientific writing:

        #1.  **Progressive Deepening:** Your new section MUST logically follow from the existing text. It should deepen the idea, moving from a general concept to specific details, or from a hypothesis to a method of testing it. Do not repeat existing information; build upon it.

        #2.  **Concrete Detail:** Be specific and avoid vague language. If you propose an experiment, name the key techniques, models, or datasets. If you describe a mechanism, explain it with sufficient detail for another expert to understand (with math, if needed).

        #3.  **Critical Thinking:** Briefly acknowledge potential challenges, limitations, or alternative approaches to your proposed section. This demonstrates foresight.

        #Based on these principles, generate a distinct, well-reasoned "next section" for the research plan. 
        #'''

        for i in range(self.config.K_EXPAND):
            print(f"    Generating expansion option {i+1}/{self.config.K_EXPAND}...")
            # Add a bit of randomness for diversity in each call
            # And instruct it to be different from previous options if any
            if node.children:
                existing_options = "\n".join([f"- {child.narrative_block[:100]}..." for child in node.children])
                dynamic_prompt = f"{base_prompt}\n\nYou have already proposed the following options. Make sure the new one is **distinct**:"
                dynamic_prompt += f"\n{existing_options}"
            else:
                dynamic_prompt = base_prompt

            dynamic_prompt += f'\n\nNow Completeing this:  \n{idea_so_far} \n\nYou should start with **Next Section**'

            response = self.llm.generate_chat_completion(
                [{"role": "user", "content": dynamic_prompt}], 
                self.config.EXPAND_MAX_LENGTH, 
                temperature=0.7 # Slightly higher temp for more diversity between calls
            )
            
            thoughts_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            thoughts = thoughts_match.group(1).strip() if thoughts_match else "No thoughts captured."
            
            content = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            content = content.split('**Next Section**', 1)[-1].split('Next Section\n', 1)[-1]

            print(content)

            if len(content) > 100: # Arbitrary threshold for substantial content
                child_node = MCTSNode(content, parent=node, thoughts=thoughts)
                node.children.append(child_node)
            else:
                print(f"Warning: Expansion option {i+1} failed to generate valid content.")
        
        if not node.children:
            print("Warning: Expansion failed to generate any valid options after K attempts.")
            node.is_terminal = True

    def simulate_and_evaluate(self, node: MCTSNode):
        # REWRITTEN for "Narrative MCTS" - evaluates the full text directly
        if not node.narrative_block: return 0.0
        
        full_text = node.get_full_text()
        print(f"  Simulating/Evaluating full text of depth {node.get_depth()}...")

        v_coherence, seq_vec = self.llm.get_prob_and_vector(full_text)
        if seq_vec is None: return 0.0

        seq_vec = seq_vec.reshape(1, -1)
        faiss.normalize_L2(seq_vec)

        # Coherence Value
        v_coherence_norm = 1 / (1 + math.exp(-v_coherence))
        
        # Novelty Value
        similarities, _ = self.novelty_db.search(seq_vec, 1)
        v_novelty = 1.0 - similarities[0][0]
        
        # Progress Value
        v_progress = 0.0 # Root node has no progress
        if node.parent:
            parent_vec = node.parent.get_vector(self.llm)
            # The node's own vector is seq_vec, which we just computed
            child_vec = seq_vec.flatten()
            if parent_vec is not None and child_vec is not None:
                # Cosine distance is 1 - cosine similarity
                v_progress = (1.0 - np.dot(parent_vec.flatten(), child_vec))

        # Hard filter for progress. If the node is too similar to its parent, prune it by returning 0.
        #if node.parent and v_progress < self.config.MIN_PROGRESS_THRESHOLD:
        #    print(f"  Warning: Node lacks progress (prog={v_progress:.3f} < {self.config.MIN_PROGRESS_THRESHOLD}). Pruning this path.")
        #    node.is_terminal = True

        value = (self.config.W_COH * v_coherence_norm + 
                 self.config.W_NOV * v_novelty + 
                 self.config.W_PROG * v_progress)
        
        print(f'  Coh={v_coherence_norm:.3f}, Nov={v_novelty:.3f}, Prog={v_progress:.3f} -> Total Value: {value:.4f}')
        # In this new model, N_ROLLOUTS is implicitly 1, as we do one direct, high-quality evaluation.
        return value

    def backpropagate(self, node: MCTSNode, value: float):
        while node is not None:
            node.N += 1
            node.Q += value
            node = node.parent

    def search(self):
        print("\n--- Phase 2: Guided Narrative Search ---")
        if self.root.is_leaf():
            self.expand(self.root)
            # Initial expansion and evaluation of the first children
            for child in self.root.children:
                value = self.simulate_and_evaluate(child)
                self.backpropagate(child, value)

        for i in range(self.config.NUM_ITERATIONS):
            print(f"--- Iteration {i+1}/{self.config.NUM_ITERATIONS} ---")
            leaf_node = self.select(self.root)
            print(f"  Select ({leaf_node.get_depth()}): ...{leaf_node.narrative_block[-80:].strip()}")

            if not leaf_node.is_terminal:
                # Expand the leaf node, creating new children
                self.expand(leaf_node)
                # Evaluate each new child
                for child in leaf_node.children:
                    if child.N == 0:
                        value = self.simulate_and_evaluate(child)
                        self.backpropagate(child, value)
            print("-" * 25)

    def get_best_sequence(self, debug=True):
        path_nodes = []
        node = self.root
        while node is not None:
            path_nodes.append(node)
            if not node.children: break
            # Choose the most visited child; if tied, choose the one with the highest value.
            node = max(node.children, key=lambda n: (n.N, n.value))

        # The final text is the concatenation of all narrative blocks on the best path
        main_text = "\n\n".join(n.narrative_block for n in path_nodes)
        
        if not debug: return main_text

        debug_parts = []
        for i, n in enumerate(path_nodes):
            debug_parts.append(f"--- Node {i} (Depth: {n.get_depth()}, Visits: {n.N}, Value: {n.value:.4f}) ---")
            if n.thoughts:
                thought_snippet = (n.thoughts[:400] + '...') if len(n.thoughts) > 400 else n.thoughts
                debug_parts.append(f"  [REASONING]:\n{thought_snippet}")
            debug_parts.append(f"  [NARRATIVE BLOCK]:\n{n.narrative_block}\n")
        debug_trace = "\n".join(debug_parts)
        
        return f"--- FINAL NARRATIVE ---\n{main_text}\n\n\n--- DEBUG TRACE (BEST PATH) ---\n{debug_trace}"

# --- 6. Helper & Main Execution ---
def build_faiss_index(vectors: np.ndarray):
    print("Building FAISS index...")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(vectors)
    index.add(vectors)
    print("FAISS index built.")
    return index

if __name__ == "__main__":
    cfg = Config()
    llm_interface = LLMInterface(cfg.MODEL_NAME, cfg.DEVICE)

    # Load pre-computed data
    model_name_for_file = os.path.basename(cfg.MODEL_NAME)
    if not model_name_for_file:
        model_name_for_file = "default_model"

    vectors_filename = f"../database/{model_name_for_file}-db/vectors_{model_name_for_file}.npy"
    faiss_index_filename = f"../database/{model_name_for_file}-db/faiss_index_{model_name_for_file}.bin"
    metadata_filename = f"../database/{model_name_for_file}-db/paper_metadata.json"

    print(f"Loading data from {vectors_filename}, {faiss_index_filename}, and {metadata_filename}...")

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
    
    theme_generator = AutomatedThemeGenerator(llm_interface, novelty_documents, vectors, cfg)
    automated_theme, theme_elaboration, concept_original_list = theme_generator.generate_theme()

    print(f"\n>>> Automatically Generated Theme: '{automated_theme}' <<<")
    print(f">>> Elaboration: '{theme_elaboration}' <<<\n")

    # The 'theme' for prompting remains the concise one.
    mcts = CG_MCTS(llm_interface, novelty_db=novelty_db, config=cfg, theme=automated_theme, init_narrative = f"{automated_theme}\n\n{theme_elaboration}")

    start_time = time.time()
    mcts.search()
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.2f} seconds.")

    final_narrative = mcts.get_best_sequence(debug=True)

    print("\n" + "="*20 + " Final Generated Narrative " + "="*20)
    print(final_narrative)
