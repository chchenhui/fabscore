
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticEntropy:
    def __init__(self, embedding_model_name: str = 'Alibaba-NLP/gte-large-en-v1.5'):
        """
        Initializes the Semantic Entropy calculator.

        Args:
            embedding_model_name (str): The name of the sentence-transformer model to use.
        """
        logging.info(f"Loading embedding model: {embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load embedding model {embedding_model_name}: {e}")
            raise

    def calculate_entropy(self, responses: list[str], distance_threshold: float = 0.2, return_diagnostics: bool = False):
        """
        Calculates the semantic entropy for a list of responses.

        Args:
            responses (list[str]): A list of N string responses.
            distance_threshold (float): The distance threshold for agglomerative clustering.
            return_diagnostics (bool): If True, returns dict with diagnostics; if False, returns float entropy.

        Returns:
            float or dict: Shannon entropy score, or dict with entropy and diagnostic metrics.
        """
        if not responses or len(responses) < 2:
            if return_diagnostics:
                response_lengths = [len(r) for r in responses] if responses else [0]
                return {
                    "semantic_entropy": 0.0,
                    "response_count": len(responses),
                    "avg_response_length": np.mean(response_lengths),
                    "min_response_length": min(response_lengths),
                    "max_response_length": max(response_lengths), 
                    "std_response_length": np.std(response_lengths),
                    "num_clusters": 1 if responses else 0,
                    "cluster_sizes": [len(responses)] if responses else []
                }
            return 0.0

        # Response length statistics
        response_lengths = [len(response) for response in responses]

        embeddings = self.embedding_model.encode(responses, convert_to_tensor=False)
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Agglomerative clustering with cosine distance
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric='cosine',
            linkage='average',
            distance_threshold=distance_threshold
        ).fit(embeddings)

        cluster_labels = clustering.labels_
        
        # Count the number of responses in each cluster
        cluster_counts = Counter(cluster_labels)
        num_responses = len(responses)
        
        # Calculate probabilities
        probabilities = [count / num_responses for count in cluster_counts.values()]
        
        # Calculate Shannon entropy
        entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
        
        # Fix floating point issue: convert -0.0 to 0.0
        if entropy == 0.0:
            entropy = 0.0
        
        if return_diagnostics:
            return {
                "semantic_entropy": entropy,
                "response_count": len(responses),
                "avg_response_length": np.mean(response_lengths),
                "min_response_length": min(response_lengths),
                "max_response_length": max(response_lengths),
                "std_response_length": np.std(response_lengths),
                "num_clusters": len(cluster_counts),
                "cluster_sizes": sorted(list(cluster_counts.values()), reverse=True)
            }
        
        return entropy

if __name__ == '__main__':
    # Example Usage
    responses_consistent = [
        "The capital of France is Paris.",
        "Paris is the capital of France.",
        "France's capital is Paris."
    ]
    
    responses_diverse = [
        "The capital of France is Paris.",
        "I think the answer is Berlin.",
        "The Eiffel Tower is a famous landmark."
    ]

    try:
        se_calculator = SemanticEntropy()
        
        entropy_consistent = se_calculator.calculate_entropy(responses_consistent)
        entropy_diverse = se_calculator.calculate_entropy(responses_diverse)
        
        print(f"Semantic Entropy for consistent responses: {entropy_consistent:.4f}")
        print(f"Semantic Entropy for diverse responses: {entropy_diverse:.4f}")

    except Exception as e:
        print(f"Could not run example: {e}")
