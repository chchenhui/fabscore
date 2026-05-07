
import numpy as np
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from Levenshtein import distance as levenshtein_distance
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaselineMetrics:
    def __init__(self, embedding_model_name: str = 'Alibaba-NLP/gte-large-en-v1.5'):
        """
        Initializes the baseline metrics calculator.

        Args:
            embedding_model_name (str): The name of the sentence-transformer model to use for embedding variance.
        """
        logging.info(f"Loading embedding model for variance calculation: {embedding_model_name}")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name, trust_remote_code=True)
            logging.info("Embedding model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load embedding model {embedding_model_name}: {e}")
            raise

    def calculate_metrics(self, responses: list[str]) -> dict:
        """
        Calculates all baseline consistency metrics for a list of responses.

        Args:
            responses (list[str]): A list of N string responses.

        Returns:
            dict: A dictionary containing the calculated scores:
                  {'avg_pairwise_bertscore', 'embedding_variance', 'levenshtein_variance'}
        """
        if not responses or len(responses) < 2:
            return {
                'avg_pairwise_bertscore': 0.0,
                'embedding_variance': 0.0,
                'levenshtein_variance': 0.0
            }

        # 1. Average Pairwise BERTScore
        try:
            # Create pairs for BERTScore
            pairs = [(responses[i], responses[j]) for i in range(len(responses)) for j in range(i + 1, len(responses))]
            cands = [p[0] for p in pairs]
            refs = [p[1] for p in pairs]
            
            if cands and refs:
                _, _, F1 = bert_score(cands, refs, lang='en', verbose=False)
                avg_pairwise_bertscore = F1.mean().item()
            else:
                avg_pairwise_bertscore = 0.0
        except Exception as e:
            logging.warning(f"Could not calculate BERTScore: {e}")
            avg_pairwise_bertscore = 0.0

        # 2. Variance of Sentence Embeddings
        try:
            embeddings = self.embedding_model.encode(responses, convert_to_tensor=False)
            embedding_variance = np.var(embeddings, axis=0).mean()
        except Exception as e:
            logging.warning(f"Could not calculate embedding variance: {e}")
            embedding_variance = 0.0

        # 3. Variance of Levenshtein Distance
        try:
            lev_distances = [levenshtein_distance(responses[i], responses[j]) 
                             for i in range(len(responses)) for j in range(i + 1, len(responses))]
            levenshtein_variance = np.var(lev_distances) if lev_distances else 0.0
        except Exception as e:
            logging.warning(f"Could not calculate Levenshtein variance: {e}")
            levenshtein_variance = 0.0

        return {
            'avg_pairwise_bertscore': avg_pairwise_bertscore,
            'embedding_variance': float(embedding_variance),
            'levenshtein_variance': float(levenshtein_variance)
        }

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
        metrics_calculator = BaselineMetrics()
        
        consistent_scores = metrics_calculator.calculate_metrics(responses_consistent)
        diverse_scores = metrics_calculator.calculate_metrics(responses_diverse)
        
        print("--- Consistent Responses ---")
        for key, value in consistent_scores.items():
            print(f"{key}: {value:.4f}")
            
        print("\n--- Diverse Responses ---")
        for key, value in diverse_scores.items():
            print(f"{key}: {value:.4f}")

    except Exception as e:
        print(f"Could not run example: {e}")
