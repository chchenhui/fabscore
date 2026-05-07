import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from dataset import SyntheticFairnessDataset
from model import ModelFactory, FairnessAwareLogisticRegression, AdversarialFairnessClassifier
from evaluate import ModelEvaluator
import pickle
import os

class FairnessTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.trained_models = {}
        self.training_results = {}

    def train_baseline_models(self, X_train, y_train, a_train):
        """
        Train baseline models without fairness constraints
        """
        print("Training baseline models...")
        baseline_models = ModelFactory.create_baseline_models()

        for name, model in baseline_models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

        return baseline_models

    def train_fairness_models(self, X_train, y_train, a_train, fairness_penalties=[0.01, 0.1, 0.5]):
        """
        Train fairness-aware models with different penalty parameters
        """
        print("Training fairness-aware models...")
        input_dim = X_train.shape[1]

        for penalty in fairness_penalties:
            print(f"Training with fairness penalty: {penalty}")

            # Fairness-aware Logistic Regression
            fair_lr = FairnessAwareLogisticRegression(fairness_penalty=penalty)
            fair_lr.fit(X_train, y_train, a_train)
            self.trained_models[f'FairnessLR_lambda_{penalty}'] = fair_lr

            # Adversarial Fairness Network
            adv_net = AdversarialFairnessClassifier(
                input_dim=input_dim,
                fairness_penalty=penalty,
                epochs=50,
                learning_rate=0.01
            )
            print(f"Training Adversarial Network with lambda={penalty}...")
            adv_net.fit(X_train, y_train, a_train)
            self.trained_models[f'AdversarialNet_lambda_{penalty}'] = adv_net

    def perform_ablation_study(self, X_train, y_train, a_train, X_test, y_test, a_test):
        """
        Perform ablation study over different fairness penalty values
        """
        print("Performing ablation study...")
        penalty_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        ablation_results = []

        evaluator = ModelEvaluator()

        for penalty in penalty_values:
            print(f"Ablation: Training with penalty = {penalty}")

            # Train fairness-aware LR
            fair_lr = FairnessAwareLogisticRegression(fairness_penalty=penalty)
            fair_lr.fit(X_train, y_train, a_train)

            # Evaluate
            results = evaluator.evaluate_model(
                fair_lr, X_test, y_test, a_test,
                model_name=f'FairnessLR_ablation_{penalty}'
            )
            results['penalty'] = penalty
            ablation_results.append(results)

        self.training_results['ablation_study'] = ablation_results
        return ablation_results

    def save_models(self, save_dir):
        """
        Save trained models to disk
        """
        os.makedirs(save_dir, exist_ok=True)

        for name, model in self.trained_models.items():
            # Skip PyTorch models for now (they need special handling)
            if 'AdversarialNet' not in name:
                with open(f"{save_dir}/{name}.pkl", 'wb') as f:
                    pickle.dump(model, f)

        print(f"Models saved to {save_dir}")

    def load_models(self, save_dir):
        """
        Load trained models from disk
        """
        loaded_models = {}
        for filename in os.listdir(save_dir):
            if filename.endswith('.pkl'):
                model_name = filename[:-4]  # Remove .pkl extension
                with open(f"{save_dir}/{filename}", 'rb') as f:
                    loaded_models[model_name] = pickle.load(f)

        self.trained_models.update(loaded_models)
        print(f"Loaded {len(loaded_models)} models from {save_dir}")
        return loaded_models


class ExperimentRunner:
    def __init__(self, n_samples=1000, bias_strength=0.3, random_state=42):
        self.n_samples = n_samples
        self.bias_strength = bias_strength
        self.random_state = random_state
        self.trainer = FairnessTrainer(random_state=random_state)
        self.evaluator = ModelEvaluator()

    def run_full_experiment(self):
        """
        Run the complete experimental pipeline
        """
        print("=" * 60)
        print("STARTING FAIRNESS EXPERIMENT")
        print("=" * 60)

        # Generate dataset
        print(f"Generating synthetic dataset (n={self.n_samples}, bias={self.bias_strength})")
        dataset = SyntheticFairnessDataset(
            n_samples=self.n_samples,
            bias_strength=self.bias_strength,
            random_state=self.random_state
        )
        X_train, X_test, y_train, y_test, a_train, a_test, scaler = dataset.get_train_test_split()

        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")

        # Dataset bias analysis
        bias_stats = dataset.compute_bias_statistics()
        print(f"Dataset bias difference: {bias_stats['bias_difference']:.3f}")

        # Train baseline models
        baseline_models = self.trainer.train_baseline_models(X_train, y_train, a_train)

        # Train fairness models
        self.trainer.train_fairness_models(X_train, y_train, a_train)

        # Evaluate all models
        print("\n" + "=" * 60)
        print("EVALUATING ALL MODELS")
        print("=" * 60)
        all_results = self.evaluator.evaluate_multiple_models(
            self.trainer.trained_models, X_test, y_test, a_test
        )

        # Ablation study
        ablation_results = self.trainer.perform_ablation_study(
            X_train, y_train, a_train, X_test, y_test, a_test
        )

        # Create results dataframe
        results_df = self.evaluator.create_results_dataframe(all_results)
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(results_df.round(4))

        # Detailed results
        self.evaluator.print_detailed_results(all_results)

        return {
            'results_df': results_df,
            'detailed_results': all_results,
            'ablation_results': ablation_results,
            'dataset_stats': bias_stats,
            'data_splits': (X_train, X_test, y_train, y_test, a_train, a_test)
        }

if __name__ == "__main__":
    # Run the complete experiment
    experiment = ExperimentRunner(n_samples=1000, bias_strength=0.3, random_state=42)
    results = experiment.run_full_experiment()

    # Save results
    results_dir = "../results"
    os.makedirs(results_dir, exist_ok=True)

    # Save results dataframe
    results['results_df'].to_csv(f"{results_dir}/model_comparison.csv", index=False)
    print(f"\nResults saved to {results_dir}/model_comparison.csv")