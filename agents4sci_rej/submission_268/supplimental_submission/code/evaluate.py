import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

class FairnessMetrics:
    def __init__(self):
        pass

    @staticmethod
    def demographic_parity(y_pred, a):
        """
        Demographic Parity: P(ŷ=1|a=0) should equal P(ŷ=1|a=1)
        Returns the absolute difference.
        """
        group_0_rate = np.mean(y_pred[a == 0])
        group_1_rate = np.mean(y_pred[a == 1])
        return abs(group_0_rate - group_1_rate)

    @staticmethod
    def equal_opportunity(y_true, y_pred, a):
        """
        Equal Opportunity: P(ŷ=1|y=1,a=0) should equal P(ŷ=1|y=1,a=1)
        Returns the absolute difference in true positive rates.
        """
        # True positive rates for each group
        group_0_mask = (a == 0) & (y_true == 1)
        group_1_mask = (a == 1) & (y_true == 1)

        if np.sum(group_0_mask) == 0 or np.sum(group_1_mask) == 0:
            return np.nan

        tpr_0 = np.mean(y_pred[group_0_mask])
        tpr_1 = np.mean(y_pred[group_1_mask])
        return abs(tpr_0 - tpr_1)

    @staticmethod
    def equalized_odds(y_true, y_pred, a):
        """
        Equalized Odds: Both TPR and FPR should be equal across groups
        Returns the maximum difference between TPR and FPR differences.
        """
        # True positive rates
        group_0_pos = (a == 0) & (y_true == 1)
        group_1_pos = (a == 1) & (y_true == 1)
        group_0_neg = (a == 0) & (y_true == 0)
        group_1_neg = (a == 1) & (y_true == 0)

        if (np.sum(group_0_pos) == 0 or np.sum(group_1_pos) == 0 or
            np.sum(group_0_neg) == 0 or np.sum(group_1_neg) == 0):
            return np.nan

        tpr_0 = np.mean(y_pred[group_0_pos])
        tpr_1 = np.mean(y_pred[group_1_pos])
        fpr_0 = np.mean(y_pred[group_0_neg])
        fpr_1 = np.mean(y_pred[group_1_neg])

        tpr_diff = abs(tpr_0 - tpr_1)
        fpr_diff = abs(fpr_0 - fpr_1)

        return max(tpr_diff, fpr_diff)

    @staticmethod
    def compute_confusion_matrices(y_true, y_pred, a):
        """
        Compute confusion matrices for each group
        """
        results = {}
        for group in [0, 1]:
            mask = (a == group)
            if np.sum(mask) > 0:
                cm = confusion_matrix(y_true[mask], y_pred[mask])
                results[f'group_{group}'] = cm
        return results

    @staticmethod
    def compute_all_metrics(y_true, y_pred, a):
        """
        Compute all fairness and performance metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        dp = FairnessMetrics.demographic_parity(y_pred, a)
        eo = FairnessMetrics.equal_opportunity(y_true, y_pred, a)
        eodds = FairnessMetrics.equalized_odds(y_true, y_pred, a)

        # Group-wise accuracies
        acc_group_0 = accuracy_score(y_true[a == 0], y_pred[a == 0]) if np.sum(a == 0) > 0 else np.nan
        acc_group_1 = accuracy_score(y_true[a == 1], y_pred[a == 1]) if np.sum(a == 1) > 0 else np.nan

        # Confusion matrices
        cms = FairnessMetrics.compute_confusion_matrices(y_true, y_pred, a)

        return {
            'accuracy': accuracy,
            'demographic_parity': dp,
            'equal_opportunity': eo,
            'equalized_odds': eodds,
            'accuracy_group_0': acc_group_0,
            'accuracy_group_1': acc_group_1,
            'confusion_matrices': cms
        }


class ModelEvaluator:
    def __init__(self):
        self.metrics = FairnessMetrics()

    def evaluate_model(self, model, X_test, y_test, a_test, model_name="Model"):
        """
        Evaluate a single model on test data
        """
        # Get predictions
        if hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
        else:
            raise ValueError("Model must have a predict method")

        # Ensure predictions are binary
        if y_pred.dtype == float:
            y_pred = (y_pred > 0.5).astype(int)

        # Compute metrics
        results = self.metrics.compute_all_metrics(y_test, y_pred, a_test)
        results['model_name'] = model_name

        return results

    def evaluate_multiple_models(self, models_dict, X_test, y_test, a_test):
        """
        Evaluate multiple models and return comparison results
        """
        all_results = []

        for model_name, model in models_dict.items():
            try:
                results = self.evaluate_model(model, X_test, y_test, a_test, model_name)
                all_results.append(results)
                print(f"Evaluated {model_name}: Accuracy={results['accuracy']:.3f}, "
                      f"DP={results['demographic_parity']:.3f}")
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")

        return all_results

    def create_results_dataframe(self, results_list):
        """
        Convert results to a pandas DataFrame for easy analysis
        """
        df_data = []
        for result in results_list:
            row = {
                'model': result['model_name'],
                'accuracy': result['accuracy'],
                'demographic_parity': result['demographic_parity'],
                'equal_opportunity': result['equal_opportunity'],
                'equalized_odds': result['equalized_odds'],
                'accuracy_group_0': result['accuracy_group_0'],
                'accuracy_group_1': result['accuracy_group_1']
            }
            df_data.append(row)

        return pd.DataFrame(df_data)

    def print_detailed_results(self, results_list):
        """
        Print detailed evaluation results
        """
        print("=" * 80)
        print("DETAILED MODEL EVALUATION RESULTS")
        print("=" * 80)

        for result in results_list:
            print(f"\nModel: {result['model_name']}")
            print("-" * 40)
            print(f"Overall Accuracy: {result['accuracy']:.4f}")
            print(f"Group 0 Accuracy: {result['accuracy_group_0']:.4f}")
            print(f"Group 1 Accuracy: {result['accuracy_group_1']:.4f}")
            print(f"Demographic Parity: {result['demographic_parity']:.4f}")
            print(f"Equal Opportunity: {result['equal_opportunity']:.4f}")
            print(f"Equalized Odds: {result['equalized_odds']:.4f}")

            # Print confusion matrices
            print("\nConfusion Matrices:")
            for group, cm in result['confusion_matrices'].items():
                print(f"{group}: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")

if __name__ == "__main__":
    # Test the evaluation metrics
    from dataset import SyntheticFairnessDataset
    from model import ModelFactory

    # Generate test data
    dataset = SyntheticFairnessDataset(n_samples=500, bias_strength=0.3)
    X_train, X_test, y_train, y_test, a_train, a_test, scaler = dataset.get_train_test_split()

    # Train baseline models
    baseline_models = ModelFactory.create_baseline_models()
    for name, model in baseline_models.items():
        model.fit(X_train, y_train)

    # Evaluate models
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_multiple_models(baseline_models, X_test, y_test, a_test)
    evaluator.print_detailed_results(results)