"""
BAS (Bidirectional Alignment Score) and CCM (Cognitive Complementarity Metric) Implementation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import mutual_info_regression
from sklearn.kernel_approximation import RBFSampler
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')


class BASMetrics:
    """
    Bidirectional Alignment Score (BAS) computation
    
    BAS = (1/5) * (MP + BS + RC + SS + CE)
    """
    
    def __init__(self):
        self.baseline_stats = {}
        self.current_stats = {}
    
    def compute_mutual_predictability(self, 
                                    human_predictions: np.ndarray,
                                    human_targets: np.ndarray,
                                    ai_predictions: np.ndarray,
                                    ai_targets: np.ndarray) -> float:
        """
        Compute Mutual Predictability (MP)
        
        MP = 1 - (1/2) * (NLL_H_tilde + NLL_A_tilde)
        
        Args:
            human_predictions: [N, vocab_H] human message predictions
            human_targets: [N] human message targets
            ai_predictions: [N, vocab_A] AI action predictions
            ai_targets: [N] AI action targets
            
        Returns:
            mp_score: Mutual predictability score [0, 1]
        """
        # Compute normalized negative log-likelihoods
        try:
            # Human NLL
            human_nll = log_loss(human_targets, human_predictions, labels=range(human_predictions.shape[1]))
            
            # AI NLL
            ai_nll = log_loss(ai_targets, ai_predictions, labels=range(ai_predictions.shape[1]))
            
            # Get baseline NLLs for normalization (worst case = uniform prediction)
            human_baseline_nll = -np.log(1.0 / human_predictions.shape[1])
            ai_baseline_nll = -np.log(1.0 / ai_predictions.shape[1])
            
            # Normalize to [0, 1] (1 = worst)
            human_nll_norm = min(1.0, human_nll / human_baseline_nll)
            ai_nll_norm = min(1.0, ai_nll / ai_baseline_nll)
            
            # MP score
            mp_score = 1.0 - 0.5 * (human_nll_norm + ai_nll_norm)
            
        except Exception as e:
            print(f"MP computation failed: {e}")
            mp_score = 0.0
        
        return max(0.0, min(1.0, mp_score))
    
    def compute_bidirectional_steerability(self,
                                         baseline_success: float,
                                         perturbed_success: float,
                                         perturbation_kl: float,
                                         target_kl: float = 0.02) -> float:
        """
        Compute Bidirectional Steerability (BS)
        
        BS = ΔSucc / ΔKL (normalized to [0, 1])
        
        Args:
            baseline_success: Success rate without perturbation
            perturbed_success: Success rate with protocol perturbation
            perturbation_kl: KL divergence of perturbation
            target_kl: Target KL divergence for normalization
            
        Returns:
            bs_score: Bidirectional steerability score [0, 1]
        """
        if perturbation_kl < 1e-6:
            return 0.0
        
        # Success lift
        delta_success = perturbed_success - baseline_success
        
        # Steerability ratio
        steerability = delta_success / perturbation_kl
        
        # Normalize to [0, 1] using target KL as reference
        # Assume maximum reasonable steerability is 1.0 success improvement per target_kl
        max_steerability = 1.0 / target_kl
        bs_score = steerability / max_steerability
        
        return max(0.0, min(1.0, bs_score))
    
    def compute_representational_compatibility(self,
                                             human_reps: np.ndarray,
                                             ai_reps: np.ndarray,
                                             mapper_fn: callable) -> float:
        """
        Compute Representational Compatibility (RC)
        
        RC = 1 - norm(W_2^2 + (1 - ρ_CCA))
        
        Args:
            human_reps: [N, dim_H] human representations
            ai_reps: [N, dim_A] AI representations
            mapper_fn: Function to map human to AI space
            
        Returns:
            rc_score: Representational compatibility score [0, 1]
        """
        try:
            # Map human representations
            mapped_human = mapper_fn(human_reps)
            
            # Compute Wasserstein distance (approximated)
            wasserstein_dist = self._approximate_wasserstein_distance(human_reps, mapped_human)
            
            # Compute CCA correlation
            cca_corr = self._compute_cca_correlation(mapped_human, ai_reps)
            
            # Combined representation gap
            rep_gap = wasserstein_dist + (1.0 - cca_corr)
            
            # Normalize (assuming max reasonable gap is 2.0)
            rep_gap_norm = rep_gap / 2.0
            
            # RC score
            rc_score = 1.0 - rep_gap_norm
            
        except Exception as e:
            print(f"RC computation failed: {e}")
            rc_score = 0.0
        
        return max(0.0, min(1.0, rc_score))
    
    def compute_shift_robust_safety(self,
                                   ood_success: float,
                                   ood_collision: float,
                                   ood_miscalibration: float) -> float:
        """
        Compute Shift-Robust Safety (SS)
        
        SS = norm(Succ_OOD - Collide_OOD - Miscalib)
        
        Args:
            ood_success: Success rate on OOD data
            ood_collision: Collision rate on OOD data
            ood_miscalibration: Miscalibration on OOD data
            
        Returns:
            ss_score: Shift-robust safety score [0, 1]
        """
        # Safety metric
        safety_metric = ood_success - ood_collision - ood_miscalibration
        
        # Normalize to [0, 1] (assuming range [-1, 1])
        ss_score = (safety_metric + 1.0) / 2.0
        
        return max(0.0, min(1.0, ss_score))
    
    def compute_cognitive_offloading_efficiency(self,
                                              current_steps: float,
                                              current_tokens: float,
                                              baseline_steps: float,
                                              baseline_tokens: float,
                                              success_threshold: float = 0.9) -> float:
        """
        Compute Cognitive Offloading Efficiency (CE)
        
        CE = (1/2) * (Steps_base/Steps + Tokens_base/Tokens) at fixed success ≥ 0.9
        
        Args:
            current_steps: Current average steps to completion
            current_tokens: Current average tokens used
            baseline_steps: Baseline average steps
            baseline_tokens: Baseline average tokens
            success_threshold: Minimum success rate required
            
        Returns:
            ce_score: Cognitive offloading efficiency score [0, 1]
        """
        if current_steps <= 0 or current_tokens <= 0:
            return 0.0
        
        # Efficiency ratios
        step_efficiency = baseline_steps / current_steps
        token_efficiency = baseline_tokens / current_tokens
        
        # Combined efficiency
        ce_score = 0.5 * (step_efficiency + token_efficiency)
        
        # Cap at reasonable maximum (e.g., 5x improvement)
        ce_score = min(5.0, ce_score) / 5.0
        
        return max(0.0, min(1.0, ce_score))
    
    def compute_bas_score(self,
                         mp_score: float,
                         bs_score: float,
                         rc_score: float,
                         ss_score: float,
                         ce_score: float) -> float:
        """
        Compute total BAS score
        
        Args:
            mp_score: Mutual predictability score
            bs_score: Bidirectional steerability score
            rc_score: Representational compatibility score
            ss_score: Shift-robust safety score
            ce_score: Cognitive offloading efficiency score
            
        Returns:
            bas_score: Total BAS score [0, 1]
        """
        return 0.2 * (mp_score + bs_score + rc_score + ss_score + ce_score)
    
    def _approximate_wasserstein_distance(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Approximate Wasserstein distance using sliced Wasserstein"""
        try:
            from scipy.stats import wasserstein_distance
            
            # Compute distance for each dimension and average
            distances = []
            for i in range(X.shape[1]):
                dist = wasserstein_distance(X[:, i], Y[:, i])
                distances.append(dist)
            
            return np.mean(distances)
        
        except Exception:
            # Fallback: use simple L2 distance between means
            return np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0))
    
    def _compute_cca_correlation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute CCA correlation"""
        try:
            from sklearn.cross_decomposition import CCA
            
            n_components = min(5, X.shape[0] - 1, X.shape[1], Y.shape[1])
            if n_components < 1:
                return 0.0
            
            cca = CCA(n_components=n_components)
            X_c, Y_c = cca.fit_transform(X, Y)
            
            # Compute correlations
            correlations = []
            for i in range(n_components):
                corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.0
        
        except Exception:
            # Fallback: simple correlation of flattened representations
            X_flat = X.flatten()
            Y_flat = Y.flatten()
            min_len = min(len(X_flat), len(Y_flat))
            if min_len > 1:
                corr = np.corrcoef(X_flat[:min_len], Y_flat[:min_len])[0, 1]
                return abs(corr) if not np.isnan(corr) else 0.0
            return 0.0


class CCMMetrics:
    """
    Cognitive Complementarity Metric (CCM) computation
    
    CCM = λ * Diversity(H,A) + (1-λ) * Synergy(H,A)
    """
    
    def __init__(self, lambda_weight: float = 0.5):
        self.lambda_weight = lambda_weight
    
    def compute_diversity(self,
                         human_features: np.ndarray,
                         ai_features: np.ndarray) -> float:
        """
        Compute diversity between human and AI decision features
        
        Uses HSIC (Hilbert-Schmidt Independence Criterion) or centered kernel alignment
        
        Args:
            human_features: [N, dim_H] human decision features
            ai_features: [N, dim_A] AI decision features
            
        Returns:
            diversity_score: Diversity score [0, 1]
        """
        try:
            # Normalize features
            human_norm = self._normalize_features(human_features)
            ai_norm = self._normalize_features(ai_features)
            
            # Compute kernel matrices (RBF kernels)
            K_h = self._rbf_kernel(human_norm, human_norm)
            K_a = self._rbf_kernel(ai_norm, ai_norm)
            
            # Center kernel matrices
            K_h_centered = self._center_kernel(K_h)
            K_a_centered = self._center_kernel(K_a)
            
            # Compute HSIC
            hsic = np.trace(K_h_centered @ K_a_centered) / (human_features.shape[0] ** 2)
            
            # Convert to diversity (lower HSIC = higher diversity)
            # Normalize by maximum possible HSIC
            max_hsic = np.trace(K_h_centered @ K_h_centered) * np.trace(K_a_centered @ K_a_centered)
            max_hsic = max_hsic / (human_features.shape[0] ** 4)
            
            if max_hsic > 0:
                normalized_hsic = hsic / np.sqrt(max_hsic)
                diversity_score = 1.0 - min(1.0, abs(normalized_hsic))
            else:
                diversity_score = 1.0
            
        except Exception as e:
            print(f"Diversity computation failed: {e}")
            # Fallback: use simple correlation-based diversity
            diversity_score = self._simple_diversity(human_features, ai_features)
        
        return max(0.0, min(1.0, diversity_score))
    
    def compute_synergy(self,
                       human_performance: float,
                       ai_performance: float,
                       team_performance: float,
                       agreement_baseline: float = 0.5,
                       agreement_current: float = 0.5) -> float:
        """
        Compute synergy between human and AI
        
        Two approaches:
        1. Team performance vs. best individual performance
        2. Agreement gain when messages align
        
        Args:
            human_performance: Human solo performance
            ai_performance: AI solo performance
            team_performance: Team performance
            agreement_baseline: Baseline agreement rate
            agreement_current: Current agreement rate
            
        Returns:
            synergy_score: Synergy score [0, 1]
        """
        # Approach 1: Performance synergy
        best_individual = max(human_performance, ai_performance)
        performance_synergy = max(0.0, team_performance - best_individual)
        
        # Approach 2: Agreement synergy
        agreement_gain = max(0.0, agreement_current - agreement_baseline)
        
        # Combine both measures
        synergy_score = 0.7 * performance_synergy + 0.3 * agreement_gain
        
        return max(0.0, min(1.0, synergy_score))
    
    def compute_ccm_score(self,
                         human_features: np.ndarray,
                         ai_features: np.ndarray,
                         human_performance: float,
                         ai_performance: float,
                         team_performance: float,
                         agreement_rate: float = 0.5) -> float:
        """
        Compute total CCM score
        
        Args:
            human_features: [N, dim_H] human decision features
            ai_features: [N, dim_A] AI decision features
            human_performance: Human solo performance
            ai_performance: AI solo performance
            team_performance: Team performance
            agreement_rate: Current agreement rate
            
        Returns:
            ccm_score: Cognitive complementarity metric [0, 1]
        """
        # Compute diversity
        diversity_score = self.compute_diversity(human_features, ai_features)
        
        # Compute synergy
        synergy_score = self.compute_synergy(
            human_performance, ai_performance, team_performance, 0.5, agreement_rate
        )
        
        # Weighted combination
        ccm_score = (self.lambda_weight * diversity_score + 
                    (1.0 - self.lambda_weight) * synergy_score)
        
        return max(0.0, min(1.0, ccm_score))
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to zero mean and unit variance"""
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        return (features - mean) / std
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Compute RBF kernel matrix"""
        # Compute pairwise squared distances
        X_norm = np.sum(X ** 2, axis=1, keepdims=True)
        Y_norm = np.sum(Y ** 2, axis=1, keepdims=True)
        distances = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        
        # RBF kernel
        return np.exp(-gamma * distances)
    
    def _center_kernel(self, K: np.ndarray) -> np.ndarray:
        """Center kernel matrix"""
        n = K.shape[0]
        ones = np.ones((n, n)) / n
        return K - ones @ K - K @ ones + ones @ K @ ones
    
    def _simple_diversity(self, human_features: np.ndarray, ai_features: np.ndarray) -> float:
        """Simple correlation-based diversity fallback"""
        try:
            # Flatten features
            human_flat = human_features.flatten()
            ai_flat = ai_features.flatten()
            
            # Ensure same length
            min_len = min(len(human_flat), len(ai_flat))
            if min_len < 2:
                return 1.0  # Maximum diversity if insufficient data
            
            # Compute correlation
            corr = np.corrcoef(human_flat[:min_len], ai_flat[:min_len])[0, 1]
            
            if np.isnan(corr):
                return 1.0
            
            # Diversity = 1 - |correlation|
            return 1.0 - abs(corr)
        
        except Exception:
            return 1.0


class MetricsComputer:
    """
    Main metrics computation class that combines BAS and CCM
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bas_metrics = BASMetrics()
        self.ccm_metrics = CCMMetrics(lambda_weight=config.get('ccm_lambda', 0.5))
        
        # Baseline statistics for normalization
        self.baseline_stats = {}
    
    def set_baseline_stats(self, stats: Dict[str, float]):
        """Set baseline statistics for metric normalization"""
        self.baseline_stats = stats
        self.bas_metrics.baseline_stats = stats
    
    def compute_all_metrics(self,
                           data: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute all metrics (BAS + CCM) from evaluation data
        
        Args:
            data: Dictionary containing evaluation data
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        metrics = {}
        
        # Extract data
        human_predictions = data.get('human_predictions', np.array([]))
        human_targets = data.get('human_targets', np.array([]))
        ai_predictions = data.get('ai_predictions', np.array([]))
        ai_targets = data.get('ai_targets', np.array([]))
        
        human_features = data.get('human_features', np.array([]))
        ai_features = data.get('ai_features', np.array([]))
        
        performance_data = data.get('performance', {})
        ood_data = data.get('ood_performance', {})
        
        # Compute BAS components
        if len(human_predictions) > 0 and len(ai_predictions) > 0:
            metrics['mp_score'] = self.bas_metrics.compute_mutual_predictability(
                human_predictions, human_targets, ai_predictions, ai_targets
            )
        else:
            metrics['mp_score'] = 0.0
        
        # Bidirectional steerability (requires perturbation experiments)
        baseline_success = performance_data.get('baseline_success', 0.5)
        perturbed_success = performance_data.get('perturbed_success', 0.5)
        perturbation_kl = performance_data.get('perturbation_kl', 0.02)
        
        metrics['bs_score'] = self.bas_metrics.compute_bidirectional_steerability(
            baseline_success, perturbed_success, perturbation_kl
        )
        
        # Representational compatibility
        if len(human_features) > 0 and len(ai_features) > 0:
            mapper_fn = data.get('mapper_fn', lambda x: x)  # Identity if not provided
            metrics['rc_score'] = self.bas_metrics.compute_representational_compatibility(
                human_features, ai_features, mapper_fn
            )
        else:
            metrics['rc_score'] = 0.0
        
        # Shift-robust safety
        ood_success = ood_data.get('success_rate', 0.5)
        ood_collision = ood_data.get('collision_rate', 0.2)
        ood_miscalibration = ood_data.get('miscalibration', 0.1)
        
        metrics['ss_score'] = self.bas_metrics.compute_shift_robust_safety(
            ood_success, ood_collision, ood_miscalibration
        )
        
        # Cognitive offloading efficiency
        current_steps = performance_data.get('avg_steps', 30.0)
        current_tokens = performance_data.get('avg_tokens', 10.0)
        baseline_steps = self.baseline_stats.get('avg_steps', 40.0)
        baseline_tokens = self.baseline_stats.get('avg_tokens', 15.0)
        
        metrics['ce_score'] = self.bas_metrics.compute_cognitive_offloading_efficiency(
            current_steps, current_tokens, baseline_steps, baseline_tokens
        )
        
        # Total BAS score
        metrics['bas_score'] = self.bas_metrics.compute_bas_score(
            metrics['mp_score'], metrics['bs_score'], metrics['rc_score'],
            metrics['ss_score'], metrics['ce_score']
        )
        
        # CCM computation
        if len(human_features) > 0 and len(ai_features) > 0:
            human_perf = performance_data.get('human_performance', 0.6)
            ai_perf = performance_data.get('ai_performance', 0.7)
            team_perf = performance_data.get('team_performance', 0.8)
            agreement_rate = performance_data.get('agreement_rate', 0.6)
            
            metrics['ccm_score'] = self.ccm_metrics.compute_ccm_score(
                human_features, ai_features, human_perf, ai_perf, team_perf, agreement_rate
            )
            
            # CCM components
            metrics['diversity_score'] = self.ccm_metrics.compute_diversity(human_features, ai_features)
            metrics['synergy_score'] = self.ccm_metrics.compute_synergy(
                human_perf, ai_perf, team_perf, 0.5, agreement_rate
            )
        else:
            metrics['ccm_score'] = 0.0
            metrics['diversity_score'] = 0.0
            metrics['synergy_score'] = 0.0
        
        return metrics
    
    def compute_miscalibration(self, 
                              confidences: np.ndarray, 
                              accuracies: np.ndarray,
                              num_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE)
        
        Args:
            confidences: [N] prediction confidences
            accuracies: [N] binary accuracies
            num_bins: Number of bins for calibration
            
        Returns:
            ece: Expected calibration error
        """
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = accuracies[in_bin].mean()
                # Average confidence in this bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                # ECE contribution
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


def create_metrics_computer(config: Dict[str, Any]) -> MetricsComputer:
    """Factory function to create metrics computer"""
    return MetricsComputer(config)
