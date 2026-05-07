"""
Statistical Testing Module for Semantic Entropy Analysis

This module implements robust statistical tests for comparing detection methods,
with special handling for degenerate score distributions that are common in 
semantic entropy applications.

Key Features:
- Wilson confidence intervals for proportions (always valid)
- DeLong tests with degeneracy detection for AUROC comparisons  
- McNemar's test for paired binary predictions
- Extensive logging of statistical assumptions and limitations
- Scientific transparency about when standard tests are inappropriate

References:
- Wilson (1927): Probable inference, the law of succession, and statistical inference
- DeLong et al. (1988): Comparing areas under correlated ROC curves
- McNemar (1947): Note on the sampling error of the difference between correlated proportions
"""

import numpy as np
import json
import logging
from typing import Tuple, Dict, List, Optional, Union
from collections import Counter
from pathlib import Path

# Statistical computing imports
import scipy.stats as stats
from scipy.stats import chi2

try:
    from statsmodels.stats.contingency_tables import mcnemar
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available - using fallback McNemar implementation")

try:
    from MLstatkit import Delong_test
    MLSTATKIT_AVAILABLE = True
    logging.info("MLstatkit available - using standard DeLong implementation")
except ImportError:
    MLSTATKIT_AVAILABLE = False
    logging.info("MLstatkit not available - using scipy bootstrap fallback")

# Setup detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DistributionAnalysis:
    """Analysis of score distribution characteristics for statistical test validity."""
    
    def __init__(self, scores: np.ndarray, labels: np.ndarray, metric_name: str):
        self.scores = np.array(scores)
        self.labels = np.array(labels)
        self.metric_name = metric_name
        self.analysis = self._analyze()
    
    def _analyze(self) -> Dict:
        """Comprehensive analysis of score distribution."""
        unique_scores = np.unique(self.scores)
        n_unique = len(unique_scores)
        n_total = len(self.scores)
        
        # Count zeros and extreme values
        n_zeros = np.sum(self.scores == 0.0)
        n_inf = np.sum(np.isinf(self.scores))
        n_nan = np.sum(np.isnan(self.scores))
        
        # Score range and spread
        finite_scores = self.scores[np.isfinite(self.scores)]
        score_range = (float(np.min(finite_scores)), float(np.max(finite_scores))) if len(finite_scores) > 0 else (0.0, 0.0)
        
        # Most common values
        score_counts = Counter(self.scores)
        most_common = score_counts.most_common(3)
        
        # Class-conditional analysis
        harmful_scores = self.scores[self.labels == 1]
        benign_scores = self.scores[self.labels == 0]
        
        analysis = {
            'metric_name': self.metric_name,
            'n_samples': n_total,
            'n_unique_scores': n_unique,
            'unique_score_ratio': n_unique / n_total,
            'n_zeros': n_zeros,
            'zero_proportion': n_zeros / n_total,
            'n_infinite': n_inf,
            'n_nan': n_nan,
            'score_range': score_range,
            'most_common_values': [(float(val), count, count/n_total) for val, count in most_common],
            'class_separation': {
                'harmful_mean': float(np.mean(harmful_scores)) if len(harmful_scores) > 0 else 0.0,
                'benign_mean': float(np.mean(benign_scores)) if len(benign_scores) > 0 else 0.0,
                'harmful_std': float(np.std(harmful_scores)) if len(harmful_scores) > 0 else 0.0,
                'benign_std': float(np.std(benign_scores)) if len(benign_scores) > 0 else 0.0
            }
        }
        
        # Degeneracy assessment
        analysis['is_degenerate'] = self._assess_degeneracy(analysis)
        analysis['delong_valid'] = not analysis['is_degenerate']['severe']
        analysis['statistical_warnings'] = self._generate_warnings(analysis)
        
        return analysis
    
    def _assess_degeneracy(self, analysis: Dict) -> Dict:
        """Assess different levels of distribution degeneracy."""
        unique_ratio = analysis['unique_score_ratio']
        zero_prop = analysis['zero_proportion']
        
        return {
            'severe': unique_ratio < 0.05 or zero_prop > 0.9,  # <5% unique or >90% zeros
            'moderate': unique_ratio < 0.1 or zero_prop > 0.7,  # <10% unique or >70% zeros
            'mild': unique_ratio < 0.2 or zero_prop > 0.5,      # <20% unique or >50% zeros
            'details': {
                'unique_score_ratio': unique_ratio,
                'zero_proportion': zero_prop,
                'effective_discrimination': unique_ratio > 0.1 and zero_prop < 0.5
            }
        }
    
    def _generate_warnings(self, analysis: Dict) -> List[str]:
        """Generate appropriate statistical warnings."""
        warnings = []
        
        if analysis['is_degenerate']['severe']:
            warnings.append("SEVERE DEGENERACY: Distribution unsuitable for DeLong AUROC confidence intervals")
            warnings.append(f"Only {analysis['n_unique_scores']}/{analysis['n_samples']} unique scores")
        elif analysis['is_degenerate']['moderate']:
            warnings.append("MODERATE DEGENERACY: DeLong test assumptions may be violated")
        
        if analysis['zero_proportion'] > 0.8:
            warnings.append(f"HIGH ZERO CONCENTRATION: {analysis['zero_proportion']*100:.1f}% of scores are exactly zero")
        
        if analysis['n_infinite'] > 0:
            warnings.append(f"INFINITE VALUES: {analysis['n_infinite']} scores are infinite")
        
        if analysis['n_nan'] > 0:
            warnings.append(f"MISSING VALUES: {analysis['n_nan']} scores are NaN")
        
        return warnings
    
    def log_analysis(self):
        """Log comprehensive distribution analysis."""
        logging.info(f"\n=== Distribution Analysis: {self.metric_name} ===")
        logging.info(f"Total samples: {self.analysis['n_samples']}")
        logging.info(f"Unique scores: {self.analysis['n_unique_scores']} ({self.analysis['unique_score_ratio']*100:.1f}%)")
        logging.info(f"Zero scores: {self.analysis['n_zeros']} ({self.analysis['zero_proportion']*100:.1f}%)")
        logging.info(f"Score range: {self.analysis['score_range']}")
        
        logging.info(f"Most common values:")
        for val, count, prop in self.analysis['most_common_values']:
            logging.info(f"  {val}: {count} occurrences ({prop*100:.1f}%)")
        
        logging.info(f"Class separation:")
        cs = self.analysis['class_separation']
        logging.info(f"  Harmful mean±std: {cs['harmful_mean']:.4f}±{cs['harmful_std']:.4f}")
        logging.info(f"  Benign mean±std: {cs['benign_mean']:.4f}±{cs['benign_std']:.4f}")
        
        if self.analysis['statistical_warnings']:
            logging.warning(f"Statistical warnings for {self.metric_name}:")
            for warning in self.analysis['statistical_warnings']:
                logging.warning(f"  {warning}")
        else:
            logging.info(f"Distribution suitable for standard statistical tests")

def calculate_wilson_ci(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for a proportion.
    
    This method is always valid regardless of sample size or success rate,
    making it ideal for FNR confidence intervals even with small samples.
    
    Args:
        successes: Number of successes (e.g., false negatives)
        trials: Total number of trials (e.g., total positive cases)
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple[float, float]: (lower_bound, upper_bound) of confidence interval
        
    References:
        Wilson, E.B. (1927). Probable inference, the law of succession, 
        and statistical inference. JASA, 22(158), 209-212.
    """
    if trials == 0:
        logging.warning("Wilson CI: No trials provided, returning (0, 0)")
        return (0.0, 0.0)
    
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha/2)
    
    p_hat = successes / trials
    n = trials
    
    # Wilson score interval formula
    denominator = 1 + (z**2 / n)
    center = (p_hat + z**2 / (2*n)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    logging.info(f"Wilson CI ({confidence*100:.0f}%): {successes}/{trials} = {p_hat:.3f} [{lower:.3f}, {upper:.3f}]")
    
    return (lower, upper)

def calculate_delong_ci_robust(y_true: np.ndarray, y_scores: np.ndarray, 
                              metric_name: str, confidence: float = 0.95) -> Dict:
    """
    Calculate AUROC with DeLong confidence interval, with robust handling of degenerate distributions.
    
    Args:
        y_true: True binary labels (0/1)
        y_scores: Predicted scores (continuous, higher = more likely positive)
        metric_name: Name of the metric for logging
        confidence: Confidence level
    
    Returns:
        Dict with AUROC, CI bounds, validity flags, and warnings
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Analyze distribution first
    dist_analysis = DistributionAnalysis(y_scores, y_true, metric_name)
    dist_analysis.log_analysis()
    
    result = {
        'metric_name': metric_name,
        'distribution_analysis': dist_analysis.analysis,
    }
    
    # Calculate AUROC regardless of distribution
    try:
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_true, y_scores)
        result['auroc'] = float(auroc)
        logging.info(f"AUROC for {metric_name}: {auroc:.3f}")
    except Exception as e:
        logging.error(f"Failed to calculate AUROC for {metric_name}: {e}")
        result['auroc'] = 0.5
        result['auroc_error'] = str(e)
    
    # Attempt DeLong CI only if distribution is suitable
    if dist_analysis.analysis['delong_valid'] and MLSTATKIT_AVAILABLE:
        try:
            # Use MLstatkit for proper DeLong implementation with confidence intervals
            # For single distribution CI, use the same scores twice and get CI
            z_score, p_value, ci_A, ci_B = Delong_test(y_true, y_scores, y_scores, return_ci=True, alpha=confidence)
            
            # Since we used same scores twice, ci_A and ci_B should be identical
            ci_lower, ci_upper = ci_A
            
            result['delong_ci'] = (float(ci_lower), float(ci_upper))
            result['delong_ci_valid'] = True
            result['delong_method'] = 'MLstatkit'
            logging.info(f"DeLong CI ({confidence*100:.0f}%) for {metric_name}: [{ci_lower:.3f}, {ci_upper:.3f}]")
        except Exception as e:
            logging.warning(f"DeLong CI failed for {metric_name}: {e}")
            result['delong_ci_valid'] = False
            result['delong_ci_error'] = str(e)
    else:
        result['delong_ci_valid'] = False
        if not dist_analysis.analysis['delong_valid']:
            result['delong_ci_error'] = "Distribution too degenerate for DeLong method"
        else:
            result['delong_ci_error'] = "MLstatkit not available"
    
    # Always provide fallback bootstrap CI (with warnings for degenerate cases)
    try:
        bootstrap_ci = bootstrap_auroc_ci(y_true, y_scores, confidence=confidence, n_bootstrap=1000)
        result['bootstrap_ci'] = bootstrap_ci
        result['bootstrap_ci_valid'] = True
        
        if dist_analysis.analysis['is_degenerate']['moderate']:
            result['bootstrap_warning'] = "Bootstrap CI may be unreliable due to degenerate distribution"
            
    except Exception as e:
        logging.warning(f"Bootstrap CI failed for {metric_name}: {e}")
        result['bootstrap_ci_valid'] = False
        result['bootstrap_ci_error'] = str(e)
    
    return result

def bootstrap_auroc_ci(y_true: np.ndarray, y_scores: np.ndarray, 
                      confidence: float = 0.95, n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for AUROC using scipy.stats.bootstrap.
    
    This is the modern, scientifically rigorous approach that properly handles
    degenerate distributions with appropriate warnings.
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import bootstrap
    
    def auroc_statistic(y_true, y_scores, axis):
        """Statistic function for scipy bootstrap."""
        # Handle the axis parameter correctly
        if axis is None:
            indices = np.arange(len(y_true))
        else:
            indices = np.arange(y_true.shape[axis])
        
        # For each bootstrap sample
        try:
            if len(np.unique(y_true)) < 2:
                return 0.5  # Return random performance for degenerate samples
            return roc_auc_score(y_true, y_scores)
        except:
            return 0.5
    
    # Prepare data for scipy bootstrap (expects tuple of arrays)
    data = (y_true, y_scores)
    
    def auroc_func(y_true_sample, y_scores_sample):
        """Function to compute AUROC for bootstrap samples."""
        # Check if sample has both classes
        if len(np.unique(y_true_sample)) < 2:
            return 0.5
        try:
            return roc_auc_score(y_true_sample, y_scores_sample)
        except:
            return 0.5
    
    try:
        # Use scipy.stats.bootstrap with BCa method (bias-corrected accelerated)
        rng = np.random.RandomState(42)  # For reproducibility
        
        # Create a bootstrap-compatible function
        def bootstrap_statistic(*args):
            return auroc_func(args[0], args[1])
        
        # Perform bootstrap
        res = bootstrap(data, bootstrap_statistic, n_resamples=n_bootstrap, 
                       confidence_level=confidence, random_state=rng, method='BCa')
        
        ci_lower = res.confidence_interval.low
        ci_upper = res.confidence_interval.high
        
        logging.info(f"Scipy Bootstrap CI ({confidence*100:.0f}%) [BCa method]: [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        return (float(ci_lower), float(ci_upper))
        
    except Exception as e:
        logging.warning(f"Scipy bootstrap failed, using fallback method: {e}")
        
        # Fallback to manual bootstrap
        bootstrap_aurocs = []
        np.random.seed(42)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
            y_boot = y_true[indices]
            scores_boot = y_scores[indices]
            
            if len(np.unique(y_boot)) < 2:
                bootstrap_aurocs.append(0.5)
                continue
                
            try:
                auroc_boot = roc_auc_score(y_boot, scores_boot)
                bootstrap_aurocs.append(auroc_boot)
            except:
                bootstrap_aurocs.append(0.5)
        
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_aurocs, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_aurocs, (1 - alpha/2) * 100)
        
        logging.info(f"Fallback Bootstrap CI ({confidence*100:.0f}%): [{ci_lower:.3f}, {ci_upper:.3f}]")
        return (float(ci_lower), float(ci_upper))

def paired_delong_test(y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray,
                      metric1_name: str, metric2_name: str) -> Dict:
    """
    Perform paired DeLong test comparing two models on the same dataset.
    
    Args:
        y_true: True binary labels
        scores1: Scores from first model/metric
        scores2: Scores from second model/metric
        metric1_name: Name of first metric
        metric2_name: Name of second metric
    
    Returns:
        Dict with test results, warnings, and validity flags
    """
    logging.info(f"\n=== Paired DeLong Test: {metric1_name} vs {metric2_name} ===")
    
    # Analyze both distributions
    dist1 = DistributionAnalysis(scores1, y_true, metric1_name)
    dist2 = DistributionAnalysis(scores2, y_true, metric2_name)
    
    result = {
        'metric1_name': metric1_name,
        'metric2_name': metric2_name,
        'distribution_analysis': {
            'metric1': dist1.analysis,
            'metric2': dist2.analysis
        }
    }
    
    # Calculate individual AUROCs
    try:
        from sklearn.metrics import roc_auc_score
        auroc1 = roc_auc_score(y_true, scores1)
        auroc2 = roc_auc_score(y_true, scores2)
        result['auroc1'] = float(auroc1)
        result['auroc2'] = float(auroc2)
        result['auroc_difference'] = float(auroc2 - auroc1)
        
        logging.info(f"AUROC {metric1_name}: {auroc1:.3f}")
        logging.info(f"AUROC {metric2_name}: {auroc2:.3f}")
        logging.info(f"Difference ({metric2_name} - {metric1_name}): {auroc2-auroc1:.3f}")
        
    except Exception as e:
        logging.error(f"Failed to calculate AUROCs: {e}")
        result['auroc_error'] = str(e)
        return result
    
    # Check if DeLong test is appropriate
    both_valid = dist1.analysis['delong_valid'] and dist2.analysis['delong_valid']
    
    if both_valid and MLSTATKIT_AVAILABLE:
        try:
            # Perform DeLong test using MLstatkit
            z_score, p_value = Delong_test(y_true, scores1, scores2)
            
            # Use the returned values directly
            stat = z_score
            
            result['delong_test'] = {
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'valid': True,
                'method': 'MLstatkit DeLong test',
                'auroc1_variance': float(delong_result.get('var1', 0)),
                'auroc2_variance': float(delong_result.get('var2', 0)),
                'covariance': float(delong_result.get('cov12', 0))
            }
            
            logging.info(f"DeLong test statistic: {stat:.3f}")
            logging.info(f"DeLong test p-value: {p_value:.6f}")
            logging.info(f"Statistically significant difference: {p_value < 0.05}")
            
        except Exception as e:
            logging.warning(f"DeLong test failed: {e}")
            result['delong_test'] = {
                'valid': False,
                'error': str(e)
            }
    else:
        # Document why test is not valid
        reasons = []
        if not dist1.analysis['delong_valid']:
            reasons.append(f"{metric1_name} distribution too degenerate")
        if not dist2.analysis['delong_valid']:
            reasons.append(f"{metric2_name} distribution too degenerate")
        if not MLSTATKIT_AVAILABLE:
            reasons.append("DeLong library not available")
        
        result['delong_test'] = {
            'valid': False,
            'reasons': reasons
        }
        
        logging.warning(f"DeLong test not performed: {'; '.join(reasons)}")
    
    return result

def mcnemar_test_paired_predictions(y_true: np.ndarray, preds1: np.ndarray, preds2: np.ndarray,
                                   method1_name: str, method2_name: str) -> Dict:
    """
    McNemar's test for comparing paired binary predictions.
    
    Particularly useful for comparing FNR between two detection methods
    when applied to the same dataset.
    
    Args:
        y_true: True binary labels
        preds1: Binary predictions from first method
        preds2: Binary predictions from second method  
        method1_name: Name of first method
        method2_name: Name of second method
    
    Returns:
        Dict with test results and contingency table
    """
    logging.info(f"\n=== McNemar Test: {method1_name} vs {method2_name} ===")
    
    y_true = np.array(y_true)
    preds1 = np.array(preds1)
    preds2 = np.array(preds2)
    
    # Build contingency table for McNemar test
    # Focus on disagreements between the two methods
    correct1 = (preds1 == y_true)
    correct2 = (preds2 == y_true)
    
    # 2x2 contingency table
    both_correct = np.sum(correct1 & correct2)
    method1_only = np.sum(correct1 & ~correct2) 
    method2_only = np.sum(~correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)
    
    contingency_table = np.array([
        [both_correct, method1_only],
        [method2_only, both_wrong]
    ])
    
    result = {
        'method1_name': method1_name,
        'method2_name': method2_name,
        'contingency_table': contingency_table.tolist(),
        'discordant_pairs': {
            'method1_only_correct': int(method1_only),
            'method2_only_correct': int(method2_only),
            'total_discordant': int(method1_only + method2_only)
        }
    }
    
    logging.info(f"Contingency table:")
    logging.info(f"  Both correct: {both_correct}")
    logging.info(f"  {method1_name} only correct: {method1_only}")
    logging.info(f"  {method2_name} only correct: {method2_only}")
    logging.info(f"  Both wrong: {both_wrong}")
    
    # Perform McNemar test
    if method1_only + method2_only == 0:
        logging.info("No discordant pairs - methods perform identically")
        result['mcnemar_test'] = {
            'statistic': 0.0,
            'p_value': 1.0,
            'valid': True,
            'note': 'No discordant pairs between methods'
        }
    else:
        try:
            if STATSMODELS_AVAILABLE:
                # Use statsmodels implementation
                mcnemar_result = mcnemar(contingency_table, exact=True)
                result['mcnemar_test'] = {
                    'statistic': float(mcnemar_result.statistic),
                    'p_value': float(mcnemar_result.pvalue),
                    'valid': True
                }
            else:
                # Fallback manual implementation
                stat = (abs(method1_only - method2_only) - 1)**2 / (method1_only + method2_only)
                p_value = 1 - chi2.cdf(stat, df=1)
                result['mcnemar_test'] = {
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'valid': True,
                    'note': 'Using fallback implementation'
                }
            
            logging.info(f"McNemar statistic: {result['mcnemar_test']['statistic']:.3f}")
            logging.info(f"McNemar p-value: {result['mcnemar_test']['p_value']:.3f}")
            
        except Exception as e:
            logging.error(f"McNemar test failed: {e}")
            result['mcnemar_test'] = {
                'valid': False,
                'error': str(e)
            }
    
    return result

def generate_binary_predictions(scores: np.ndarray, threshold: float) -> np.ndarray:
    """
    Generate binary predictions from continuous scores using a threshold.
    
    Args:
        scores: Continuous scores (higher = more likely positive)
        threshold: Decision threshold
    
    Returns:
        Binary predictions (0/1)
    """
    return (np.array(scores) >= threshold).astype(int)

def format_metric_with_ci(value: float, ci_lower: float, ci_upper: float, 
                         decimals: int = 3) -> str:
    """
    Format metric with confidence interval for reporting.
    
    Args:
        value: Point estimate
        ci_lower: Lower CI bound  
        ci_upper: Upper CI bound
        decimals: Number of decimal places
    
    Returns:
        Formatted string like "0.625 [0.543, 0.707]"
    """
    return f"{value:.{decimals}f} [{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"

def summarize_statistical_analysis(results: Dict) -> str:
    """
    Generate a summary of statistical analysis results for reporting.
    
    Args:
        results: Dictionary containing all statistical test results
    
    Returns:
        Formatted summary string
    """
    summary_lines = []
    summary_lines.append("=== STATISTICAL ANALYSIS SUMMARY ===")
    
    # Distribution summaries
    if 'distributions' in results:
        summary_lines.append("\nDistribution Analysis:")
        for metric, dist in results['distributions'].items():
            if dist.get('is_degenerate', {}).get('severe', False):
                summary_lines.append(f"  {metric}: SEVERELY DEGENERATE ({dist['zero_proportion']*100:.1f}% zeros)")
            else:
                summary_lines.append(f"  {metric}: Normal distribution")
    
    # Test validity summary
    if 'tests' in results:
        summary_lines.append("\nStatistical Test Validity:")
        for test_name, test_result in results['tests'].items():
            if test_result.get('valid', False):
                summary_lines.append(f"  {test_name}: VALID")
            else:
                reason = test_result.get('error', 'Unknown reason')
                summary_lines.append(f"  {test_name}: INVALID ({reason})")
    
    return "\n".join(summary_lines)

if __name__ == "__main__":
    # Basic module test
    logging.info("Statistical Tests Module - Basic Functionality Test")
    
    # Test Wilson CI
    ci_lower, ci_upper = calculate_wilson_ci(44, 60)
    logging.info(f"Wilson CI test: 44/60 = [{ci_lower:.3f}, {ci_upper:.3f}]")
    
    # Test with synthetic data
    np.random.seed(42)
    y_true = np.array([0]*50 + [1]*50)
    y_scores_normal = np.random.beta(2, 5, 50).tolist() + np.random.beta(5, 2, 50).tolist()
    y_scores_degenerate = [0.0]*95 + [0.5]*5
    
    # Test normal distribution
    result_normal = calculate_delong_ci_robust(y_true, y_scores_normal, "Normal Distribution")
    
    # Test degenerate distribution  
    result_degenerate = calculate_delong_ci_robust(y_true, y_scores_degenerate, "Degenerate Distribution")
    
    logging.info("Module test completed successfully")