"""
Power analysis for pharmaceutical forecasting experiments.
Following Linus principle: Know your statistical power or your results are meaningless.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from statsmodels.stats.power import ttest_power, tt_solve_power
from statsmodels.stats.proportion import power_proportions_2indep, proportion_effectsize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class PowerAnalysisConfig:
    """Configuration for power analysis."""
    
    # Effect sizes
    mape_improvement: float = 0.10  # 10% improvement in MAPE
    year2_ape_improvement: float = 0.15  # 15% improvement in Year 2 APE
    peak_ape_improvement: float = 0.12  # 12% improvement in Peak APE
    
    # Statistical parameters
    alpha: float = 0.05  # Type I error rate
    power: float = 0.80  # Desired power (1 - Type II error)
    
    # Sample size constraints
    min_sample_size: int = 50  # Minimum viable sample size
    max_sample_size: int = 200  # Maximum practical sample size
    
    # Multiple comparisons
    n_comparisons: int = 5  # Number of model comparisons
    correction_method: str = 'holm'  # Multiple comparison correction
    
    # Bootstrap parameters
    n_bootstrap: int = 5000
    confidence_level: float = 0.95


class PowerAnalyzer:
    """
    Power analysis for pharmaceutical forecasting experiments.
    Calculates required sample sizes and achieved power for various effect sizes.
    """
    
    def __init__(self, config: PowerAnalysisConfig = None):
        self.config = config or PowerAnalysisConfig()
        self.results = {}
    
    def calculate_required_sample_size(self, effect_size: float, 
                                     alpha: float = None, 
                                     power: float = None) -> int:
        """
        Calculate required sample size for a given effect size.
        
        Args:
            effect_size: Standardized effect size (Cohen's d)
            alpha: Type I error rate
            power: Desired power
        
        Returns:
            Required sample size
        """
        alpha = alpha or self.config.alpha
        power = power or self.config.power
        
        # Use statsmodels for power calculation
        n_required = tt_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(n_required))
    
    def calculate_achieved_power(self, effect_size: float, sample_size: int,
                               alpha: float = None) -> float:
        """
        Calculate achieved power for a given effect size and sample size.
        
        Args:
            effect_size: Standardized effect size
            sample_size: Sample size
            alpha: Type I error rate
        
        Returns:
            Achieved power
        """
        alpha = alpha or self.config.alpha
        
        achieved_power = ttest_power(
            effect_size=effect_size,
            nobs=sample_size,
            alpha=alpha,
            alternative='two-sided'
        )
        
        return achieved_power
    
    def mape_effect_size_to_cohens_d(self, mape_improvement: float, 
                                    baseline_mape: float = 0.30) -> float:
        """
        Convert MAPE improvement to Cohen's d.
        
        Args:
            mape_improvement: Relative improvement in MAPE (e.g., 0.10 for 10%)
            baseline_mape: Baseline MAPE (default 30% for industry standard)
        
        Returns:
            Cohen's d effect size
        """
        # MAPE improvement as standardized effect
        # Assuming normal distribution of MAPE differences
        effect_size = mape_improvement * baseline_mape / (baseline_mape * 0.5)  # Rough approximation
        return effect_size
    
    def analyze_mape_power(self, baseline_mape: float = 0.30) -> Dict[str, any]:
        """
        Power analysis for MAPE improvement.
        
        Args:
            baseline_mape: Baseline MAPE (default 30% for industry standard)
        
        Returns:
            Power analysis results
        """
        effect_size = self.mape_effect_size_to_cohens_d(
            self.config.mape_improvement, baseline_mape
        )
        
        # Calculate required sample size
        n_required = self.calculate_required_sample_size(effect_size)
        
        # Calculate achieved power for different sample sizes
        sample_sizes = [50, 75, 100, 125, 150, 200]
        achieved_powers = []
        
        for n in sample_sizes:
            power = self.calculate_achieved_power(effect_size, n)
            achieved_powers.append(power)
        
        return {
            'metric': 'MAPE',
            'baseline_mape': baseline_mape,
            'improvement': self.config.mape_improvement,
            'effect_size': effect_size,
            'required_sample_size': n_required,
            'sample_sizes': sample_sizes,
            'achieved_powers': achieved_powers,
            'power_curve': list(zip(sample_sizes, achieved_powers))
        }
    
    def analyze_year2_ape_power(self, baseline_year2_ape: float = 0.35) -> Dict[str, any]:
        """
        Power analysis for Year 2 APE improvement.
        """
        effect_size = self.mape_effect_size_to_cohens_d(
            self.config.year2_ape_improvement, baseline_year2_ape
        )
        
        n_required = self.calculate_required_sample_size(effect_size)
        
        sample_sizes = [50, 75, 100, 125, 150, 200]
        achieved_powers = [self.calculate_achieved_power(effect_size, n) for n in sample_sizes]
        
        return {
            'metric': 'Year2_APE',
            'baseline_year2_ape': baseline_year2_ape,
            'improvement': self.config.year2_ape_improvement,
            'effect_size': effect_size,
            'required_sample_size': n_required,
            'sample_sizes': sample_sizes,
            'achieved_powers': achieved_powers,
            'power_curve': list(zip(sample_sizes, achieved_powers))
        }
    
    def analyze_peak_ape_power(self, baseline_peak_ape: float = 0.40) -> Dict[str, any]:
        """
        Power analysis for Peak APE improvement.
        """
        effect_size = self.mape_effect_size_to_cohens_d(
            self.config.peak_ape_improvement, baseline_peak_ape
        )
        
        n_required = self.calculate_required_sample_size(effect_size)
        
        sample_sizes = [50, 75, 100, 125, 150, 200]
        achieved_powers = [self.calculate_achieved_power(effect_size, n) for n in sample_sizes]
        
        return {
            'metric': 'Peak_APE',
            'baseline_peak_ape': baseline_peak_ape,
            'improvement': self.config.peak_ape_improvement,
            'effect_size': effect_size,
            'required_sample_size': n_required,
            'sample_sizes': sample_sizes,
            'achieved_powers': achieved_powers,
            'power_curve': list(zip(sample_sizes, achieved_powers))
        }
    
    def analyze_multiple_comparisons_power(self) -> Dict[str, any]:
        """
        Power analysis accounting for multiple comparisons.
        """
        # Adjust alpha for multiple comparisons
        if self.config.correction_method == 'bonferroni':
            adjusted_alpha = self.config.alpha / self.config.n_comparisons
        elif self.config.correction_method == 'holm':
            # Holm-Bonferroni is less conservative
            adjusted_alpha = self.config.alpha / (self.config.n_comparisons + 1)
        else:
            adjusted_alpha = self.config.alpha
        
        # Calculate required sample size with adjusted alpha
        effect_size = self.mape_effect_size_to_cohens_d(self.config.mape_improvement)
        n_required = self.calculate_required_sample_size(effect_size, adjusted_alpha)
        
        return {
            'correction_method': self.config.correction_method,
            'n_comparisons': self.config.n_comparisons,
            'original_alpha': self.config.alpha,
            'adjusted_alpha': adjusted_alpha,
            'effect_size': effect_size,
            'required_sample_size': n_required,
            'power_loss': self.config.power - self.calculate_achieved_power(effect_size, n_required, adjusted_alpha)
        }
    
    def comprehensive_power_analysis(self) -> Dict[str, any]:
        """
        Comprehensive power analysis for all metrics.
        """
        results = {
            'config': self.config,
            'mape_analysis': self.analyze_mape_power(),
            'year2_ape_analysis': self.analyze_year2_ape_power(),
            'peak_ape_analysis': self.analyze_peak_ape_power(),
            'multiple_comparisons_analysis': self.analyze_multiple_comparisons_power()
        }
        
        # Summary recommendations
        max_required_n = max(
            results['mape_analysis']['required_sample_size'],
            results['year2_ape_analysis']['required_sample_size'],
            results['peak_ape_analysis']['required_sample_size'],
            results['multiple_comparisons_analysis']['required_sample_size']
        )
        
        results['recommendations'] = {
            'minimum_sample_size': max_required_n,
            'recommended_sample_size': max(max_required_n, self.config.min_sample_size),
            'power_adequate': max_required_n <= self.config.max_sample_size,
            'multiple_comparisons_impact': results['multiple_comparisons_analysis']['power_loss']
        }
        
        return results
    
    def plot_power_curves(self, results: Dict[str, any], save_path: str = None):
        """
        Plot power curves for different metrics.
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Power Analysis for Pharmaceutical Forecasting', fontsize=16)
            
            # MAPE power curve
            mape_data = results['mape_analysis']
            axes[0, 0].plot(mape_data['sample_sizes'], mape_data['achieved_powers'], 'b-o')
            axes[0, 0].axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')
            axes[0, 0].axvline(x=mape_data['required_sample_size'], color='g', linestyle='--', label='Required N')
            axes[0, 0].set_xlabel('Sample Size')
            axes[0, 0].set_ylabel('Power')
            axes[0, 0].set_title('MAPE Improvement Power')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Year 2 APE power curve
            y2_data = results['year2_ape_analysis']
            axes[0, 1].plot(y2_data['sample_sizes'], y2_data['achieved_powers'], 'g-o')
            axes[0, 1].axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')
            axes[0, 1].axvline(x=y2_data['required_sample_size'], color='g', linestyle='--', label='Required N')
            axes[0, 1].set_xlabel('Sample Size')
            axes[0, 1].set_ylabel('Power')
            axes[0, 1].set_title('Year 2 APE Improvement Power')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Peak APE power curve
            peak_data = results['peak_ape_analysis']
            axes[1, 0].plot(peak_data['sample_sizes'], peak_data['achieved_powers'], 'r-o')
            axes[1, 0].axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')
            axes[1, 0].axvline(x=peak_data['required_sample_size'], color='g', linestyle='--', label='Required N')
            axes[1, 0].set_xlabel('Sample Size')
            axes[1, 0].set_ylabel('Power')
            axes[1, 0].set_title('Peak APE Improvement Power')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Combined power curve
            all_sample_sizes = mape_data['sample_sizes']
            min_powers = [min(mape_data['achieved_powers'][i], 
                            y2_data['achieved_powers'][i], 
                            peak_data['achieved_powers'][i]) 
                         for i in range(len(all_sample_sizes))]
            
            axes[1, 1].plot(all_sample_sizes, min_powers, 'k-o', linewidth=2)
            axes[1, 1].axhline(y=0.8, color='r', linestyle='--', label='Target Power (0.8)')
            axes[1, 1].set_xlabel('Sample Size')
            axes[1, 1].set_ylabel('Minimum Power')
            axes[1, 1].set_title('Combined Power (Worst Case)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Power curves saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Skipping power curve plots.")
    
    def generate_power_report(self, results: Dict[str, any]) -> str:
        """
        Generate a comprehensive power analysis report.
        """
        report = []
        report.append("=" * 60)
        report.append("POWER ANALYSIS REPORT")
        report.append("Pharmaceutical Forecasting Experiment")
        report.append("=" * 60)
        report.append("")
        
        # Configuration
        report.append("CONFIGURATION:")
        report.append(f"  Alpha (Type I error): {self.config.alpha}")
        report.append(f"  Target Power: {self.config.power}")
        report.append(f"  Multiple Comparisons: {self.config.n_comparisons}")
        report.append(f"  Correction Method: {self.config.correction_method}")
        report.append("")
        
        # MAPE Analysis
        mape_data = results['mape_analysis']
        report.append("MAPE IMPROVEMENT POWER ANALYSIS:")
        report.append(f"  Baseline MAPE: {mape_data['baseline_mape']:.1%}")
        report.append(f"  Target Improvement: {mape_data['improvement']:.1%}")
        report.append(f"  Effect Size (Cohen's d): {mape_data['effect_size']:.3f}")
        report.append(f"  Required Sample Size: {mape_data['required_sample_size']}")
        report.append("")
        
        # Year 2 APE Analysis
        y2_data = results['year2_ape_analysis']
        report.append("YEAR 2 APE IMPROVEMENT POWER ANALYSIS:")
        report.append(f"  Baseline Year 2 APE: {y2_data['baseline_year2_ape']:.1%}")
        report.append(f"  Target Improvement: {y2_data['improvement']:.1%}")
        report.append(f"  Effect Size (Cohen's d): {y2_data['effect_size']:.3f}")
        report.append(f"  Required Sample Size: {y2_data['required_sample_size']}")
        report.append("")
        
        # Peak APE Analysis
        peak_data = results['peak_ape_analysis']
        report.append("PEAK APE IMPROVEMENT POWER ANALYSIS:")
        report.append(f"  Baseline Peak APE: {peak_data['baseline_peak_ape']:.1%}")
        report.append(f"  Target Improvement: {peak_data['improvement']:.1%}")
        report.append(f"  Effect Size (Cohen's d): {peak_data['effect_size']:.3f}")
        report.append(f"  Required Sample Size: {peak_data['required_sample_size']}")
        report.append("")
        
        # Multiple Comparisons
        mc_data = results['multiple_comparisons_analysis']
        report.append("MULTIPLE COMPARISONS IMPACT:")
        report.append(f"  Adjusted Alpha: {mc_data['adjusted_alpha']:.4f}")
        report.append(f"  Power Loss: {mc_data['power_loss']:.3f}")
        report.append(f"  Required Sample Size: {mc_data['required_sample_size']}")
        report.append("")
        
        # Recommendations
        recs = results['recommendations']
        report.append("RECOMMENDATIONS:")
        report.append(f"  Minimum Sample Size: {recs['minimum_sample_size']}")
        report.append(f"  Recommended Sample Size: {recs['recommended_sample_size']}")
        report.append(f"  Power Adequate: {'YES' if recs['power_adequate'] else 'NO'}")
        report.append(f"  Multiple Comparisons Impact: {recs['multiple_comparisons_impact']:.3f}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


def run_power_analysis(config: PowerAnalysisConfig = None) -> Dict[str, any]:
    """
    Run comprehensive power analysis.
    
    Args:
        config: Power analysis configuration
    
    Returns:
        Power analysis results
    """
    config = config or PowerAnalysisConfig()
    analyzer = PowerAnalyzer(config)
    
    results = analyzer.comprehensive_power_analysis()
    
    # Generate report
    report = analyzer.generate_power_report(results)
    print(report)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Power Analysis for Pharmaceutical Forecasting")
    print("=" * 50)
    
    # Run power analysis
    results = run_power_analysis()
    
    # Plot power curves
    analyzer = PowerAnalyzer()
    analyzer.plot_power_curves(results, "power_analysis_curves.png")
    
    print("✅ Power analysis complete!")
