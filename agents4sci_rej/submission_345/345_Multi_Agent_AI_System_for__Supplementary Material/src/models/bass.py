"""
Bass Diffusion Model implementation for pharmaceutical adoption forecasting.

The Bass model describes the adoption of new products through innovation and imitation effects.
For pharmaceutical products, this captures both early adopters (innovation) and followers (imitation).

Key parameters:
- m: Market size (total addressable market)
- p: Innovation coefficient (external influence)
- q: Imitation coefficient (internal influence/word-of-mouth)

Example usage:
    >>> import numpy as np
    >>> adopters = bass_adopters(T=40, m=1200000, p=0.03, q=0.40)
    >>> print(f"Peak adoption quarter: {np.argmax(adopters) + 1}")
    >>> print(f"Total adoption: {adopters.sum():.0f}")
"""
import numpy as np
from typing import Tuple, Optional
import warnings

def bass_adopters(T: int, m: float, p: float, q: float) -> np.ndarray:
    """
    Calculate quarterly adopters using Bass diffusion model.
    
    Args:
        T: Number of time periods (quarters)
        m: Market size (total addressable market)
        p: Innovation coefficient (0 < p < 1)
        q: Imitation coefficient (0 < q < 1)
    
    Returns:
        np.ndarray: Number of new adopters per period
        
    Raises:
        ValueError: If parameters are out of valid range
        
    Example:
        >>> adopters = bass_adopters(T=40, m=1200000, p=0.03, q=0.40)
        >>> adopters.shape
        (40,)
        >>> adopters.sum() <= 1200000  # Total adoption <= market size
        True
    """
    # Validate inputs
    if T <= 0:
        raise ValueError("T must be positive")
    if m <= 0:
        raise ValueError("Market size m must be positive")
    if not (0 < p < 1):
        raise ValueError("Innovation coefficient p must be between 0 and 1")
    if not (0 < q < 1):
        raise ValueError("Imitation coefficient q must be between 0 and 1")
    
    # Initialize arrays
    N = np.zeros(T, dtype=float)  # New adopters per period
    cum = 0.0  # Cumulative adopters
    
    for t in range(T):
        # Bass model hazard function
        # f(t) = (p + q * cum/m) * (1 - cum/m)
        if cum >= m:
            # Market saturated
            N[t] = 0.0
        else:
            adoption_rate = (p + q * (cum / m)) * (1 - cum / m)
            N[t] = max(0.0, m * adoption_rate)
            cum += N[t]
            
            # Ensure we don't exceed market size
            if cum > m:
                N[t] -= (cum - m)
                cum = m
    
    return N

def bass_cumulative(T: int, m: float, p: float, q: float) -> np.ndarray:
    """
    Calculate cumulative adopters using Bass diffusion model.
    
    Args:
        T: Number of time periods
        m: Market size
        p: Innovation coefficient
        q: Imitation coefficient
    
    Returns:
        np.ndarray: Cumulative adopters over time
    """
    adopters = bass_adopters(T, m, p, q)
    return np.cumsum(adopters)

def bass_peak_time(m: float, p: float, q: float) -> float:
    """
    Calculate the theoretical peak adoption time for Bass model.
    
    Args:
        m: Market size
        p: Innovation coefficient
        q: Imitation coefficient
    
    Returns:
        float: Time of peak adoption (in same units as model periods)
    """
    if p <= 0 or q <= 0:
        return float('inf')
    
    # Analytical solution for peak time
    # t_peak = ln(q/p) / (p + q)
    return np.log(q / p) / (p + q)

def estimate_bass_parameters(adoption_data: np.ndarray, 
                           market_size: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Estimate Bass model parameters from observed adoption data.
    
    This is a simplified estimation method. For production use, consider
    more sophisticated fitting methods (e.g., nonlinear least squares).
    
    Args:
        adoption_data: Observed new adopters per period
        market_size: Known market size (if None, estimated from data)
    
    Returns:
        Tuple[float, float, float]: Estimated (m, p, q) parameters
    """
    if len(adoption_data) < 3:
        raise ValueError("Need at least 3 periods of data for parameter estimation")
    
    # Estimate market size if not provided
    if market_size is None:
        # Simple heuristic: assume current cumulative is ~50% of market at peak
        cumulative = np.cumsum(adoption_data)
        peak_idx = np.argmax(adoption_data)
        if peak_idx > 0:
            market_size = cumulative[peak_idx] * 2
        else:
            market_size = cumulative[-1] * 3
    
    # Simple moment-based estimation (could be improved)
    cumulative = np.cumsum(adoption_data)
    
    # Estimate p from early periods (before imitation kicks in)
    if len(adoption_data) > 1 and cumulative[0] > 0:
        p_est = adoption_data[0] / market_size
    else:
        p_est = 0.01  # Default
    
    # Estimate q from peak timing
    peak_time = np.argmax(adoption_data) + 1
    if peak_time > 1 and p_est > 0:
        # From t_peak = ln(q/p) / (p + q), solve for q
        # This is approximate
        q_est = p_est * np.exp(p_est * peak_time)
        q_est = min(q_est, 0.8)  # Cap at reasonable value
    else:
        q_est = 0.3  # Default
    
    # Ensure valid ranges
    p_est = max(0.001, min(0.1, p_est))
    q_est = max(0.1, min(0.8, q_est))
    
    return market_size, p_est, q_est

def validate_bass_output(adopters: np.ndarray, m: float, 
                        tolerance: float = 1e-6) -> bool:
    """
    Validate Bass model output for consistency.
    
    Args:
        adopters: New adopters per period
        m: Market size
        tolerance: Numerical tolerance for checks
    
    Returns:
        bool: True if validation passes
        
    Raises:
        AssertionError: If validation fails
    """
    # Check non-negative
    assert np.all(adopters >= -tolerance), "Adopters should be non-negative"
    
    # Check cumulative doesn't exceed market size
    cumulative = np.cumsum(adopters)
    assert cumulative[-1] <= m + tolerance, f"Total adoption {cumulative[-1]} exceeds market size {m}"
    
    # Check monotonic cumulative (weakly increasing)
    assert np.all(np.diff(cumulative) >= -tolerance), "Cumulative adoption should be monotonic"
    
    return True

if __name__ == "__main__":
    # Demo usage with typical pharmaceutical parameters
    print("Bass Diffusion Model Demo")
    print("=" * 40)
    
    # Parameters for a respiratory drug (like ensifentrine)
    T = 40  # 10 years quarterly
    m = 1_200_000  # Eligible COPD patients
    p = 0.03  # Innovation coefficient
    q = 0.40  # Imitation coefficient
    
    # Generate adoption curve
    adopters = bass_adopters(T, m, p, q)
    cumulative = bass_cumulative(T, m, p, q)
    
    # Validate output
    validate_bass_output(adopters, m)
    
    # Print key metrics
    peak_quarter = np.argmax(adopters) + 1
    peak_adopters = adopters.max()
    total_adoption = adopters.sum()
    
    print(f"Market size: {m:,}")
    print(f"Peak adoption quarter: {peak_quarter}")
    print(f"Peak quarterly adopters: {peak_adopters:,.0f}")
    print(f"Total adoption: {total_adoption:,.0f}")
    print(f"Market penetration: {100 * total_adoption / m:.1f}%")
    
    # Show first few quarters
    print(f"\nFirst 8 quarters:")
    for i in range(min(8, len(adopters))):
        print(f"  Q{i+1}: {adopters[i]:,.0f} new, {cumulative[i]:,.0f} cumulative")
