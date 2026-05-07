"""
Analysis package for marketplace simulation

This package provides tools for analyzing marketplace simulation data,
including market metrics, bidding behavior, and emergent agent patterns.

Main entrypoint:
    from src.analysis import MarketplaceAnalysis
    analyzer = MarketplaceAnalysis('simulation_results.json')
    results = analyzer.run_full_analysis()
"""

from .market_analysis import MarketplaceAnalysis

__all__ = ['MarketplaceAnalysis']