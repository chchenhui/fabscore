"""
Constants and configuration values for the Commercial Forecast Agent.

All magic numbers and business logic constants are centralized here
to eliminate hardcoded values scattered throughout the codebase.
"""

# Price-to-access tier mapping thresholds (USD per month)
PRICE_THRESHOLDS = {
    'OPEN': 1000,    # Below $1000/month = broad formulary access
    'PA': 2500,      # $1000-2500/month = prior authorization required  
    'NICHE': float('inf')  # Above $2500/month = highly restricted access
}

# Gross-to-net percentages by access tier
GTN_BY_TIER = {
    'OPEN': 0.75,    # 75% - minimal rebates for broad access
    'PA': 0.65,      # 65% - moderate rebates for PA tier
    'NICHE': 0.55    # 55% - heavy rebates for restricted access
}

# Adoption ceiling by access tier (maximum market penetration)
ADOPTION_CEILING_BY_TIER = {
    'OPEN': 1.0,     # 100% - no access restrictions
    'PA': 0.6,       # 60% - prior auth limits adoption
    'NICHE': 0.25    # 25% - highly restricted specialty use only
}

# Default financial parameters
DEFAULT_WACC = 0.10              # 10% weighted average cost of capital
DEFAULT_COGS_PCT = 0.15          # 15% cost of goods sold
DEFAULT_ADHERENCE_RATE = 0.85    # 85% patient adherence/persistence
DEFAULT_PRICE_EROSION = 0.02     # 2% annual price erosion

# Bass diffusion model defaults
DEFAULT_BASS_P = 0.03            # Innovation coefficient (external influence)
DEFAULT_BASS_Q = 0.40            # Imitation coefficient (word-of-mouth)

# Simulation parameters
DEFAULT_TIME_HORIZON_YEARS = 10   # Standard forecast horizon
DEFAULT_MONTE_CARLO_RUNS = 10000  # Monte Carlo simulation count
DEFAULT_QUARTERLY_PERIODS = 40    # 10 years * 4 quarters

# Market sizing defaults
DEFAULT_MARKET_SIZE = 1_200_000   # Eligible patients (US COPD example)

# SG&A model parameters
DEFAULT_SGA_LAUNCH_ANNUAL = 350_000_000  # $350M annual launch SG&A
DEFAULT_SGA_DECAY_TARGET = 0.5           # Decay to 50% of launch spend

# Uncertainty parameters for Monte Carlo
DEFAULT_UNCERTAINTY = {
    'gtn_pct': 0.05,           # 5% standard deviation
    'adherence_rate': 0.1,     # 10% standard deviation  
    'price_erosion': 0.01,     # 1% standard deviation
    'bass_p': 0.01,            # Innovation coeff uncertainty
    'bass_q': 0.05             # Imitation coeff uncertainty
}

# Validation thresholds
MAX_INDENTATION_LEVELS = 3        # Linus rule: max 3 levels
MIN_FUNCTION_LINES = 5            # Minimum function size
MAX_FUNCTION_LINES = 50           # Maximum function size (refactor if larger)

# Display formatting
CURRENCY_FORMAT = "${:,.0f}"      # Standard currency display
PERCENTAGE_FORMAT = "{:.1%}"      # Standard percentage display
LARGE_NUMBER_FORMAT = "{:,.0f}"   # Large number formatting