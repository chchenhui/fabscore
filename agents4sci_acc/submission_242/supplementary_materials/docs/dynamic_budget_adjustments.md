# Dynamic Budget Adjustments - Technical Documentation

## Overview

The Dynamic Budget Adjustment system allows clients to make explicit, reflection-driven decisions about job budgets. This replaces the previous automatic budget increase system with a more realistic, GPT-driven approach.

## Key Components

### 1. Client Budget Strategy Adjustments
- **General Budget Multiplier**: Affects all future job postings
- **Field**: `client.budget_multiplier` (default: 1.0)
- **Range**: 0.5x to 2.0x (configurable in `ReflectionAdjustmentEngine`)
- **Application**: Applied during job generation to `budget_amount`

### 2. Specific Job Budget Adjustments
- **Target**: Individual unfilled jobs by job ID
- **Mechanism**: Direct modification of `job.budget_amount`
- **Context**: Provided via unfilled job details in client reflection prompts
- **Format**: `[{"job_id": "job_123", "increase_percentage": 20, "reasoning": "why"}]`

### 3. Unfilled Job Tracking
- **Field**: `job.rounds_unfilled` (integer)
- **Initialization**: 0 when job created
- **Updates**: Incremented each round by `_update_unfilled_jobs_tracking()`
- **Usage**: Provides context for client budget decisions

### 4. Probability-Based Reflections
- **Parameter**: `reflection_probability` (float, 0.0 to 1.0)
- **Default**: 0.1 (10% chance per agent per round)
- **Implementation**: `random.random() < self.reflection_probability`
- **Benefit**: More realistic reflection frequency vs fixed intervals

## Technical Implementation

### Reflection Processing Flow

```python
# 1. Client reflection includes unfilled job context
unfilled_jobs = [job for job in self.active_jobs if job.client_id == client.id]
for job in unfilled_jobs:
    activities.append({
        'type': 'unfilled_job',
        'job_id': job.id,
        'title': job.title,
        'current_budget': job.budget_amount,
        'rounds_unfilled': job.rounds_unfilled,
        'timeline': job.timeline,
        'category': job.category
    })

# 2. GPT decides on budget adjustments
validated_reflection = validate_reflection(reflection_text)

# 3. Apply general budget strategy
if validated_reflection.budget_adjustment:
    budget_result = reflection_adjustment_engine.apply_client_budget_adjustment(
        client, validated_reflection.budget_adjustment, round_num
    )

# 4. Apply specific job budget adjustments  
if validated_reflection.job_budget_adjustments:
    job_results = self._apply_specific_job_budget_adjustments(
        validated_reflection.job_budget_adjustments, round_num
    )
```

### Pydantic Model Structure

```python
class ReflectionResponse(BaseModel):
    key_insights: List[str]
    strategy_adjustments: List[str] 
    market_observations: List[str]
    
    # Optional client fields
    budget_adjustment: Optional[Dict] = None  # General strategy
    job_budget_adjustments: Optional[List[Dict]] = None  # Specific jobs
    requirements_adjustment: Optional[Dict] = None
    
    # Optional freelancer fields
    rate_adjustment: Optional[Dict] = None
```

### Simplified Prompt Design

The client reflection prompt was streamlined from ~800 tokens to ~300 tokens:

```python
f"""You are {client_data.get('company_name')} ({client_data.get('company_size')} company, {client_data.get('budget_philosophy')} budget approach).

RECENT ACTIVITY:
{performance_summary}{activities_text}{unfilled_jobs_text}

Reflect and adjust your hiring strategy:

1. BUDGET STRATEGY: increase/decrease/keep your general budget for future jobs
2. UNFILLED JOBS: Optionally increase specific job budgets (format: [{"job_id": "job_123", "increase_percentage": 20, "reasoning": "why"}])
3. REQUIREMENTS: Make job requirements more/same/less specific

Return JSON with ALL fields:
{{"key_insights": ["insight1", "insight2"],
"strategy_adjustments": ["adjustment1", "adjustment2"], 
"budget_adjustment": {{"change": "increase/decrease/keep", "reasoning": "why"}},
"job_budget_adjustments": [],
"requirements_adjustment": {{"specificity": "more/same/less", "reasoning": "why"}},
"market_observations": ["observation1", "observation2"]}}"""
```

## Benefits

### Research Benefits
- **Realistic Budget Behavior**: Clients make conscious decisions rather than automatic responses
- **Market Response Studies**: How explicit budget increases affect bid volume and quality
- **Strategy Evolution**: Track how client budget strategies adapt over time
- **Economic Modeling**: More authentic price discovery mechanisms

### Technical Benefits
- **Performance**: Shorter prompts reduce token usage and processing time
- **Maintainability**: Clear separation between general and specific budget adjustments
- **Flexibility**: Configurable reflection probability allows tuning simulation speed vs detail
- **Testability**: Discrete components with clear interfaces and comprehensive test coverage

## Configuration

### Command Line Options
```bash
# Configure reflection probability
python run_marketplace.py --reflection-probability 0.25  # 25% chance per agent per round

# Disable reflections entirely for speed
python run_marketplace.py --disable-reflections

# Legacy parameter removed
# --reflection-frequency N  # DEPRECATED - use --reflection-probability instead
```

### Engine Configuration
```python
class ReflectionAdjustmentEngine:
    def __init__(self):
        # Freelancer rate bounds
        self.min_hourly_rate = 5.0
        self.max_hourly_rate = 500.0
        
        # Client budget multiplier bounds  
        self.min_budget_multiplier = 0.5
        self.max_budget_multiplier = 2.0
        
        # Significance threshold (5% minimum change)
        self.min_change_threshold = 0.05
```

## Testing

Comprehensive test coverage includes:
- `tests/test_client_job_budget_adjustments.py`: New functionality tests
- `tests/test_reflection_adjustments.py`: Engine testing
- `tests/test_validation_and_assumptions.py`: Pydantic validation
- `tests/test_prompt_code_consistency.py`: Prompt-code alignment

All tests pass and demonstrate the system working as designed.
