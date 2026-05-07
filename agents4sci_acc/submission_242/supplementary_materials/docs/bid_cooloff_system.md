# Bid Cooloff System

The **Bid Cooloff System** is a sophisticated mechanism that enables freelancers to reconsider previously declined job opportunities after a configurable waiting period. This system models realistic marketplace behavior where freelancers can revisit opportunities with updated strategies, market knowledge, or pricing approaches.

## 🎯 Overview

### Core Concept
When a freelancer declines a job (decision: "no"), they are temporarily blocked from bidding on that same job for a specified number of rounds (the "cooloff period"). After the cooloff expires, the freelancer can reconsider and potentially bid on the job if it remains unfilled.

### Key Benefits
- **Market Efficiency**: Prevents permanent job isolation, improving overall market fill rates
- **Agent Learning**: Enables freelancers to apply new strategies to previously declined opportunities
- **Realistic Behavior**: Mirrors real-world scenarios where freelancers revisit declined opportunities
- **Research Depth**: Enables studies of long-term decision evolution and strategy adaptation

## ⚙️ Technical Implementation

### Decision Tracking Structure
Each freelancer-job decision is tracked with:
```python
{
    (freelancer_id, job_id): {
        'decision': 'yes' | 'no',
        'round': int  # Round when decision was made
    }
}
```

### Core Logic
```python
def get_jobs_for_freelancer(self, freelancer, available_jobs, current_round):
    """Filter jobs based on cooloff logic"""
    for job in available_jobs:
        decision_key = (freelancer.id, job.id)
        
        if decision_key not in self.freelancer_job_decisions:
            # Never seen this job before - always available
            include_job = True
        else:
            decision_info = self.freelancer_job_decisions[decision_key]
            last_decision = decision_info.get('decision', 'no')
            last_round = decision_info.get('round', 0)
            rounds_since_decision = current_round - last_round
            
            # Allow re-bidding if:
            # 1. They declined ('no') AND cooloff enabled AND cooloff period has passed
            if (last_decision == 'no' and 
                self.bid_cooloff_rounds > 0 and 
                rounds_since_decision >= self.bid_cooloff_rounds):
                include_job = True
                print(f"🔄 {freelancer.name}: Cooloff expired for {job.title}")
            else:
                include_job = False  # Still in cooloff or permanently blocked
    
    return filtered_jobs
```

### Decision Categories

#### 🆕 **New Jobs**
- **Status**: Always available
- **Logic**: No previous decision exists
- **Behavior**: Freelancer sees job for the first time

#### ❌ **Declined Jobs (Within Cooloff)**
- **Status**: Temporarily blocked
- **Logic**: `decision == 'no'` AND `rounds_since_decision < cooloff_period`
- **Behavior**: Job hidden from freelancer

#### 🔄 **Declined Jobs (Cooloff Expired)**
- **Status**: Available again
- **Logic**: `decision == 'no'` AND `rounds_since_decision >= cooloff_period`
- **Behavior**: Job visible again, freelancer can reconsider

#### ✅ **Previously Accepted Jobs**
- **Status**: Permanently blocked
- **Logic**: `decision == 'yes'`
- **Behavior**: Never shown again (already bid once)

#### 🚫 **Disabled Cooloff**
- **Status**: Permanently blocked after any decision
- **Logic**: `bid_cooloff_rounds == 0`
- **Behavior**: Traditional "one chance only" model

## 🎛️ Configuration Options

### Command Line Arguments
```bash
# Default cooloff (5 rounds)
python run_marketplace.py --bid-cooloff-rounds 5

# Short cooloff (3 rounds)
python run_marketplace.py --bid-cooloff-rounds 3

# Disabled cooloff (permanent decisions)
python run_marketplace.py --bid-cooloff-rounds 0

# Long cooloff (10 rounds)
python run_marketplace.py --bid-cooloff-rounds 10
```

### Marketplace Initialization
```python
marketplace = TrueGPTMarketplace(
    num_freelancers=10,
    num_clients=5,
    rounds=20,
    bid_cooloff_rounds=5,  # 5-round cooloff period
    # ... other parameters
)
```

## 📊 Research Applications

### 1. Strategy Adaptation Studies
```python
# Research Question: How do freelancers adapt strategies over time?
# Setup: Long simulation with moderate cooloff
python run_marketplace.py --freelancers 15 --rounds 25 --bid-cooloff-rounds 5
```

**Metrics to Analyze:**
- Re-bidding success rates vs initial bidding success rates
- Changes in freelancer minimum rates between initial and repeat considerations
- Time patterns: when do freelancers typically reconsider declined jobs?

### 2. Market Efficiency Analysis
```python
# Research Question: Does bid cooloff improve market fill rates?
# Setup: Comparative study with/without cooloff
python run_marketplace.py --rounds 20 --bid-cooloff-rounds 5  # With cooloff
python run_marketplace.py --rounds 20 --bid-cooloff-rounds 0  # Without cooloff
```

**Metrics to Compare:**
- Job fill rates over time
- Total unfilled jobs at simulation end
- Market health scores and saturation indicators

### 3. Learning Trajectory Studies
```python
# Research Question: How does market learning affect re-bidding behavior?
# Setup: Enable reflections with cooloff
python run_marketplace.py --reflection-probability 0.3 --bid-cooloff-rounds 5
```

**Analysis Focus:**
- Freelancer reflection insights between decline and re-consideration
- Rate adjustments during cooloff periods
- Success correlation with reflection frequency

### 4. Temporal Market Dynamics
```python
# Research Question: How do cooloff dynamics evolve over extended periods?
# Setup: Long simulation with comprehensive tracking
python run_marketplace.py --rounds 50 --bid-cooloff-rounds 7 --freelancers 20
```

**Research Areas:**
- Long-term job persistence and eventual filling patterns
- Freelancer specialization development through repeated exposure
- Market maturation effects on re-bidding behavior

## 📈 Data Collection

### Key Metrics Tracked
- **`freelancer_job_decisions`**: Complete decision history with round tracking
- **`reflection_sessions`**: Freelancer insights and strategy adjustments
- **`round_data`**: Market health metrics including cooloff effects
- **`all_decisions`**: Comprehensive log of all bidding decisions

### Example Analysis Code
```python
from src.analysis.market_analysis import MarketplaceAnalysis

# Load simulation with cooloff data
analyzer = MarketplaceAnalysis()
analyzer.load_simulation_data("results/cooloff_simulation.json")

# Analyze re-bidding patterns
rebid_analysis = analyzer.analyze_rebidding_patterns()
cooloff_efficiency = analyzer.calculate_cooloff_efficiency()
strategy_evolution = analyzer.track_freelancer_adaptation()

# Generate cooloff-specific visualizations
analyzer.plot_rebidding_success_rates()
analyzer.plot_cooloff_timeline_analysis()
analyzer.plot_strategy_adaptation_patterns()
```

## 🔬 Experimental Design Examples

### Experiment 1: Cooloff Period Optimization
```bash
# Test different cooloff periods
for cooloff in 1 3 5 7 10; do
    python run_marketplace.py --bid-cooloff-rounds $cooloff --rounds 20 \
        --output-prefix "cooloff_${cooloff}_rounds"
done
```

### Experiment 2: Agent Density vs Cooloff Effectiveness
```bash
# Test cooloff effects under different market densities
python run_marketplace.py --freelancers 5 --clients 3 --bid-cooloff-rounds 5   # Low density
python run_marketplace.py --freelancers 15 --clients 8 --bid-cooloff-rounds 5  # High density
```

### Experiment 3: Reflection Integration
```bash
# Test cooloff with different reflection strategies
python run_marketplace.py --reflection-probability 0.1 --bid-cooloff-rounds 5   # Low reflection
python run_marketplace.py --reflection-probability 0.4 --bid-cooloff-rounds 5   # High reflection
```

## 🚀 Advanced Features

### Backward Compatibility
The system automatically handles legacy decision data:
```python
# Old format (string)
freelancer_job_decisions[key] = 'no'

# Automatically converted to new format
freelancer_job_decisions[key] = {'decision': 'no', 'round': 0}
```

### Logging and Monitoring
```python
# Cooloff expiration logging
print(f"🔄 {freelancer.name}: Cooloff expired for {job.title} (declined {rounds_ago} rounds ago)")

# Configuration logging at startup
print(f"🔄 Bid Cooloff: Freelancers can re-bid after {cooloff_rounds} rounds")
print(f"🔄 Bid Cooloff: Disabled (freelancers cannot re-bid on previously declined jobs)")
```

### Performance Considerations
- **Memory**: O(F × J) where F = freelancers, J = unique jobs seen
- **Computation**: O(F × A) per round where A = active jobs
- **Storage**: Decision history persisted in simulation results JSON

## 🎯 Best Practices

### For Researchers
1. **Control Groups**: Always run simulations with/without cooloff for comparison
2. **Sufficient Rounds**: Use 15+ rounds to see cooloff effects manifest
3. **Multiple Trials**: Run multiple simulations with same parameters for statistical significance
4. **Parameter Sweeps**: Test different cooloff periods to find optimal values

### For Implementation
1. **Clear Logging**: Monitor cooloff expirations to verify system behavior
2. **Data Validation**: Ensure decision tracking integrity across rounds
3. **Performance Monitoring**: Track memory usage with large freelancer populations
4. **Configuration Documentation**: Document cooloff settings in result metadata

## 🔮 Future Enhancements

### Potential Extensions
- **Dynamic Cooloff**: Adjust cooloff periods based on market conditions
- **Job-Specific Cooloff**: Different cooloff periods for different job types
- **Graduated Re-entry**: Longer cooloffs for repeated declines
- **Client-Driven Cooloff**: Allow clients to set job-specific cooloff periods

### Research Opportunities
- **Cross-Platform Validation**: Compare with real marketplace data
- **Human Behavior Studies**: Test cooloff effects with human subjects
- **Market Design**: Optimize cooloff parameters for different market conditions
- **Agent Diversity**: Study cooloff effects across different agent personalities

---

*This documentation covers the complete bid cooloff system as implemented in the True GPT Marketplace framework. The system enables sophisticated studies of agent adaptation, market efficiency, and long-term decision evolution in AI-powered economic simulations.*
