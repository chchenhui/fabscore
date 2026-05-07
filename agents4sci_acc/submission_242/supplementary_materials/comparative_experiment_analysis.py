#!/usr/bin/env python3
"""
Comprehensive comparison between 1-bid and 3-bid experiments
Demonstrates framework parameter sensitivity
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def load_experiment_data(file_path):
    """Load and parse experiment data"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def analyze_experiment(data, experiment_name):
    """Extract key metrics from experiment data"""
    config = data['simulation_config']
    round_data = data['round_data']
    freelancer_profiles = data['freelancer_profiles']
    reputation_data = data['reputation_data']['freelancers']
    
    # Basic metrics
    total_jobs = sum(r['jobs_posted'] for r in round_data)
    total_bids = sum(r['total_bids'] for r in round_data)
    total_filled = round_data[-1]['jobs_filled'] if round_data else 0  # Final cumulative count
    
    # Round-by-round analysis
    rounds_df = pd.DataFrame([{
        'round': r['round'],
        'jobs_posted': r['jobs_posted'],
        'total_bids': r['total_bids'],
        'jobs_filled_this_round': r.get('jobs_filled_this_round', r['jobs_filled']),  # Use jobs_filled_this_round if available
        'jobs_filled_cumulative': r['jobs_filled'],
        'bid_rejection_rate': r['bid_rejection_metrics']['bid_rejection_rate'],
        'participation_rate': r['market_activity']['freelancer_participation_rate'],
        'avg_bids_per_job': r['bid_distribution']['avg_bids_per_job'],
        'supply_demand_ratio': r['market_health']['supply_demand_ratio']
    } for r in round_data])
    
    # Freelancer analysis
    active_bidders = sum(1 for f in freelancer_profiles.values() if f['total_bids'] > 0)
    successful_freelancers = sum(1 for f in freelancer_profiles.values() if f['total_hired'] > 0)
    
    # Reputation distribution
    tiers = {}
    for freelancer_id, rep in reputation_data.items():
        tier = rep['tier']
        tiers[tier] = tiers.get(tier, 0) + 1
    
    # Temporal trends (early vs late)
    early_rounds = rounds_df[rounds_df['round'] <= 50]
    late_rounds = rounds_df[rounds_df['round'] > 100]
    
    return {
        'experiment_name': experiment_name,
        'config': config,
        'total_jobs': total_jobs,
        'total_bids': total_bids,
        'total_filled': total_filled,
        'fill_rate': total_filled / total_jobs if total_jobs > 0 else 0,
        'bid_efficiency': total_filled / total_bids if total_bids > 0 else 0,
        'avg_bids_per_job': total_bids / total_jobs if total_jobs > 0 else 0,
        'avg_bid_rejection_rate': rounds_df['bid_rejection_rate'].mean(),
        'avg_participation_rate': rounds_df['participation_rate'].mean(),
        'active_freelancers': active_bidders,
        'successful_freelancers': successful_freelancers,
        'freelancer_hiring_rate': successful_freelancers / len(freelancer_profiles),
        'reputation_tiers': tiers,
        'temporal_trends': {
            'early_participation': early_rounds['participation_rate'].mean(),
            'late_participation': late_rounds['participation_rate'].mean(),
            'early_rejection_rate': early_rounds['bid_rejection_rate'].mean(),
            'late_rejection_rate': late_rounds['bid_rejection_rate'].mean(),
            'early_bids_per_job': early_rounds['avg_bids_per_job'].mean(),
            'late_bids_per_job': late_rounds['avg_bids_per_job'].mean()
        }
    }

def compare_experiments():
    """Main comparison function"""
    
    # File paths
    file_1_bid = "/Users/silviaterragni/dev/simulated_marketplace/results/simuleval/true_gpt_simulation_20250903_183837.json"
    file_3_bid = "/Users/silviaterragni/dev/simulated_marketplace/results/simuleval/true_gpt_simulation_20250903_114208.json"
    
    print("🔬 FRAMEWORK PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Load both experiments
    print(f"📂 Loading experiments...")
    data_1_bid = load_experiment_data(file_1_bid)
    data_3_bid = load_experiment_data(file_3_bid)
    
    if not data_1_bid or not data_3_bid:
        print("❌ Failed to load experiment data")
        return
        
    # Analyze both experiments
    print(f"📊 Analyzing experiments...")
    analysis_1_bid = analyze_experiment(data_1_bid, "1-Bid Constraint")
    analysis_3_bid = analyze_experiment(data_3_bid, "3-Bid Constraint")
    
    # Print comparison
    print(f"\n🎯 EXPERIMENT COMPARISON")
    print("=" * 60)
    
    print(f"\n📈 CORE METRICS:")
    print(f"{'Metric':<25} {'1-Bid':<15} {'3-Bid':<15} {'Difference'}")
    print("-" * 60)
    print(f"{'Jobs Posted':<25} {analysis_1_bid['total_jobs']} {analysis_3_bid['total_jobs']} {analysis_3_bid['total_jobs'] - analysis_1_bid['total_jobs']:+}")
    print(f"{'Bids Submitted':<25} {analysis_1_bid['total_bids']} {analysis_3_bid['total_bids']} {analysis_3_bid['total_bids'] - analysis_1_bid['total_bids']:+}")
    print(f"{'Jobs Filled':<25} {analysis_1_bid['total_filled']} {analysis_3_bid['total_filled']} {analysis_3_bid['total_filled'] - analysis_1_bid['total_filled']:+}")
    print(f"{'Fill Rate':<25} {analysis_1_bid['fill_rate']:.1%} {analysis_3_bid['fill_rate']:.1%} {(analysis_3_bid['fill_rate'] - analysis_1_bid['fill_rate'])*100:+.1f}%")
    print(f"{'Bid Efficiency':<25} {analysis_1_bid['bid_efficiency']:.1%} {analysis_3_bid['bid_efficiency']:.1%} {(analysis_3_bid['bid_efficiency'] - analysis_1_bid['bid_efficiency'])*100:+.1f}%")
    print(f"{'Bids per Job':<25} {analysis_1_bid['avg_bids_per_job']:.2f} {analysis_3_bid['avg_bids_per_job']:.2f} {analysis_3_bid['avg_bids_per_job'] - analysis_1_bid['avg_bids_per_job']:+.2f}")
    
    print(f"\n🎪 MARKET BEHAVIOR:")
    print(f"{'Metric':<25} {'1-Bid':<15} {'3-Bid':<15} {'Difference'}")
    print("-" * 60)
    print(f"{'Avg Participation':<25} {analysis_1_bid['avg_participation_rate']:.1%} {analysis_3_bid['avg_participation_rate']:.1%} {(analysis_3_bid['avg_participation_rate'] - analysis_1_bid['avg_participation_rate'])*100:+.1f}%")
    print(f"{'Avg Rejection Rate':<25} {analysis_1_bid['avg_bid_rejection_rate']:.1%} {analysis_3_bid['avg_bid_rejection_rate']:.1%} {(analysis_3_bid['avg_bid_rejection_rate'] - analysis_1_bid['avg_bid_rejection_rate'])*100:+.1f}%")
    print(f"{'Active Freelancers':<25} {analysis_1_bid['active_freelancers']} {analysis_3_bid['active_freelancers']} {analysis_3_bid['active_freelancers'] - analysis_1_bid['active_freelancers']:+}")
    print(f"{'Successful Freelancers':<25} {analysis_1_bid['successful_freelancers']} {analysis_3_bid['successful_freelancers']} {analysis_3_bid['successful_freelancers'] - analysis_1_bid['successful_freelancers']:+}")
    print(f"{'Freelancer Hiring Rate':<25} {analysis_1_bid['freelancer_hiring_rate']:.1%} {analysis_3_bid['freelancer_hiring_rate']:.1%} {(analysis_3_bid['freelancer_hiring_rate'] - analysis_1_bid['freelancer_hiring_rate'])*100:+.1f}%")
    
    print(f"\n⏰ TEMPORAL DYNAMICS:")
    print(f"{'Metric':<25} {'1-Bid Early':<15} {'1-Bid Late':<15} {'3-Bid Early':<15} {'3-Bid Late':<15}")
    print("-" * 75)
    t1 = analysis_1_bid['temporal_trends']
    t3 = analysis_3_bid['temporal_trends']
    print(f"{'Participation Rate':<25} {t1['early_participation']:.1%} {t1['late_participation']:.1%} {t3['early_participation']:.1%} {t3['late_participation']:.1%}")
    print(f"{'Rejection Rate':<25} {t1['early_rejection_rate']:.1%} {t1['late_rejection_rate']:.1%} {t3['early_rejection_rate']:.1%} {t3['late_rejection_rate']:.1%}")
    print(f"{'Bids per Job':<25} {t1['early_bids_per_job']:.2f} {t1['late_bids_per_job']:.2f} {t3['early_bids_per_job']:.2f} {t3['late_bids_per_job']:.2f}")
    
    print(f"\n🏆 REPUTATION DISTRIBUTION:")
    all_tiers = set(analysis_1_bid['reputation_tiers'].keys()) | set(analysis_3_bid['reputation_tiers'].keys())
    for tier in sorted(all_tiers):
        count_1 = analysis_1_bid['reputation_tiers'].get(tier, 0)
        count_3 = analysis_3_bid['reputation_tiers'].get(tier, 0)
        pct_1 = count_1 / 200 * 100  # 200 freelancers
        pct_3 = count_3 / 200 * 100
        print(f"  {tier:<15} {count_1} ({pct_1:.1f}%)    vs    {count_3} ({pct_3:.1f}%)")
    
    # Framework insights
    print(f"\n💡 FRAMEWORK INSIGHTS:")
    print("=" * 60)
    
    # Surprising finding: results are very similar!
    fill_rate_diff = abs(analysis_3_bid['fill_rate'] - analysis_1_bid['fill_rate'])
    if fill_rate_diff < 0.02:  # Less than 2% difference
        print("🔍 SURPRISING FINDING: Fill rates nearly identical!")
        print(f"   1-bid: {analysis_1_bid['fill_rate']:.1%}, 3-bid: {analysis_3_bid['fill_rate']:.1%}")
        print("   This suggests freelancers were highly strategic even with more bid budget!")
        
    bids_per_job_diff = analysis_3_bid['avg_bids_per_job'] - analysis_1_bid['avg_bids_per_job']
    if bids_per_job_diff > 0:
        print(f"\n📈 BID INTENSITY: +{bids_per_job_diff:.2f} more bids per job with 3-bid constraint")
        print("   Framework captures increased market activity with relaxed constraints")
        
    # Success efficiency
    eff_1 = analysis_1_bid['bid_efficiency']
    eff_3 = analysis_3_bid['bid_efficiency']
    if eff_1 > eff_3:
        print(f"\n🎯 BID EFFICIENCY: 1-bid constraint more efficient ({eff_1:.1%} vs {eff_3:.1%})")
        print("   Constraint forces higher-quality bidding decisions")
    elif eff_3 > eff_1:
        print(f"\n🎯 BID EFFICIENCY: 3-bid constraint more efficient ({eff_3:.1%} vs {eff_1:.1%})")
        print("   More opportunities lead to better matches")
    
    # Save detailed comparison
    comparison_data = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'experiment_1_bid': analysis_1_bid,
        'experiment_3_bid': analysis_3_bid,
        'key_findings': {
            'fill_rate_difference': analysis_3_bid['fill_rate'] - analysis_1_bid['fill_rate'],
            'bid_efficiency_difference': analysis_3_bid['bid_efficiency'] - analysis_1_bid['bid_efficiency'],
            'market_activity_difference': analysis_3_bid['avg_bids_per_job'] - analysis_1_bid['avg_bids_per_job'],
            'participation_difference': analysis_3_bid['avg_participation_rate'] - analysis_1_bid['avg_participation_rate']
        }
    }
    
    print(f"\n💾 Saving detailed comparison...")
    with open('framework_parameter_sensitivity_analysis.json', 'w') as f:
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        json.dump(convert_numpy(comparison_data), f, indent=2)
    
    print("✅ Analysis complete!")
    
    return comparison_data

if __name__ == "__main__":
    compare_experiments()
