#!/usr/bin/env python3
"""
Category Breakdown Analysis for LLM-LLM Configuration

Analyzes freelancer distribution, job distribution, and trends by skill category
in the LLM-LLM marketplace simulation.
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px

def analyze_llm_llm_by_category():
    """Comprehensive category-based analysis of LLM-LLM configuration"""
    
    print("=" * 80)
    print("CATEGORY BREAKDOWN ANALYSIS - LLM-LLM CONFIGURATION")
    print("=" * 80)
    
    # Load LLM-LLM simulation data
    llm_llm_file = "results/simuleval/true_gpt_simulation_20250905_001404.json"
    
    try:
        with open(llm_llm_file, 'r') as f:
            simulation_data = json.load(f)
        print(f"✅ Loaded simulation data: {llm_llm_file}")
    except FileNotFoundError:
        print(f"❌ File not found: {llm_llm_file}")
        return
    
    # Extract data
    freelancers = simulation_data.get('freelancer_profiles', {})
    all_jobs = simulation_data.get('all_jobs', [])
    all_bids = simulation_data.get('all_bids', [])
    hiring_outcomes = simulation_data.get('hiring_outcomes', [])
    round_data = simulation_data.get('round_data', [])
    
    # Extract client data from round_data (clients are embedded in job postings)
    clients_by_category = defaultdict(set)
    for job in all_jobs:
        client_id = job.get('client_id', 'unknown')
        job_category = job.get('category', 'Unknown')
        clients_by_category[job_category].add(client_id)
    
    # Calculate total unique clients
    all_clients = set()
    for client_set in clients_by_category.values():
        all_clients.update(client_set)
    
    print(f"📊 Data Overview:")
    print(f"   - Freelancers: {len(freelancers)}")
    print(f"   - Clients: {len(all_clients)}")
    print(f"   - Jobs: {len(all_jobs)}")
    print(f"   - Bids: {len(all_bids)}")
    print(f"   - Hiring outcomes: {len(hiring_outcomes)}")
    print(f"   - Rounds: {len(round_data)}")
    
    # Use existing categories from freelancer and job profiles
    
    # Analyze freelancer distribution by existing category
    print(f"\n🧑‍💼 FREELANCER DISTRIBUTION BY CATEGORY")
    print("-" * 60)
    
    freelancer_categories = defaultdict(list)
    freelancer_category_counts = Counter()
    
    for freelancer_id, freelancer in freelancers.items():
        # Use existing category assignment
        freelancer_category = freelancer.get('category', 'Unknown')
        freelancer_categories[freelancer_category].append(freelancer_id)
        freelancer_category_counts[freelancer_category] += 1
    
    # Display freelancer distribution
    for category, count in freelancer_category_counts.most_common():
        percentage = (count / len(freelancers)) * 100
        print(f"   {category:<25}: {count:3d} freelancers ({percentage:5.1f}%)")
    
    # Analyze job distribution by category
    print(f"\n💼 JOB DISTRIBUTION BY CATEGORY")
    print("-" * 60)
    
    job_categories = defaultdict(list)
    job_category_counts = Counter()
    
    for job in all_jobs:
        job_category = job.get('category', 'Unknown')
        job_categories[job_category].append(job['id'])
        job_category_counts[job_category] += 1
    
    # Display job distribution
    for category, count in job_category_counts.most_common():
        percentage = (count / len(all_jobs)) * 100
        print(f"   {category:<25}: {count:3d} jobs ({percentage:5.1f}%)")
    
    # Analyze client distribution by category
    print(f"\n🏢 CLIENT DISTRIBUTION BY CATEGORY")
    print("-" * 60)
    
    client_category_counts = Counter()
    for category, client_set in clients_by_category.items():
        client_category_counts[category] = len(client_set)
    
    # Display client distribution
    for category, count in client_category_counts.most_common():
        percentage = (count / len(all_clients)) * 100
        print(f"   {category:<25}: {count:3d} clients ({percentage:5.1f}%)")
    
    # Analyze bidding patterns by category
    print("\n🎯 BIDDING PATTERNS BY CATEGORY")
    print("-" * 60)
    
    # Create bid analysis by job category
    bids_by_job_category = defaultdict(list)
    for bid in all_bids:
        job_id = bid.get('job_id')
        # Find job category
        job_category = None
        for job in all_jobs:
            if job['id'] == job_id:
                job_category = job.get('category', 'Unknown')
                break
        if job_category:
            bids_by_job_category[job_category].append(bid)
    
    # Calculate bidding metrics by category
    print(f"{'Category':<25} {'Jobs':<6} {'Bids':<6} {'Avg Bids/Job':<12} {'Fill Rate':<10}")
    print("-" * 70)
    
    category_metrics = {}
    
    for category in job_category_counts.keys():
        jobs_in_category = job_category_counts[category]
        bids_in_category = len(bids_by_job_category[category])
        avg_bids_per_job = bids_in_category / jobs_in_category if jobs_in_category > 0 else 0
        
        # Calculate fill rate for this category (unique jobs filled, not total outcomes)
        filled_job_ids = set()
        for outcome in hiring_outcomes:
            if outcome.get('selected_freelancer') not in [None, 'none']:
                job_id = outcome.get('job_id')
                # Find job category
                for job in all_jobs:
                    if job['id'] == job_id and job.get('category') == category:
                        filled_job_ids.add(job_id)
                        break
        
        filled_jobs = len(filled_job_ids)
        fill_rate = (filled_jobs / jobs_in_category) * 100 if jobs_in_category > 0 else 0
        
        category_metrics[category] = {
            'jobs': jobs_in_category,
            'bids': bids_in_category,
            'avg_bids_per_job': avg_bids_per_job,
            'fill_rate': fill_rate,
            'filled_jobs': filled_jobs
        }
        
        print(f"{category:<25} {jobs_in_category:<6} {bids_in_category:<6} {avg_bids_per_job:<12.2f} {fill_rate:<10.1f}%")
    
    # Analyze freelancer success by their actual category
    print(f"\n🏆 FREELANCER SUCCESS BY THEIR ACTUAL CATEGORY")
    print("-" * 80)
    
    # Get hired freelancers
    hired_freelancers = set()
    freelancer_job_wins = defaultdict(int)
    for outcome in hiring_outcomes:
        selected = outcome.get('selected_freelancer')
        if selected and selected != 'none':
            hired_freelancers.add(selected)
            freelancer_job_wins[selected] += 1
    
    # Success by freelancer category
    freelancer_success = defaultdict(lambda: {'total': 0, 'hired': 0, 'jobs_won': 0})
    for fid, freelancer in freelancers.items():
        cat = freelancer.get('category', 'Unknown')
        freelancer_success[cat]['total'] += 1
        if fid in hired_freelancers:
            freelancer_success[cat]['hired'] += 1
            freelancer_success[cat]['jobs_won'] += freelancer_job_wins[fid]
    
    print(f"{'Freelancer Category':<30} {'Total':<6} {'Hired':<6} {'Success Rate':<12} {'Jobs Won':<10}")
    print("-" * 80)
    
    for cat, metrics in sorted(freelancer_success.items()):
        total = metrics['total']
        hired = metrics['hired']
        success_rate = (hired / total) * 100 if total > 0 else 0
        jobs_won = metrics['jobs_won']
        print(f'{cat:<30} {total:<6} {hired:<6} {success_rate:<12.1f}% {jobs_won:<10}')
    
    # Temporal trends by category
    print(f"\n📈 TEMPORAL TRENDS BY CATEGORY")
    print("-" * 60)
    
    analyze_temporal_trends_by_category(round_data, all_jobs, job_categories)
    
    # Cross-category hiring patterns
    print(f"\n🔄 CROSS-CATEGORY HIRING PATTERNS")
    print("-" * 60)
    
    analyze_cross_category_hiring(hiring_outcomes, all_jobs, freelancers)
    
    # Generate visualizations
    create_category_visualizations(
        freelancer_category_counts, 
        job_category_counts, 
        category_metrics, 
        freelancer_success
    )
    
    # Create Sankey diagram for cross-category hiring flows
    create_sankey_flow_diagram(hiring_outcomes, all_jobs, freelancers)
    
    # Additional market analyses
    analyze_competition_patterns(all_jobs, all_bids, freelancers, job_category_counts)
    analyze_cross_category_flows(hiring_outcomes, all_jobs, freelancers)
    analyze_reputation_tiers(simulation_data)
    analyze_rejection_patterns_with_charts(hiring_outcomes, all_bids)
    
    # Reputation system impact analysis
    print("\n" + "="*60)
    print("REPUTATION SYSTEM IMPACT ANALYSIS")
    print("="*60)
    
    output_dir = Path("analysis_results/category_breakdown")
    reputation_data = analyze_reputation_impact(simulation_data, output_dir)
    
    print(f"\n✅ Category breakdown analysis complete!")
    print(f"📊 Visualizations saved to: analysis_results/category_breakdown/")


def analyze_temporal_trends_by_category(round_data, all_jobs, job_categories):
    """Analyze how different categories perform over time"""
    
    # Group jobs by round and category
    jobs_by_round_category = defaultdict(lambda: defaultdict(int))
    
    for job in all_jobs:
        job_round = job.get('round', 1)
        job_category = job.get('category', 'Unknown')
        jobs_by_round_category[job_round][job_category] += 1
    
    # Show trends for top 5 categories
    top_categories = Counter()
    for round_jobs in jobs_by_round_category.values():
        for category, count in round_jobs.items():
            top_categories[category] += count
    
    print(f"Job posting trends for top 5 categories:")
    print(f"{'Round':<6}", end='')
    top_5_categories = [cat for cat, _ in top_categories.most_common(5)]
    for category in top_5_categories:
        print(f"{category[:15]:<16}", end='')
    print()
    print("-" * (6 + 16 * 5))
    
    # Show first 10 rounds
    for round_num in sorted(jobs_by_round_category.keys())[:10]:
        print(f"{round_num:<6}", end='')
        for category in top_5_categories:
            count = jobs_by_round_category[round_num][category]
            print(f"{count:<16}", end='')
        print()

def analyze_cross_category_hiring(hiring_outcomes, all_jobs, freelancers):
    """Analyze hiring patterns across different job-freelancer category combinations"""
    
    cross_hires = defaultdict(lambda: defaultdict(int))
    
    for outcome in hiring_outcomes:
        selected = outcome.get('selected_freelancer')
        if selected and selected != 'none':
            job_id = outcome.get('job_id')
            
            # Find job category
            job_cat = None
            for job in all_jobs:
                if job['id'] == job_id:
                    job_cat = job.get('category', 'Unknown')
                    break
            
            # Find freelancer category
            freelancer_cat = freelancers.get(selected, {}).get('category', 'Unknown')
            
            if job_cat and freelancer_cat:
                cross_hires[job_cat][freelancer_cat] += 1
    
    # Show significant cross-category patterns
    print(f"(Job Category → Freelancer Category)")
    print()
    
    for job_cat, freelancer_hires in cross_hires.items():
        total_hires = sum(freelancer_hires.values())
        if total_hires >= 10:  # Only show categories with significant hiring
            print(f'{job_cat} jobs ({total_hires} total hires):')
            for freelancer_cat, count in sorted(freelancer_hires.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    pct = (count / total_hires) * 100
                    match_type = '✓ Direct Match' if job_cat == freelancer_cat else '✗ Cross-category'
                    print(f'  → {freelancer_cat:<28} {count:2d} hires ({pct:4.1f}%) {match_type}')
            print()


def create_sankey_flow_diagram(hiring_outcomes, all_jobs, freelancers):
    """Create a Sankey diagram showing freelancer category → job category hiring flows"""
    
    print(f"\n🌊 CREATING SANKEY FLOW DIAGRAM")
    print("-" * 60)
    
    # Create output directory
    output_dir = Path("analysis_results/category_breakdown")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all hiring flows
    flows = defaultdict(int)
    
    for outcome in hiring_outcomes:
        selected = outcome.get('selected_freelancer')
        if selected and selected != 'none':
            job_id = outcome.get('job_id')
            
            # Find job category
            job_cat = None
            for job in all_jobs:
                if job['id'] == job_id:
                    job_cat = job.get('category', 'Unknown')
                    break
            
            # Find freelancer category
            freelancer_cat = freelancers.get(selected, {}).get('category', 'Unknown')
            
            if job_cat and freelancer_cat:
                flows[(freelancer_cat, job_cat)] += 1
    
    # Prepare data for Sankey diagram
    freelancer_categories = sorted(set(flow[0] for flow in flows.keys()))
    job_categories = sorted(set(flow[1] for flow in flows.keys()))
    
    # Create node lists and labels
    all_categories = freelancer_categories + job_categories
    node_labels = [f"Freelancer: {cat}" for cat in freelancer_categories] + [f"Job: {cat}" for cat in job_categories]
    
    # Create source, target, and value lists for Sankey
    source = []
    target = []
    value = []
    
    for (freelancer_cat, job_cat), count in flows.items():
        if count > 0:  # Only include actual flows
            source_idx = freelancer_categories.index(freelancer_cat)
            target_idx = len(freelancer_categories) + job_categories.index(job_cat)
            
            source.append(source_idx)
            target.append(target_idx)
            value.append(count)
    
    # Create color scheme - distinct colors for each category
    # Use a color palette with enough distinct colors
    color_palette = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#AED6F1', '#A9DFBF'
    ]
    
    # Ensure we have enough colors by cycling if needed
    while len(color_palette) < len(freelancer_categories):
        color_palette.extend(color_palette)
    
    # Create node colors - same color for matching freelancer and job categories
    node_colors = []
    
    # Freelancer nodes (left side)
    for i, cat in enumerate(freelancer_categories):
        node_colors.append(color_palette[i % len(color_palette)])
    
    # Job nodes (right side) - use same colors as corresponding freelancer categories
    for cat in job_categories:
        if cat in freelancer_categories:
            # Use same color as the corresponding freelancer category
            idx = freelancer_categories.index(cat)
            node_colors.append(color_palette[idx % len(color_palette)])
        else:
            # Use a default color for job categories without corresponding freelancer category
            node_colors.append('#CCCCCC')
    
    # Create link colors using the freelancer category colors
    link_colors = []
    for i, (freelancer_cat, job_cat) in enumerate([(freelancer_categories[s], job_categories[t - len(freelancer_categories)]) 
                                                   for s, t in zip(source, target)]):
        # Use the freelancer category color for the link
        freelancer_idx = freelancer_categories.index(freelancer_cat)
        base_color = color_palette[freelancer_idx % len(color_palette)]
        
        # Convert hex to rgba and adjust opacity based on match type
        hex_color = base_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        if freelancer_cat == job_cat:
            # Same category: more opaque
            link_colors.append(f'rgba({r}, {g}, {b}, 0.6)')
        else:
            # Cross-category: more transparent
            link_colors.append(f'rgba({r}, {g}, {b}, 0.3)')
    
    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_colors
        )
    )])
    
    fig.update_layout(
        title_text="Freelancer Category → Job Category Hiring Flows<br><sub>Each category has a distinct color. Darker flows = Same category, Lighter flows = Cross-category</sub>",
        font_size=12,
        width=1200,
        height=800
    )
    
    # Save as HTML
    sankey_file = output_dir / "category_hiring_flows.html"
    fig.write_html(str(sankey_file))
    
    # Also save as PNG (requires kaleido)
    try:
        png_file = output_dir / "category_hiring_flows.png"
        fig.write_image(str(png_file), width=1200, height=800, scale=2)
        print(f"📊 Sankey diagram saved as PNG: {png_file}")
    except Exception as e:
        print(f"⚠️  Could not save PNG (install kaleido for PNG export): {e}")
    
    print(f"📊 Interactive Sankey diagram saved: {sankey_file}")
    print(f"💡 Open the HTML file in your browser to interact with the diagram")
    
    # Print summary statistics
    same_category_hires = sum(count for (f_cat, j_cat), count in flows.items() if f_cat == j_cat)
    cross_category_hires = sum(count for (f_cat, j_cat), count in flows.items() if f_cat != j_cat)
    total_hires = same_category_hires + cross_category_hires
    
    print(f"\n📈 Hiring Flow Summary:")
    print(f"   Same-category hires: {same_category_hires} ({same_category_hires/total_hires*100:.1f}%)")
    print(f"   Cross-category hires: {cross_category_hires} ({cross_category_hires/total_hires*100:.1f}%)")
    print(f"   Total hires: {total_hires}")


def create_category_visualizations(freelancer_counts, job_counts, category_metrics, freelancer_success):
    """Create visualizations for category analysis"""
    
    # Create output directory
    output_dir = Path("analysis_results/category_breakdown")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Freelancer and Job Distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Freelancer distribution
    categories = list(freelancer_counts.keys())[:8]  # Top 8 categories
    counts = [freelancer_counts[cat] for cat in categories]
    
    ax1.bar(range(len(categories)), counts, alpha=0.7)
    ax1.set_title('Freelancer Distribution by Category', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Number of Freelancers')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Job distribution
    job_categories = list(job_counts.keys())[:8]  # Top 8 categories
    job_counts_list = [job_counts[cat] for cat in job_categories]
    
    ax2.bar(range(len(job_categories)), job_counts_list, alpha=0.7, color='orange')
    ax2.set_title('Job Distribution by Category', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Category')
    ax2.set_ylabel('Number of Jobs')
    ax2.set_xticks(range(len(job_categories)))
    ax2.set_xticklabels(job_categories, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "freelancer_job_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Category Performance Metrics - Focus on Fill Rate and Competition Level Only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    categories = list(category_metrics.keys())
    fill_rates = [category_metrics[cat]['fill_rate'] for cat in categories]
    avg_bids = [category_metrics[cat]['avg_bids_per_job'] for cat in categories]
    
    # Fill rates
    ax1.bar(range(len(categories)), fill_rates, alpha=0.7, color='green')
    ax1.set_title('Fill Rate by Job Category', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Fill Rate (%)')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Average bids per job (Competition Level)
    ax2.bar(range(len(categories)), avg_bids, alpha=0.7, color='red')
    ax2.set_title('Competition Level by Job Category', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Average Bids per Job')
    ax2.set_xticks(range(len(categories)))
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "category_performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved:")
    print(f"   - {output_dir}/freelancer_job_distribution.png")
    print(f"   - {output_dir}/category_performance_metrics.png")


def analyze_competition_patterns(all_jobs, all_bids, freelancers, job_category_counts):
    """Analyze competition patterns and bidding behavior across categories"""
    
    print(f"\n🏁 COMPETITION PATTERNS BY CATEGORY")
    print("-" * 60)
    
    competition_data = {}
    
    for category, job_count in job_category_counts.items():
        # Get jobs and bids for this category
        category_jobs = [job for job in all_jobs if job.get('category') == category]
        job_ids = [job['id'] for job in category_jobs]
        category_bids = [bid for bid in all_bids if bid.get('job_id') in job_ids]
        
        if not category_bids:
            continue
        
        total_bids = len(category_bids)
        avg_bids_per_job = total_bids / job_count if job_count > 0 else 0
        
        # Count unique bidders and repeat bidders
        freelancer_bid_counts = Counter(bid.get('freelancer_id') for bid in category_bids)
        unique_bidders = len(freelancer_bid_counts)
        
        # Calculate repeat bidder rate
        jobs_per_freelancer = defaultdict(set)
        for bid in category_bids:
            jobs_per_freelancer[bid.get('freelancer_id')].add(bid.get('job_id'))
        
        repeat_bidders = sum(1 for jobs_set in jobs_per_freelancer.values() if len(jobs_set) > 1)
        repeat_rate = repeat_bidders / unique_bidders if unique_bidders > 0 else 0
        
        # Check if bidding is well-distributed (no single freelancer dominates)
        max_bids_by_one = max(freelancer_bid_counts.values()) if freelancer_bid_counts else 0
        max_share = max_bids_by_one / total_bids if total_bids > 0 else 0
        
        competition_data[category] = {
            'avg_bids_per_job': avg_bids_per_job,
            'unique_bidders': unique_bidders,
            'repeat_rate': repeat_rate,
            'max_freelancer_share': max_share,
            'total_bids': total_bids
        }
    
    # Display results
    print(f"{'Category':<25} {'Bids/Job':<9} {'Unique':<7} {'Repeat%':<8} {'MaxShare%':<10}")
    print("-" * 70)
    
    for category, data in sorted(competition_data.items()):
        print(f"{category:<25} {data['avg_bids_per_job']:<9.2f} {data['unique_bidders']:<7} "
              f"{data['repeat_rate']*100:<8.1f} {data['max_freelancer_share']*100:<10.1f}")
    
    # Summary insights
    high_competition = [cat for cat, data in competition_data.items() if data['avg_bids_per_job'] > 6]
    well_distributed = [cat for cat, data in competition_data.items() if data['max_freelancer_share'] < 0.3]
    
    print(f"\n📊 Competition Insights:")
    print(f"High competition categories (>6 bids/job): {', '.join(high_competition)}")
    print(f"Well-distributed bidding (<30% max share): {len(well_distributed)}/{len(competition_data)} categories")
    
    return competition_data


def analyze_cross_category_flows(hiring_outcomes, all_jobs, freelancers):
    """Analyze cross-category hiring flows and patterns"""
    
    print(f"\n🔄 CROSS-CATEGORY HIRING FLOWS")
    print("-" * 60)
    
    # Track hiring flows between categories
    flows = defaultdict(int)
    category_totals = defaultdict(int)
    
    for outcome in hiring_outcomes:
        selected = outcome.get('selected_freelancer')
        if selected and selected != 'none':
            job_id = outcome.get('job_id')
            
            # Find job and freelancer categories
            job_cat = None
            for job in all_jobs:
                if job['id'] == job_id:
                    job_cat = job.get('category', 'Unknown')
                    break
            
            freelancer_cat = freelancers.get(selected, {}).get('category', 'Unknown')
            
            if job_cat and freelancer_cat:
                flows[(freelancer_cat, job_cat)] += 1
                category_totals[freelancer_cat] += 1
    
    # Calculate specialization rates (same-category hiring)
    print("Category Specialization (% hired within same category):")
    print("-" * 60)
    
    specialization_data = {}
    for freelancer_cat, total_hires in category_totals.items():
        same_category_hires = flows.get((freelancer_cat, freelancer_cat), 0)
        specialization_rate = same_category_hires / total_hires if total_hires > 0 else 0
        cross_category_rate = 1 - specialization_rate
        
        specialization_data[freelancer_cat] = {
            'specialization_rate': specialization_rate,
            'cross_category_rate': cross_category_rate,
            'total_hires': total_hires
        }
        
        print(f"{freelancer_cat:<30}: {specialization_rate*100:5.1f}% within category, "
              f"{cross_category_rate*100:5.1f}% cross-category")
    
    # Show significant cross-category flows
    print(f"\nMajor Cross-Category Flows (>5% of freelancer category):")
    print("-" * 60)
    
    significant_flows = []
    for (freelancer_cat, job_cat), count in flows.items():
        if freelancer_cat != job_cat:  # Only cross-category
            total_from_category = category_totals[freelancer_cat]
            if total_from_category > 0:
                flow_rate = count / total_from_category
                if flow_rate > 0.05:  # >5% threshold
                    significant_flows.append((freelancer_cat, job_cat, flow_rate, count))
    
    significant_flows.sort(key=lambda x: x[2], reverse=True)
    
    for freelancer_cat, job_cat, rate, count in significant_flows:
        print(f"{freelancer_cat:<20} → {job_cat:<20}: {rate*100:5.1f}% ({count} hires)")
    
    # Summary statistics
    total_hires = sum(category_totals.values())
    same_category_total = sum(flows.get((cat, cat), 0) for cat in category_totals.keys())
    cross_category_total = total_hires - same_category_total
    
    print(f"\n📊 Overall Flow Summary:")
    print(f"Same-category hires: {same_category_total} ({same_category_total/total_hires*100:.1f}%)")
    print(f"Cross-category hires: {cross_category_total} ({cross_category_total/total_hires*100:.1f}%)")
    
    return specialization_data, significant_flows


def analyze_reputation_tiers(simulation_data):
    """Analyze final reputation tier distribution"""
    
    print(f"\n🏆 FINAL REPUTATION TIER DISTRIBUTION")
    print("-" * 60)
    
    # Get reputation data
    reputation_data = simulation_data.get('reputation_data', {})
    freelancer_reputation = reputation_data.get('freelancers', {})
    
    if not freelancer_reputation:
        print("No reputation data available")
        return
    
    # Count tiers
    tier_counts = Counter()
    for freelancer_id, rep_info in freelancer_reputation.items():
        tier = rep_info.get('tier', 'Unknown')
        tier_counts[tier] += 1
    
    total_freelancers = len(freelancer_reputation)
    
    print(f"Freelancer Reputation Tiers (Total: {total_freelancers}):")
    print("-" * 60)
    
    for tier, count in tier_counts.most_common():
        percentage = (count / total_freelancers) * 100
        print(f"   {tier:<15}: {count:3d} freelancers ({percentage:5.1f}%)")
    
    # Create pie chart
    output_dir = Path("analysis_results/category_breakdown")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if tier_counts:
        plt.figure(figsize=(8, 8))
        tiers = list(tier_counts.keys())
        counts = list(tier_counts.values())
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFD700'][:len(tiers)]
        
        plt.pie(counts, labels=tiers, autopct='%1.1f%%', startangle=90, colors=colors)
        plt.title('Final Freelancer Reputation Tier Distribution', fontsize=14, fontweight='bold')
        plt.axis('equal')
        
        plt.savefig(output_dir / "reputation_tier_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Reputation tier pie chart saved: {output_dir}/reputation_tier_distribution.png")
    
    return tier_counts


def analyze_rejection_patterns_with_charts(hiring_outcomes, all_bids):
    """Analyze rejection patterns and create pie charts"""
    
    print(f"\n🚫 REJECTION PATTERN ANALYSIS")
    print("-" * 60)
    
    # Analyze hiring outcomes
    hired_count = 0
    rejected_by_client = 0
    no_bids_received = 0
    rejection_reasons = []
    
    for outcome in hiring_outcomes:
        selected = outcome.get('selected_freelancer')
        reasoning = outcome.get('reasoning', '').lower()
        
        if selected and selected != 'none':
            hired_count += 1
        elif 'no bid' in reasoning or 'no bids' in reasoning:
            no_bids_received += 1
        else:
            rejected_by_client += 1
            rejection_reasons.append(outcome.get('reasoning', ''))
    
    total_jobs = len(hiring_outcomes)
    
    print(f"Job Outcome Distribution (Total: {total_jobs} jobs):")
    print("-" * 60)
    print(f"   Successful hires    : {hired_count:3d} ({hired_count/total_jobs*100:5.1f}%)")
    print(f"   Client rejections   : {rejected_by_client:3d} ({rejected_by_client/total_jobs*100:5.1f}%)")
    print(f"   No bids received    : {no_bids_received:3d} ({no_bids_received/total_jobs*100:5.1f}%)")
    
    # Analyze rejection reasons if available
    if rejection_reasons:
        print(f"\nSample rejection reasons:")
        for i, reason in enumerate(rejection_reasons[:3], 1):
            print(f"   {i}. \"{reason[:100]}{'...' if len(reason) > 100 else ''}\"")
    
    # Create outcome pie chart
    output_dir = Path("analysis_results/category_breakdown")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Job outcomes pie chart
    plt.figure(figsize=(10, 5))
    
    # First subplot: Job outcomes
    plt.subplot(1, 2, 1)
    outcomes = ['Successful Hires', 'Client Rejections', 'No Bids Received']
    counts = [hired_count, rejected_by_client, no_bids_received]
    colors = ['#99FF99', '#FF9999', '#FFCC99']
    
    plt.pie(counts, labels=outcomes, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Job Outcome Distribution', fontsize=12, fontweight='bold')
    
    # Second subplot: Success vs All Rejections
    plt.subplot(1, 2, 2)
    success_vs_rejection = ['Successful Hires', 'All Rejections']
    success_vs_counts = [hired_count, rejected_by_client + no_bids_received]
    colors2 = ['#99FF99', '#FF9999']
    
    plt.pie(success_vs_counts, labels=success_vs_rejection, autopct='%1.1f%%', startangle=90, colors=colors2)
    plt.title('Success vs Rejection Rate', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "job_outcome_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Job outcome charts saved: {output_dir}/job_outcome_analysis.png")
    
    # Bid success analysis
    total_bids = len(all_bids)
    if total_bids > 0:
        bid_success_rate = hired_count / total_bids * 100
        print(f"\n📈 Bid Success Metrics:")
        print(f"   Total bids placed   : {total_bids}")
        print(f"   Successful bids     : {hired_count}")
        print(f"   Bid success rate    : {bid_success_rate:.1f}%")
    
    return {
        'total_jobs': total_jobs,
        'hired_count': hired_count,
        'rejected_by_client': rejected_by_client,
        'no_bids_received': no_bids_received,
        'total_bids': total_bids
    }


def analyze_reputation_impact(simulation_data, output_dir):
    """Analyze the impact of reputation tiers on hiring success and provide examples"""
    
    # Get freelancer final states
    freelancers = simulation_data['freelancer_profiles']
    hiring_outcomes = simulation_data['hiring_outcomes']
    
    # Calculate tier progression
    tier_counts = {'New': 0, 'Established': 0, 'Expert': 0, 'Elite': 0}
    freelancer_progression = []
    
    for freelancer_id, freelancer in freelancers.items():
        total_hired = freelancer.get('total_hired', 0)
        completed_jobs = freelancer.get('completed_jobs', 0)
        
        # Calculate tier based on criteria from paper
        if total_hired < 3:
            tier = 'New'
        elif total_hired <= 6 and (completed_jobs / total_hired >= 0.6 if total_hired > 0 else False):
            tier = 'Established'
        elif total_hired <= 14 and (completed_jobs / total_hired >= 0.75 if total_hired > 0 else False):
            tier = 'Expert'
        elif total_hired >= 15 and (completed_jobs / total_hired >= 0.85 if total_hired > 0 else False):
            tier = 'Elite'
        else:
            tier = 'New'  # Fallback for those who don't meet advancement criteria
        
        tier_counts[tier] += 1
        freelancer_progression.append({
            'id': freelancer_id,
            'name': freelancer.get('name', 'Unknown'),
            'tier': tier,
            'total_hired': total_hired,
            'completed_jobs': completed_jobs,
            'completion_rate': completed_jobs / total_hired if total_hired > 0 else 0
        })
    
    print(f"\n📊 FREELANCER TIER PROGRESSION:")
    total_freelancers = sum(tier_counts.values())
    for tier, count in tier_counts.items():
        percentage = (count / total_freelancers) * 100
        print(f"   {tier:12}: {count:3d} freelancers ({percentage:5.1f}%)")
    
    # Find examples of hiring decisions that mention reputation or success rates
    print(f"\n🔍 SEARCHING FOR REPUTATION-BASED HIRING DECISIONS:")
    reputation_mentions = []
    
    for outcome in hiring_outcomes:
        if outcome.get('selected_freelancer') and outcome.get('reasoning'):
            reasoning = outcome['reasoning'].lower()
            if any(keyword in reasoning for keyword in ['reputation', 'experience', 'success', 'track record', 'established', 'expert', 'proven']):
                # Get freelancer info
                freelancer_id = outcome['selected_freelancer']
                freelancer_info = next((f for f in freelancer_progression if f['id'] == freelancer_id), None)
                
                reputation_mentions.append({
                    'job_id': outcome['job_id'],
                    'freelancer_id': freelancer_id,
                    'freelancer_tier': freelancer_info['tier'] if freelancer_info else 'Unknown',
                    'reasoning': outcome['reasoning'][:200] + "..." if len(outcome['reasoning']) > 200 else outcome['reasoning']
                })
    
    print(f"   Found {len(reputation_mentions)} hiring decisions mentioning reputation factors")
    
    # Show examples by tier
    tier_examples = {'New': [], 'Established': [], 'Expert': [], 'Elite': []}
    for mention in reputation_mentions:
        tier = mention['freelancer_tier']
        if tier in tier_examples and len(tier_examples[tier]) < 2:  # Limit to 2 examples per tier
            tier_examples[tier].append(mention)
    
    for tier in ['Elite', 'Expert', 'Established', 'New']:
        if tier_examples[tier]:
            print(f"\n   {tier} Freelancer Hiring Examples:")
            for i, example in enumerate(tier_examples[tier], 1):
                print(f"   {i}. Job: {example['job_id']}")
                print(f"      Reasoning: {example['reasoning']}")
    
    # Calculate hiring success rates by tier
    print(f"\n📈 HIRING SUCCESS RATES BY TIER:")
    tier_stats = {tier: {'hired': 0, 'total_bids': 0} for tier in tier_counts.keys()}
    
    # This would require analyzing all bids, which is complex
    # For now, show progression statistics
    promoted_freelancers = sum(tier_counts[tier] for tier in ['Established', 'Expert', 'Elite'])
    promotion_rate = (promoted_freelancers / total_freelancers) * 100
    
    print(f"   Freelancers who advanced beyond 'New': {promoted_freelancers}/{total_freelancers} ({promotion_rate:.1f}%)")
    
    # Elite achievements
    elite_freelancers = [f for f in freelancer_progression if f['tier'] == 'Elite']
    if elite_freelancers:
        print("\n🏆 ELITE FREELANCER ACHIEVEMENTS:")
        for freelancer in elite_freelancers[:3]:  # Show top 3
            print(f"   {freelancer['name']}: {freelancer['total_hired']} jobs, {freelancer['completion_rate']:.1%} success rate")
    
    # Create pie chart for paper
    create_reputation_pie_chart(tier_counts, output_dir)
    
    return {
        'tier_distribution': tier_counts,
        'reputation_mentions': len(reputation_mentions),
        'promotion_rate': promotion_rate,
        'elite_count': tier_counts['Elite']
    }


def create_reputation_pie_chart(tier_counts, output_dir):
    """Create a professional pie chart for reputation tier distribution for the paper"""
    
    # Create papers/figures directory for paper figures
    paper_output_dir = Path("papers/figures")
    paper_output_dir.mkdir(parents=True, exist_ok=True)
    
    if not tier_counts or sum(tier_counts.values()) == 0:
        print("No reputation data available for pie chart")
        return
    
    # Calculate percentages
    total_freelancers = sum(tier_counts.values())
    data = []
    for tier in ['New', 'Established', 'Expert', 'Elite']:  # Ordered by progression
        count = tier_counts.get(tier, 0)
        if count > 0:
            percentage = (count / total_freelancers) * 100
            data.append((tier, percentage))
    
    if not data:
        print("No valid tier data for pie chart")
        return
    
    # Set style for consistent, professional appearance
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels, values = zip(*data)
    
    # Use progression colors (light to dark)
    colors = ['#ffcccc', '#ff9999', '#ff6666', '#cc0000'][:len(labels)]
    
    # Create pie chart with percentage labels (use legend to avoid overlap)
    wedges, texts, autotexts = ax.pie(values, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    
    # Improve text formatting for percentages
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # Add legend instead of direct labels to avoid overlap
    ax.legend(wedges, labels, title="Reputation Tiers", 
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
    
    ax.set_title('Freelancer Reputation Tier Distribution\n(After 100 Rounds)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    plt.tight_layout()
    
    # Save to papers/figures for the paper
    paper_filename = paper_output_dir / "reputation_tier_distribution.png"
    plt.savefig(paper_filename, dpi=300, bbox_inches='tight')
    
    # Also save to analysis results
    analysis_filename = output_dir / "reputation_tier_distribution.png"
    plt.savefig(analysis_filename, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    print(f"📊 Reputation tier pie chart saved for paper: {paper_filename}")
    print(f"📊 Reputation tier pie chart saved for analysis: {analysis_filename}")


if __name__ == "__main__":
    analyze_llm_llm_by_category()
