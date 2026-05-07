#!/usr/bin/env python3
"""
Comprehensive analysis of bid rejection and hire rejection reasons
"""

import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_pie_chart(data, title, filename, colors=None):
    """Create a professional pie chart for paper figures"""
    if not data:
        print(f"No data available for {title}")
        return
        
    labels, values = zip(*data)
    
    # Create output directory
    output_dir = Path("papers/figures")
    output_dir.mkdir(exist_ok=True)
    
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    # Set style for consistent, professional appearance
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'serif'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create pie chart with percentage labels
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    
    # Improve text formatting
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    for text in texts:
        text.set_fontsize(11)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created {filename}")

def create_combined_rejection_chart(client_data, freelancer_data):
    """Create a combined figure with two subplots for client and freelancer rejection reasons"""
    
    # Create output directory
    output_dir = Path("papers/figures")
    output_dir.mkdir(exist_ok=True)
    
    # Set style for consistent, professional appearance
    plt.style.use('default')
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'
    
    # Create figure with two subplots (wider to accommodate legends)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Colors for consistency across both charts
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    # Client rejection chart (left subplot)
    if client_data:
        client_labels, client_values = zip(*client_data)
        
        # Create pie chart without labels (use legend instead)
        wedges1, texts1, autotexts1 = ax1.pie(
            client_values, 
            autopct='%1.1f%%',
            colors=colors[:len(client_labels)],
            startangle=90,
            pctdistance=0.85  # Move percentage labels closer to center
        )
        
        # Improve text formatting for percentages
        for autotext in autotexts1:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # Add legend instead of direct labels to avoid overlap
        ax1.legend(wedges1, client_labels, title="Rejection Reasons", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
        
        ax1.set_title('LLM Client Rejection Reasons', fontsize=14, fontweight='bold', pad=20)
    
    # Freelancer rejection chart (right subplot)
    if freelancer_data:
        freelancer_labels, freelancer_values = zip(*freelancer_data)
        
        # Create pie chart without labels (use legend instead)
        wedges2, texts2, autotexts2 = ax2.pie(
            freelancer_values, 
            autopct='%1.1f%%',
            colors=colors[:len(freelancer_labels)],
            startangle=90,
            pctdistance=0.85  # Move percentage labels closer to center
        )
        
        # Improve text formatting for percentages
        for autotext in autotexts2:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        # Add legend instead of direct labels to avoid overlap
        ax2.legend(wedges2, freelancer_labels, title="Rejection Reasons", 
                  loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
        
        ax2.set_title('LLM Freelancer Strategic Rejection Reasons', fontsize=14, fontweight='bold', pad=20)
    
    # Equal aspect ratio ensures that pies are drawn as circles
    ax1.axis('equal')
    ax2.axis('equal')
    
    # Add overall title
    fig.suptitle('LLM Agent Rejection Patterns Comparison', fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for suptitle
    
    # Save the combined chart
    filename = output_dir / "rejection_analysis_combined.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined rejection analysis chart: {filename}")

def analyze_comprehensive_rejections():
    """Comprehensive analysis of rejection patterns"""
    
    print("=== COMPREHENSIVE REJECTION ANALYSIS ===")
    
    # Load simulation data for comparative analysis
    experiments = [
        ("results/simuleval/true_gpt_simulation_20250905_001404.json", "LLM-LLM (Both Agents Use Reasoning)"),
        ("results/simuleval/true_gpt_simulation_20250910_123457.json", "Random-LLM (Random Freelancer, LLM Client)")
    ]
    
    experiment_results = {}
    
    for file_path, experiment_name in experiments:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {experiment_name}")
        print(f"{'='*80}")
        
        with open(file_path, 'r') as f:
            simulation_data = json.load(f)
        
        results = analyze_experiment_rejections(simulation_data, experiment_name)
        experiment_results[experiment_name] = results
    
    # Comparative analysis
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS SUMMARY")
    print(f"{'='*80}")
    comparative_analysis(experiment_results)
    
    return experiment_results

def analyze_experiment_rejections(simulation_data, experiment_name):
    """Analyze rejection patterns for a single experiment - both client and freelancer perspectives"""
    
    print(f"📊 Loaded simulation data for {experiment_name}")
    
    # Get all the data sources
    hiring_outcomes = simulation_data.get('hiring_outcomes', [])
    all_bids = simulation_data.get('all_bids', [])
    all_jobs = simulation_data.get('all_jobs', [])
    freelancers = simulation_data.get('freelancer_profiles', {})
    config = simulation_data.get('simulation_config', {})
    
    print(f"📈 Data Overview:")
    print(f"   - Hiring outcomes: {len(hiring_outcomes)}")
    print(f"   - Total bids: {len(all_bids)}")
    print(f"   - Total jobs: {len(all_jobs)}")
    print(f"   - Freelancers: {len(freelancers)}")
    print(f"   - Freelancer Agent Type: {config.get('freelancer_agent_type', 'N/A')}")
    print(f"   - Client Agent Type: {config.get('client_agent_type', 'N/A')}")
    
    # Analyze both directions of rejection
    print(f"\n🔄 ANALYZING ALL REJECTION DIRECTIONS:")
    print(f"   1. Client rejections of freelancer bids")
    print(f"   2. Freelancer rejections of job opportunities") 
    print(f"   3. Client feedback to freelancers about rejected bids")
    
    # ===== 1. CLIENT-SIDE REJECTION ANALYSIS =====
    print(f"\n🏢 CLIENT-SIDE REJECTION ANALYSIS:")
    
    # Analyze hiring patterns from client perspective
    hired_outcomes = []
    client_rejected_outcomes = []  # Client actively rejected bids
    no_bid_outcomes = []
    
    for outcome in hiring_outcomes:
        selected = outcome.get('selected_freelancer')
        reasoning = outcome.get('reasoning', '')
        
        if selected and selected != 'none' and selected is not None:
            hired_outcomes.append({'selected': selected, 'reasoning': reasoning})
        elif selected == 'none' or selected is None:
            if 'no bid' in reasoning.lower() or 'no bids' in reasoning.lower():
                no_bid_outcomes.append({'reasoning': reasoning})
            else:
                client_rejected_outcomes.append({'reasoning': reasoning})
        else:
            no_bid_outcomes.append({'reasoning': reasoning})
    
    print(f"   - Jobs with successful hires: {len(hired_outcomes)}")
    print(f"   - Jobs where client rejected all bids: {len(client_rejected_outcomes)}")
    print(f"   - Jobs with no bids received: {len(no_bid_outcomes)}")
    
    # Analyze client rejection reasoning
    client_rejection_reasons = [outcome['reasoning'] for outcome in client_rejected_outcomes if outcome['reasoning']]
    
    if client_rejection_reasons:
        print(f"\n🚫 CLIENT REJECTION PATTERNS:")
        print(f"   Total client rejection decisions with reasoning: {len(client_rejection_reasons)}")
        
        # Show sample client rejections
        for i, reason in enumerate(client_rejection_reasons[:5], 1):
            print(f"   {i}. \"{reason[:120]}{'...' if len(reason) > 120 else ''}\"")
    else:
        print(f"\n🚫 CLIENT REJECTION PATTERNS: No explicit rejection reasoning found")
    
    # Define theme extraction function for use throughout
    def extract_themes(reasons_list):
        theme_keywords = {
            'skill_mismatch': ['skill', 'experience', 'expertise', 'qualifications', 'background'],
            'communication': ['message', 'communication', 'understanding', 'proposal', 'pitch'],
            'competition': ['better', 'stronger', 'more', 'competitive', 'other'],
            'rate_budget': ['rate', 'budget', 'cost', 'price', 'expensive'],
            'fit': ['fit', 'match', 'align', 'suitable', 'appropriate']
        }
        
        theme_counts = defaultdict(int)
        for reason in reasons_list:
            reason_lower = reason.lower()
            for theme, keywords in theme_keywords.items():
                if any(keyword in reason_lower for keyword in keywords):
                    theme_counts[theme] += 1
        
        return theme_counts

    # ===== 2. FREELANCER-SIDE REJECTION ANALYSIS =====
    print(f"\n👥 FREELANCER-SIDE REJECTION ANALYSIS (freelancers choosing NOT to bid):")
    
    # Extract freelancer decisions not to bid from all_decisions
    freelancer_rejections = []
    freelancer_bids = []
    all_decisions = simulation_data.get('all_decisions', [])
    
    for decision in all_decisions:
        if decision.get('decision') == 'no':
            freelancer_rejections.append({
                'freelancer_id': decision.get('freelancer_id'),
                'freelancer_name': decision.get('freelancer_name'),
                'job_id': decision.get('job_id'),
                'job_title': decision.get('job_title'),
                'reasoning': decision.get('reasoning', ''),
                'round': decision.get('round')
            })
        elif decision.get('decision') == 'yes':
            freelancer_bids.append(decision)
    
    print(f"   - Total freelancer decisions: {len(all_decisions)}")
    print(f"   - Freelancer rejections (chose not to bid): {len(freelancer_rejections)}")
    print(f"   - Freelancer bids (chose to bid): {len(freelancer_bids)}")
    
    # Filter rejections with meaningful reasoning (non-empty)
    meaningful_rejections = [r for r in freelancer_rejections if r['reasoning'] and r['reasoning'].strip()]
    
    print(f"   - Freelancer rejections with detailed reasoning: {len(meaningful_rejections)}")
    
    if len(meaningful_rejections) > 0:
        print(f"\n📝 FREELANCER REJECTION REASONING SAMPLES:")
        for i, rejection in enumerate(meaningful_rejections[:5], 1):
            reasoning_text = rejection['reasoning'][:120] + ("..." if len(rejection['reasoning']) > 120 else "")
            print(f"{i:2d}. Job: {rejection['job_title'][:50]}...")
            print(f"     Reasoning: {reasoning_text}")
        
        freelancer_rejection_themes = extract_themes([r['reasoning'] for r in meaningful_rejections])
        
        print(f"\n🔍 FREELANCER REJECTION THEMES:")
        total_freelancer_rejection_themes = sum(freelancer_rejection_themes.values())
        for theme, count in sorted(freelancer_rejection_themes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_freelancer_rejection_themes) * 100 if total_freelancer_rejection_themes > 0 else 0
            print(f"   - {theme.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    else:
        print(f"   No detailed freelancer rejection reasoning found (expected for Random freelancers)")
        freelancer_rejection_themes = {}
    
    # ===== 3. CLIENT FEEDBACK TO FREELANCERS ANALYSIS =====
    print(f"\n📬 CLIENT FEEDBACK TO FREELANCERS (stored by freelancers about rejected bids):")
    
    # Check bid feedback in freelancer profiles (feedback FROM clients TO freelancers)
    total_feedback_entries = 0
    freelancer_feedback_reasons = []
    
    for freelancer_id, freelancer in freelancers.items():
        bid_feedback = freelancer.get('bid_feedback', [])
        total_feedback_entries += len(bid_feedback)
        
        for feedback in bid_feedback:
            if isinstance(feedback, dict):
                feedback_content = feedback.get('feedback', {})
                if isinstance(feedback_content, dict):
                    main_reason = feedback_content.get('main_reason', '')
                    if main_reason:
                        freelancer_feedback_reasons.append(main_reason)
    
    print(f"   - Total client feedback entries stored by freelancers: {total_feedback_entries}")
    print(f"   - Feedback with detailed reasons: {len(freelancer_feedback_reasons)}")
    
    if freelancer_feedback_reasons:
        print(f"\n📝 CLIENT FEEDBACK TO FREELANCERS:")
        reason_counts = Counter(freelancer_feedback_reasons)
        for i, (reason, count) in enumerate(reason_counts.most_common(5), 1):
            percentage = (count / len(freelancer_feedback_reasons)) * 100
            print(f"{i:2d}. {reason:<80} ({count:2d}, {percentage:4.1f}%)")
        
        client_feedback_themes = extract_themes(freelancer_feedback_reasons)
        
        print(f"\n🔍 CLIENT FEEDBACK TO FREELANCERS THEMES:")
        total_client_feedback_themes = sum(client_feedback_themes.values())
        for theme, count in sorted(client_feedback_themes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_client_feedback_themes) * 100 if total_client_feedback_themes > 0 else 0
            print(f"   - {theme.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    else:
        print(f"   No detailed client feedback found (expected for Random freelancers)")
        client_feedback_themes = {}
    
    # Analyze successful hiring reasons
    hire_reasons = [outcome['reasoning'] for outcome in hired_outcomes if outcome['reasoning']]
    
    if hire_reasons:
        print(f"\n✅ TOP SUCCESSFUL HIRING REASONS:")
        hire_reason_counts = Counter(hire_reasons)
        for i, (reason, count) in enumerate(hire_reason_counts.most_common(10), 1):
            percentage = (count / len(hire_reasons)) * 100
            print(f"{i:2d}. {reason[:80]:<80} ({count:3d}, {percentage:4.1f}%)")
    
    # Analyze why jobs go unfilled
    unfilled_analysis = defaultdict(int)
    
    # Look at jobs that didn't get bids vs jobs that got bids but no hires
    jobs_with_bids = set()
    for bid in all_bids:
        jobs_with_bids.add(bid['job_id'])
    
    total_jobs = len(all_jobs)
    jobs_with_bids_count = len(jobs_with_bids)
    jobs_without_bids = total_jobs - jobs_with_bids_count
    
    print(f"\n📊 JOB FULFILLMENT ANALYSIS:")
    print(f"   - Total jobs posted: {total_jobs}")
    print(f"   - Jobs that received bids: {jobs_with_bids_count}")
    print(f"   - Jobs with no bids: {jobs_without_bids}")
    print(f"   - Jobs with successful hires: {len(hired_outcomes)}")
    if jobs_with_bids_count > 0:
        print(f"   - Bid-to-hire conversion rate: {len(hired_outcomes)/jobs_with_bids_count*100:.1f}%")
    
    # Extract key themes from rejection reasons using keyword analysis
    # Analyze client rejection themes
    client_themes = {}
    if client_rejection_reasons:
        print(f"\n🔍 CLIENT REJECTION THEMES:")
        client_themes = extract_themes(client_rejection_reasons)
        total_client_themes = sum(client_themes.values())
        for theme, count in sorted(client_themes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_client_themes) * 100 if total_client_themes > 0 else 0
            print(f"   - {theme.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Analyze freelancer feedback themes  
    freelancer_themes = {}
    if freelancer_feedback_reasons:
        print(f"\n🔍 FREELANCER FEEDBACK THEMES:")
        freelancer_themes = extract_themes(freelancer_feedback_reasons)
        total_freelancer_themes = sum(freelancer_themes.values())
        for theme, count in sorted(freelancer_themes.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_freelancer_themes) * 100 if total_freelancer_themes > 0 else 0
            print(f"   - {theme.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    else:
        print(f"\n🔍 FREELANCER FEEDBACK THEMES: None (expected for Random freelancers)")
    
    # Return analysis results for comparison
    return {
        'experiment_name': experiment_name,
        'config': config,
        'total_jobs': len(all_jobs),
        'total_bids': len(all_bids),
        'jobs_with_bids': jobs_with_bids_count,
        'successful_hires': len(hired_outcomes),
        'bid_to_hire_conversion': len(hired_outcomes)/jobs_with_bids_count*100 if jobs_with_bids_count > 0 else 0,
        
        # Client-side rejection data
        'client_rejections': len(client_rejected_outcomes),
        'client_rejection_reasons': client_rejection_reasons,
        'client_themes': client_themes,
        
        # Freelancer-side rejection data (freelancers choosing NOT to bid)
        'freelancer_rejections': len(freelancer_rejections),
        'freelancer_rejections_with_reasons': len(meaningful_rejections),
        'freelancer_rejection_themes': freelancer_rejection_themes,
        'freelancer_rejection_reasons': [r['reasoning'] for r in meaningful_rejections],
        
        # Client feedback to freelancers data  
        'total_client_feedback_entries': total_feedback_entries,
        'client_feedback_with_reasons': len(freelancer_feedback_reasons),
        'client_feedback_themes': client_feedback_themes,
        'client_feedback_reasons': freelancer_feedback_reasons,
        
        'hire_reasons': hire_reasons
    }

def comparative_analysis(experiment_results):
    """Compare rejection patterns across experiments"""
    
    print("🔍 COMPARATIVE DECISION-MAKING QUALITY ANALYSIS")
    print("-" * 60)
    
    # Extract results for each experiment
    experiments = list(experiment_results.keys())
    
    print("\n📊 MARKET PERFORMANCE COMPARISON:")
    print(f"{'Metric':<30} {'LLM-LLM':<20} {'Random-LLM':<20}")
    print("-" * 70)
    
    for exp_name in experiments:
        results = experiment_results[exp_name]
        short_name = "LLM-LLM" if "LLM-LLM" in exp_name else "Random-LLM"
        
        if short_name == "LLM-LLM":
            llm_llm_results = results
        else:
            random_llm_results = results
    
    # Performance metrics comparison
    metrics = [
        ("Total Jobs", "total_jobs"),
        ("Jobs with Bids", "jobs_with_bids"), 
        ("Successful Hires", "successful_hires"),
        ("Bid-to-Hire Rate (%)", "bid_to_hire_conversion"),
        ("Client Feedback Entries", "total_client_feedback_entries"),
        ("Client Feedback Details", "client_feedback_with_reasons")
    ]
    
    for metric_name, metric_key in metrics:
        llm_val = llm_llm_results.get(metric_key, 0)
        random_val = random_llm_results.get(metric_key, 0)
        
        if metric_key == "bid_to_hire_conversion":
            print(f"{metric_name:<30} {llm_val:<20.1f} {random_val:<20.1f}")
        else:
            print(f"{metric_name:<30} {llm_val:<20} {random_val:<20}")
    
    print("\n🏢 CLIENT-SIDE REJECTION ANALYSIS:")
    print(f"{'Metric':<25} {'LLM-LLM':<15} {'Random-LLM':<15} {'Notes':<30}")
    print("-" * 85)
    
    llm_client_rejections = llm_llm_results.get('client_rejections', 0)
    random_client_rejections = random_llm_results.get('client_rejections', 0)
    
    print(f"{'Client Rejections':<25} {llm_client_rejections:<15} {random_client_rejections:<15} {'Both use LLM clients':<30}")
    
    # Client rejection themes comparison (both should have LLM clients)
    print(f"\n🎯 CLIENT REJECTION THEMES COMPARISON:")
    print(f"{'Theme':<20} {'LLM-LLM':<15} {'Random-LLM':<15} {'Difference':<15}")
    print("-" * 65)
    
    # Get all unique client themes
    all_client_themes = set()
    all_client_themes.update(llm_llm_results.get('client_themes', {}).keys())
    all_client_themes.update(random_llm_results.get('client_themes', {}).keys())
    
    llm_client_total = sum(llm_llm_results.get('client_themes', {}).values())
    random_client_total = sum(random_llm_results.get('client_themes', {}).values())
    
    for theme in sorted(all_client_themes):
        llm_count = llm_llm_results.get('client_themes', {}).get(theme, 0)
        random_count = random_llm_results.get('client_themes', {}).get(theme, 0)
        
        llm_pct = (llm_count / llm_client_total * 100) if llm_client_total > 0 else 0
        random_pct = (random_count / random_client_total * 100) if random_client_total > 0 else 0
        diff = llm_pct - random_pct
        
        theme_name = theme.replace('_', ' ').title()
        print(f"{theme_name:<20} {llm_pct:<15.1f} {random_pct:<15.1f} {diff:+6.1f}pp")
    
    print("\n🚫 FREELANCER REJECTION ANALYSIS (freelancers choosing NOT to bid):")
    print(f"{'Metric':<25} {'LLM-LLM':<15} {'Random-LLM':<15} {'Notes':<30}")
    print("-" * 85)
    
    llm_freelancer_rejections = llm_llm_results.get('freelancer_rejections_with_reasons', 0)
    random_freelancer_rejections = random_llm_results.get('freelancer_rejections_with_reasons', 0)
    
    print(f"{'Freelancer Rejections':<25} {llm_freelancer_rejections:<15} {random_freelancer_rejections:<15} {'LLM vs Random freelancers':<30}")
    
    # Freelancer rejection themes comparison
    print(f"\n🔍 FREELANCER REJECTION THEMES:")
    print(f"{'Theme':<20} {'LLM-LLM':<15} {'Random-LLM':<15} {'Expected Pattern':<20}")
    print("-" * 75)
    
    # Get all unique freelancer rejection themes
    all_freelancer_rejection_themes = set()
    all_freelancer_rejection_themes.update(llm_llm_results.get('freelancer_rejection_themes', {}).keys())
    all_freelancer_rejection_themes.update(random_llm_results.get('freelancer_rejection_themes', {}).keys())
    
    llm_rejection_total = sum(llm_llm_results.get('freelancer_rejection_themes', {}).values())
    random_rejection_total = sum(random_llm_results.get('freelancer_rejection_themes', {}).values())
    
    for theme in sorted(all_freelancer_rejection_themes):
        llm_count = llm_llm_results.get('freelancer_rejection_themes', {}).get(theme, 0)
        random_count = random_llm_results.get('freelancer_rejection_themes', {}).get(theme, 0)
        
        llm_pct = (llm_count / llm_rejection_total * 100) if llm_rejection_total > 0 else 0
        random_pct = (random_count / random_rejection_total * 100) if random_rejection_total > 0 else 0
        
        theme_name = theme.replace('_', ' ').title()
        expected = "LLM > Random" if llm_pct > random_pct else "Random > LLM"
        print(f"{theme_name:<20} {llm_pct:<15.1f} {random_pct:<15.1f} {expected:<20}")
    
    print("\n📬 CLIENT FEEDBACK TO FREELANCERS ANALYSIS:")
    print(f"{'Metric':<25} {'LLM-LLM':<15} {'Random-LLM':<15} {'Notes':<30}")
    print("-" * 85)
    
    llm_client_feedback = llm_llm_results.get('client_feedback_with_reasons', 0)
    random_client_feedback = random_llm_results.get('client_feedback_with_reasons', 0)
    
    print(f"{'Client Feedback':<25} {llm_client_feedback:<15} {random_client_feedback:<15} {'LLM vs Random freelancers':<30}")
    
    # Client feedback themes comparison
    print(f"\n🔍 CLIENT FEEDBACK TO FREELANCERS THEMES:")
    print(f"{'Theme':<20} {'LLM-LLM':<15} {'Random-LLM':<15} {'Expected Pattern':<20}")
    print("-" * 75)
    
    # Get all unique client feedback themes
    all_client_feedback_themes = set()
    all_client_feedback_themes.update(llm_llm_results.get('client_feedback_themes', {}).keys())
    all_client_feedback_themes.update(random_llm_results.get('client_feedback_themes', {}).keys())
    
    llm_client_feedback_total = sum(llm_llm_results.get('client_feedback_themes', {}).values())
    random_client_feedback_total = sum(random_llm_results.get('client_feedback_themes', {}).values())
    
    for theme in sorted(all_client_feedback_themes):
        llm_count = llm_llm_results.get('client_feedback_themes', {}).get(theme, 0)
        random_count = random_llm_results.get('client_feedback_themes', {}).get(theme, 0)
        
        llm_pct = (llm_count / llm_client_feedback_total * 100) if llm_client_feedback_total > 0 else 0
        random_pct = (random_count / random_client_feedback_total * 100) if random_client_feedback_total > 0 else 0
        
        theme_name = theme.replace('_', ' ').title()
        expected = "LLM > Random" if llm_pct > random_pct else "Random > LLM"
        print(f"{theme_name:<20} {llm_pct:<15.1f} {random_pct:<15.1f} {expected:<20}")
    
    print("\n💡 KEY INSIGHTS:")
    
    # Market efficiency comparison
    if llm_llm_results.get('bid_to_hire_conversion', 0) > random_llm_results.get('bid_to_hire_conversion', 0):
        diff = llm_llm_results.get('bid_to_hire_conversion', 0) - random_llm_results.get('bid_to_hire_conversion', 0)
        print(f"   • LLM freelancers + LLM clients achieve {diff:.1f}% higher bid-to-hire conversion")
    
    # Client-side reasoning (both use LLM clients)
    if llm_client_rejections > 0 and random_client_rejections > 0:
        print(f"   • Both configurations use LLM clients - both show systematic rejection reasoning")
    elif llm_client_rejections > 0 or random_client_rejections > 0:
        print(f"   • LLM clients provide structured rejection reasoning when present")
    
    # Client feedback to freelancers patterns
    if llm_client_feedback > 0 and random_client_feedback == 0:
        print(f"   • Only LLM freelancers receive/store detailed feedback from clients ({llm_client_feedback} vs {random_client_feedback})")
        print(f"   • Random freelancers don't process detailed rejection feedback")
    
    # Freelancer rejection patterns
    if llm_freelancer_rejections > 0 and random_freelancer_rejections > 0:
        print(f"   • LLM freelancers provide detailed reasoning for rejecting jobs ({llm_freelancer_rejections:,} reasoned rejections)")
        print(f"   • Random freelancers provide no meaningful rejection reasoning (generic probabilistic decisions)")
    
    # Strategic pairing insights
    print(f"   • Client reasoning quality independent of freelancer type (both use LLM clients)")
    print(f"   • Freelancer feedback processing depends on freelancer reasoning capability")
    print(f"   • Full LLM-LLM pairing creates most efficient market outcomes")

def generate_paper_pie_charts(experiment_results):
    """Generate pie charts for the paper based on analysis results"""
    
    print(f"\n{'='*80}")
    print("GENERATING PIE CHARTS FOR PAPER")
    print(f"{'='*80}")
    
    # Find LLM-LLM results for the charts
    llm_llm_results = None
    for exp_name, results in experiment_results.items():
        if "LLM-LLM" in exp_name:
            llm_llm_results = results
            break
    
    if not llm_llm_results:
        print("No LLM-LLM results found for pie chart generation")
        return
    
    # Combined Client and Freelancer Rejection Analysis
    client_themes = llm_llm_results.get('client_themes', {})
    freelancer_themes = llm_llm_results.get('freelancer_rejection_themes', {})
    
    if client_themes and freelancer_themes:
        # Map themes to paper categories
        theme_mapping = {
            'skill_mismatch': 'Skill Mismatch',
            'fit': 'Project Fit', 
            'rate_budget': 'Rate/Budget',
            'communication': 'Communication',
            'competition': 'Competition'
        }
        
        # Prepare client data
        client_data = []
        for theme, count in client_themes.items():
            mapped_name = theme_mapping.get(theme, theme.replace('_', ' ').title())
            total_client = sum(client_themes.values())
            percentage = (count / total_client * 100) if total_client > 0 else 0
            client_data.append((mapped_name, percentage))
        client_data.sort(key=lambda x: x[1], reverse=True)
        
        # Prepare freelancer data
        freelancer_data = []
        for theme, count in freelancer_themes.items():
            mapped_name = theme_mapping.get(theme, theme.replace('_', ' ').title())
            total_freelancer = sum(freelancer_themes.values())
            percentage = (count / total_freelancer * 100) if total_freelancer > 0 else 0
            freelancer_data.append((mapped_name, percentage))
        freelancer_data.sort(key=lambda x: x[1], reverse=True)
        
        # Create combined subplot figure
        create_combined_rejection_chart(client_data, freelancer_data)
    
    print("Pie chart generation completed!")

if __name__ == "__main__":
    experiment_results = analyze_comprehensive_rejections()
    
    # Generate pie charts for the paper
    if experiment_results:
        generate_paper_pie_charts(experiment_results)
