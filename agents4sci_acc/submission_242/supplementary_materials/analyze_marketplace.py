#!/usr/bin/env python3
"""
Run marketplace simulation analysis

This script provides a command-line interface for running the marketplace analysis pipeline.
"""

import argparse
import logging
from src.analysis import MarketplaceAnalysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run analysis pipeline"""
    parser = argparse.ArgumentParser(description="Run marketplace simulation analysis")
    parser.add_argument(
        "--simulation-file",
        type=str,
        default="results/simuleval/true_gpt_simulation_20250814_111659.json",
        help="Path to simulation results JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip generating visualizations"
    )
    parser.add_argument(
        "--skip-reports",
        action="store_true",
        help="Skip generating reports"
    )
    parser.add_argument(
        "--generate-trend-plots",
        action="store_true",
        help="Generate historical market trend plots"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = MarketplaceAnalysis(args.simulation_file)
        
        # Run analysis
        market_metrics = analyzer.analyze_market_metrics()
        results = {
            'simulation_overview': market_metrics.get('simulation_overview', {}),
            'market_metrics': market_metrics,
            'bidding_behavior': analyzer.analyze_bidding_behavior(),
            'agent_learning': analyzer.analyze_agent_learning(),
            'reputation_analysis': analyzer.analyze_reputation_dynamics()
        }
        
        # Generate visualizations
        if not args.skip_visualizations:
            analyzer.generate_visualizations(f"{args.output_dir}/figures")
        
        # Generate trend plots if requested
        if args.generate_trend_plots:
            from src.analysis.visualization.market_trend_plots import plot_market_trends_from_file
            print("\nGenerating historical trend plots...")
            plot_market_trends_from_file(args.simulation_file, f"{args.output_dir}/figures")
            print("✅ Trend plots saved to results/figures/")
        
        # Generate reports (this saves analysis_results.json)
        if not args.skip_reports:
            reports = analyzer.generate_reports(args.output_dir)
            results['reports'] = reports
            
        # Generate paper figures after reports are saved
        if not args.skip_visualizations:
            from src.analysis.paper_figures import generate_paper_figures
            generate_paper_figures(
                simulation_file=args.simulation_file,
                analysis_file=f"{args.output_dir}/analysis_results.json",
                output_dir=f"{args.output_dir}/figures"
            )
            
            # Read the saved analysis results to get complete data
            import json
            try:
                with open(f"{args.output_dir}/analysis_results.json", "r") as f:
                    saved_results = json.load(f)
                    results.update(saved_results)
            except FileNotFoundError:
                pass
        
        # Print summary
        print("\n=== Analysis Results Summary ===")
        
        if 'market_metrics' in results:
            metrics = results['market_metrics']
            if 'unfilled_jobs' in metrics:
                stats = metrics['unfilled_jobs']['overall_stats']
                print("\nMarket Performance:")
                print(f"- Fill rate: {stats['fill_rate']*100:.1f}%")
                print(f"- Total jobs: {stats['total_jobs']}")
                print(f"- Unfilled jobs: {stats['unfilled_jobs']}")
            
            # Show saturation analysis
            if 'saturation_analysis' in metrics:
                saturation = metrics['saturation_analysis']
                print("\nMarket Saturation Analysis:")
                
                # Market health score with trends
                if 'market_health_score' in saturation:
                    health = saturation['market_health_score']
                    if 'health_grade' in health:
                        grade = health.get('health_grade', 'unknown')
                        score = health.get('overall_health_score', 0)
                        trend = health.get('health_trend', 'unknown')
                        print(f"- Market health: {grade} ({score:.2f})")
                        if trend != 'unknown':
                            print(f"  • Health trend: {trend}")
                
                # Enhanced bid volume analysis
                if 'bid_volume_analysis' in saturation:
                    bid_vol = saturation['bid_volume_analysis']
                    if 'average_bids_per_job' in bid_vol:
                        avg_bids = bid_vol['average_bids_per_job']
                        risk = bid_vol.get('saturation_risk_level', 'unknown')
                        competitiveness = bid_vol.get('current_competitiveness', 'unknown')
                        comp_trend = bid_vol.get('competition_trend', 'unknown')
                        
                        print(f"- Avg bids per job: {avg_bids:.1f}")
                        print(f"- Saturation risk: {risk}")
                        print(f"- Market competitiveness: {competitiveness}")
                        if comp_trend != 'unknown':
                            print(f"  • Competition trend: {comp_trend}")
                
                # Enhanced bid rejection analysis
                if 'bid_rejection_analysis' in saturation:
                    rejection = saturation['bid_rejection_analysis']
                    if 'average_rejection_rate' in rejection:
                        rate = rejection['average_rejection_rate']
                        trend = rejection.get('trend_direction', 'unknown')
                        print(f"- Avg bid rejection rate: {rate*100:.1f}%")
                        if trend != 'unknown':
                            print(f"  • Rejection rate trend: {trend}")
                
                # Enhanced outcome diversity with historical data
                if 'freelancer_outcome_diversity' in saturation:
                    diversity = saturation['freelancer_outcome_diversity']
                    if 'current_gini_coefficient' in diversity:
                        gini = diversity['current_gini_coefficient']
                        level = diversity.get('diversity_level', 'unknown')
                        freelancers_with_work = diversity.get('freelancers_with_work', 0)
                        total_freelancers = diversity.get('total_freelancers', 0)
                        gini_trend = diversity.get('gini_trend', 'unknown')
                        
                        print(f"- Outcome diversity: {level} ({gini:.2f})")
                        print(f"- Freelancers with work: {freelancers_with_work}/{total_freelancers}")
                        if gini_trend != 'unknown':
                            print(f"  • Diversity trend: {gini_trend}")
                
                # Enhanced fatigue indicators
                if 'freelancer_fatigue_indicators' in saturation:
                    fatigue = saturation['freelancer_fatigue_indicators']
                    if 'overall_fatigue_level' in fatigue:
                        level = fatigue['overall_fatigue_level']
                        trend = fatigue.get('fatigue_trend', 'unknown')
                        print(f"- Freelancer fatigue: {level}")
                        if trend != 'unknown':
                            print(f"  • Fatigue trend: {trend}")
                
                # Enhanced client activity patterns
                if 'client_activity_patterns' in saturation:
                    activity = saturation['client_activity_patterns']
                    if 'engagement_assessment' in activity:
                        assessment = activity['engagement_assessment']
                        rate = activity.get('average_activity_rate', 0)
                        trend = activity.get('engagement_trend', 'unknown')
                        print(f"- Client engagement: {assessment} ({rate*100:.1f}% activity rate)")
                        if trend != 'unknown':
                            print(f"  • Client engagement trend: {trend}")
        
        if 'bidding_behavior' in results:
            behavior = results['bidding_behavior']
            if 'decision_patterns' in behavior:
                patterns = behavior['decision_patterns']
                print("\nBidding Behavior:")
                print(f"- Total decisions: {patterns.get('total_decisions', 0)}")
                print(f"- Bid rate: {patterns.get('bid_rate', 0)*100:.1f}%")
        
        # Show reputation analysis
        if 'reputation_analysis' in results:
            reputation = results['reputation_analysis']
            print("\nReputation Dynamics:")
            
            # Historical reputation progression
            if 'reputation_progression' in reputation:
                progression = reputation['reputation_progression']
                
                # Freelancer progression
                if 'freelancers' in progression:
                    fl_prog = progression['freelancers']
                    print(f"- Freelancer Learning: {fl_prog.get('agents_tracked', 0)} agents tracked")
                    print(f"  • Tier Mobility: {fl_prog.get('tier_mobility_rate', 0)*100:.1f}% promoted ({fl_prog.get('agents_promoted', 0)} agents)")
                    print(f"  • Learning Trajectory: {fl_prog.get('learning_trajectory', 'unknown')}")
                    print(f"  • Avg Score Improvement: {fl_prog.get('average_score_improvement', 0):.3f}")
                
                # Client progression
                if 'clients' in progression:
                    cl_prog = progression['clients']
                    print(f"- Client Learning: {cl_prog.get('agents_tracked', 0)} agents tracked")
                    print(f"  • Tier Mobility: {cl_prog.get('tier_mobility_rate', 0)*100:.1f}% promoted ({cl_prog.get('agents_promoted', 0)} agents)")
                    print(f"  • Learning Trajectory: {cl_prog.get('learning_trajectory', 'unknown')}")
            
            # Current reputation distribution
            if 'current_reputation_distribution' in reputation:
                current = reputation['current_reputation_distribution']
                
                # Freelancer current state
                if 'freelancers' in current:
                    freelancer_rep = current['freelancers']
                    if 'tier_distribution' in freelancer_rep:
                        tier_dist = freelancer_rep['tier_distribution']
                        print(f"- Current Freelancer Tiers: New={tier_dist.get('New', 0)}, Established={tier_dist.get('Established', 0)}, Expert={tier_dist.get('Expert', 0)}, Elite={tier_dist.get('Elite', 0)}")
                    
                    if 'average_completion_rate' in freelancer_rep:
                        print(f"- Avg Completion Rate: {freelancer_rep['average_completion_rate']*100:.1f}%")
                
                # Client current state  
                if 'clients' in current:
                    client_rep = current['clients']
                    if 'tier_distribution' in client_rep:
                        tier_dist = client_rep['tier_distribution']
                        print(f"- Current Client Tiers: New={tier_dist.get('New', 0)}, Established={tier_dist.get('Established', 0)}, Expert={tier_dist.get('Expert', 0)}, Elite={tier_dist.get('Elite', 0)}")
                    
                    if 'average_hire_success_rate' in client_rep:
                        print(f"- Avg Client Hire Success: {client_rep['average_hire_success_rate']*100:.1f}%")
            
            # Reputation impact metrics
            if 'reputation_impact' in reputation:
                impact = reputation['reputation_impact']
                if 'tier_hiring_advantage' in impact:
                    advantage = impact['tier_hiring_advantage']
                    print(f"- Elite vs New Hire Rate: {advantage.get('elite_vs_new_ratio', 1.0):.1f}x advantage")
                
                if 'reputation_mobility' in impact:
                    mobility = impact['reputation_mobility']
                    print(f"- Market Mobility: {mobility.get('freelancers_promoted', 0)} freelancers, {mobility.get('clients_promoted', 0)} clients advanced")
            
            print(f"- Historical Data Points: {reputation.get('historical_data_points', 0)} rounds")
        
        print("\nAgent Statistics:")
        # Get actual number of unique freelancers from simulation overview
        total_freelancers = results.get('simulation_overview', {}).get('total_freelancers', 0)
        total_bids = results.get('simulation_overview', {}).get('total_bids', 0)
        if total_freelancers > 0:
            print(f"- Total freelancers: {total_freelancers}")
            print(f"- Total bids made: {total_bids}")
            print(f"- Avg bids per freelancer: {total_bids/total_freelancers:.1f}")
        
        # Show categories with bidding activity
        if 'market_metrics' in results:
            metrics = results['market_metrics']
            if 'skill_distribution' in metrics and 'category_coverage' in metrics['skill_distribution']:
                active_categories = sum(1 for cat in metrics['skill_distribution']['category_coverage'].values() 
                                   if cat.get('avg_bids_per_job', 0) > 0)
                total_categories = len(metrics['skill_distribution']['category_coverage'])
                print(f"- Active categories: {active_categories}/{total_categories}")
        
        if 'reports' in results:
            print("\nReports generated:")
            for report_type, path in results['reports'].items():
                print(f"- {report_type}: {path}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.exception(f"Error during analysis: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
