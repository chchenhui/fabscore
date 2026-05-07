"""
Freelancer-related prompts for the marketplace simulation.
Contains prompts for bidding decisions, job completion evaluation, and feedback generation.
"""

from typing import Dict, List


def generate_job_completion_prompt(job: Dict, freelancer: Dict, duration_rounds: int) -> str:
    """Generate prompt for evaluating job completion."""
    
    return f"""You are evaluating the completion of a freelance job.

Job: {job['title']}
Description: {job.get('description', 'No description available')}
Duration: {job['timeline']} ({duration_rounds} rounds)
Required Skills: {', '.join(job.get('skills_required', []))}

Freelancer: {freelancer['name']}
Skills: {', '.join(freelancer['skills'])}

Most freelancers with relevant skills successfully complete their jobs. Determine if the job was completed successfully based on how well the freelancer's skills align with the job requirements.

Return a JSON object with:
- success: true/false (true if freelancer has adequate skills for the job)"""


def create_bidding_decision_prompt(freelancer, job, success_display, experience_context, strategy_context, bids_per_round, reputation_info=None) -> str:
    """Generate prompt for freelancer bidding decisions."""
    
    # Build reputation context if available
    reputation_context = ""
    if reputation_info:
        # Build category expertise display
        expertise_display = "None yet" if not reputation_info.category_expertise else ", ".join(
            f"{cat}: {level:.1%}" for cat, level in sorted(reputation_info.category_expertise.items(), key=lambda x: x[1], reverse=True)[:3]
        )
        
        # Determine competitive position based on tier
        competitive_descriptions = {
            "New": "just starting to build your reputation",
            "Established": "becoming a recognized freelancer", 
            "Expert": "highly competitive with strong credibility",
            "Elite": "among the top freelancers with premium positioning"
        }
        competitiveness = competitive_descriptions.get(reputation_info.tier.value, "building your reputation")
        
        reputation_context = f"""
REPUTATION: {reputation_info.tier.value} ({reputation_info.job_completion_rate:.0%} success rate, ${reputation_info.total_earnings:.0f} earned)
Top skills: {expertise_display}
Market position: {competitiveness}"""

    return f"""You are {freelancer.name}, a freelancer. 

YOUR PROFILE:
You are {freelancer.personality} and {freelancer.background}. Your motivation is {freelancer.motivation}.
Your skills include {', '.join(freelancer.skills)}.
You charge a minimum of ${freelancer.min_hourly_rate}/hour and prefer {freelancer.preferred_project_length} length projects.

YOUR TRACK RECORD:
You've submitted {freelancer.total_bids} bids and been hired {freelancer.total_hired} times. {success_display}
{experience_context}

STRATEGIC INSIGHTS:
{strategy_context}{reputation_context}

BIDDING CONSTRAINTS:
You have {bids_per_round} bid(s) left this round.

JOB OPPORTUNITY:
Title: {job.title}
Company: {getattr(job, 'client', {}).get('name', 'Unknown Company')}
Category: {job.category}
Description: {job.description}
Skills Required: {', '.join(job.skills_required)}
Budget: {job.budget_type} - ${job.budget_amount}
Timeline: {job.timeline}

DECISION: Should you bid on this job? Consider your skills, the budget, and whether this could be a good opportunity.

Return a JSON object with these exact fields:
- decision: "yes" or "no" 
- reasoning: short explanation for your decision
- message: brief pitch to the client if bidding (required if decision is "yes")"""


def generate_feedback_prompt(job, hiring_decision, freelancer_bid) -> str:
    """Generate prompt for providing feedback to unsuccessful bidders.
    
    Args:
        job: Job dictionary
        hiring_decision: Dictionary with hiring decision details
        freelancer_bid: Dictionary of the specific bid receiving feedback
    """
    
    # Handle reasoning with proper fallback for None/empty values
    reasoning = hiring_decision.get('reasoning') 
    if not reasoning:
        reasoning = 'No reasoning provided'
    
    return f"""You are a client providing constructive feedback to an unsuccessful bidder.

JOB: {job['title']}
JOB BUDGET: ${job.get('budget_amount', 'unknown')}/hour
THEIR MESSAGE: "{freelancer_bid.get('message', 'No message') if freelancer_bid else 'No message'}"

HIRING DECISION: {reasoning}

Provide helpful feedback explaining why they weren't selected and suggestions for improvement.

Return a JSON object with these exact fields:
- main_reason: primary reason for not selecting them
- pitch_feedback: feedback on their message/pitch
- advice: specific advice for improving future bids (single paragraph)"""