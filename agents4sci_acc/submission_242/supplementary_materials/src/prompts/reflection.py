"""
Reflection prompts for agents to analyze their performance and adjust strategies.
"""

from typing import Dict, List


def create_freelancer_reflection_prompt(
    freelancer_data: Dict,
    recent_activities: List[Dict],
    performance_summary: str
) -> str:
    """Generate reflection prompt for freelancers."""
    
    activities_text = ""
    if recent_activities:
        activities_text = "\nRECENT ACTIVITIES:\n"
        for activity in recent_activities:
            if activity['type'] == 'bid':
                reasoning = f" ({activity['reasoning']})" if activity.get('reasoning') else ""
                activities_text += f"- Bid on job {activity['job_id']}: {activity['outcome']}{reasoning}\n"
            elif activity['type'] == 'bid_opportunity':
                if activity['response'] == 'yes':
                    decision_text = "bid submitted"
                else:
                    decision_text = "passed (chose not to bid)"
                reasoning = f" - {activity['reasoning']}" if activity.get('reasoning') else ""
                activities_text += f"- Reviewed job {activity['job_id']}: {decision_text}{reasoning}\n"
    
    return f"""You are {freelancer_data['name']}, reflecting on your recent marketplace experience.

YOUR BACKGROUND:
- Skills: {', '.join(freelancer_data.get('skills', []))}
- Experience Level: {freelancer_data.get('experience_level', 'Not specified')}
- Rate: ${freelancer_data.get('min_hourly_rate', 0)}/hour
- Total Bids: {freelancer_data.get('total_bids', 0)}
- Total Hired: {freelancer_data.get('total_hired', 0)}

RECENT PERFORMANCE:
{performance_summary}{activities_text}

Reflect on your recent experiences and performance. What patterns do you notice? What should you adjust about your strategy?

MANDATORY RATE DECISION: You MUST evaluate your current hourly rate of ${freelancer_data.get('min_hourly_rate', 0)}/hour:

IF you're getting rejected because your rate is too high → DECREASE your rate
IF you're getting very few opportunities → DECREASE your rate to be more competitive  
IF you're getting hired easily and demand is high → INCREASE your rate
IF your current rate is working well → KEEP your current rate

Be decisive - choose a specific action and provide a concrete new rate.

Examples of rate_adjustment responses:
- {{"change": "decrease", "new_rate": 45, "reasoning": "Current rate of $65/hr is too high for most jobs I see"}}
- {{"change": "increase", "new_rate": 85, "reasoning": "Getting hired quickly, market can support higher rates"}}
- {{"change": "keep", "new_rate": 65, "reasoning": "Current rate is appropriate for my skill level"}}

Return a JSON object with these exact fields (ALL REQUIRED):
- key_insights: array of 1-2 main insights from recent experiences
- strategy_adjustments: array of 1-2 planned changes to your strategy
- rate_adjustment: {{"change": "increase/decrease/keep", "new_rate": number, "reasoning": "brief explanation"}}
- market_observations: array of 1-2 observations about market conditions"""


def create_client_reflection_prompt(
    client_data: Dict,
    recent_activities: List[Dict],
    performance_summary: str
) -> str:
    """Generate reflection prompt for clients."""
    
    activities_text = ""
    unfilled_jobs_text = ""
    
    if recent_activities:
        activities_text = "\nRECENT ACTIVITIES:\n"
        unfilled_jobs = []
        
        for activity in recent_activities:
            if activity['type'] == 'job_post':
                activities_text += f"- Posted job {activity['job_id']}: {activity['outcome']}\n"
            elif activity['type'] == 'hiring_decision':
                outcome_text = "hired someone" if activity['outcome'] == 'hire' else "decided not to hire"
                reasoning = f" - {activity['reasoning']}" if activity.get('reasoning') else ""
                activities_text += f"- Job {activity['job_id']}: {outcome_text}{reasoning}\n"
            elif activity['type'] == 'unfilled_job':
                unfilled_jobs.append(activity)
        
        # Add detailed unfilled jobs section
        if unfilled_jobs:
            unfilled_jobs_text = "\n\nUNFILLED JOBS (REQUIRING YOUR DECISION):\n"
            for job in unfilled_jobs:
                unfilled_jobs_text += f"- {job['title']} (Job {job['job_id']})\n"
                unfilled_jobs_text += f"  Current Budget: ${job['current_budget']:.0f}\n"
                unfilled_jobs_text += f"  Unfilled for: {job['rounds_unfilled']} rounds\n"
                unfilled_jobs_text += f"  Category: {job['category']}, Timeline: {job['timeline']}\n\n"
    
    return f"""You are {client_data.get('company_name', 'Unknown Company')} ({client_data.get('company_size', 'Unknown')} company, {client_data.get('budget_philosophy', 'balanced')} budget approach).

RECENT ACTIVITY:
{performance_summary}{activities_text}{unfilled_jobs_text}

Reflect and adjust your hiring strategy:

1. BUDGET STRATEGY: increase/decrease/keep your general budget for future jobs
2. UNFILLED JOBS: Optionally increase specific job budgets (format: [{{"job_id": "job_123", "increase_percentage": 20, "reasoning": "why"}}])

Return JSON with ALL fields:
{{"key_insights": ["insight1", "insight2"],
"strategy_adjustments": ["adjustment1", "adjustment2"], 
"budget_adjustment": {{"change": "increase/decrease/keep", "reasoning": "why"}},
"job_budget_adjustments": [{{"job_id": "job_123", "increase_percentage": 20, "reasoning": "why"}}],
"market_observations": ["observation1", "observation2"]}}"""