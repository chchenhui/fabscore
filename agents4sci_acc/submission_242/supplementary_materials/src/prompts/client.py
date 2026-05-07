"""
Client-related prompts for the marketplace simulation.
Contains prompts for job posting creation and hiring decisions.
"""




def create_gpt_job_posting_prompt(client, selected_category) -> str:
    """Generate prompt for GPT-powered job posting creation."""
    return f"""You are the hiring manager at {client.company_name}, a {client.company_size} company specializing in {selected_category.title}.

YOUR COMPANY BACKGROUND:
{client.background}

You need to create a job posting for a {selected_category.title} project. Your company has a {client.budget_philosophy} budget approach and a {client.hiring_style} hiring style.

Create a realistic job posting that includes:
- Clear project requirements
- Specific skills needed
- Appropriate budget (market rate for {selected_category.title} is around ${selected_category.avg_rate}/hour
- Timeline for completion (short/medium/long term)
- Any special requirements (certifications, portfolio, tools expertise)

Return your job posting as JSON with these fields:
- title
- description
- skills_required (array of strings)
- budget_type ("fixed" or "hourly")
- budget_amount (in USD)
- timeline
- special_requirements (array of strings)
- category (set this to: "{selected_category.title}")"""


def create_hiring_decision_prompt(client, job, bid_summaries, reflection_prompt) -> str:
    """Generate prompt for client hiring decisions."""
    prompt = f"""You are the hiring manager at {client.company_name}, a {client.company_size} company.

ABOUT YOUR COMPANY:
{client.background}
You have a {client.budget_philosophy} approach to budgets and a {client.hiring_style} hiring style.

YOUR JOB POSTING:
Title: {job.title}
Company: {client.company_name}
Category: {job.category}
Description: {job.description}
Required Skills: {', '.join(job.skills_required)}
Budget: {job.budget_type} - ${job.budget_amount}
Timeline: {job.timeline}

You've received {len(bid_summaries)} bids to review:
"""

    for i, summary in enumerate(bid_summaries, 1):
        prompt += f"""
Bid #{i}:
- Freelancer: {summary['freelancer_name']}
- Skills: {', '.join(summary['skills'])}
- Track record: {summary['success_rate']}
- Message: "{summary['message']}"
"""

    prompt += f"""{reflection_prompt}

DECISION: Review the bids and decide which freelancer would be the best fit for your company and project.

Consider:
1. Do their skills match your requirements?
2. Does their message show they understand the project?
3. Do they have a good track record for success?

Return a JSON object with these exact fields:
- selected_bid_number: number of chosen bid (1, 2, 3, etc.) or 0 for "none"
- selected_freelancer: name of chosen freelancer or "none"
- reasoning: short explanation of your choice
- confidence_level: "high", "medium", or "low"
- learnings: key insight from this decision (optional)
- future_implications: brief note on strategic impact (optional)

IMPORTANT: Use selected_bid_number (1, 2, 3, etc.) to indicate your choice, not the freelancer name.
    """
    return prompt