"""
Base prompts for creating agent personas in the marketplace simulation.
"""

from typing import Dict, List


def generate_freelancer_persona_prompt(batch_count: int, category_info: List[Dict]) -> str:
    """Generate prompt for creating freelancer personas."""
    return f"""Create {batch_count} diverse freelancer personas for a marketplace simulation. 

IMPORTANT: To avoid bias, you MUST use anonymous identifiers:
- name: MUST be exactly "Freelancer A", "Freelancer B", "Freelancer C", etc. (alphabetical order)
- Do NOT use names that could indicate gender, ethnicity, or cultural background

Each freelancer should be unique with:
- Different skill combinations across these categories: {', '.join(cat['name'] for cat in category_info)}
- Varied experience levels (entry, mid, senior)  
- Diverse personalities and motivations
- Realistic hourly rates based on skills/experience (range: $15-100/hour)
- Varied project length preferences

Make them authentic but avoid stereotypes. Focus on professional qualifications and work styles.

Return as JSON array with EXACTLY these fields for each freelancer:
{{
  "name": "Freelancer A",
  "category": "Category Name", # one among the categories: {', '.join(cat['name'] for cat in category_info)}
  "skills": ["skill1", "skill2", "skill3"],
  "min_hourly_rate": 50,
  "personality": "Description of work style and personality",
  "motivation": "What drives them professionally",
  "background": "Professional experience and background",
  "preferred_project_length": "Duration preference (e.g., 'medium', 'long')"
}}

ALL FIELDS ARE REQUIRED. Do not omit any field."""


def generate_client_persona_prompt(clients_needed: int, business_categories: list = None) -> str:
    """Generate prompt for creating client personas."""
    
    if business_categories:
        # Multiple categories - assign them in order
        categories_list = ', '.join(business_categories)
        category_instruction = f"""
Create diverse businesses using these specific categories in order: {categories_list}
- Each company should specialize in their assigned category
- Varied company sizes (startup, small, medium, large enterprise)
- Different budget philosophies (cost-conscious, value-focused, premium-focused)
- Diverse hiring styles (quick decisions, thorough evaluation, relationship-focused)
- Different risk tolerance levels
- Unique company cultures and values

Each company background should reflect their specialization in their assigned category."""
            
        # Show example for first category, note that it varies
        category_field = f'"business_category": "{business_categories[0]}" // Use the assigned category for each client'
    else:
        category_instruction = """Each client should represent a different type of business with:
- Varied company sizes (startup, small, medium, large enterprise)
- Different budget philosophies (cost-conscious, value-focused, premium-focused)
- Diverse hiring styles (quick decisions, thorough evaluation, relationship-focused)
- Various business categories (use these job categories: SOFTWARE, DATA_SCIENCE, MARKETING, DESIGN, ACCOUNTING, ADMINISTRATIVE, LEGAL, ENGINEERING, TRANSLATION, WRITING, CUSTOMER_SUPPORT)
- Different risk tolerance levels
- Unique company cultures and values"""
        
        category_field = '"business_category": "SOFTWARE"'
    
    return f"""Create {clients_needed} diverse client company personas for a marketplace simulation.

IMPORTANT: To avoid bias, use neutral business identifiers:
- company_name: Use neutral formats like "Company A", "TechCorp B", "Solutions Inc C", etc.  
- Do NOT use names that could indicate specific regions or cultural associations

{category_instruction}

Focus on business characteristics and operational needs rather than cultural identifiers.

Return as JSON array with EXACTLY these fields for each client:
{{
  "company_name": "Company A",
  "company_size": "startup",
  "budget_philosophy": "cost-conscious", // or "value-focused" or "premium-focused"
  "hiring_style": "quick decisions",
  "background": "Industry background and specialization",
  {category_field}
}}

ALL FIELDS ARE REQUIRED. Do not omit any field."""