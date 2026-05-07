#!/usr/bin/env python3
"""
TRUE GPT-Powered Marketplace Simulation

This addresses the core goal by using GPT for EVERYTHING:
1. GPT creates freelancer personas
2. GPT creates job postings  
3. GPT makes all bidding decisions
4. GPT makes all hiring decisions
5. GPT reflects on and adapts strategies

No templates - pure emergent behavior!
"""

import json
import random
import argparse
import time
from datetime import datetime
from typing import Dict, List
import logging
import sys
from pathlib import Path
import concurrent.futures

# Use the same configuration system as simuleval_core
from config.llm_config import LLMConfigManager, create_openai_client
from caching.token_tracker import get_token_tracker
from config.agent_cache import AgentCache
from .job_categories import JobCategory, category_manager
from .agent_reflection import reflection_manager
from .simple_reputation import simple_reputation_manager
from .entities import Freelancer, Client, Job, Bid, HiringDecision
from .ranking_algorithm import JobRankingCalculator
from .reflection_adjustments import reflection_adjustment_engine
from .market_metrics_utils import (
    calculate_health_grade,
    calculate_competitiveness_level,
    calculate_work_distribution_gini,
    calculate_market_health_score
)
from prompts import (
    generate_freelancer_persona_prompt,
    generate_client_persona_prompt,
    create_gpt_job_posting_prompt,
    create_bidding_decision_prompt,
    create_hiring_decision_prompt,
    generate_feedback_prompt,
    create_freelancer_reflection_prompt,
    create_client_reflection_prompt,
    validate_bidding_decision,
    validate_hiring_decision,
    validate_feedback,
    validate_reflection,
    validate_freelancer_personas,
    validate_client_personas,
    validate_job_posting,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set up OpenAI client using the proper config with caching for cost savings
config_manager = LLMConfigManager()
llm_config = config_manager.get_config()
openai_client = create_openai_client(llm_config)

class TrueGPTMarketplace:
    """Marketplace where GPT generates everything from scratch"""
    
    def __init__(
            self, num_freelancers: int = 5, num_clients: int = 3, rounds: int = 8, bids_per_round: int = 1,
            jobs_per_freelancer_per_round: int = 3, job_selection_method: str = 'random', relevance_mode: str = 'strict',
            use_cache: bool = True, reflection_probability: float = 0.1, enable_reflections: bool = True,
            max_workers: int = 10, bid_cooloff_rounds: int = 5, max_active_jobs: int = 3, quiet_mode: bool = False,
            job_posting_cooldown_min: int = 3, job_posting_cooldown_max: int = 10,
            # Baseline agent configuration
            freelancer_agent_type: str = 'llm', client_agent_type: str = 'llm',
            baseline_greedy_undercut: float = 0.9, random_freelancer_bid_probability: float = 0.5,
            # Reproducibility
            random_seed: int = None):
        self.num_freelancers = num_freelancers
        self.num_clients = num_clients  
        self.rounds = rounds
        self.bids_per_round = bids_per_round  # Each freelancer gets this many bids per round - controls strategic scarcity
        self.jobs_per_freelancer_per_round = jobs_per_freelancer_per_round  # Max jobs shown to each freelancer per round
        self.job_selection_method = job_selection_method  # 'random' or 'relevance'
        self.relevance_mode = relevance_mode  # 'strict', 'moderate', 'relaxed'
        self.max_active_jobs = max_active_jobs  # Maximum concurrent jobs per freelancer
        # Validate job posting cooldown parameters
        if job_posting_cooldown_min > job_posting_cooldown_max:
            raise ValueError(f"job_posting_cooldown_min ({job_posting_cooldown_min}) cannot be greater than job_posting_cooldown_max ({job_posting_cooldown_max})")
        
        self.job_posting_cooldown_min = job_posting_cooldown_min  # Minimum rounds before client can post again
        self.job_posting_cooldown_max = job_posting_cooldown_max  # Maximum rounds before client can post again
        
        # Initialize the ranking calculator
        self.ranking_calculator = JobRankingCalculator(relevance_mode)
        
        self.freelancers = []
        self.clients = []
        self.all_jobs = []
        self.active_jobs = []  # Jobs currently available for bidding (not yet filled)
        self.all_bids = []
        # PERFORMANCE: O(1) lookup dictionaries to avoid O(n) linear searches
        self.jobs_by_id = {}  # {job_id: Job} for fast lookups
        self.freelancers_by_id = {}  # {freelancer_id: Freelancer} for fast lookups
        # Decision tracking simplified for performance - only keep minimal data
        self.all_decisions = []
        self.hiring_outcomes = []
        self.round_data = []
        self.reflection_sessions = []  # Track all reflection sessions
        self.current_round = 0  # Track current simulation round
        
        # Caching system
        self.cache = AgentCache() if use_cache else None
        
        # Keep reference to client and config
        self.llm_config = llm_config
        self.openai_client = openai_client
        
        # Performance optimization settings
        self.reflection_probability = reflection_probability  # Probability each agent reflects per round (0.0 to 1.0)
        self.enable_reflections = enable_reflections  # Can disable entirely for speed
        self.max_workers = max_workers  # For parallel processing
        self.bid_cooloff_rounds = bid_cooloff_rounds  # Rounds before freelancer can re-bid on same job
        self.quiet_mode = quiet_mode  # Reduce logging for better performance
        
        # Baseline agent configuration
        self.freelancer_agent_type = freelancer_agent_type  # 'llm', 'random', 'greedy', 'no_reputation'
        self.client_agent_type = client_agent_type  # 'llm', 'random', 'greedy', 'no_reputation'
        self.baseline_greedy_undercut = baseline_greedy_undercut  # Undercut percentage for greedy agents (0.9 = 90% of budget)
        self.random_freelancer_bid_probability = random_freelancer_bid_probability  # Probability for random freelancers to bid (0.5 = 50%)
        self.random_seed = random_seed  # Store random seed for reproducibility
        
        # Validate agent types
        valid_freelancer_types = ['llm', 'random', 'greedy', 'no_reputation']
        valid_client_types = ['llm', 'random', 'greedy', 'no_reputation']
        if self.freelancer_agent_type not in valid_freelancer_types:
            raise ValueError(f"Invalid freelancer_agent_type: {self.freelancer_agent_type}. Must be one of: {valid_freelancer_types}")
        if self.client_agent_type not in valid_client_types:
            raise ValueError(f"Invalid client_agent_type: {self.client_agent_type}. Must be one of: {valid_client_types}")
        
        # Validate random freelancer bid probability
        if not (0.0 <= random_freelancer_bid_probability <= 1.0):
            raise ValueError(f"random_freelancer_bid_probability must be between 0.0 and 1.0, got {random_freelancer_bid_probability}")
        
        # Track freelancer decisions with round information for cooloff
        self.freelancer_job_decisions = {}  # {(freelancer_id, job_id): {'decision': str, 'round': int}}
        
        # Performance optimization: Set logging levels
        if self.quiet_mode:
            # Quiet mode: only show warnings and errors
            logging.getLogger().setLevel(logging.WARNING)
            logging.getLogger("caching.token_tracker").setLevel(logging.WARNING)
            logging.getLogger("marketplace.true_gpt_marketplace").setLevel(logging.WARNING)
            logging.getLogger("marketplace.reflection_adjustments").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
    
    def calculate_job_relevance(self, freelancer, job):
        """Calculate job relevance for a freelancer (delegate to ranking calculator)."""
        return self.ranking_calculator.calculate_job_relevance(freelancer, job)
    
    def _make_openai_request(self, **kwargs):
        """Make an OpenAI API request (wrapper for testing)."""
        return openai_client.chat.completions.create(**kwargs)
    
    def get_jobs_for_freelancer(self, freelancer: Freelancer, available_jobs: List[Job], current_round: int) -> List[Job]:
        """Get available jobs for a freelancer (applying cooloff logic for re-bidding)"""
        freelancer_id = freelancer.id
        
        # PERFORMANCE OPTIMIZATION: Pre-filter using freelancer's decision keys to avoid O(n²)
        # Instead of looping through all jobs, only check jobs this freelancer has decided on
        freelancer_decided_jobs = {job_id for (f_id, job_id) in self.freelancer_job_decisions.keys() if f_id == freelancer_id}
        
        available_jobs_filtered = []
        for job in available_jobs:
            if job.id not in freelancer_decided_jobs:
                # Never seen this job before - always available
                available_jobs_filtered.append(job)
            else:
                # Check if cooloff period has passed for jobs they declined
                decision_key = (freelancer_id, job.id)
                decision_info = self.freelancer_job_decisions[decision_key]
                
                # If it's stored as old format (just decision string), convert it
                if isinstance(decision_info, str):
                    decision_info = {'decision': decision_info, 'round': 0}
                    self.freelancer_job_decisions[decision_key] = decision_info
                
                last_decision = decision_info.get('decision', 'no')
                last_round = decision_info.get('round', 0)
                rounds_since_decision = current_round - last_round
                
                # Allow re-bidding if:
                # 1. They declined ('no') AND cooloff period has passed AND cooloff is enabled (> 0)
                # 2. Job is still unfilled (which it must be since it's in available_jobs)
                if (last_decision == 'no' and 
                    self.bid_cooloff_rounds > 0 and 
                    rounds_since_decision >= self.bid_cooloff_rounds):
                    available_jobs_filtered.append(job)
                    # Only log cooloff expiration in non-quiet mode to reduce spam
                    if not self.quiet_mode:
                        print(f"    🔄 {freelancer.name}: Cooloff expired for {job.title} (declined {rounds_since_decision} rounds ago)")
                # If they said 'yes' before, they already bid - no re-bidding allowed
                # If cooloff is disabled (0) or hasn't passed, skip this job
        
        return available_jobs_filtered
    
    def select_jobs_for_freelancer(self, freelancer: Freelancer, unseen_jobs: List[Job]) -> List[Job]:
        """Select which jobs to show to a freelancer based on configured selection method and agent type"""
        import random
        
        # If we don't have too many jobs, show them all
        if len(unseen_jobs) <= self.jobs_per_freelancer_per_round:
            return unseen_jobs
        
        # For greedy and no_reputation agents, always show highest budget jobs
        if self.freelancer_agent_type in ['greedy', 'no_reputation']:
            # Sort by budget (descending) and take top jobs
            sorted_jobs = sorted(unseen_jobs, key=lambda job: job.budget_amount, reverse=True)
            return sorted_jobs[:self.jobs_per_freelancer_per_round]
        
        # Apply selection method for other agent types
        if self.job_selection_method == 'relevance':
            # PERFORMANCE OPTIMIZATION: Limit relevance calculations to avoid exponential slowdown
            # Only calculate relevance for a reasonable subset to maintain performance
            max_relevance_jobs = min(20, len(unseen_jobs))  # Limit expensive calculations
            
            if len(unseen_jobs) > max_relevance_jobs:
                # Pre-filter using budget to reduce expensive relevance calculations
                budget_sorted = sorted(unseen_jobs, key=lambda job: job.budget_amount, reverse=True)
                relevance_candidates = budget_sorted[:max_relevance_jobs]
            else:
                relevance_candidates = unseen_jobs
            
            # Score the limited candidate set
            job_scores = []
            for job in relevance_candidates:
                relevance_score = self.ranking_calculator.calculate_job_relevance(freelancer, job)
                job_scores.append((job, relevance_score))
            
            # Sort by relevance score (descending) and take top jobs
            job_scores.sort(key=lambda x: x[1], reverse=True)
            return [job for job, score in job_scores[:self.jobs_per_freelancer_per_round]]
        else:
            # Random sampling (default)
            return random.sample(unseen_jobs, self.jobs_per_freelancer_per_round)

        
    def generate_freelancer_personas(self) -> List[Dict]:
        """Use GPT to create diverse, authentic freelancer personas"""
        
        # Try to load from cache first with smart sampling
        cached_profiles = []
        if self.cache:
            cached_profiles = self.cache.load_freelancer_profiles() or []
            logger.info(f"Found {len(cached_profiles)} cached freelancer profiles")
        
        if len(cached_profiles) >= self.num_freelancers:
            logger.info(f"Using {self.num_freelancers} cached freelancer profiles")
            # Take random sample if we have more than needed
            if len(cached_profiles) > self.num_freelancers:
                freelancers = random.sample(cached_profiles, self.num_freelancers)
            else:
                freelancers = cached_profiles[:self.num_freelancers]
            
            # Convert cached dictionaries to Freelancer objects
            freelancer_objects = []
            for i, freelancer_data in enumerate(freelancers):
                freelancer_obj = Freelancer(
                    id=f"freelancer_{i+1}",
                    name=freelancer_data.get('name', f"Freelancer {i+1}"),
                    category=freelancer_data.get('category', 'Web, Mobile & Software Dev'),
                    skills=freelancer_data.get('skills', []),
                    min_hourly_rate=float(freelancer_data.get('min_hourly_rate', 25)),
                    personality=freelancer_data.get('personality', ''),
                    motivation=freelancer_data.get('motivation', ''),
                    background=freelancer_data.get('background', ''),
                    preferred_project_length=freelancer_data.get('preferred_project_length', 'any'),
                    total_bids=0,
                    total_hired=0,
                    completed_jobs=0,
                    bids_this_round=0,
                    active_jobs=0,
                    max_active_jobs=self.max_active_jobs,
                    ongoing_jobs=[],
                    job_history=[],
                    bid_feedback=[]
                )
                freelancer_objects.append(freelancer_obj)
            
            self.freelancers = freelancer_objects
            
            # PERFORMANCE: Populate freelancer lookup dictionary for O(1) access
            for freelancer in freelancer_objects:
                self.freelancers_by_id[freelancer.id] = freelancer
            
            # Initialize reputation for all cached freelancers
            logger.info(f"🔧 Initializing reputation for {len(self.freelancers)} cached freelancers...")
            for freelancer in self.freelancers:
                logger.info(f"   Initializing: {freelancer.id} ({freelancer.name})")
                simple_reputation_manager.initialize_freelancer(
                    freelancer.id, 
                    skills=freelancer.skills,
                    freelancer_category=freelancer.category
                )
            
            return freelancer_objects
        
        # Generate additional profiles if needed in batches
        needed_profiles = self.num_freelancers - len(cached_profiles)
        batch_size = 10  # Generate 10 profiles at a time
        all_new_profiles = []
        
        for batch_start in range(0, needed_profiles, batch_size):
            batch_count = min(batch_size, needed_profiles - batch_start)
            logger.info(f"🤖 GPT generating batch of {batch_count} freelancer personas ({batch_start + 1}-{batch_start + batch_count} of {needed_profiles})...")
            
            # Get all categories and their key information
            category_info = []
            for category in JobCategory:
                cat_def = category_manager.get_category(category)
                category_info.append({
                    'name': cat_def.name,
                    'core_skills': cat_def.core_skills[:3],
                    'avg_rate': cat_def.metrics.avg_hourly_rate
                })
        
        try:
            # Generate profiles in batches until we have enough
            while len(all_new_profiles) < needed_profiles:
                remaining = needed_profiles - len(all_new_profiles)
                current_batch_size = min(batch_size, remaining)
                
                logger.info(f"🤖 GPT generating batch of {current_batch_size} freelancer personas ({len(all_new_profiles) + 1}-{len(all_new_profiles) + current_batch_size} of {needed_profiles})...")
                
                # Use OpenAI client for freelancer generation
                current_prompt = generate_freelancer_persona_prompt(current_batch_size, category_info)
                response = openai_client.chat.completions.create(
                    model=llm_config.model,
                    messages=[{"role": "user", "content": current_prompt}],
                    temperature=0.8,  # Higher creativity for diverse personas
                    max_tokens=10000,  # Note: max_tokens instead of max_completion_tokens for direct API call
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                
                # Validate using Pydantic
                validated_personas = validate_freelancer_personas(content)
                if not validated_personas:
                    logger.error("Could not validate freelancer personas, retrying...")
                    continue
                
                # Convert validated personas to profile dicts
                valid_profiles = []
                for persona in validated_personas:
                    profile = persona.model_dump()
                    # Add required tracking fields
                    profile['total_bids'] = 0
                    profile['total_hired'] = 0
                    profile['bids_this_round'] = 0
                    valid_profiles.append(profile)
                
                all_new_profiles.extend(valid_profiles)
                logger.info(f"Added {len(valid_profiles)} valid profiles from batch")
                
                # Add a small delay between batches
                time.sleep(1)
            
            # Combine cached and newly generated profiles
            all_freelancers = cached_profiles + all_new_profiles
            
            # Validate we have enough profiles
            if len(all_freelancers) < self.num_freelancers:
                error_msg = f"Failed to generate enough freelancer profiles. Got {len(all_freelancers)}, needed {self.num_freelancers}. Experiment cannot continue."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Take random sample if we have more than needed
            if len(all_freelancers) > self.num_freelancers:
                all_freelancers = random.sample(all_freelancers, self.num_freelancers)
            
            # Convert dictionaries to Freelancer objects and assign sequential IDs
            freelancer_objects = []
            for i, freelancer_data in enumerate(all_freelancers):
                freelancer_obj = Freelancer(
                    id=f"freelancer_{i+1}",
                    name=freelancer_data.get('name', f"Freelancer {i+1}"),
                    category=freelancer_data.get('category', 'Web, Mobile & Software Dev'),
                    skills=freelancer_data.get('skills', []),
                    min_hourly_rate=float(freelancer_data.get('min_hourly_rate', 25)),
                    personality=freelancer_data.get('personality', ''),
                    motivation=freelancer_data.get('motivation', ''),
                    background=freelancer_data.get('background', ''),
                    preferred_project_length=freelancer_data.get('preferred_project_length', 'any'),
                    total_bids=0,
                    total_hired=0,
                    completed_jobs=0,
                    bids_this_round=0,
                    active_jobs=0,
                    max_active_jobs=self.max_active_jobs,
                    ongoing_jobs=[],
                    job_history=[],
                    bid_feedback=[]
                )
                freelancer_objects.append(freelancer_obj)
            
            self.freelancers = freelancer_objects
            
            # PERFORMANCE: Populate freelancer lookup dictionary for O(1) access
            for freelancer in freelancer_objects:
                self.freelancers_by_id[freelancer.id] = freelancer
            
            # Initialize reputation for all freelancers
            logger.info(f"🔧 Initializing reputation for {len(self.freelancers)} freelancers...")
            for freelancer in self.freelancers:
                logger.info(f"   Initializing: {freelancer.id} ({freelancer.name})")
                simple_reputation_manager.initialize_freelancer(
                    freelancer.id, 
                    skills=freelancer.skills,
                    freelancer_category=freelancer.category
                )
            
            logger.info(f"✅ Using {len(cached_profiles)} cached + {len(all_new_profiles)} new = {len(self.freelancers)} total freelancer personas")
            
            # Cache the newly generated profiles
            if self.cache and all_new_profiles:
                self.cache.save_freelancer_profiles(all_new_profiles, append_to_existing=True)
            
            return self.freelancers
            
        except Exception as e:
            logger.exception(f"Error generating freelancer personas: {e}")
            # Fallback to minimal template if GPT fails
            return self._fallback_freelancers()
    
    def generate_client_personas(self) -> List[Dict]:
        """Use GPT to create diverse client personas and companies with caching"""
        
        # Try to load from cache first
        if self.cache:
            cached_profiles = self.cache.load_client_profiles(self.num_clients)
            if cached_profiles and len(cached_profiles) >= self.num_clients:
                # Use cached profiles (sample if we have more than needed)
                selected_profiles = cached_profiles[:self.num_clients] if len(cached_profiles) > self.num_clients else cached_profiles
                
                # Convert cached dictionaries to Client objects
                client_objects = []
                for i, client_data in enumerate(selected_profiles):
                    client_obj = Client(
                        id=f"client_{i+1}",
                        company_name=client_data.get('company_name', f"Company {i+1}"),
                        company_size=client_data.get('company_size', 'medium'),
                        budget_philosophy=client_data.get('budget_philosophy', 'balanced'),
                        hiring_style=client_data.get('hiring_style', 'professional'),
                        background=client_data.get('background', ''),
                        business_category=client_data.get('business_category', 'SOFTWARE'),
                        next_job_round=0
                    )
                    client_objects.append(client_obj)
                
                self.clients = client_objects
                
                # Initialize reputation for all cached clients
                logger.info(f"🔧 Initializing reputation for {len(self.clients)} cached clients...")
                for client in self.clients:
                    logger.info(f"   Initializing: {client.id} ({client.company_name})")
                    simple_reputation_manager.initialize_client(client.id)
                
                logger.info(f"Using {len(client_objects)} cached client profiles")
                return client_objects
            
            # If we have some cached but need more
            if cached_profiles:
                clients_needed = self.num_clients - len(cached_profiles)
                logger.info(f"🤖 GPT generating {clients_needed} additional client personas...")
            else:
                clients_needed = self.num_clients
                logger.info(f"🤖 GPT generating {clients_needed} client personas...")
        else:
            clients_needed = self.num_clients
            logger.info(f"🤖 GPT generating {clients_needed} client personas...")
            cached_profiles = []
        
        # Generate clients in batches to avoid JSON syntax errors
        batch_size = 5  # Smaller batches for better JSON reliability
        all_new_clients = []
        
        try:
            while len(all_new_clients) < clients_needed:
                remaining = clients_needed - len(all_new_clients)
                current_batch_size = min(batch_size, remaining)
                
                # Sample random business categories for this batch (one per client)
                sampled_categories = [random.choice(list(JobCategory)).name for _ in range(current_batch_size)]
                
                logger.info(f"🤖 GPT generating batch of {current_batch_size} client personas with categories {sampled_categories} ({len(all_new_clients) + 1}-{len(all_new_clients) + current_batch_size} of {clients_needed})...")
                
                prompt = generate_client_persona_prompt(current_batch_size, sampled_categories)
                response = self._make_openai_request(
                    model=llm_config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    max_completion_tokens=1500,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                
                # Validate using Pydantic
                validated_clients = validate_client_personas(content)
                if not validated_clients:
                    logger.error("Could not validate client personas, retrying...")
                    continue
                
                # Convert validated clients to profile dicts
                valid_profiles = []
                for client in validated_clients:
                    profile = client.model_dump()
                    valid_profiles.append(profile)
                
                all_new_clients.extend(valid_profiles)
                logger.info(f"Added {len(valid_profiles)} valid client profiles from batch")
                
                # Add a small delay between batches
                time.sleep(1)
            
            # Combine cached and newly generated profiles
            all_clients = (cached_profiles or []) + all_new_clients
            
            # Validate we have enough profiles
            if len(all_clients) < self.num_clients:
                error_msg = f"Failed to generate enough client profiles. Got {len(all_clients)}, needed {self.num_clients}. Experiment cannot continue."
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Convert dictionaries to Client objects and assign IDs
            client_objects = []
            selected_clients = all_clients[:self.num_clients]
            for i, client_data in enumerate(selected_clients):
                client_obj = Client(
                    id=f"client_{i+1}",
                    company_name=client_data.get('company_name', f"Company {i+1}"),
                    company_size=client_data.get('company_size', 'medium'),
                    budget_philosophy=client_data.get('budget_philosophy', 'balanced'),
                    hiring_style=client_data.get('hiring_style', 'professional'),
                    background=client_data.get('background', ''),
                    business_category=client_data.get('business_category', 'SOFTWARE'),
                    next_job_round=0
                )
                client_objects.append(client_obj)
            
            self.clients = client_objects
            logger.info(f"DEBUG: About to initialize reputation for {len(self.clients)} clients")
            
            # Initialize reputation for all clients
            logger.info(f"🔧 Initializing reputation for {len(self.clients)} clients...")
            for client in self.clients:
                logger.info(f"   Initializing: {client.id} ({client.company_name})")
                simple_reputation_manager.initialize_client(client.id)
            
            logger.info(f"✅ Using {len(cached_profiles or [])} cached + {len(all_new_clients)} new = {len(self.clients)} total client personas")
            
            # Cache the newly generated profiles (append to existing)
            if self.cache and all_new_clients:
                self.cache.save_client_profiles(all_new_clients, append_to_existing=True)
            
            return self.clients
                
        except Exception as e:
            logger.exception(f"Error generating client personas: {e}")
            return self._fallback_clients()
    
    def generate_job_posting(self, client: Dict, round_num: int) -> Dict:
        """Use GPT to create authentic job postings based on client persona and job categories"""
        
        # Try to load from cache first
        if self.cache:
            cached_templates = self.cache.load_job_templates()
            if cached_templates:
                # We want more diversity in jobs, so let's generate a new one
                # Only use cache if we find an exact match for both company and category
                company_templates = []
                for template in cached_templates:
                    template_company = template.get('company_name', '')
                    template_category = template.get('category', '').lower()
                    if (template_company == client.company_name and 
                        template_category == client.background.lower()):
                        company_templates.append(template)
                
                if company_templates and random.random() < 0.3:  # 30% chance to reuse a template
                    # Use a random template from the filtered list
                    template = random.choice(company_templates)
                    
                    # Deep copy to avoid modifying cache
                    job = template.copy()
                    
                    # Update company-specific fields
                    job['company_name'] = client.company_name
                    
                    # Inherit category from client's business domain
                    job_category = JobCategory[client.business_category.upper()]
                    job['category'] = job_category.value[0]  # Get the display name from the tuple
                    
                    # Get category definition for budget calculations
                    category_def = category_manager.get_category(job_category)
                    # Customize budget based on client's philosophy and market conditions
                    base_rate = category_def.metrics.avg_hourly_rate
                    client_budget_philosophy = client.budget_philosophy.lower()
                    
                    # Add more variation to budgets
                    budget_adjustment = 1.0  # Default no adjustment
                    if 'premium' in client_budget_philosophy or 'premium-focused' in client_budget_philosophy:
                        budget_adjustment = random.uniform(1.1, 1.3)  # 10-30% above market
                    elif 'value' in client_budget_philosophy or 'value-focused' in client_budget_philosophy:
                        budget_adjustment = random.uniform(0.85, 0.95)  # 5-15% below market  
                    else:  # cost-conscious (default)
                        budget_adjustment = random.uniform(0.75, 0.85)  # 15-25% below market
                        
                    # Add some market-driven variation
                    market_adjustment = random.uniform(0.95, 1.05)  # ±5% random market fluctuation
                    
                    # Apply client's dynamic budget multiplier from reflections
                    final_budget = base_rate * budget_adjustment * market_adjustment * client.budget_multiplier
                    job['budget_amount'] = max(job.get('budget_amount', 0), final_budget)
                    # Convert to Job object
                    job_obj = Job(
                        id=f"job_{round_num}_{client.id}",
                        client_id=client.id,
                        title=job.get('title', 'Untitled Job'),
                        description=job.get('description', ''),
                        category=job.get('category', JobCategory.SOFTWARE.value[0]),
                        skills_required=job.get('skills_required', []),
                        budget_type=job.get('budget_type', 'fixed'),
                        budget_amount=float(job.get('budget_amount', 0)),
                        timeline=job.get('timeline', 'Flexible'),
                        special_requirements=job.get('special_requirements', ''),
                        posted_time=datetime.now()
                    )
                    
                    logger.info(f"Using modified cached job template for {client.company_name}")
                    return job_obj
        
        # Select a category based on client's background and skills needed
        selected_category = random.choice(list(JobCategory))
        
        prompt = create_gpt_job_posting_prompt(client, selected_category)
        
        try:
            response = self._make_openai_request(
                model=llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=800,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if not self.quiet_mode:
                logger.info(f"🔍 {llm_config.model} job response: '{content}'")
            
            # Validate using Pydantic
            validated_job = validate_job_posting(content)
            if validated_job:
                job = validated_job.model_dump()
                
                # Apply client's dynamic budget multiplier from reflections
                original_budget = float(job.get('budget_amount', 0))
                adjusted_budget = original_budget * client.budget_multiplier
                
                # Convert to Job object
                job_obj = Job(
                    id=f"job_{round_num}_{client.id}",
                    client_id=client.id,
                    title=job.get('title', 'Untitled Job'),
                    description=job.get('description', ''),
                    category=job.get('category', JobCategory.SOFTWARE.value[0]),
                    skills_required=job.get('skills_required', []),
                    budget_type=job.get('budget_type', 'fixed'),
                    budget_amount=adjusted_budget,
                    timeline=job.get('timeline', 'Flexible'),
                    special_requirements=job.get('special_requirements', ''),
                    posted_time=datetime.now()
                )
                
                # Cache the new job template
                if self.cache:
                    template = job.copy()
                    # Keep company info but remove instance-specific fields
                    template['company_name'] = client.company_name
                    template['category'] = job.get('category', JobCategory.SOFTWARE.value[0])
                    template.pop('id', None)
                    template.pop('client_id', None)
                    template.pop('posted_time', None)
                    self.cache.save_job_templates([template], append_to_existing=True)
                
                return job_obj
            else:
                logger.error(f"❌ No JSON found in job response: {content}")
                raise ValueError("Could not extract JSON from GPT response")
                
        except Exception as e:
            logger.exception(f"Error generating job posting: {e}")
            return self._fallback_job(client, round_num)
    

    def freelancer_bidding_decision(self, freelancer: Freelancer, job: Job, current_round: int = 0) -> Dict:
        """Main bidding decision method that delegates to appropriate agent type."""
        # Delegate to the appropriate agent type
        if self.freelancer_agent_type == 'llm':
            return self._freelancer_bidding_decision_llm(freelancer, job, current_round)
        elif self.freelancer_agent_type == 'random':
            return self._freelancer_bidding_decision_random(freelancer, job, current_round)
        elif self.freelancer_agent_type == 'greedy':
            return self._freelancer_bidding_decision_greedy(freelancer, job, current_round)
        elif self.freelancer_agent_type == 'no_reputation':
            return self._freelancer_bidding_decision_no_reputation(freelancer, job, current_round)
        else:
            raise ValueError(f"Unknown freelancer agent type: {self.freelancer_agent_type}")

    def _freelancer_bidding_decision_llm(self, freelancer: Freelancer, job: Job, current_round: int = 0) -> Dict:
        """Original LLM-powered bidding decision logic."""
        # Note: Cooloff logic is handled in get_jobs_for_freelancer, so if we reach here,
        # the freelancer is allowed to consider this job (either new or cooloff expired)

        # Check bid budget first
        if not freelancer.can_bid(self.bids_per_round):
            return {
                "decision": "no", 
                "reasoning": f"Already used up bid budget ({self.bids_per_round} bids per round)"
            }
        
        # Get reflection history and insights
        reflections = reflection_manager.get_agent_reflections(freelancer.id, days=30)  # Last 30 days
        learning_progress = reflection_manager.get_learning_progress(freelancer.id)
        
        # Build context about freelancer's history and learning
        if freelancer.total_bids == 0:
            success_display = "New freelancer (no previous bids)"
            experience_context = "You are new to the platform and eager to build your reputation."
            strategy_context = "Consider starting with projects that closely match your core skills."
        else:
            success_rate = freelancer.total_hired / freelancer.total_bids
            success_display = f"{success_rate:.1%} success rate"
            
            # Combine reflection insights with bid feedback
            insights = []
            
            # Add reflection-based insights
            if reflections:
                latest_reflection = reflections[-1]
                
                # Add key insights from simplified reflection system
                if latest_reflection.key_insights:
                    insights.extend(latest_reflection.key_insights[:2])  # Add top 2 insights
                
                # Add learning progress insights
                if learning_progress:
                    recent_insights = learning_progress.get('recent_insights', [])
                    if recent_insights:
                        insights.extend(recent_insights[:2])
            
            # Add recent bid feedback (rarely available since clients don't usually provide feedback)
            if hasattr(freelancer, 'bid_feedback') and freelancer.bid_feedback:
                recent_feedback = freelancer.bid_feedback[-3:]  # Last 3 pieces of feedback
                for fb in recent_feedback:
                    if 'feedback' in fb:
                        feedback_themes = [
                            fb['feedback'].get('main_reason', ''),
                            fb['feedback'].get('pitch_feedback', '')
                        ]
                        insights.extend(filter(None, feedback_themes))
                        insights.append("Rare client feedback received - valuable learning opportunity")
            
            experience_context = "; ".join(insights) if insights else "You are building experience with each bid."
            
            # Build strategy context from reflections and job history
            strategy_points = []
            
            # Add reflection-based strategy
            if reflections:
                latest_reflection = reflections[-1]
                if latest_reflection.planned_adjustments:
                    strategy_points.extend(latest_reflection.planned_adjustments[:2])
            
            # Add job history insights
            if hasattr(freelancer, 'job_history') and freelancer.job_history:
                completed_jobs = [j for j in freelancer.job_history if j.get('success', False)]
                if completed_jobs:
                    recent_success = completed_jobs[-1]
                    strategy_points.append(
                        f"Recent success: {recent_success.get('rating', 0)}/5 rating - {recent_success.get('feedback', 'No feedback')}"
                    )
            
            strategy_context = ". ".join(strategy_points) if strategy_points else "Focus on projects where you can demonstrate clear value."
        
        # Get reputation data for adaptive prompts
        reputation_info = simple_reputation_manager.get_freelancer_reputation(freelancer.id)
        
        prompt = create_bidding_decision_prompt(freelancer, job, success_display, experience_context, strategy_context, self.bids_per_round, reputation_info)
        
        try:
            response = self._make_openai_request(
                model=llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=1000,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            if not self.quiet_mode:
                logger.info(f"🔍 {llm_config.model} bidding response: '{content}'")
            
            # Validate response using Pydantic
            validated_response = validate_bidding_decision(content)
            if validated_response:
                # Use the validated object directly
                if validated_response.decision == 'yes':
                    # Return dict with validated data plus metadata for bidding
                    return {
                        'decision': validated_response.decision,
                        'reasoning': validated_response.reasoning,
                        'message': validated_response.message,
                        'freelancer_id': freelancer.id,
                        'job_id': job.id,
                        'submission_time': datetime.now()
                    }
                else:
                    # Return simple dict for no-bid decision
                    return {
                        'decision': validated_response.decision,
                        'reasoning': validated_response.reasoning
                    }
            else:
                raise ValueError("Could not validate GPT response")
                
        except Exception as e:
            logger.exception(f"Error in bidding decision: {e}")
            return {"decision": "no", "reasoning": "Technical error in decision making"}

    def _freelancer_bidding_decision_random(self, freelancer: Freelancer, job: Job, current_round: int = 0) -> Dict:
        """Random bidding baseline: bid randomly on jobs with random bid values."""
        import random
        
        # Check bid budget first
        if not freelancer.can_bid(self.bids_per_round):
            return {
                "decision": "no", 
                "reasoning": f"Already used up bid budget ({self.bids_per_round} bids per round)"
            }
        
        # Random decision: configurable probability to bid
        decision = 'yes' if random.random() < self.random_freelancer_bid_probability else 'no'
        
        if decision == 'yes':
            # Random bid value: uniform draw from 50-150% of job budget
            bid_multiplier = random.uniform(0.5, 1.5)
            bid_amount = job.budget_amount * bid_multiplier

            return {
                'decision': decision,
                'reasoning': f'Random bidding baseline - random decision to bid ({self.random_freelancer_bid_probability*100:.0f}% probability)',
                'message': f'I am very interested in this job. I think I can deliver value given my expertise in this field. My rate is: ${bid_amount:.2f}',
                'freelancer_id': freelancer.id,
                'job_id': job.id,
                'submission_time': datetime.now()
            }
        else:
            return {
                'decision': decision,
                'reasoning': f'Random bidding baseline - random decision not to bid ({self.random_freelancer_bid_probability*100:.0f}% probability)'
            }

    def _freelancer_bidding_decision_greedy(self, freelancer: Freelancer, job: Job, current_round: int = 0) -> Dict:
        """Greedy heuristic baseline: always bid on highest budget job with undercut pricing."""
        # Check bid budget first
        if not freelancer.can_bid(self.bids_per_round):
            return {
                "decision": "no", 
                "reasoning": f"Already used up bid budget ({self.bids_per_round} bids per round)"
            }
        
        # Greedy strategy: always bid (assume job selection already filtered for highest budget)
        decision = 'yes'
        
        # Greedy strategy: undercut by percentage to win jobs
        bid_amount = job.budget_amount * self.baseline_greedy_undercut
        reasoning = f'Greedy baseline - undercutting at {self.baseline_greedy_undercut*100:.0f}% of budget (${bid_amount:.2f})'
        
        return {
            'decision': decision,
            'reasoning': reasoning,
            'message': f'Greedy baseline bid. Rate: ${bid_amount:.2f}',
            'freelancer_id': freelancer.id,
            'job_id': job.id,
            'submission_time': datetime.now()
        }

    def _freelancer_bidding_decision_no_reputation(self, freelancer: Freelancer, job: Job, current_round: int = 0) -> Dict:
        """No-reputation baseline: greedy strategy but ignore reputation entirely."""
        # Same as greedy but without any reputation considerations
        return self._freelancer_bidding_decision_greedy(freelancer, job, current_round)
    
    def client_hiring_decision(self, client: Client, job: Job, bids: List[Bid]) -> HiringDecision:
        """Main hiring decision method that delegates to appropriate agent type."""
        # Delegate to the appropriate agent type
        if self.client_agent_type == 'llm':
            return self._client_hiring_decision_llm(client, job, bids)
        elif self.client_agent_type == 'random':
            return self._client_hiring_decision_random(client, job, bids)
        elif self.client_agent_type == 'greedy':
            return self._client_hiring_decision_greedy(client, job, bids)
        elif self.client_agent_type == 'no_reputation':
            return self._client_hiring_decision_no_reputation(client, job, bids)
        else:
            raise ValueError(f"Unknown client agent type: {self.client_agent_type}")

    def _client_hiring_decision_llm(self, client: Client, job: Job, bids: List[Bid]) -> HiringDecision:
        """Original LLM-powered hiring decision logic."""
        
        if not bids:
            return HiringDecision(
                job_id=job.id,
                client_id=client.id,
                selected_freelancer=None,
                reasoning="No bids received",
                timestamp=datetime.now()
            )
        
        # Prepare bid summaries for the client
        bid_summaries = []
        
        for bid in bids:
            # PERFORMANCE: Use O(1) dictionary lookup instead of O(n) linear search
            freelancer = self.freelancers_by_id.get(bid.freelancer_id)
            if not freelancer:
                continue
                
            # Get sophisticated reputation data
            reputation = simple_reputation_manager.get_freelancer_reputation(freelancer.id)
            
            if reputation:
                # Use comprehensive reputation system
                tier_descriptions = {
                    "New": "New freelancer (eager to build reputation)",
                    "Established": f"Established freelancer ({reputation.successful_jobs} successful projects, {reputation.job_completion_rate:.0%} completion rate)",
                    "Expert": f"Expert freelancer ({reputation.successful_jobs} successful projects, {reputation.job_completion_rate:.0%} completion rate)",
                    "Elite": f"Elite freelancer ({reputation.successful_jobs} successful projects, {reputation.job_completion_rate:.0%} completion rate)"
                }
                success_display = tier_descriptions.get(reputation.tier.value, "Unknown reputation")
                
                # Add category expertise for this job if available
                job_category = getattr(job, 'category', None)
                if job_category and job_category in reputation.category_expertise:
                    expertise_level = reputation.category_expertise[job_category]
                    if expertise_level > 0.7:
                        success_display += f" (Specialized in {job_category})"
                    elif expertise_level > 0.4:
                        success_display += f" (Experienced in {job_category})"
            else:
                # Fallback to basic calculation if reputation not available
                if freelancer.total_hired == 0:
                    success_display = "New freelancer (eager to build reputation)"
                else:
                    success_rate = freelancer.completed_jobs / freelancer.total_hired
                    success_display = f"Track record: {success_rate:.0%} completion rate"
            
            bid_summaries.append({
                "freelancer_name": freelancer.name,
                "skills": freelancer.skills,
                "message": bid.message,
                "success_rate": success_display
            })
        


        # Get reflection history and insights
        reflections = reflection_manager.get_agent_reflections(client.id, days=30)  # Last 30 days
        learning_progress = reflection_manager.get_learning_progress(client.id)
        
        # Get previous hiring decisions for reflection (limit to recent decisions to prevent death spiral)
        previous_decisions = [h for h in self.hiring_outcomes 
                            if h.client_id == client.id][-3:]  # Last 3 decisions only
        
        reflection_prompt = "\nREFLECT ON YOUR EXPERIENCE AND LEARNING:\n"
        
        # Add reflection-based insights
        if reflections:
            latest_reflection = reflections[-1]
            
            # Add key insights from reflection
            if latest_reflection.key_insights:
                reflection_prompt += "\nRECENT LEARNINGS:\n"
                for insight in latest_reflection.key_insights:
                    reflection_prompt += f"- {insight}\n"
            
            # Add planned adjustments
            if latest_reflection.planned_adjustments:
                reflection_prompt += "\nPLANNED ADJUSTMENTS:\n"
                for adjustment in latest_reflection.planned_adjustments:
                    reflection_prompt += f"- {adjustment}\n"
        
        # Add learning progress insights
        if learning_progress:
            top_insights = learning_progress.get('top_insights', [])
            if top_insights:
                reflection_prompt += "\nKEY INSIGHTS FROM EXPERIENCE:\n"
                for insight in top_insights[:3]:
                    reflection_prompt += f"- {insight}\n"
        
        # Add previous decisions
        if previous_decisions:
            reflection_prompt += "\nRECENT HIRING DECISIONS:\n"
            for decision in previous_decisions:
                # PERFORMANCE: Use O(1) dictionary lookup instead of O(n) linear search
                # FIX: Use different variable name to avoid overwriting job parameter
                decision_job = self.jobs_by_id.get(decision.job_id)
                if decision_job:
                    reflection_prompt += f"""
- {decision_job.title}:
  Decision: {"Hired " + decision.selected_freelancer if decision.selected_freelancer else "No hire"}
  Reasoning: {decision.reasoning}
  Outcome: Pending
  Learning: No learning recorded
"""

        # Create hiring prompt (outside all conditional blocks)
        prompt = create_hiring_decision_prompt(client, job, bid_summaries, reflection_prompt)
        
        try:
            response = self._make_openai_request(
                model=llm_config.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
                max_completion_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            
            # Validate response using Pydantic
            validated_response = validate_hiring_decision(content)
            if validated_response:
                # Use validated object directly
                selected_freelancer_id = None
                
                if validated_response.selected_bid_number > 0 and validated_response.selected_bid_number <= len(bids):
                    # Get the freelancer from the selected bid (1-indexed)
                    selected_bid = bids[validated_response.selected_bid_number - 1]
                    selected_freelancer_id = selected_bid.freelancer_id
                else:
                    # Fallback: try to match by name if bid number is not valid
                    if validated_response.selected_freelancer != 'none':
                        for bid in bids:
                            # PERFORMANCE: Use O(1) dictionary lookup instead of O(n) linear search
                            freelancer = self.freelancers_by_id.get(bid.freelancer_id)
                            if freelancer and freelancer.name.lower() == validated_response.selected_freelancer.lower():
                                selected_freelancer_id = bid.freelancer_id
                                break
                
                # Create and return HiringDecision entity
                return HiringDecision(
                    job_id=job.id,
                    client_id=client.id,
                    selected_freelancer=selected_freelancer_id,
                    reasoning=validated_response.reasoning,
                    timestamp=datetime.now()
                )
            else:
                raise ValueError("Could not extract JSON from GPT response")
                
        except Exception as e:
            logger.exception(f"Error in hiring decision: {e}")
            return HiringDecision(
                job_id=job.id,
                client_id=client.id,
                selected_freelancer=None,
                reasoning="Technical error in hiring decision",
                timestamp=datetime.now()
            )

    def _client_hiring_decision_random(self, client: Client, job: Job, bids: List[Bid]) -> HiringDecision:
        """Random hiring baseline: 50% chance to pick a random bid, otherwise don't hire."""
        import random
        
        if not bids:
            return HiringDecision(
                job_id=job.id,
                client_id=client.id,
                selected_freelancer=None,
                reasoning="No bids received",
                timestamp=datetime.now()
            )
        
        # Random decision: 50% chance to hire anyone at all
        if random.random() < 0.5:
            # Random hiring: pick a random bid
            selected_bid = random.choice(bids)
            
            return HiringDecision(
                job_id=job.id,
                client_id=client.id,
                selected_freelancer=selected_bid.freelancer_id,
                reasoning=f"Random hiring baseline - randomly selected from {len(bids)} bids",
                timestamp=datetime.now()
            )
        else:
            # Random decision not to hire
            return HiringDecision(
                job_id=job.id,
                client_id=client.id,
                selected_freelancer=None,
                reasoning=f"Random hiring baseline - randomly decided not to hire (had {len(bids)} bids available)",
                timestamp=datetime.now()
            )

    def _client_hiring_decision_greedy(self, client: Client, job: Job, bids: List[Bid]) -> HiringDecision:
        """Greedy hiring baseline: always pick the lowest-priced bid that doesn't exceed budget."""
        if not bids:
            return HiringDecision(
                job_id=job.id,
                client_id=client.id,
                selected_freelancer=None,
                reasoning="No bids received",
                timestamp=datetime.now()
            )
        
        # Greedy strategy: pick the lowest bid that doesn't exceed job budget
        valid_bids = [bid for bid in bids if bid.proposed_rate <= job.budget_amount]
        
        if not valid_bids:
            # No bids within budget - don't hire anyone
            return HiringDecision(
                job_id=job.id,
                client_id=client.id,
                selected_freelancer=None,
                reasoning=f"Greedy baseline - no bids within budget (${job.budget_amount:.2f})",
                timestamp=datetime.now()
            )
        
        # Select the lowest priced valid bid
        lowest_bid = min(valid_bids, key=lambda b: b.proposed_rate)
        
        return HiringDecision(
            job_id=job.id,
            client_id=client.id,
            selected_freelancer=lowest_bid.freelancer_id,
            reasoning=f"Greedy baseline - selected lowest bid (${lowest_bid.proposed_rate:.2f}) from {len(valid_bids)} valid bids",
            timestamp=datetime.now()
        )

    def _client_hiring_decision_no_reputation(self, client: Client, job: Job, bids: List[Bid]) -> HiringDecision:
        """No-reputation baseline: greedy strategy but ignore reputation entirely."""
        # Same as greedy but without any reputation considerations
        return self._client_hiring_decision_greedy(client, job, bids)
    
    def _extract_json_from_response(self, content: str, is_array: bool = False) -> str:
        """Extract JSON from GPT response, handling markdown code blocks"""
        
        # Try to find JSON within markdown code blocks first
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                json_content = content[start:end].strip()
                return json_content
        
        # Fallback to looking for raw JSON
        if is_array:
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end > start:
                return content[start:end]
        else:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return content[start:end]
        
        return ""
    
    def _process_completed_jobs(self, round_num: int) -> None:
        """Process job completions and reset freelancer bid budgets."""
        start_time = time.time()
        
        # Reset all freelancer bid budgets first
        for freelancer in self.freelancers:
            freelancer.reset_round()  # Reset bids_this_round = 0
        
        # Prepare job completion tasks for parallel processing
        completion_tasks = []
        for freelancer in self.freelancers:
            for job in freelancer.ongoing_jobs:
                rounds_elapsed = round_num - job['start_round']
                if rounds_elapsed >= job['duration_rounds']:
                    # Get the full job details
                    # PERFORMANCE: Use O(1) dictionary lookup instead of O(n) linear search
                    full_job = self.jobs_by_id.get(job['job_id'])
                    if full_job:
                        completion_tasks.append((freelancer, job, full_job))
        
        # Process job completions in parallel
        completion_results = []
        if completion_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                completion_futures = [executor.submit(self._process_job_completion_task, task) for task in completion_tasks]
                
                for future in concurrent.futures.as_completed(completion_futures):
                    try:
                        result = future.result()
                        if result:
                            completion_results.append(result)
                    except Exception as e:
                        logger.exception(f"Error in job completion: {e}")
        
        # Process results by freelancer
        freelancer_completions = {}
        for result in completion_results:
            freelancer_id = result['freelancer_id']
            if freelancer_id not in freelancer_completions:
                freelancer_completions[freelancer_id] = []
            freelancer_completions[freelancer_id].append(result)
        
        # Update freelancer stats and reputation
        for freelancer_id, completions in freelancer_completions.items():
            # PERFORMANCE: Use O(1) dictionary lookup instead of O(n) linear search
            freelancer = self.freelancers_by_id.get(freelancer_id)
            if freelancer:
                for completion in completions:
                    # Update freelancer stats using the entity methods
                    freelancer.complete_job(
                        job_id=completion['job_id'],
                        success=completion['success']
                    )
                    
                    # Update reputation based on job completion performance
                    simple_reputation_manager.update_freelancer_after_job(
                        freelancer.id,
                        earnings=completion['earnings'],
                        success=completion['success'],
                        job_category=completion['job_category']
                    )
                
                print(f"  ✨ {freelancer.name} completed {len(completions)} jobs")
        
        elapsed_time = time.time() - start_time
        if completion_tasks:
            print(f"    ⚡ Job completions processed in {elapsed_time:.1f}s ({len(completion_tasks)} jobs)")
    
    def _process_job_completion_task(self, task_data) -> Dict:
        """Process a single job completion task in parallel."""
        freelancer, job, full_job = task_data
        
        try:
            # NOTE: For now, all jobs are considered successful to avoid unfair client judgment
            # criteria that were hurting freelancer reputation progression. This could be
            # improved in the future with more sophisticated evaluation mechanisms.
            success = True
            
            # Calculate earnings
            estimated_earnings = getattr(full_job, 'budget_amount', 50.0) * job['duration_rounds']
            
            return {
                'freelancer_id': freelancer.id,
                'job_id': job['job_id'],
                'success': success,
                'earnings': estimated_earnings,
                'job_category': getattr(full_job, 'category', None)
            }
            
        except Exception as e:
            logger.exception(f"Failed to process job completion for {freelancer.name}: {e}")
            return {
                'freelancer_id': freelancer.id,
                'job_id': job['job_id'],
                'success': True,  # Even on error, default to success
                'earnings': 50.0,
                'job_category': None
            }
    
    def _generate_new_jobs(self, round_num: int) -> List[Job]:
        """Generate new job postings from clients in parallel."""
        start_time = time.time()
        new_jobs = []
        
        # Prepare job posting tasks for parallel processing
        job_posting_tasks = []
        for client in self.clients:
            # Each client has a cooldown period of 3-10 rounds between jobs
            if not client.can_post_job(round_num):
                continue
            job_posting_tasks.append((client, round_num))
        
        # Process job postings in parallel
        if job_posting_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                job_futures = [executor.submit(self._process_job_posting_task, task) for task in job_posting_tasks]
                
                for future in concurrent.futures.as_completed(job_futures):
                    try:
                        result = future.result()
                        if result:
                            job = result['job']
                            client = result['client']
                            new_jobs.append(job)
                            self.all_jobs.append(job)
                            self.active_jobs.append(job)
                            # PERFORMANCE: Add to lookup dictionary for O(1) access
                            self.jobs_by_id[job.id] = job
                            print(f"  📝 {client.company_name}: {job.title}")
                            
                            # Set next job round with random cooldown
                            client.set_next_job_round(round_num, random.randint(self.job_posting_cooldown_min, self.job_posting_cooldown_max))
                    except Exception as e:
                        logger.exception(f"Error in job posting: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"  📊 {len(self.active_jobs)} total active jobs ({len(new_jobs)} new this round)")
        print(f"    ⚡ Job posting completed in {elapsed_time:.1f}s ({len(job_posting_tasks)} clients)")
        return new_jobs
    
    def _process_job_posting_task(self, task_data) -> Dict:
        """Process a single job posting task in parallel."""
        client, round_num = task_data
        
        try:
            job = self.generate_job_posting(client, round_num)
            return {
                'job': job,
                'client': client
            }
        except Exception as e:
            logger.exception(f"Failed to generate job posting for {client.company_name}: {e}")
            return None
    
    def _process_freelancer_bidding(self, round_num: int) -> tuple:
        """Process freelancer bidding decisions and return bids and decisions."""
        start_time = time.time()
        round_bids = []
        round_decisions = []
        
        # Prepare bidding tasks for parallel processing
        bidding_tasks = []
        for freelancer in self.freelancers:
            # Get available jobs for this freelancer (applying cooloff logic)
            unseen_jobs = self.get_jobs_for_freelancer(freelancer, self.active_jobs, round_num)
            
            if not unseen_jobs:
                print(f"  👀 {freelancer.name}: No new jobs to review")
                continue
            
            # Select which jobs to show based on selection method
            selected_jobs = self.select_jobs_for_freelancer(freelancer, unseen_jobs)
            
            # Show debugging info with relevance scores if using relevance-based selection
            # PERFORMANCE: Only calculate relevance for debugging in non-quiet mode
            if self.job_selection_method == 'relevance' and len(selected_jobs) > 0 and not self.quiet_mode:
                # Calculate relevance scores for debugging
                relevance_scores = [self.ranking_calculator.calculate_job_relevance(freelancer, job) for job in selected_jobs]
                avg_relevance = sum(relevance_scores) / len(relevance_scores)
                print(f"  👀 {freelancer.name}: Reviewing {len(selected_jobs)} job(s) (avg relevance: {avg_relevance:.2f})")
            elif not self.quiet_mode:
                print(f"  👀 {freelancer.name}: Reviewing {len(selected_jobs)} job(s)")
            
            # Add task for this freelancer's bidding decisions
            bidding_tasks.append((freelancer, selected_jobs, round_num))
        
        # Process bidding decisions in parallel
        if bidding_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                bidding_futures = [executor.submit(self._process_freelancer_bidding_task, task) for task in bidding_tasks]
                
                for future in concurrent.futures.as_completed(bidding_futures):
                    try:
                        freelancer_results = future.result()
                        if freelancer_results:
                            round_bids.extend(freelancer_results['bids'])
                            round_decisions.extend(freelancer_results['decisions'])
                            
                            # Update freelancer job decisions
                            for decision_key, decision_value in freelancer_results['job_decisions'].items():
                                self.freelancer_job_decisions[decision_key] = decision_value
                                
                    except Exception as e:
                        logger.exception(f"Error in freelancer bidding: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"    ⚡ Bidding completed in {elapsed_time:.1f}s ({len(self.freelancers)} freelancers, {len(round_bids)} bids)")
        
        return round_bids, round_decisions
    
    def _process_hiring_decisions(self, round_num: int, round_bids: List[Bid]) -> None:
        """Process client hiring decisions in parallel."""
        start_time = time.time()
        
        # Group bids by job
        jobs_with_bids = {}
        for bid in round_bids:
            job_id = bid.job_id
            if job_id not in jobs_with_bids:
                jobs_with_bids[job_id] = []
            jobs_with_bids[job_id].append(bid)
        
        # Prepare hiring decision tasks for parallel processing
        hiring_tasks = []
        for job_id, job_bids in jobs_with_bids.items():
            # PERFORMANCE: Use O(1) dictionary lookup instead of O(n) linear search
            job = self.jobs_by_id.get(job_id)
            if job is None:
                # Fallback: search in active_jobs if not in dictionary (shouldn't happen)
                job = next((j for j in self.active_jobs if j.id == job_id), None)
                if job is None:
                    logger.warning(f"Job {job_id} not found for hiring decision - skipping")
                    continue
            
            client = next(c for c in self.clients if c.id == job.client_id)
            hiring_tasks.append((client, job, job_bids))
        
        # Process hiring decisions in parallel
        hiring_results = []
        if hiring_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                hiring_futures = [executor.submit(self._process_hiring_decision_task, task) for task in hiring_tasks]
                
                for future in concurrent.futures.as_completed(hiring_futures):
                    try:
                        result = future.result()
                        if result:
                            hiring_results.append(result)
                    except Exception as e:
                        logger.exception(f"Error in hiring decision: {e}")
        
        # Process results sequentially to maintain consistency
        for result in hiring_results:
            hiring_decision = result['hiring_decision']
            job = result['job']
            job_bids = result['job_bids']
            
            self.hiring_outcomes.append(hiring_decision)
            
            if hiring_decision.selected_freelancer:
                # Find the hired freelancer (could be by ID or name)
                selected_id = hiring_decision.selected_freelancer
                hired_freelancer = None
                
                # Try to find by ID first, then by name
                for f in self.freelancers:
                    if f.id == selected_id or f.name == selected_id:
                        hired_freelancer = f
                        break
                
                if hired_freelancer:
                    print(f"    ✅ {job.title}: Hired {hired_freelancer.name}")
                    # Remove filled job from active jobs
                    self.active_jobs = [j for j in self.active_jobs if j.id != job.id]
                    
                    # Clean up decision history for completed job to prevent memory bloat
                    # This is critical for performance - decision history can grow to millions of entries
                    job_decisions_to_remove = [key for key in self.freelancer_job_decisions.keys() if key[1] == job.id]
                    for key in job_decisions_to_remove:
                        del self.freelancer_job_decisions[key]
                    
                    # PERFORMANCE: Remove from lookup dictionary to keep it lean
                    if job.id in self.jobs_by_id:
                        del self.jobs_by_id[job.id]
                    
                    # Update freelancer hire count and active jobs using entity method
                    hired_freelancer.add_hire()
                    
                    # Update reputation: hired freelancer gets reputation boost
                    simple_reputation_manager.update_freelancer_after_bid(
                        hired_freelancer.id, 
                        was_hired=True, 
                        job_category=getattr(job, 'category', None)
                    )
                    
                    # Update reputation: client gets successful hire
                    client = next(c for c in self.clients if c.id == job.client_id)
                    simple_reputation_manager.update_client_after_post(
                        client.id,
                        was_filled=True,
                        job_budget=getattr(job, 'budget_amount', 50.0)
                    )
                    
                    # Update reputation: unsuccessful bidders don't get hired
                    for bid in job_bids:
                        if bid.freelancer_id != hired_freelancer.id:
                            simple_reputation_manager.update_freelancer_after_bid(
                                bid.freelancer_id,
                                was_hired=False,
                                job_category=getattr(job, 'category', None)
                            )
                    
                    # Provide feedback to unsuccessful bidders (only for LLM agents who can use it)
                    feedback_tasks = []
                    if self.freelancer_agent_type == 'llm':
                        for bid in job_bids:
                            if bid.freelancer_id != hired_freelancer.id:
                                if random.random() < 0.15:  # 15% chance of feedback (realistic)
                                    feedback_tasks.append((bid, job_bids, hiring_decision, job, round_num))
                                    print(f"    📝 Rare feedback queued for {bid.freelancer_id}")
                    
                    # Process feedback in parallel if any
                    if feedback_tasks:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(feedback_tasks))) as executor:
                            feedback_futures = [executor.submit(self._process_feedback_task, task) for task in feedback_tasks]
                            
                            for future in concurrent.futures.as_completed(feedback_futures):
                                try:
                                    future.result()  # Just wait for completion
                                except Exception as e:
                                    logger.exception(f"Error in feedback generation: {e}")
                    
                    # Add job to ongoing jobs with start round and duration
                    self._schedule_ongoing_job(hired_freelancer, job, round_num)
                else:
                    print(f"    ⚠️ {job.title}: Could not find hired freelancer {selected_id}")
            else:
                print(f"    ❌ {job.title}: No hire")
                
                # Update reputation: client didn't hire anyone  
                client = next(c for c in self.clients if c.id == job.client_id)
                simple_reputation_manager.update_client_after_post(
                    client.id,
                    was_filled=False,
                    job_budget=getattr(job, 'budget_amount', 50.0)
                )
                
                # Update reputation: all bidders didn't get hired
                for bid in job_bids:
                    simple_reputation_manager.update_freelancer_after_bid(
                        bid.freelancer_id,
                        was_hired=False,
                        job_category=getattr(job, 'category', None)
                    )
        
        # Also add "No bids received" outcomes for jobs without bids
        for job in self.active_jobs:
            if job.id not in jobs_with_bids:
                no_bid_outcome = HiringDecision(
                    job_id=job.id,
                    client_id=job.client_id,
                    selected_freelancer=None,
                    reasoning="No bids received",
                    timestamp=datetime.now()
                )
                self.hiring_outcomes.append(no_bid_outcome)
        
        elapsed_time = time.time() - start_time
        print(f"    ⚡ Hiring decisions completed in {elapsed_time:.1f}s ({len(hiring_tasks)} jobs with bids)")
        
        self.all_bids.extend(round_bids)
    
    def _process_hiring_decision_task(self, task_data) -> Dict:
        """Process a single hiring decision task in parallel."""
        client, job, job_bids = task_data
        
        # Safety check: ensure job is not None
        if job is None:
            logger.error("Received None job in hiring decision task - skipping")
            return None
        
        try:
            hiring_decision = self.client_hiring_decision(client, job, job_bids)
            return {
                'hiring_decision': hiring_decision,
                'job': job,
                'job_bids': job_bids
            }
        except Exception as e:
            logger.exception(f"Failed to make hiring decision for {job.title}: {e}")
            # Return a fallback decision
            return {
                'hiring_decision': HiringDecision(
                    job_id=job.id,
                    client_id=client.id,
                    selected_freelancer=None,
                    reasoning="Technical error in hiring decision",
                    timestamp=datetime.now()
                ),
                'job': job,
                'job_bids': job_bids
            }
    
    def _process_feedback_task(self, task_data) -> None:
        """Process a single feedback task in parallel."""
        bid, job_bids, hiring_decision, job, round_num = task_data
        return self._provide_bid_feedback(bid, job_bids, hiring_decision, job, round_num)
    
    def _provide_bid_feedback(self, bid: Bid, job_bids: List[Bid], hiring_decision: HiringDecision, job: Job, round_num: int) -> None:
        """Provide feedback to unsuccessful bidders."""
        # Generate feedback prompt with only the necessary data
        feedback_prompt = generate_feedback_prompt(job.to_dict(), hiring_decision.to_dict(), bid.to_dict())

        feedback_response = self.openai_client.chat.completions.create(
            model=self.llm_config.model,
            messages=[{"role": "user", "content": feedback_prompt}],
            response_format={"type": "json_object"}
        )
        
        content = feedback_response.choices[0].message.content
        validated_feedback = validate_feedback(content)
        
        # Store feedback in freelancer's history using entity method
        # PERFORMANCE: Use O(1) dictionary lookup instead of O(n) linear search
        unsuccessful_freelancer = self.freelancers_by_id.get(bid.freelancer_id)
        if unsuccessful_freelancer:
            if validated_feedback:
                # Only convert to dict when needed for storage
                feedback_dict = {
                    'main_reason': validated_feedback.main_reason,
                    'pitch_feedback': validated_feedback.pitch_feedback,
                    'advice': validated_feedback.advice
                }
            else:
                # Fallback if validation fails
                feedback_dict = {
                    'main_reason': 'Unable to provide specific feedback',
                    'pitch_feedback': 'Message was reviewed',
                    'advice': 'Continue improving your proposals and ensure they clearly demonstrate your understanding of the project requirements.'
                }
            
            unsuccessful_freelancer.add_feedback({
                'job_id': job.id,
                'round': round_num,
                'feedback': feedback_dict
            })

    def _schedule_ongoing_job(self, hired_freelancer: Freelancer, job: Job, round_num: int) -> None:
        """Schedule an ongoing job for a hired freelancer."""
        # Convert timeline to number of rounds (1 month = 1 round, targeting 4 rounds average)
        timeline = job.timeline.lower()
        
        # Handle various timeline formats
        if '-' in timeline:
            # Handle ranges like "3-6 months"
            parts = timeline.split('-')
            if len(parts) == 2:
                try:
                    start = int(''.join(filter(str.isdigit, parts[0])))
                    end = int(''.join(filter(str.isdigit, parts[1])))
                    duration = (start + end) // 2
                except ValueError:
                    duration = 4  # Default to 4 rounds
            else:
                duration = 4  # Default to 4 rounds
        else:
            try:
                duration = int(''.join(filter(str.isdigit, timeline)))
            except ValueError:
                duration = 4  # Default to 4 rounds
                
        if 'week' in timeline:
            duration_rounds = max(1, duration // 4)  # 4 weeks = 1 round
        elif 'month' in timeline:
            duration_rounds = max(1, duration)  # Each month = 1 round
        else:
            duration_rounds = 4  # Default to 4 rounds
        
        job_start = {
            'job_id': job.id,
            'start_round': round_num,
            'duration_rounds': duration_rounds
        }
        hired_freelancer.ongoing_jobs.append(job_start)
    
    def _create_reflections(self, round_num: int, round_decisions: List[Dict], new_jobs: List[Job], available_jobs: List[Job]) -> None:
        """Create reflection sessions for agents based on individual probability checks."""
        
        # Skip reflections if disabled
        if not self.enable_reflections:
            print("\n  ⚡ Skipping reflections (disabled for speed)")
            return
            
        print(f"\n  🤔 Agents considering reflections (round {round_num})...")
        
        # Process reflections in parallel for better performance
        start_time = time.time()
        round_reflections = []  # Collect all reflection sessions for this round
        
        # Prepare freelancer reflection tasks (probability-based)
        freelancer_tasks = []
        reflecting_freelancers = []
        for freelancer in self.freelancers:
            if random.random() < self.reflection_probability:
                recent_activities = self._get_freelancer_activities(freelancer, round_decisions, available_jobs, round_num)
                freelancer_tasks.append((freelancer, recent_activities, round_num))
                reflecting_freelancers.append(freelancer.name)
        
        # Process freelancer reflections in parallel
        if freelancer_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                freelancer_futures = [executor.submit(self._process_freelancer_reflection, task) for task in freelancer_tasks]
                
                for future in concurrent.futures.as_completed(freelancer_futures):
                    try:
                        result = future.result()
                        if result and result.get('insights'):
                            print(f"    💡 {result['name']}: {result['insights'][0]}")
                            if 'reflection_session' in result:
                                result['reflection_session']['round'] = round_num
                                round_reflections.append(result['reflection_session'])
                    except Exception as e:
                        logger.exception(f"Error in freelancer reflection: {e}")
        
        # Prepare client reflection tasks (probability-based)
        client_tasks = []
        reflecting_clients = []
        for client in self.clients:
            if random.random() < self.reflection_probability:
                recent_activities = self._get_client_activities(client, new_jobs)
                client_tasks.append((client, recent_activities, round_num))
                reflecting_clients.append(client.company_name)
        
        # Process client reflections in parallel
        if client_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                client_futures = [executor.submit(self._process_client_reflection, task) for task in client_tasks]
                
                for future in concurrent.futures.as_completed(client_futures):
                    try:
                        result = future.result()
                        if result and result.get('insights'):
                            print(f"    💡 {result['name']}: {result['insights'][0]}")
                            if 'reflection_session' in result:
                                result['reflection_session']['round'] = round_num
                                round_reflections.append(result['reflection_session'])
                    except Exception as e:
                        logger.exception(f"Error in client reflection: {e}")
        
        # Store all reflections from this round
        self.reflection_sessions.extend(round_reflections)
        
        elapsed_time = time.time() - start_time
        total_reflecting = len(freelancer_tasks) + len(client_tasks)
        probability_percent = self.reflection_probability * 100
        print(f"    ⚡ Reflections completed in {elapsed_time:.1f}s ({total_reflecting} agents reflected, {probability_percent:.0f}% probability)")
    
    def _update_unfilled_jobs_tracking(self, round_num: int) -> None:
        """Track how long jobs have been unfilled (for client reflection context)."""
        if not self.active_jobs:
            return
        
        # Simply track how long each job has been unfilled
        for job in self.active_jobs:
            job.rounds_unfilled += 1
    
    def _apply_specific_job_budget_adjustments(self, job_adjustments: List[Dict], round_num: int) -> List[Dict]:
        """Apply specific budget adjustments to individual jobs based on client reflection."""
        adjustment_results = []
        
        for adjustment in job_adjustments:
            job_id = adjustment.get('job_id')
            increase_percentage = adjustment.get('increase_percentage', 0)
            reasoning = adjustment.get('reasoning', 'Client-requested budget increase')
            
            # Find the job in active jobs
            job = next((j for j in self.active_jobs if j.id == job_id), None)
            if job and increase_percentage > 0:
                old_budget = job.budget_amount
                new_budget = old_budget * (1 + increase_percentage / 100)
                job.budget_amount = new_budget
                
                adjustment_results.append({
                    'job_id': job_id,
                    'old_budget': old_budget,
                    'new_budget': new_budget,
                    'increase_percentage': increase_percentage,
                    'reasoning': reasoning,
                    'round': round_num
                })
                
                print(f"    💰 Job {job_id} budget: ${old_budget:.0f} → ${new_budget:.0f} (+{increase_percentage}%)")
        
        return adjustment_results
    
    def _get_freelancer_activities(self, freelancer: Freelancer, round_decisions: List[Dict], available_jobs: List[Job], round_num: int) -> List[Dict]:
        """Get recent activities for a freelancer."""
        recent_activities = []
        
        # Add bid decisions
        recent_bids = [d for d in round_decisions if d['freelancer_id'] == freelancer.id]
        recent_activities.extend([{
            'type': 'bid',
            'job_id': bid['job_id'],
            'outcome': 'success' if bid['decision'] == 'yes' else 'pass',
            'reasoning': bid.get('reasoning', ''),
            'earnings': 0  # Earnings tracked separately after job completion
        } for bid in recent_bids])
        
        # Add job opportunities seen (get reasoning from all_decisions)
        for job in available_jobs:
            if (freelancer.id, job.id) in self.freelancer_job_decisions:
                # Find the decision record with reasoning
                # PERFORMANCE: Skip expensive search through all_decisions 
                # This data is already available in freelancer_job_decisions
                decision_record = None
                decision_info = self.freelancer_job_decisions.get((freelancer.id, job.id), {})
                # Handle both old string format and new dict format
                if isinstance(decision_info, str):
                    response = decision_info
                else:
                    response = decision_info.get('decision', 'no')
                reasoning = decision_record.get('reasoning', '') if decision_record else ''
                
                recent_activities.append({
                    'type': 'bid_opportunity',
                    'job_id': job.id,
                    'response': response,
                    'reasoning': reasoning
                })
        
        return recent_activities
    
    def _get_client_activities(self, client, new_jobs: List[Job]) -> List[Dict]:
        """Get recent activities for a client."""
        recent_activities = []
        
        # Add job postings
        client_jobs = [j for j in new_jobs if j.client_id == client.id]
        recent_activities.extend([{
            'type': 'job_post',
            'job_id': job.id,
            'outcome': 'filled' if any(h.job_id == job.id and h.selected_freelancer 
                                    for h in self.hiring_outcomes) else 'unfilled'
        } for job in client_jobs])
        
        # Add hiring decisions
        client_decisions = [h for h in self.hiring_outcomes if h.client_id == client.id]
        
        for decision in client_decisions:
            recent_activities.append({
                'type': 'hiring_decision',
                'job_id': decision.job_id,
                'outcome': 'hire' if decision.selected_freelancer else 'no_hire',
                'reasoning': decision.reasoning
            })
        
        # Add unfilled jobs with their current status
        client_unfilled_jobs = [job for job in self.active_jobs if job.client_id == client.id]
        for job in client_unfilled_jobs:
            recent_activities.append({
                'type': 'unfilled_job',
                'job_id': job.id,
                'title': job.title,
                'current_budget': job.budget_amount,
                'rounds_unfilled': job.rounds_unfilled,
                'timeline': job.timeline,
                'category': job.category
            })
        
        return recent_activities
    
    def _process_freelancer_reflection(self, task_data) -> Dict:
        """Process a single freelancer reflection in parallel."""
        freelancer, recent_activities, round_num = task_data
        
        try:
            performance_summary = f"Made {len([a for a in recent_activities if a['type'] == 'bid'])} bids in round {round_num}"
            
            reflection_prompt = create_freelancer_reflection_prompt(
                freelancer.to_dict(),
                recent_activities,
                performance_summary
            )
            
            response = self._make_openai_request(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": reflection_prompt}],
                max_tokens=500,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            reflection_text = response.choices[0].message.content
            
            # Validate reflection using Pydantic
            validated_reflection = validate_reflection(reflection_text)
            if validated_reflection:
                # Use validated object directly
                key_insights = validated_reflection.key_insights
                planned_adjustments = validated_reflection.strategy_adjustments
                
                # Process rate adjustment if present
                rate_adjustment_result = None
                if validated_reflection.rate_adjustment:
                    rate_adjustment_result = reflection_adjustment_engine.apply_freelancer_rate_adjustment(
                        freelancer, validated_reflection.rate_adjustment, round_num
                    )
            else:
                # Fallback if validation fails
                key_insights = ["Reflected on recent bidding performance"]
                planned_adjustments = ["Will continue monitoring market opportunities"]
                rate_adjustment_result = None
            
            # Create simplified reflection session
            reflection = reflection_manager.create_reflection_session(
                agent_id=freelancer.id,
                agent_type='freelancer',
                recent_activities=recent_activities
            )
            
            # Update with GPT insights
            reflection.key_insights = key_insights
            reflection.planned_adjustments = planned_adjustments
            
            result = {
                'name': freelancer.name,
                'insights': key_insights,
                'adjustments': planned_adjustments,
                'reflection_session': {
                    'agent_id': freelancer.id,
                    'agent_type': 'freelancer',
                    'timestamp': str(reflection.timestamp),
                    'performance_summary': reflection.performance_summary,
                    'key_insights': key_insights,
                    'planned_adjustments': planned_adjustments
                }
            }
            
            if rate_adjustment_result:
                result['rate_adjustment'] = rate_adjustment_result
                result['reflection_session']['rate_adjustment'] = rate_adjustment_result
            
            return result
            
        except Exception as e:
            logger.exception(f"Failed to create reflection for {freelancer.name}: {e}")
            # Fallback to simple reflection
            reflection = reflection_manager.create_reflection_session(
                agent_id=freelancer.id,
                agent_type='freelancer',
                recent_activities=recent_activities
            )
            return {'name': freelancer.name, 'insights': [], 'adjustments': []}
    
    def _process_client_reflection(self, task_data) -> Dict:
        """Process a single client reflection in parallel."""
        client, recent_activities, round_num = task_data
        
        try:
            client_jobs = [a for a in recent_activities if a['type'] == 'job_post']
            performance_summary = f"Posted {len(client_jobs)} jobs in round {round_num}"
            
            reflection_prompt = create_client_reflection_prompt(
                client.to_dict(),
                recent_activities,
                performance_summary
            )
            
            response = self._make_openai_request(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": reflection_prompt}],
                max_tokens=500,
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            reflection_text = response.choices[0].message.content
            
            # Validate reflection using Pydantic
            validated_reflection = validate_reflection(reflection_text)
            if validated_reflection:
                # Use validated object directly
                key_insights = validated_reflection.key_insights
                planned_adjustments = validated_reflection.strategy_adjustments
                
                # Process budget adjustment if present
                budget_adjustment_result = None
                if validated_reflection.budget_adjustment:
                    budget_adjustment_result = reflection_adjustment_engine.apply_client_budget_adjustment(
                        client, validated_reflection.budget_adjustment, round_num
                    )
                
                # Process specific job budget adjustments if present
                job_budget_results = []
                if validated_reflection.job_budget_adjustments:
                    job_budget_results = self._apply_specific_job_budget_adjustments(
                        validated_reflection.job_budget_adjustments, round_num
                    )
            else:
                # Fallback if validation fails
                key_insights = ["Reflected on recent hiring performance"]
                planned_adjustments = ["Will continue improving job posting strategy"]
                budget_adjustment_result = None
                job_budget_results = []
            
            # Create simplified reflection session
            reflection = reflection_manager.create_reflection_session(
                agent_id=client.id,
                agent_type='client',
                recent_activities=recent_activities
            )
            
            # Update with GPT insights
            reflection.key_insights = key_insights
            reflection.planned_adjustments = planned_adjustments
            
            result = {
                'name': client.company_name,
                'insights': key_insights,
                'adjustments': planned_adjustments,
                'reflection_session': {
                    'agent_id': client.id,
                    'agent_type': 'client',
                    'timestamp': str(reflection.timestamp),
                    'performance_summary': reflection.performance_summary,
                    'key_insights': key_insights,
                    'planned_adjustments': planned_adjustments
                }
            }
            
            if budget_adjustment_result:
                result['budget_adjustment'] = budget_adjustment_result
                result['reflection_session']['budget_adjustment'] = budget_adjustment_result
            
            if job_budget_results:
                result['job_budget_adjustments'] = job_budget_results
                result['reflection_session']['job_budget_adjustments'] = job_budget_results
            
            return result
            
        except Exception as e:
            logger.exception(f"Failed to create reflection for {client.company_name}: {e}")
            # Fallback to simple reflection
            reflection = reflection_manager.create_reflection_session(
                agent_id=client.id,
                agent_type='client',
                recent_activities=recent_activities
            )
            return {'name': client.company_name, 'insights': [], 'adjustments': []}
    

    def _process_freelancer_bidding_task(self, task_data) -> Dict:
        """Process bidding decisions for a single freelancer in parallel."""
        freelancer, available_jobs, round_num = task_data
        
        task_bids = []
        task_decisions = []
        task_job_decisions = {}
        
        # Parallelize the freelancer's job decisions for faster processing
        # Use a fraction of max_workers since we already have freelancer-level parallelization
        job_workers = min(max(1, self.max_workers // 4), len(available_jobs))
        job_decision_futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=job_workers) as executor:
            for job in available_jobs:
                future = executor.submit(self.freelancer_bidding_decision, freelancer, job, round_num)
                job_decision_futures.append((future, job))
        
        # Process completed decisions
        for future, job in job_decision_futures:
            try:
                decision = future.result()
                
                # Always store the decision (yes or no) for analysis
                decision_record = {
                    'freelancer_id': freelancer.id,
                    'freelancer_name': freelancer.name,
                    'job_id': job.id,
                    'job_title': job.title,
                    'decision': decision.get('decision'),
                    'reasoning': decision.get('reasoning'),
                    'round': round_num,
                    'submission_time': datetime.now().isoformat()
                }
                
                # Remember this decision with round info for cooloff system
                decision_key = (freelancer.id, job.id)
                task_job_decisions[decision_key] = {
                    'decision': decision.get('decision'),
                    'round': round_num
                }
                
                if decision.get('decision') == 'yes':
                    # Create a Bid entity for successful decisions
                    bid = Bid(
                        freelancer_id=freelancer.id,
                        job_id=job.id,
                        proposed_rate=job.budget_amount,  # Use job budget as the rate
                        message=decision.get('message', ''),
                        submission_time=datetime.now()
                    )
                    
                    # Add bid-specific fields for decision tracking
                    decision_record.update({
                        'message': decision.get('message')
                    })
                    task_bids.append(bid)
                    freelancer.add_bid()  # Use entity method to track bids
                    # Only log individual bids in non-quiet mode
                    if not self.quiet_mode:
                        print(f"    💰 {freelancer.name} used their bid on {job.title}")
                
                task_decisions.append(decision_record)
                
                # Check if freelancer has reached their bidding limit
                if freelancer.bids_this_round >= self.bids_per_round:
                    print(f"    🚫 {freelancer.name} reached bid limit ({self.bids_per_round})")
                    # Process any remaining decisions without bidding
                    continue
                
            except Exception as e:
                logger.exception(f"Error processing job decision for {freelancer.name}: {e}")
                # Add failed decision for tracking
                decision_record = {
                    'freelancer_id': freelancer.id,
                    'freelancer_name': freelancer.name,
                    'job_id': job.id,
                    'job_title': job.title,
                    'decision': 'no',
                    'reasoning': 'Technical error in decision making',
                    'round': round_num,
                    'submission_time': datetime.now().isoformat()
                }
                task_decisions.append(decision_record)
        
        return {
            'bids': task_bids,
            'decisions': task_decisions,
            'job_decisions': task_job_decisions,
            'freelancer_name': freelancer.name
        }
    
    def _record_round_data(self, round_num: int, new_jobs: List[Job], round_bids: List[Dict]) -> None:
        """Record comprehensive round statistics for saturation analysis."""
        
        # Basic metrics
        jobs_filled_this_round = len([h for h in self.hiring_outcomes
                                     if h.selected_freelancer and
                                     any(j.id == h.job_id for j in new_jobs)])
        
        # Calculate bid rejection metrics
        total_bids_this_round = len(round_bids)
        successful_bids_this_round = jobs_filled_this_round  # Each job hires at most 1 freelancer
        rejected_bids_this_round = total_bids_this_round - successful_bids_this_round
        bid_rejection_rate = rejected_bids_this_round / total_bids_this_round if total_bids_this_round > 0 else 0
        
        # Calculate bid distribution metrics
        job_bid_counts = {}
        for bid in round_bids:
            job_id = bid.job_id if hasattr(bid, 'job_id') else bid.get('job_id')
            job_bid_counts[job_id] = job_bid_counts.get(job_id, 0) + 1
        
        bid_volumes = list(job_bid_counts.values()) if job_bid_counts else [0]
        
        # Freelancer activity metrics  
        active_bidders = len(set(bid.freelancer_id if hasattr(bid, 'freelancer_id') else bid.get('freelancer_id') 
                               for bid in round_bids))
        
        # Client activity metrics  
        active_clients = len(set(job.client_id for job in new_jobs))
        
        # Calculate client engagement metrics
        client_posting_frequency = self._calculate_client_posting_frequency(round_num)
        clients_on_cooldown = len([c for c in self.clients if not c.can_post_job(round_num)])
        
        # Market efficiency metrics
        jobs_with_no_bids = len(new_jobs) - len(job_bid_counts)
        jobs_with_excess_bids = sum(1 for count in bid_volumes if count > 5)  # Threshold for "excess"
        
        # Fatigue analysis removed for performance - can be re-added later if needed
        fatigue_signals = {
            'high_rejection_rate_freelancers': 0,
            'declining_participation': 0,
            'bid_selectivity_increase': 0
        }  # Store for saturation risk calculation
        
        round_data = {
            'round': round_num,
            'jobs_posted': len(new_jobs),
            'total_bids': len(round_bids),
            'jobs_filled': len([h for h in self.hiring_outcomes if h.selected_freelancer]),
            'jobs_filled_this_round': jobs_filled_this_round,
            
            # Bid rejection metrics
            'bid_rejection_metrics': {
                'total_bids_this_round': total_bids_this_round,
                'successful_bids_this_round': successful_bids_this_round,
                'rejected_bids_this_round': rejected_bids_this_round,
                'bid_rejection_rate': bid_rejection_rate,
                'bid_success_rate': 1 - bid_rejection_rate
            },
            
            # Saturation metrics
            'bid_distribution': {
                'min_bids_per_job': min(bid_volumes),
                'max_bids_per_job': max(bid_volumes),
                'avg_bids_per_job': sum(bid_volumes) / len(bid_volumes) if bid_volumes else 0,
                'jobs_with_no_bids': jobs_with_no_bids,
                'jobs_with_excess_bids': jobs_with_excess_bids
            },
            
            # Activity metrics
            'market_activity': {
                'active_bidders': active_bidders,
                'active_clients': active_clients,
                'clients_on_cooldown': clients_on_cooldown,
                'client_posting_frequency': client_posting_frequency,
                'freelancer_participation_rate': active_bidders / len(self.freelancers),
                'client_activity_rate': active_clients / len(self.clients)
            },
            
            # Fatigue and sentiment indicators
            'fatigue_indicators': fatigue_signals,
            
            # Market health indicators
            'market_health': {
                'supply_demand_ratio': len(self.freelancers) / len(self.active_jobs) if len(self.active_jobs) > 0 else float('inf'),
                'competition_intensity': sum(bid_volumes) / len(new_jobs) if len(new_jobs) > 0 else 0,
                'market_saturation_risk': self._calculate_saturation_risk(round_num),
                'health_score': calculate_market_health_score(self.round_data) if self.round_data else 0.0,
                'health_grade': calculate_health_grade(calculate_market_health_score(self.round_data) if self.round_data else 0.0),
                'competitiveness_level': calculate_competitiveness_level(sum(bid_volumes) / len(new_jobs) if len(new_jobs) > 0 else 0)
            },
            
            # Outcome diversity metrics
            'outcome_diversity': {
                'freelancers_with_work': len([f for f in self.freelancers if f.active_jobs > 0]),
                'total_freelancers': len(self.freelancers),
                'work_distribution_gini': calculate_work_distribution_gini([f.active_jobs for f in self.freelancers]),
                'participation_rate': active_bidders / len(self.freelancers)
            },
            
            # Reputation snapshots for this round
            'reputation_snapshot': {
                'freelancers': {f.id: self._serialize_reputation(simple_reputation_manager.get_freelancer_reputation(f.id)) for f in self.freelancers},
                'clients': {c.id: self._serialize_reputation(simple_reputation_manager.get_client_reputation(c.id)) for c in self.clients}
            }
        }
        
        self.round_data.append(round_data)
    
    # Fatigue analysis methods removed for performance optimization
    # Can be re-added later if needed for research purposes
    
    def _calculate_client_posting_frequency(self, round_num: int) -> Dict:
        """Calculate client posting frequency metrics."""
        frequency_metrics = {
            'average_posts_per_client': 0.0,
            'most_active_client_posts': 0,
            'least_active_client_posts': 0,
            'posting_distribution': {}
        }
        
        if round_num > 0:
            # PERFORMANCE: Count posts per client without creating filtered lists
            client_post_counts = {}
            for client in self.clients:
                client_post_counts[client.id] = 0
            
            # Single pass through all jobs to count per client
            for job in self.all_jobs:
                if job.client_id in client_post_counts:
                    client_post_counts[job.client_id] += 1
            
            post_counts = list(client_post_counts.values())
            if post_counts:
                frequency_metrics.update({
                    'average_posts_per_client': sum(post_counts) / len(post_counts),
                    'most_active_client_posts': max(post_counts),
                    'least_active_client_posts': min(post_counts),
                    'posting_distribution': client_post_counts
                })
        
        return frequency_metrics
    
    def _calculate_saturation_risk(self, round_num: int) -> float:
        """Calculate overall market saturation risk score (0.0 to 1.0)."""
        risk_factors = []
        
        # Factor 1: Supply-demand imbalance
        if len(self.active_jobs) > 0:
            supply_demand_ratio = len(self.freelancers) / len(self.active_jobs)
            supply_risk = min(1.0, (supply_demand_ratio - 2.0) / 3.0)  # Risk increases after 2:1 ratio
            risk_factors.append(max(0.0, supply_risk))
        
        # Factor 2: Bid concentration (too many bids per job)
        if self.round_data:
            recent_round = self.round_data[-1]
            avg_bids = recent_round.get('bid_distribution', {}).get('avg_bids_per_job', 0)
            bid_concentration_risk = min(1.0, (avg_bids - 3.0) / 7.0)  # Risk after 3 bids/job
            risk_factors.append(max(0.0, bid_concentration_risk))
        
        # Factor 3: Declining fill rate
        if len(self.round_data) >= 3:
            recent_fill_rates = [r.get('jobs_filled_this_round', 0) / max(r.get('jobs_posted', 1), 1) 
                               for r in self.round_data[-3:]]
            if len(recent_fill_rates) >= 2:
                fill_rate_trend = recent_fill_rates[-1] - recent_fill_rates[0]
                declining_fill_risk = max(0.0, -fill_rate_trend)  # Negative trend = risk
                risk_factors.append(min(1.0, declining_fill_risk * 2))
        
        # Factor 4: High fatigue signals
        if hasattr(self, '_last_fatigue_signals'):
            fatigue_score = (
                self._last_fatigue_signals.get('high_rejection_rate_freelancers', 0) / len(self.freelancers) +
                self._last_fatigue_signals.get('declining_participation', 0) +
                self._last_fatigue_signals.get('bid_selectivity_increase', 0)
            ) / 3.0
            risk_factors.append(fatigue_score)
        
        # Overall risk is the average of all factors
        return sum(risk_factors) / len(risk_factors) if risk_factors else 0.0
    

    
    def _save_simulation_results(self, intermediate: bool = False, final: bool = False, round_completed: int = None) -> Dict:
        """Save simulation results to file, overwriting the same file for intermediate saves."""
        
        # Compile current results
        results = {
            'simulation_config': {
                # Core parameters
                'freelancers': self.num_freelancers,
                'clients': self.num_clients,
                'rounds': self.rounds,
                
                # Bidding mechanics
                'bids_per_round': self.bids_per_round,
                'jobs_per_freelancer_per_round': self.jobs_per_freelancer_per_round,
                'bid_cooloff_rounds': self.bid_cooloff_rounds,
                
                # Job posting mechanics
                'job_posting_cooldown_min': self.job_posting_cooldown_min,
                'job_posting_cooldown_max': self.job_posting_cooldown_max,
                
                # Job selection and matching
                'job_selection_method': self.job_selection_method,
                'relevance_mode': self.relevance_mode,
                
                # Agent behavior
                'reflection_probability': self.reflection_probability,
                'enable_reflections': self.enable_reflections,
                
                # System configuration
                'use_cache': self.cache is not None,
                'max_workers': self.max_workers,
                'max_active_jobs': self.max_active_jobs,
                
                # Baseline agent configuration
                'freelancer_agent_type': self.freelancer_agent_type,
                'client_agent_type': self.client_agent_type,
                'baseline_greedy_undercut': self.baseline_greedy_undercut,
                'random_freelancer_bid_probability': self.random_freelancer_bid_probability,
                
                # Execution metadata
                'timestamp': self.simulation_timestamp,
                'random_seed': getattr(self, 'random_seed', None),
                'type': f'{self.freelancer_agent_type}_{self.client_agent_type}_powered',
                'status': 'completed' if final else f'in_progress_round_{round_completed}'
            },
            'freelancer_profiles': {f.id: f.to_dict() for f in self.freelancers},
            'client_profiles': {c.id: c.to_dict() for c in self.clients},
            'all_jobs': [job.to_dict() for job in self.all_jobs],
            'all_bids': [bid.to_dict() for bid in self.all_bids],
            'all_decisions': self.all_decisions,  # Include ALL freelancer decisions
            'hiring_outcomes': [outcome.to_dict() if hasattr(outcome, 'to_dict') else outcome for outcome in self.hiring_outcomes],
            'round_data': self.round_data,
            'reflection_sessions': self.reflection_sessions,  # Include all reflection sessions
            'reputation_data': {
                'freelancers': {f.id: self._serialize_reputation(simple_reputation_manager.get_freelancer_reputation(f.id)) for f in self.freelancers},
                'clients': {c.id: self._serialize_reputation(simple_reputation_manager.get_client_reputation(c.id)) for c in self.clients}
            }
        }
        
        # Save to the same file (overwrite)
        with open(self.results_filename, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        if intermediate:
            print(f"  💾 Intermediate results saved: Round {round_completed} complete")
        elif final:
            print(f"\n💾 Final results saved: {self.results_filename}")
            
            # Final analysis
            total_jobs = len(self.all_jobs)
            total_bids = len(self.all_bids)
            filled_jobs = len([h for h in self.hiring_outcomes if h.selected_freelancer])
            
            print("\n📊 FINAL RESULTS:")
            print(f"  Jobs posted: {total_jobs}")
            print(f"  Bids submitted: {total_bids}")
            print(f"  Jobs filled: {filled_jobs}")
            print(f"  Fill rate: {filled_jobs/total_jobs*100:.1f}%" if total_jobs > 0 else "  Fill rate: N/A")
            print(f"  Bids per job: {total_bids/total_jobs:.2f}" if total_jobs > 0 else "  Bids per job: N/A")
            
            print("\n🎯 SUCCESS: True emergent behavior simulation complete!")
            
            # Note: Direct OpenAI client (no API response caching for better speed)
        
        return results

    def run_simulation(self) -> Dict:
        """Run the complete GPT-powered simulation"""
        # Initialize token tracker for this simulation
        self.token_tracker = get_token_tracker()
        
        # Initialize the simulation results file
        self.simulation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        Path("results/simuleval").mkdir(parents=True, exist_ok=True)
        self.results_filename = f"results/simuleval/true_gpt_simulation_{self.simulation_timestamp}.json"
        
        print("🚀 STARTING TRUE GPT-POWERED MARKETPLACE SIMULATION")
        print("=" * 60)
        print(f"📁 Results will be saved to: {self.results_filename}")
        
        # 1. Generate personas using GPT
        print("1️⃣ GPT generating freelancer personas...")
        self.freelancers = self.generate_freelancer_personas()
        
        # PERFORMANCE: Populate freelancer lookup dictionary for O(1) access
        for freelancer in self.freelancers:
            self.freelancers_by_id[freelancer.id] = freelancer
        
        print("2️⃣ GPT generating client personas...")
        self.clients = self.generate_client_personas()
        
        # Show what GPT created
        print("\n👥 GPT CREATED FREELANCERS:")
        for f in self.freelancers:
            print(f"  {f.name} - {', '.join(f.skills[:2])} - ${f.min_hourly_rate}/hr")
        
        print("\n🏢 GPT CREATED CLIENTS:")
        for c in self.clients:
            print(f"  {c.company_name} - {c.budget_philosophy}")
        
        # Save initial state (personas created)
        self._save_simulation_results(intermediate=True, round_completed=0)
        
        # 2. Run marketplace rounds
        print(f"\n3️⃣ Running {self.rounds} marketplace rounds...")
        
        for round_num in range(1, self.rounds + 1):
            print(f"\n--- Round {round_num} ---")
            
            # Update current round tracker
            self.current_round = round_num
            
            # Step 1: Process completed jobs and reset freelancer bid budgets
            self._process_completed_jobs(round_num)
            
            # Step 1.5: Track unfilled jobs duration  
            self._update_unfilled_jobs_tracking(round_num)
            
            # Step 2: Generate new job postings from clients
            new_jobs = self._generate_new_jobs(round_num)
            
            # Step 3: Process freelancer bidding decisions
            round_bids, round_decisions = self._process_freelancer_bidding(round_num)
            
            # Add round decisions to all_decisions for basic tracking only
            self.all_decisions.extend(round_decisions)
            
            # Step 4: Process client hiring decisions
            self._process_hiring_decisions(round_num, round_bids)
            
            # Step 5: Create reflection sessions for all agents
            available_jobs = self.active_jobs  # All jobs currently active
            self._create_reflections(round_num, round_decisions, new_jobs, available_jobs)
            
            # Step 6: Record round statistics
            self._record_round_data(round_num, new_jobs, round_bids)
            
            # Step 7: Save intermediate results periodically (not every round for performance)
            if not self.quiet_mode or round_num % 25 == 0 or round_num == self.rounds:
                # Save every 25 rounds in normal mode, or every 25 rounds + final in quiet mode
                self._save_simulation_results(intermediate=True, round_completed=round_num)
        
        # Save final results using the same filename pattern established during intermediate saves
        return self._save_simulation_results(final=True)
    
    def _fallback_freelancers(self) -> List[Dict]:
        """Minimal fallback if GPT fails"""
        return [
            {
                'id': 'freelancer_1',
                'name': 'Alex Chen',
                'skills': ['data science'],
                'min_hourly_rate': 25,
                'personality': 'Analytical',
                'motivation': 'Building portfolio',
                'background': 'Junior data scientist',
                'total_bids': 0,
                'total_hired': 0,
                'bids_this_round': 0
            }
        ]
    
    def _fallback_clients(self) -> List[Client]:
        """Minimal fallback if GPT fails"""
        client_data = {
                'id': 'client_1',
                'company_name': 'TechStartup',
                'company_size': 'startup',
                'budget_philosophy': 'value-focused',
                'hiring_style': 'quick decisions',
                'background': 'Growing software development company focused on web and mobile applications'
            }
        
        client_obj = Client(
            id=client_data['id'],
            company_name=client_data['company_name'],
            company_size=client_data['company_size'],
            budget_philosophy=client_data['budget_philosophy'],
            hiring_style=client_data['hiring_style'],
            background=client_data['background'],
            next_job_round=0
        )
        
        return [client_obj]
    
    def _fallback_job(self, client: Dict, round_num: int) -> Job:
        """Fallback job based on client profile if GPT fails"""
        # Get category-specific skills
        category = JobCategory.SOFTWARE.value[0]  # Default category for fallback jobs
        skills = {
            JobCategory.SOFTWARE.value[0]: ['software development', 'web development', 'mobile development'],
            JobCategory.DESIGN.value[0]: ['graphic design', 'UI/UX design', 'branding'],
            JobCategory.DATA_SCIENCE.value[0]: ['data analysis', 'machine learning', 'statistics'],
            JobCategory.WRITING.value[0]: ['content writing', 'copywriting', 'editing'],
            JobCategory.ACCOUNTING.value[0]: ['bookkeeping', 'financial analysis', 'tax preparation'],
            JobCategory.ADMIN.value[0]: ['data entry', 'email management', 'virtual assistance'],
            JobCategory.CUSTOMER_SERVICE.value[0]: ['customer support', 'help desk', 'technical support'],
            JobCategory.ENGINEERING.value[0]: ['CAD design', 'structural engineering', 'technical drawing'],
            JobCategory.IT.value[0]: ['network administration', 'system administration', 'cloud infrastructure'],
            JobCategory.LEGAL.value[0]: ['legal writing', 'contract law', 'compliance'],
            JobCategory.SALES.value[0]: ['digital marketing', 'social media', 'content marketing'],
            JobCategory.TRANSLATION.value[0]: ['translation', 'localization', 'proofreading']
        }.get(category, ['project management', 'consulting', 'research'])
        
        # Set budget based on philosophy and category
        base_rate = next((cat.value[1] for cat in JobCategory if cat.value[0] == category), 50)
        philosophy = client.budget_philosophy.lower()
        if 'premium' in philosophy:
            budget = base_rate * 1.3  # 30% above average
        elif 'value' in philosophy:
            budget = base_rate * 0.9  # 10% below average
        else:  # balanced
            budget = base_rate
            
        # Create Job object
        return Job(
            id=f"job_{round_num}_{client.id}",
            client_id=client.id,
            title=f"{category} {random.choice(['Specialist', 'Expert', 'Professional'])} Needed",
            description=f"Looking for a skilled professional in {category}. Must have experience in {', '.join(skills[:2])}.",
            skills_required=skills[:3],  # Take top 3 skills
            budget_type='fixed' if random.random() < 0.6 else 'hourly',  # 60% chance of fixed
            budget_amount=budget,
            timeline=random.choice(['2 weeks', '1 month', '3 months']),
            special_requirements=[f"Experience in {category} required."],
            category=category,
            posted_time=datetime.now()
        )
    
    def _serialize_reputation(self, reputation) -> Dict:
        """Serialize reputation object to JSON-compatible dict"""
        if reputation is None:
            return None
        
        # Convert dataclass to dict
        result = reputation.__dict__.copy()
        
        # Handle enum serialization
        if 'tier' in result and hasattr(result['tier'], 'value'):
            result['tier'] = result['tier'].value
        
        # Handle datetime serialization
        for key, value in result.items():
            if hasattr(value, 'isoformat'):  # datetime object
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                # Ensure all dict keys are strings (for category_expertise etc.)
                result[key] = {str(k): v for k, v in value.items()}
        
        return result

def parse_arguments():
    """Parse command line arguments for flexible experimentation"""
    parser = argparse.ArgumentParser(
        description="True GPT-Powered Marketplace Simulation - Study emergent economic behavior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--freelancers', '-f', type=int, default=5,
        help='Number of freelancer agents to create'
    )
    
    parser.add_argument(
        '--clients', '-c', type=int, default=3,
        help='Number of client companies to create'
    )
    
    parser.add_argument(
        '--rounds', '-r', type=int, default=6,
        help='Number of marketplace rounds to simulate'
    )
    
    parser.add_argument(
        '--bids-per-round', '-b', type=int, default=1,
        help='Number of bids each freelancer can make per round (controls strategic scarcity)'
    )
    
    parser.add_argument(
        '--jobs-per-freelancer', '-j', type=int, default=3,
        help='Maximum number of jobs shown to each freelancer per round (realistic attention limit)'
    )
    
    parser.add_argument(
        '--job-selection', choices=['random', 'relevance'], default='random',
        help='How to select jobs shown to freelancers: random sampling or relevance-based ranking'
    )
    
    parser.add_argument(
        '--relevance-mode', choices=['strict', 'moderate', 'relaxed'], default='strict',
        help='Relevance algorithm strictness when using relevance-based job selection'
    )
    
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Disable caching - force fresh persona generation'
    )
    
    parser.add_argument(
        '--scenario', choices=['scarcity', 'moderate', 'abundant'], 
        help='Predefined scenarios: scarcity (1 bid), moderate (2 bids), abundant (5 bids)'
    )
    
    # Performance optimization arguments
    parser.add_argument(
        '--reflection-probability', type=float, default=0.1,
        help='Probability (0.0-1.0) that each agent reflects per round (0.1=10%% chance per agent per round)'
    )
    
    parser.add_argument(
        '--disable-reflections', action='store_true',
        help='Disable agent reflections entirely for maximum speed'
    )
    
    parser.add_argument(
        '--bid-cooloff-rounds', type=int, default=5,
        help='Number of rounds before freelancer can re-bid on jobs they previously declined (0=disable cooloff)'
    )
    
    parser.add_argument(
        '--job-posting-cooldown-min', type=int, default=3,
        help='Minimum number of rounds before client can post another job'
    )
    
    parser.add_argument(
        '--job-posting-cooldown-max', type=int, default=10,
        help='Maximum number of rounds before client can post another job'
    )
    
    parser.add_argument(
        '--max-workers', type=int, default=15,
        help='Maximum parallel workers for API calls (higher = faster but more resource intensive)'
    )
    
    parser.add_argument(
        '--max-active-jobs', type=int, default=3,
        help='Maximum number of concurrent jobs per freelancer (affects reputation progression speed)'
    )
    
    parser.add_argument(
        '--quiet', action='store_true',
        help='Enable quiet mode - reduces logging and file I/O for maximum performance'
    )
    
    # Baseline agent configuration arguments
    parser.add_argument(
        '--freelancer-agent-type', choices=['llm', 'random', 'greedy', 'no_reputation'], default='llm',
        help='Type of freelancer agent: llm (default), random (50%% bid chance), greedy (highest budget jobs), no_reputation (greedy without reputation)'
    )
    
    parser.add_argument(
        '--client-agent-type', choices=['llm', 'random', 'greedy', 'no_reputation'], default='llm',
        help='Type of client agent: llm (default), random (random bid selection), greedy (lowest bid), no_reputation (greedy without reputation)'
    )
    
    parser.add_argument(
        '--baseline-greedy-undercut', type=float, default=0.9,
        help='Undercut percentage for greedy agents (0.9 = 90%% of job budget). Must be between 0.1 and 1.0'
    )
    
    parser.add_argument(
        '--random-freelancer-bid-probability', type=float, default=0.5,
        help='Probability for random freelancers to bid on jobs (0.5 = 50%% chance). Must be between 0.0 and 1.0'
    )
    
    # Pre-configured baseline scenarios
    parser.add_argument(
        '--baseline-scenario', choices=['random', 'greedy', 'no_reputation', 'llm_no_reputation', 'llm_no_reflection'], 
        help='Pre-configured baseline scenarios: random (pure noise), greedy (rational heuristic), no_reputation (greedy ignoring reputation), llm_no_reputation (LLM without reputation), llm_no_reflection (LLM without reflection)'
    )
    
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducible experiments (default: None for random seed)'
    )

    
    return parser.parse_args()

def main():
    """Run the true GPT-powered simulation with configurable parameters"""
    args = parse_arguments()
    
    # Set random seed for reproducibility
    if args.seed is not None:
        import random
        import numpy as np
        random.seed(args.seed)
        try:
            np.random.seed(args.seed)
        except ImportError:
            pass  # numpy not available, skip
        print(f"🎲 Random seed set to: {args.seed}")
    
    # Handle predefined scenarios
    if args.scenario:
        scenario_configs = {
            'scarcity': {'bids_per_round': 1},
            'moderate': {'bids_per_round': 2}, 
            'abundant': {'bids_per_round': 5}
        }
        bids_per_round = scenario_configs[args.scenario]['bids_per_round']
        print(f"🎭 Running '{args.scenario}' scenario")
    else:
        bids_per_round = args.bids_per_round
    
    # Handle baseline scenarios
    freelancer_agent_type = args.freelancer_agent_type
    client_agent_type = args.client_agent_type
    baseline_greedy_undercut = args.baseline_greedy_undercut
    random_freelancer_bid_probability = args.random_freelancer_bid_probability
    reflection_probability = args.reflection_probability
    enable_reflections = not args.disable_reflections
    
    if args.baseline_scenario:
        baseline_configs = {
            'random': {
                'freelancer_agent_type': 'random',
                'client_agent_type': 'random',
                'enable_reflections': False
            },
            'greedy': {
                'freelancer_agent_type': 'greedy',
                'client_agent_type': 'greedy',
                'enable_reflections': False
            },
            'no_reputation': {
                'freelancer_agent_type': 'no_reputation',
                'client_agent_type': 'no_reputation',
                'enable_reflections': False
            },
            'llm_no_reputation': {
                'freelancer_agent_type': 'llm',
                'client_agent_type': 'llm',
                'enable_reflections': True,
                # Note: We'll disable reputation usage in prompts for this scenario
            },
            'llm_no_reflection': {
                'freelancer_agent_type': 'llm',
                'client_agent_type': 'llm', 
                'enable_reflections': False
            }
        }
        
        config = baseline_configs[args.baseline_scenario]
        freelancer_agent_type = config['freelancer_agent_type']
        client_agent_type = config['client_agent_type']
        enable_reflections = config['enable_reflections']
        print(f"🤖 Running '{args.baseline_scenario}' baseline scenario")
    
    # Validate baseline_greedy_undercut parameter
    if not (0.1 <= baseline_greedy_undercut <= 1.0):
        print(f"❌ ERROR: baseline_greedy_undercut must be between 0.1 and 1.0, got {baseline_greedy_undercut}")
        sys.exit(1)
    
    # Validate random_freelancer_bid_probability parameter
    if not (0.0 <= random_freelancer_bid_probability <= 1.0):
        print(f"❌ ERROR: random_freelancer_bid_probability must be between 0.0 and 1.0, got {random_freelancer_bid_probability}")
        sys.exit(1)
    
    print(f"📊 Configuration: {args.freelancers} freelancers, {args.clients} clients, {args.rounds} rounds")
    print(f"💰 Bid Budget: {bids_per_round} bid(s) per freelancer per round")
    print(f"👀 Job Visibility: {args.jobs_per_freelancer} job(s) shown per freelancer per round")
    print(f"🔄 Max Active Jobs: {args.max_active_jobs} concurrent project(s) per freelancer")
    print(f"🎯 Job Selection: {args.job_selection} method")
    if args.job_selection == 'relevance':
        print(f"🧠 Relevance Mode: {args.relevance_mode}")
    
    # Agent type information
    print(f"🤖 Agent Types: Freelancers={freelancer_agent_type}, Clients={client_agent_type}")
    if freelancer_agent_type in ['random', 'greedy', 'no_reputation'] or client_agent_type in ['random', 'greedy', 'no_reputation']:
        print(f"📈 Baseline Config: Greedy undercut={baseline_greedy_undercut*100:.0f}%")
    
    # Performance optimization info
    if not enable_reflections:
        print("⚡ Performance: Reflections DISABLED for maximum speed")
    else:
        probability_percent = reflection_probability * 100
        print(f"🤔 Performance: {probability_percent:.0f}% chance per agent per round to reflect")
    print(f"⚡ Performance: {args.max_workers} parallel workers")
    if args.quiet:
        print("🔇 Quiet Mode: ENABLED - reduced logging and file I/O for maximum speed")
    if args.bid_cooloff_rounds > 0:
        print(f"🔄 Bid Cooloff: Freelancers can re-bid after {args.bid_cooloff_rounds} rounds")
    else:
        print("🔄 Bid Cooloff: Disabled (freelancers cannot re-bid on previously declined jobs)")
    
    try:
        marketplace = TrueGPTMarketplace(
            num_freelancers=args.freelancers,
            num_clients=args.clients,
            rounds=args.rounds,
            bids_per_round=bids_per_round,
            jobs_per_freelancer_per_round=args.jobs_per_freelancer,
            job_selection_method=args.job_selection,
            relevance_mode=args.relevance_mode,
            use_cache=not args.no_cache,
            reflection_probability=reflection_probability,
            enable_reflections=enable_reflections,
            bid_cooloff_rounds=args.bid_cooloff_rounds,
            max_workers=args.max_workers,
            max_active_jobs=args.max_active_jobs,
            quiet_mode=args.quiet,
            job_posting_cooldown_min=args.job_posting_cooldown_min,
            job_posting_cooldown_max=args.job_posting_cooldown_max,
            # Baseline agent configuration
            freelancer_agent_type=freelancer_agent_type,
            client_agent_type=client_agent_type,
            baseline_greedy_undercut=baseline_greedy_undercut,
            random_freelancer_bid_probability=random_freelancer_bid_probability,
            # Reproducibility
            random_seed=args.seed
        )
        
        results = marketplace.run_simulation()
        
        # Save token usage report
        token_tracker = get_token_tracker()
        token_tracker.save_report()
        
        return results
        
    except RuntimeError as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nExperiment aborted due to insufficient profiles. Please try:")
        print("1. Running with fewer freelancers/clients")
        print("2. Checking API connectivity")
        print("3. Clearing the cache with --no-cache flag")
        sys.exit(1)

if __name__ == "__main__":
    main()
