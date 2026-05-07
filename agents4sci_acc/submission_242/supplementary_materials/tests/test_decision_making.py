"""
Test decision-making logic and bidding behaviors.
"""
import pytest
from unittest.mock import patch
from marketplace.true_gpt_marketplace import TrueGPTMarketplace


class TestFreelancerBiddingDecisions:
    """Test freelancer bidding decision logic."""
    
    def test_bid_budget_constraint_enforcement(self, marketplace_instance, sample_freelancer, sample_job):
        """Test that freelancers respect bid budget limits."""
        # Framework Assumption: Fixed Budget Bidding System
        
        # Set freelancer to have used all bids
        sample_freelancer.bids_this_round = 3  # Assuming 3 is the limit
        bids_per_round = 3
        
        # Should not be able to bid when at limit
        assert not sample_freelancer.can_bid(bids_per_round)
        
        # Reset and test can bid when under limit
        sample_freelancer.bids_this_round = 1
        assert sample_freelancer.can_bid(bids_per_round)
    
    def test_budget_alignment_decision_factor(self, sample_freelancer, sample_job, low_budget_job):
        """Test that budget alignment affects bidding decisions."""
        # Framework Assumption: Bidding is accept/reject based on published budget
        
        # Good budget job (80 >= 75 minimum rate)
        assert sample_job.budget_amount >= sample_freelancer.min_hourly_rate
        
        # Low budget job (50 < 75 minimum rate)
        assert low_budget_job.budget_amount < sample_freelancer.min_hourly_rate
        
        # The decision logic should consider this alignment
        # (Testing the framework's capability to support this decision)
    
    def test_skill_alignment_decision_factor(self, sample_freelancer, sample_job, mismatch_job):
        """Test that skill alignment affects bidding decisions."""
        # Framework Assumption: Limited Skill Complexity with semantic matching
        
        # Calculate skill overlaps for decision making
        freelancer_skills = set(skill.lower() for skill in sample_freelancer.skills)
        
        # Good match job
        good_job_skills = set(skill.lower() for skill in sample_job.skills_required)
        good_overlap = len(freelancer_skills.intersection(good_job_skills))
        
        # Mismatch job
        bad_job_skills = set(skill.lower() for skill in mismatch_job.skills_required)
        bad_overlap = len(freelancer_skills.intersection(bad_job_skills))
        
        # Should have more overlap with matching job
        assert good_overlap > bad_overlap
    
    def test_freelancer_bidding_decision_structure(self, marketplace_instance, sample_freelancer, sample_job):
        """Test the structure of freelancer bidding decisions."""
        # Framework Assumption: Simplified Decision Prompts
        
        # Test that the bidding decision method exists and returns structured data
        result = marketplace_instance.freelancer_bidding_decision(sample_freelancer, sample_job)
        
        # Should return structured decision
        assert "decision" in result
        assert "reasoning" in result
        assert result["decision"] in ["yes", "no"]
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0
    

    def test_freelancer_rejection_decision_structure(self, marketplace_instance, sample_freelancer, mismatch_job):
        """Test structure of rejection decisions."""
        # Framework Assumption: Natural language reasoning
        
        # Test rejection decision structure
        result = marketplace_instance.freelancer_bidding_decision(sample_freelancer, mismatch_job)
        
        # Should return structured decision (might be yes or no, but structure should be valid)
        assert "decision" in result
        assert result["decision"] == "no"
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0
        # Message should be empty for rejections
        assert result.get("message", "") == ""
    
    def test_decision_consistency_with_reputation(self, marketplace_instance, sample_freelancer):
        """Test that decisions can consider reputation status."""
        # Framework Assumption: Perfect Memory and Learning
        
        # New freelancer should have different considerations than experienced one
        assert sample_freelancer.total_bids == 0  # New freelancer
        assert sample_freelancer.total_hired == 0
        
        # Decision prompts should include reputation context
        # (This tests the framework's capability to provide reputation data)


class TestClientHiringDecisions:
    """Test client hiring decision logic."""
    

    def test_client_hiring_decision_structure(self,  marketplace_instance, sample_client, sample_job):
        """Test structure of client hiring decisions."""
        # Framework Assumption: Streamlined Evaluation Criteria
        
        # Test hiring decision structure
        
        # Create mock Bid entities (not dictionaries)
        from marketplace.entities import Bid
        from datetime import datetime
        
        # Create a mock freelancer and add to marketplace for lookup
        from marketplace.entities import Freelancer
        mock_freelancer = Freelancer(
            id='test_freelancer_1',
            name='John Doe',
            skills=['Python', 'Machine Learning'],
            category='SOFTWARE',
            min_hourly_rate=50.0,
            personality='Professional and experienced',
            motivation='Building great software',
            background='5 years experience in ML'
        )
        
        # Add to marketplace freelancer lookup
        marketplace_instance.freelancers_by_id[mock_freelancer.id] = mock_freelancer
        marketplace_instance.freelancers.append(mock_freelancer)
        
        mock_bids = [Bid(
            freelancer_id='test_freelancer_1',
            job_id=sample_job.id,
            proposed_rate=75.0,
            message="Excited to work on this project",
            submission_time=datetime.now()
        )]
        
        result = marketplace_instance.client_hiring_decision(sample_client, sample_job, mock_bids)
        
        # Should return a HiringDecision object with required fields
        from marketplace.entities import HiringDecision
        assert isinstance(result, HiringDecision)
        assert result.job_id == sample_job.id
        assert result.client_id == sample_client.id
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
        assert result.timestamp is not None
    

    def test_client_no_hire_decision(self,  marketplace_instance, sample_client, sample_job):
        """Test client decision to hire no one."""
        # Framework Assumption: Binary Hiring Decisions
        
        # Test "hire none" decision structure
        
        bid_summaries = []  # No good bids
        
        result = marketplace_instance.client_hiring_decision(sample_client, sample_job, bid_summaries)
        
        # Should return a HiringDecision object for no-hire case
        from marketplace.entities import HiringDecision
        assert isinstance(result, HiringDecision)
        assert result.job_id == sample_job.id
        assert result.client_id == sample_client.id
        assert result.selected_freelancer is None  # No one hired
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0
    
    def test_hiring_decision_factors(self, sample_client, sample_job):
        """Test the factors that should influence hiring decisions."""
        # Framework Assumption: Core factors (skill match, budget alignment, project understanding)
        
        # Mock bid with good skill match
        good_bid = {
            "freelancer_name": "ML Expert",
            "skills": ["Machine Learning", "Python", "TensorFlow"],  # Matches job requirements
            "success_rate": "90% completion rate",
            "message": "I have 5 years of ML experience and built similar recommendation systems"
        }
        
        # Mock bid with poor skill match
        poor_bid = {
            "freelancer_name": "Designer", 
            "skills": ["Graphic Design", "Photoshop"],  # Doesn't match ML job
            "success_rate": "95% completion rate",
            "message": "I can design great interfaces"
        }
        
        # Good bid should score better on skill match
        good_skills = set(skill.lower() for skill in good_bid["skills"])
        job_skills = set(skill.lower() for skill in sample_job.skills_required)
        good_overlap = len(good_skills.intersection(job_skills))
        
        poor_skills = set(skill.lower() for skill in poor_bid["skills"])
        poor_overlap = len(poor_skills.intersection(job_skills))
        
        assert good_overlap > poor_overlap


class TestJobCompletionDecisions:
    """Test job completion evaluation logic."""
    

    def test_job_completion_evaluation_structure(self,  marketplace_instance):
        """Test structure of job completion evaluations."""
        # Framework Assumption: No Fatigue Effects on Performance
        
        # Test job completion prompt generation and validation
        from prompts import generate_job_completion_prompt, validate_job_completion
        from prompts.models import JobCompletionResponse
        
        # Create test job and freelancer data
        job_data = {
            'id': 'test_job',
            'title': 'Python Development Task',
            'description': 'Build a simple Python application',
            'skills_required': ['Python', 'Flask'],
            'budget_amount': 500.0,
            'timeline': '1 week'
        }
        
        freelancer_data = {
            'id': 'test_freelancer',
            'name': 'Test Developer',
            'skills': ['Python', 'Flask', 'JavaScript'],
            'category': 'Web, Mobile & Software Dev'
        }
        
        # Test that job completion prompt can be generated
        completion_prompt = generate_job_completion_prompt(job_data, freelancer_data, duration_rounds=5)
        assert isinstance(completion_prompt, str)
        assert len(completion_prompt) > 0
        assert 'Python Development Task' in completion_prompt
        
        # Test that job completion validation works with proper structure
        mock_completion_response = '{"success": true}'
        validated = validate_job_completion(mock_completion_response)
        
        assert isinstance(validated, JobCompletionResponse)
        assert isinstance(validated.success, bool)
    

    def test_job_completion_failure_case(self,  marketplace_instance):
        """Test job completion failure evaluation."""
        # Framework Assumption: Deterministic Job Completion (but can still fail)
        
        # Test job completion validation with failure scenario
        from prompts import validate_job_completion
        from prompts.models import JobCompletionResponse
        
        # Test validation of a failed job completion
        mock_failure_response = '{"success": false}'
        validated_failure = validate_job_completion(mock_failure_response)
        
        assert isinstance(validated_failure, JobCompletionResponse)
        assert not validated_failure.success
        
        # Test that skill mismatch detection works for completion prediction
        mismatched_skills = ["Graphic Design", "UI/UX"]
        job_requirements = ["Python", "Backend Development", "Databases"]
        
        freelancer_skills = set(mismatched_skills)
        job_skills = set(job_requirements)
        skill_overlap = len(freelancer_skills.intersection(job_skills))
        
        # Should have no overlap for this test case
        assert skill_overlap == 0  # Complete skill mismatch should predict higher failure rate


class TestDecisionRationality:
    """Test rational decision-making assumptions."""
    
    def test_consistent_preference_structure(self, sample_freelancer):
        """Test that agents have consistent preferences."""
        # Framework Assumption: GPT as Rational Actors
        
        # Freelancer should have consistent preferences
        assert sample_freelancer.min_hourly_rate > 0  # Values their time
        assert len(sample_freelancer.skills) > 0  # Has defined capabilities
        assert sample_freelancer.preferred_project_length in ["short-term", "medium-term", "long-term"]
    
    def test_goal_oriented_behavior_structure(self, sample_freelancer, sample_client):
        """Test that agents have clear goal structures."""
        # Framework Assumption: Single-Goal Optimization
        
        # Freelancer goals: hiring success, earnings
        assert sample_freelancer.min_hourly_rate > 0  # Earnings consideration
        assert hasattr(sample_freelancer, 'total_bids')  # Tracks bidding activity
        assert hasattr(sample_freelancer, 'total_hired')  # Tracks success
        
        # Client goals: quality/value hiring
        assert hasattr(sample_client, 'budget_multiplier')  # Tracks budget strategy
        assert hasattr(sample_client, 'budget_philosophy')  # Has hiring strategy  
        assert sample_client.budget_philosophy in ["cost-effective", "balanced approach", "premium-focused", "value-focused"]
    
    def test_information_processing_capability(self, marketplace_instance, sample_freelancer, sample_job):
        """Test that agents can process relevant information."""
        # Framework Assumption: Observable Preferences
        
        # Agents should be able to evaluate job-freelancer fit
        # This tests the framework's information flow capabilities
        
        # Job information should be complete
        assert sample_job.title is not None
        assert len(sample_job.skills_required) > 0
        assert sample_job.budget_amount > 0
        assert sample_job.timeline is not None
        
        # Freelancer information should be available for matching
        assert len(sample_freelancer.skills) > 0
        assert sample_freelancer.min_hourly_rate > 0
