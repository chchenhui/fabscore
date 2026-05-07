"""
Test the core entities (Freelancer, Client, Job) and their behaviors.
"""
import pytest
from datetime import datetime, timedelta
from marketplace.entities import Freelancer, Client, Job


class TestFreelancer:
    """Test Freelancer entity and behaviors."""
    
    def test_freelancer_creation(self, sample_freelancer):
        """Test basic freelancer creation and properties."""
        # Framework Assumption: Homogeneous Agent Types (all can bid on any job)
        assert sample_freelancer.id == "freelancer_1"
        assert sample_freelancer.name == "John Doe"
        assert sample_freelancer.min_hourly_rate == 75.0
        assert "Python" in sample_freelancer.skills
        assert sample_freelancer.total_bids == 0
        assert sample_freelancer.total_hired == 0
        assert sample_freelancer.ongoing_jobs == []
    
    def test_freelancer_can_bid_budget_limit(self):
        """Test bid budget constraints."""
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Testing"],
            min_hourly_rate=50.0,
            personality="Test personality",
            motivation="Test motivation",
            background="Test background",
            preferred_project_length="short-term"
        )
        
        # Should be able to bid when under limit
        assert freelancer.can_bid(max_bids_per_round=3)
        
        # Track a bid
        freelancer.bids_this_round = 2
        assert freelancer.can_bid(max_bids_per_round=3)
        
        # At limit
        freelancer.bids_this_round = 3
        assert not freelancer.can_bid(max_bids_per_round=3)
    
    def test_freelancer_use_bid(self):
        """Test using a bid updates counter."""
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Testing"],
            min_hourly_rate=50.0,
            personality="Test personality",
            motivation="Test motivation", 
            background="Test background",
            preferred_project_length="short-term"
        )
        
        initial_bids = freelancer.bids_this_round
        freelancer.add_bid()
        assert freelancer.bids_this_round == initial_bids + 1
        assert freelancer.total_bids == 1
    
    def test_freelancer_reset_round(self):
        """Test round reset functionality."""
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Testing"],
            min_hourly_rate=50.0,
            personality="Test personality",
            motivation="Test motivation",
            background="Test background",
            preferred_project_length="short-term"
        )
        
        # Use some bids
        freelancer.bids_this_round = 3
        
        # Reset round
        freelancer.reset_round()
        
        assert freelancer.bids_this_round == 0


class TestClient:
    """Test Client entity and behaviors."""
    
    def test_client_creation(self, sample_client):
        """Test basic client creation and properties."""
        assert sample_client.id == "client_1"
        assert sample_client.company_name == "TechCorp"
        assert sample_client.budget_philosophy == "balanced approach"
        assert sample_client.next_job_round == 0
    
    def test_client_can_post_job(self):
        """Test job posting cooldown logic."""
        client = Client(
            id="test_client",
            company_name="Test Corp",
            company_size="small startup",
            budget_philosophy="cost-effective",
            hiring_style="quick decisions",
            background="Test company background",
            business_category="SOFTWARE"
        )
        
        # Should be able to post in round 0
        assert client.can_post_job(current_round=0)
        
        # Set next job round to future
        client.next_job_round = 5
        assert not client.can_post_job(current_round=3)
        assert client.can_post_job(current_round=5)
        assert client.can_post_job(current_round=6)
    
    def test_client_cooldown_mechanism(self):
        """Test client cooldown after posting jobs."""
        client = Client(
            id="test_client",
            company_name="Test Corp",
            company_size="small startup",
            budget_philosophy="cost-effective",
            hiring_style="quick decisions",
            background="Test company background",
            business_category="SOFTWARE"
        )
        
        # Set cooldown
        client.set_next_job_round(current_round=5, cooldown=3)
        
        assert client.next_job_round == 8


class TestJob:
    """Test Job entity and properties."""
    
    def test_job_creation(self, sample_job):
        """Test basic job creation and properties."""
        # Framework Assumption: Perfect Job Information
        assert sample_job.id == "job_1"
        assert sample_job.client_id == "client_1"
        assert sample_job.title == "Machine Learning Engineer"
        assert sample_job.budget_amount == 80.0
        assert sample_job.budget_type == "hourly"
        assert "Python" in sample_job.skills_required
        assert len(sample_job.special_requirements) == 2
    
    def test_job_serialization(self, sample_job):
        """Test job can be serialized to dict."""
        # Framework Assumption: Measurable Outcomes
        job_dict = sample_job.to_dict()
        
        assert isinstance(job_dict, dict)
        assert job_dict["id"] == sample_job.id
        assert job_dict["title"] == sample_job.title
        assert job_dict["budget_amount"] == sample_job.budget_amount
    
    def test_job_timeline_parsing(self):
        """Test job timeline parsing for completion estimation."""
        # Framework Assumption: Deterministic Job Completion
        job_short = Job(
            id="job_short",
            client_id="client_1", 
            title="Quick Task",
            description="Short description",
            skills_required=["Testing"],
            budget_type="fixed",
            budget_amount=500.0,
            timeline="1-2 weeks",  # Short term
            special_requirements=["None"],
            category="Testing",
            posted_time=datetime.now()
        )
        
        job_long = Job(
            id="job_long",
            client_id="client_1",
            title="Long Project", 
            description="Long description",
            skills_required=["Testing"],
            budget_type="hourly",
            budget_amount=75.0,
            timeline="6-12 months",  # Long term
            special_requirements=["Experience required"],
            category="Testing",
            posted_time=datetime.now()
        )
        
        # Timeline strings should be preserved for evaluation
        assert "week" in job_short.timeline.lower()
        assert "month" in job_long.timeline.lower()


class TestEntityInteractions:
    """Test interactions between entities."""
    
    def test_freelancer_budget_alignment(self, sample_freelancer, sample_job, low_budget_job):
        """Test budget-based decision logic."""
        # Framework Assumption: Perfect Budget Information
        
        # Job with good budget (80 > 75 minimum)
        assert sample_job.budget_amount >= sample_freelancer.min_hourly_rate
        
        # Job with low budget (50 < 75 minimum) 
        assert low_budget_job.budget_amount < sample_freelancer.min_hourly_rate
    
    def test_skill_requirement_matching(self, sample_freelancer, sample_job, mismatch_job):
        """Test basic skill matching concepts."""
        # Framework Assumption: Limited Skill Complexity
        
        # Good match: freelancer has ML and Python, job requires ML and Python
        freelancer_skills = set(sample_freelancer.skills)
        job_skills = set(sample_job.skills_required)
        overlap = freelancer_skills.intersection(job_skills)
        assert len(overlap) >= 2  # Should have Python and ML overlap
        
        # Poor match: freelancer has tech skills, job requires design skills
        mismatch_skills = set(mismatch_job.skills_required)
        mismatch_overlap = freelancer_skills.intersection(mismatch_skills)
        assert len(mismatch_overlap) == 0  # No overlap expected
    
    def test_job_category_classification(self, sample_job, mismatch_job):
        """Test job categorization system."""
        # Framework Assumption: Limited Skill Complexity
        assert sample_job.category == "Data Science & Analytics"
        assert mismatch_job.category == "Design & Creative"
        assert sample_job.category != mismatch_job.category
    
    def test_client_serialization(self, sample_client):
        """Test client can be serialized to dict."""
        # Framework Assumption: Measurable Outcomes
        client_dict = sample_client.to_dict()
        
        assert isinstance(client_dict, dict)
        assert client_dict["id"] == sample_client.id
        assert client_dict["company_name"] == sample_client.company_name
        assert client_dict["budget_philosophy"] == sample_client.budget_philosophy
