"""
Pytest configuration and shared fixtures for marketplace tests.
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketplace.entities import Freelancer, Client, Job
from marketplace.ranking_algorithm import JobRankingCalculator
from marketplace.true_gpt_marketplace import TrueGPTMarketplace
from datetime import datetime


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_freelancer():
    """Create a basic freelancer for testing."""
    return Freelancer(
        id="freelancer_1",
        name="John Doe",
        category="Data Science & Analytics",
        skills=["Python", "Machine Learning", "Data Analysis"],
        min_hourly_rate=75.0,
        personality="Detail-oriented and analytical",
        motivation="Building expertise in AI/ML",
        background="Software engineer with 3 years experience",
        preferred_project_length="medium-term"
    )


@pytest.fixture
def sample_client():
    """Create a basic client for testing."""
    return Client(
        id="client_1",
        company_name="TechCorp",
        company_size="medium-sized startup",
        budget_philosophy="balanced approach",
        hiring_style="thorough evaluation",
        background="AI-focused startup developing machine learning solutions",
        business_category="SOFTWARE"
    )


@pytest.fixture
def sample_job():
    """Create a basic job posting for testing."""
    return Job(
        id="job_1",
        client_id="client_1",
        title="Machine Learning Engineer",
        description="Develop ML models for recommendation system",
        skills_required=["Python", "Machine Learning", "TensorFlow"],
        budget_type="hourly",
        budget_amount=80.0,
        timeline="3-6 months",
        special_requirements=["PhD or Masters in CS", "3+ years ML experience"],
        category="Data Science & Analytics",
        posted_time=datetime.now()
    )


@pytest.fixture
def mismatch_job():
    """Create a job that doesn't match the sample freelancer's skills."""
    return Job(
        id="job_2",
        client_id="client_1",
        title="Graphic Designer",
        description="Design marketing materials and brand assets",
        skills_required=["Adobe Photoshop", "Graphic Design", "Branding"],
        budget_type="hourly",
        budget_amount=45.0,
        timeline="2-4 weeks",
        special_requirements=["Portfolio of design work", "Experience with print design"],
        category="Design & Creative",
        posted_time=datetime.now()
    )


@pytest.fixture
def low_budget_job():
    """Create a job with budget below freelancer's minimum rate."""
    return Job(
        id="job_3",
        client_id="client_1",
        title="Data Analyst",
        description="Analyze customer data and create reports",
        skills_required=["Python", "Data Analysis", "SQL"],
        budget_type="hourly",
        budget_amount=50.0,  # Below freelancer's $75 minimum
        timeline="1-2 months",
        special_requirements=["Experience with pandas", "SQL proficiency"],
        category="Data Science & Analytics",
        posted_time=datetime.now()
    )


@pytest.fixture
def ranking_calculator():
    """Create a job ranking calculator."""
    return JobRankingCalculator(relevance_mode='moderate')


# ReputationManager fixture removed as reputation system is not being used


@pytest.fixture
def marketplace_instance():
    """Create a minimal marketplace instance for testing."""
    return TrueGPTMarketplace(
        num_freelancers=5,
        num_clients=3,
        rounds=2,
        bids_per_round=2,
        jobs_per_freelancer_per_round=3,
        job_selection_method="relevance",
        relevance_mode="moderate",
        enable_reflections=False,
        max_workers=1  # Single threaded for testing
    )


@pytest.fixture
def multiple_freelancers():
    """Create multiple freelancers with different skills for ranking tests."""
    return [
        Freelancer(
            id="ml_expert",
            name="Alice ML",
            category="Data Science & Analytics",
            skills=["Machine Learning", "Python", "TensorFlow", "Deep Learning", "Neural Networks"],
            min_hourly_rate=120.0,
            personality="Expert-level practitioner",
            motivation="Leading ML innovation",
            background="PhD in ML, 8 years experience",
            preferred_project_length="long-term"
        ),
        Freelancer(
            id="python_dev",
            name="Bob Python",
            category="Web, Mobile & Software Dev",
            skills=["Python", "Django", "Web Development", "APIs"],
            min_hourly_rate=85.0,
            personality="Full-stack developer",
            motivation="Building scalable applications",
            background="5 years web development",
            preferred_project_length="medium-term"
        ),
        Freelancer(
            id="data_analyst",
            name="Carol Data",
            category="Data Science & Analytics",
            skills=["Data Analysis", "SQL", "Excel", "Tableau"],
            min_hourly_rate=65.0,
            personality="Analytical and detail-oriented",
            motivation="Extracting insights from data",
            background="3 years business intelligence",
            preferred_project_length="short-term"
        ),
        Freelancer(
            id="designer",
            name="Dave Design",
            category="Design & Creative",
            skills=["Graphic Design", "Adobe Photoshop", "Branding", "UI/UX"],
            min_hourly_rate=70.0,
            personality="Creative and user-focused",
            motivation="Creating beautiful interfaces",
            background="4 years design experience",
            preferred_project_length="medium-term"
        )
    ]


@pytest.fixture
def ml_focused_job():
    """Create a machine learning focused job for ranking tests."""
    return Job(
        id="ml_job",
        client_id="client_1",
        title="Senior ML Engineer - Deep Learning",
        description="Build advanced neural networks for computer vision",
        skills_required=["Machine Learning", "Deep Learning", "Python", "TensorFlow", "Neural Networks"],
        budget_type="hourly",
        budget_amount=130.0,
        timeline="6-12 months",
        special_requirements=["PhD preferred", "Published research", "5+ years ML experience"],
        category="Data Science & Analytics",
        posted_time=datetime.now()
    )
