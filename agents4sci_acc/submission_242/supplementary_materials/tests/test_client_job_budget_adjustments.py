"""
Test client-driven specific job budget adjustments functionality.
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketplace.true_gpt_marketplace import TrueGPTMarketplace
from marketplace.entities import Job, Client


class TestClientJobBudgetAdjustments:
    """Test the new client-driven job budget adjustment system."""
    
    def test_job_rounds_unfilled_tracking(self):
        """Test that jobs track how many rounds they've been unfilled."""
        job = Job(
            id="test_job_1",
            client_id="client_1", 
            title="Test Job",
            description="Test description",
            category="SOFTWARE",
            skills_required=["Python"],
            budget_type="hourly",
            budget_amount=50.0,
            timeline="1 month",
            special_requirements="None"
        )
        
        # Job should start with 0 rounds unfilled
        assert job.rounds_unfilled == 0
        
        # After tracking, should increment
        job.rounds_unfilled += 1
        assert job.rounds_unfilled == 1
        
        job.rounds_unfilled += 1
        assert job.rounds_unfilled == 2
    
    def test_unfilled_jobs_tracking_method(self):
        """Test the marketplace method that tracks unfilled jobs."""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1,
            num_clients=1,
            rounds=2,
            enable_reflections=False  # Disable for speed
        )
        
        # Create some test jobs
        job1 = Job(
            id="job_1",
            client_id="client_1",
            title="Job 1", 
            description="Test",
            category="SOFTWARE",
            skills_required=["Python"],
            budget_type="hourly",
            budget_amount=50.0,
            timeline="1 month",
            special_requirements="None"
        )
        
        job2 = Job(
            id="job_2", 
            client_id="client_1",
            title="Job 2",
            description="Test",
            category="DATA_SCIENCE", 
            skills_required=["ML"],
            budget_type="fixed",
            budget_amount=1000.0,
            timeline="2 weeks",
            special_requirements="None"
        )
        
        marketplace.active_jobs = [job1, job2]
        
        # Track unfilled jobs
        marketplace._update_unfilled_jobs_tracking(round_num=1)
        
        # Both jobs should have incremented rounds_unfilled
        assert job1.rounds_unfilled == 1
        assert job2.rounds_unfilled == 1
        
        # Track again
        marketplace._update_unfilled_jobs_tracking(round_num=2)
        
        assert job1.rounds_unfilled == 2
        assert job2.rounds_unfilled == 2
    
    def test_specific_job_budget_adjustments(self):
        """Test applying specific budget adjustments to individual jobs."""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1,
            num_clients=1, 
            rounds=2,
            enable_reflections=False
        )
        
        # Create test jobs
        job1 = Job(
            id="job_urgent_1",
            client_id="client_1",
            title="Urgent Task",
            description="Critical project",
            category="SOFTWARE",
            skills_required=["Python"],
            budget_type="hourly", 
            budget_amount=100.0,
            timeline="1 week",
            special_requirements="None"
        )
        
        job2 = Job(
            id="job_normal_2",
            client_id="client_1",
            title="Normal Task", 
            description="Regular project",
            category="DESIGN",
            skills_required=["Photoshop"],
            budget_type="fixed",
            budget_amount=500.0,
            timeline="1 month",
            special_requirements="None"
        )
        
        marketplace.active_jobs = [job1, job2]
        
        # Test job budget adjustments
        adjustments = [
            {
                "job_id": "job_urgent_1",
                "increase_percentage": 25,
                "reasoning": "Critical project needs immediate attention"
            },
            {
                "job_id": "job_normal_2", 
                "increase_percentage": 10,
                "reasoning": "Slight budget increase to attract candidates"
            }
        ]
        
        results = marketplace._apply_specific_job_budget_adjustments(adjustments, round_num=3)
        
        # Check results
        assert len(results) == 2
        
        # Check job 1 adjustment
        job1_result = next(r for r in results if r['job_id'] == 'job_urgent_1')
        assert job1_result['old_budget'] == 100.0
        assert job1_result['new_budget'] == 125.0  # 100 * 1.25
        assert job1_result['increase_percentage'] == 25
        assert job1_result['reasoning'] == "Critical project needs immediate attention"
        
        # Check job 2 adjustment  
        job2_result = next(r for r in results if r['job_id'] == 'job_normal_2')
        assert job2_result['old_budget'] == 500.0
        assert job2_result['new_budget'] == 550.0  # 500 * 1.10
        assert job2_result['increase_percentage'] == 10
        
        # Verify actual job objects were updated
        assert job1.budget_amount == 125.0
        assert job2.budget_amount == 550.0
    
    def test_job_budget_adjustments_no_effect_for_missing_jobs(self):
        """Test that adjustments for non-existent jobs are ignored."""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1,
            num_clients=1,
            rounds=2,
            enable_reflections=False
        )
        
        # Empty active jobs
        marketplace.active_jobs = []
        
        adjustments = [
            {
                "job_id": "nonexistent_job",
                "increase_percentage": 50,
                "reasoning": "This job doesn't exist"
            }
        ]
        
        results = marketplace._apply_specific_job_budget_adjustments(adjustments, round_num=1)
        
        # Should return empty results
        assert len(results) == 0
    
    def test_job_budget_adjustments_zero_percentage(self):
        """Test that zero or negative percentage adjustments are ignored."""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1,
            num_clients=1,
            rounds=2, 
            enable_reflections=False
        )
        
        job = Job(
            id="test_job",
            client_id="client_1",
            title="Test Job",
            description="Test",
            category="SOFTWARE", 
            skills_required=["Python"],
            budget_type="hourly",
            budget_amount=75.0,
            timeline="1 week",
            special_requirements="None"
        )
        
        marketplace.active_jobs = [job]
        
        adjustments = [
            {
                "job_id": "test_job",
                "increase_percentage": 0,  # Zero increase
                "reasoning": "No change needed"
            }
        ]
        
        results = marketplace._apply_specific_job_budget_adjustments(adjustments, round_num=1)
        
        # Should return empty results (zero increase ignored)
        assert len(results) == 0
        
        # Job budget should remain unchanged
        assert job.budget_amount == 75.0
    
    def test_client_activities_include_unfilled_jobs(self):
        """Test that client activities include unfilled job information."""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1,
            num_clients=1,
            rounds=2,
            enable_reflections=False
        )
        
        # Create a test client
        client = Client(
            id="client_1",
            company_name="Test Company",
            company_size="startup",
            budget_philosophy="balanced",
            hiring_style="professional",
            background="Tech company",
            business_category="SOFTWARE"
        )
        
        # Create unfilled jobs
        job1 = Job(
            id="unfilled_job_1",
            client_id="client_1",
            title="Senior Developer",
            description="Need experienced dev",
            category="SOFTWARE",
            skills_required=["Python", "Django"],
            budget_type="hourly",
            budget_amount=80.0,
            timeline="3 months",
            special_requirements="5+ years experience",
            rounds_unfilled=3
        )
        
        job2 = Job(
            id="unfilled_job_2", 
            client_id="client_1",
            title="Data Analyst",
            description="Data analysis role",
            category="DATA_SCIENCE",
            skills_required=["SQL", "Python"],
            budget_type="fixed",
            budget_amount=2000.0,
            timeline="1 month", 
            special_requirements="Statistics background",
            rounds_unfilled=1
        )
        
        marketplace.active_jobs = [job1, job2]
        
        # Get client activities
        activities = marketplace._get_client_activities(client, [])
        
        # Should have 2 unfilled job activities
        unfilled_activities = [a for a in activities if a['type'] == 'unfilled_job']
        assert len(unfilled_activities) == 2
        
        # Check unfilled job 1 activity
        job1_activity = next(a for a in unfilled_activities if a['job_id'] == 'unfilled_job_1')
        assert job1_activity['title'] == "Senior Developer"
        assert job1_activity['current_budget'] == 80.0
        assert job1_activity['rounds_unfilled'] == 3
        assert job1_activity['timeline'] == "3 months"
        assert job1_activity['category'] == "SOFTWARE"
        
        # Check unfilled job 2 activity
        job2_activity = next(a for a in unfilled_activities if a['job_id'] == 'unfilled_job_2')
        assert job2_activity['title'] == "Data Analyst"
        assert job2_activity['current_budget'] == 2000.0
        assert job2_activity['rounds_unfilled'] == 1
        assert job2_activity['timeline'] == "1 month"
        assert job2_activity['category'] == "DATA_SCIENCE"
