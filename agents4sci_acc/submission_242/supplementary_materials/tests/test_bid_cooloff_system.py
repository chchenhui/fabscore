"""
Tests for the bid cooloff system - allows freelancers to re-bid on jobs after N rounds.
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketplace.entities import Freelancer, Client, Job
from marketplace.true_gpt_marketplace import TrueGPTMarketplace


class TestBidCooloffSystem:
    """Test the bid cooloff functionality"""

    @pytest.fixture
    def cooloff_marketplace(self):
        """Create marketplace with short cooloff period for testing"""
        return TrueGPTMarketplace(
            num_freelancers=2,
            num_clients=1,
            rounds=10,
            bids_per_round=1,
            jobs_per_freelancer_per_round=3,
            job_selection_method="random",
            relevance_mode="moderate",
            enable_reflections=False,
            max_workers=1,
            bid_cooloff_rounds=3  # 3 round cooloff for testing
        )

    @pytest.fixture
    def no_cooloff_marketplace(self):
        """Create marketplace with cooloff disabled"""
        return TrueGPTMarketplace(
            num_freelancers=2,
            num_clients=1,
            rounds=10,
            bids_per_round=1,
            jobs_per_freelancer_per_round=3,
            job_selection_method="random",
            relevance_mode="moderate",
            enable_reflections=False,
            max_workers=1,
            bid_cooloff_rounds=0  # Cooloff disabled
        )

    @pytest.fixture
    def test_freelancer(self):
        """Create a test freelancer"""
        return Freelancer(
            id="freelancer_test",
            name="Test Freelancer",

            category="Web, Mobile & Software Dev",
            skills=["Python", "Testing"],
            min_hourly_rate=50.0,
            personality="Test personality",
            motivation="Test motivation",
            background="Test background"
        )

    @pytest.fixture
    def test_job(self):
        """Create a test job"""
        return Job(
            id="job_test",
            client_id="client_test",
            title="Test Job",
            description="Test job description",
            skills_required=["Python", "Testing"],
            budget_type="hourly",
            budget_amount=60.0,
            timeline="1 month",
            special_requirements=["Test requirement"],
            category="Software Development",
            posted_time=datetime.now()
        )

    def test_cooloff_parameter_initialization(self, cooloff_marketplace, no_cooloff_marketplace):
        """Test that cooloff parameter is properly initialized"""
        assert cooloff_marketplace.bid_cooloff_rounds == 3
        assert no_cooloff_marketplace.bid_cooloff_rounds == 0

    def test_decision_tracking_structure(self, cooloff_marketplace, test_freelancer, test_job):
        """Test that decisions are tracked with proper round information"""
        # Simulate a decision being made
        decision_key = (test_freelancer.id, test_job.id)
        decision_info = {'decision': 'no', 'round': 1}
        cooloff_marketplace.freelancer_job_decisions[decision_key] = decision_info

        # Verify structure
        stored_decision = cooloff_marketplace.freelancer_job_decisions[decision_key]
        assert isinstance(stored_decision, dict)
        assert stored_decision['decision'] == 'no'
        assert stored_decision['round'] == 1

    def test_get_jobs_for_freelancer_new_job(self, cooloff_marketplace, test_freelancer, test_job):
        """Test that new jobs are always available"""
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=1
        )
        assert len(available_jobs) == 1
        assert available_jobs[0].id == test_job.id

    def test_get_jobs_for_freelancer_declined_within_cooloff(self, cooloff_marketplace, test_freelancer, test_job):
        """Test that declined jobs are blocked during cooloff period"""
        # Record a "no" decision in round 1
        decision_key = (test_freelancer.id, test_job.id)
        cooloff_marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # Check availability in round 2 (within cooloff)
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=2
        )
        assert len(available_jobs) == 0

        # Check availability in round 3 (still within cooloff)
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=3
        )
        assert len(available_jobs) == 0

    def test_get_jobs_for_freelancer_declined_after_cooloff(self, cooloff_marketplace, test_freelancer, test_job):
        """Test that declined jobs become available again after cooloff expires"""
        # Record a "no" decision in round 1
        decision_key = (test_freelancer.id, test_job.id)
        cooloff_marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # Check availability in round 4 (cooloff expired: 4-1 = 3 >= 3)
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=4
        )
        assert len(available_jobs) == 1
        assert available_jobs[0].id == test_job.id

        # Check availability in round 5 (well past cooloff)
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=5
        )
        assert len(available_jobs) == 1

    def test_get_jobs_for_freelancer_accepted_job_blocked(self, cooloff_marketplace, test_freelancer, test_job):
        """Test that jobs with 'yes' decisions are permanently blocked"""
        # Record a "yes" decision in round 1
        decision_key = (test_freelancer.id, test_job.id)
        cooloff_marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'yes',
            'round': 1
        }

        # Check availability in round 5 (well past cooloff)
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=5
        )
        assert len(available_jobs) == 0  # Should still be blocked

    def test_get_jobs_for_freelancer_disabled_cooloff(self, no_cooloff_marketplace, test_freelancer, test_job):
        """Test that when cooloff is disabled, decisions are permanent"""
        # Record a "no" decision in round 1
        decision_key = (test_freelancer.id, test_job.id)
        no_cooloff_marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # Check availability in round 10 (should still be blocked)
        available_jobs = no_cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=10
        )
        assert len(available_jobs) == 0

    def test_get_jobs_for_freelancer_backward_compatibility(self, cooloff_marketplace, test_freelancer, test_job):
        """Test that old string format decisions are converted to new format"""
        # Record an old-style decision (just string)
        decision_key = (test_freelancer.id, test_job.id)
        cooloff_marketplace.freelancer_job_decisions[decision_key] = 'no'

        # Call the method which should convert the format
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=1
        )

        # Check that it was converted to new format
        decision_info = cooloff_marketplace.freelancer_job_decisions[decision_key]
        assert isinstance(decision_info, dict)
        assert decision_info['decision'] == 'no'
        assert decision_info['round'] == 0

    def test_get_jobs_for_freelancer_multiple_jobs(self, cooloff_marketplace, test_freelancer):
        """Test filtering with multiple jobs and mixed decision states"""
        # Create multiple test jobs
        job1 = Job(
            id="job_1", client_id="client_1", title="Job 1", description="Desc 1",
            skills_required=["Python"], budget_type="hourly", budget_amount=50.0,
            timeline="1 month", special_requirements=[], category="Software",
            posted_time=datetime.now()
        )
        job2 = Job(
            id="job_2", client_id="client_1", title="Job 2", description="Desc 2",
            skills_required=["Python"], budget_type="hourly", budget_amount=60.0,
            timeline="2 months", special_requirements=[], category="Software",
            posted_time=datetime.now()
        )
        job3 = Job(
            id="job_3", client_id="client_1", title="Job 3", description="Desc 3",
            skills_required=["Python"], budget_type="hourly", budget_amount=70.0,
            timeline="3 months", special_requirements=[], category="Software",
            posted_time=datetime.now()
        )

        # Set up decision history
        cooloff_marketplace.freelancer_job_decisions = {
            (test_freelancer.id, job1.id): {'decision': 'no', 'round': 1},   # Within cooloff
            (test_freelancer.id, job2.id): {'decision': 'yes', 'round': 1},  # Permanently blocked
            # job3 has no decision - should be available
        }

        # Test in round 2 (within cooloff for job1)
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [job1, job2, job3], current_round=2
        )
        # Only job3 should be available
        assert len(available_jobs) == 1
        assert available_jobs[0].id == job3.id

        # Test in round 4 (cooloff expired for job1)
        available_jobs = cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [job1, job2, job3], current_round=4
        )
        # job1 and job3 should be available, job2 still blocked
        assert len(available_jobs) == 2
        available_ids = {job.id for job in available_jobs}
        assert "job_1" in available_ids
        assert "job_3" in available_ids
        assert "job_2" not in available_ids

    def test_cooloff_logging_output(self, cooloff_marketplace, test_freelancer, test_job, capsys):
        """Test that cooloff expiration is logged properly"""
        # Record a "no" decision in round 1
        decision_key = (test_freelancer.id, test_job.id)
        cooloff_marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # Call method after cooloff expires
        cooloff_marketplace.get_jobs_for_freelancer(
            test_freelancer, [test_job], current_round=4
        )

        # Check that the cooloff expiration was logged
        captured = capsys.readouterr()
        assert "🔄" in captured.out
        assert "Cooloff expired" in captured.out
        assert test_freelancer.name in captured.out
        assert test_job.title in captured.out
        assert "declined 3 rounds ago" in captured.out


class TestBidCooloffIntegration:
    """Test bid cooloff integration with the broader marketplace system"""

    def test_freelancer_bidding_decision_method_signature(self):
        """Test that freelancer_bidding_decision accepts current_round parameter"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=1,
            enable_reflections=False, max_workers=1
        )
        
        freelancer = Freelancer(
            id="test_freelancer", name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        
        job = Job(
            id="test_job", client_id="test_client", title="Test Job",
            description="Test", skills_required=["Python"], budget_type="hourly",
            budget_amount=60.0, timeline="1 month", special_requirements=[],
            category="Software", posted_time=datetime.now()
        )

        # This should not raise an error about missing current_round parameter
        try:
            # Mock the actual GPT call since we're just testing the method signature
            # The openai_client is imported at module level, so we patch it there
            import marketplace.true_gpt_marketplace as marketplace_module
            with patch.object(marketplace_module, 'openai_client') as mock_client:
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = '{"decision": "no", "reasoning": "test"}'
                mock_client.chat.completions.create.return_value = mock_response
                
                # This call should work with the current_round parameter
                result = marketplace.freelancer_bidding_decision(freelancer, job, current_round=5)
                assert 'decision' in result
        except TypeError as e:
            pytest.fail(f"Method signature test failed: {e}")

    def test_process_freelancer_bidding_task_round_tracking(self):
        """Test that the bidding task properly tracks round information"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=1,
            enable_reflections=False, max_workers=1, bid_cooloff_rounds=3
        )
        
        freelancer = Freelancer(
            id="test_freelancer", name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        
        job = Job(
            id="test_job", client_id="test_client", title="Test Job",
            description="Test", skills_required=["Python"], budget_type="hourly",
            budget_amount=60.0, timeline="1 month", special_requirements=[],
            category="Software", posted_time=datetime.now()
        )

        # Mock the GPT response
        with patch.object(marketplace, 'freelancer_bidding_decision') as mock_decision:
            mock_decision.return_value = {
                'decision': 'no',
                'reasoning': 'Test reasoning'
            }
            
            # Call the bidding task
            task_data = (freelancer, [job], 5)  # Round 5
            result = marketplace._process_freelancer_bidding_task(task_data)
            
            # Check that job decisions include round information
            assert 'job_decisions' in result
            decision_key = (freelancer.id, job.id)
            assert decision_key in result['job_decisions']
            
            decision_info = result['job_decisions'][decision_key]
            assert isinstance(decision_info, dict)
            assert decision_info['decision'] == 'no'
            assert decision_info['round'] == 5

    def test_marketplace_argument_parsing_integration(self):
        """Test that bid-cooloff-rounds argument is properly handled"""
        from marketplace.true_gpt_marketplace import parse_arguments
        
        # Test default value
        with patch('sys.argv', ['script_name']):
            args = parse_arguments()
            assert args.bid_cooloff_rounds == 5  # Default value

        # Test custom value
        with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', '10']):
            args = parse_arguments()
            assert args.bid_cooloff_rounds == 10

        # Test disabled cooloff
        with patch('sys.argv', ['script_name', '--bid-cooloff-rounds', '0']):
            args = parse_arguments()
            assert args.bid_cooloff_rounds == 0

    def test_marketplace_initialization_with_cooloff(self):
        """Test that marketplace properly initializes with cooloff parameter"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=2, num_clients=1, rounds=5,
            bid_cooloff_rounds=7, enable_reflections=False, max_workers=1
        )
        
        assert marketplace.bid_cooloff_rounds == 7
        assert isinstance(marketplace.freelancer_job_decisions, dict)


class TestBidCooloffEdgeCases:
    """Test edge cases and error conditions for bid cooloff"""

    def test_cooloff_with_zero_rounds(self):
        """Test cooloff behavior when set to 0 (disabled)"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=1,
            bid_cooloff_rounds=0, enable_reflections=False, max_workers=1
        )
        
        freelancer = Freelancer(
            id="test_freelancer", name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        
        job = Job(
            id="test_job", client_id="test_client", title="Test Job",
            description="Test", skills_required=["Python"], budget_type="hourly",
            budget_amount=60.0, timeline="1 month", special_requirements=[],
            category="Software", posted_time=datetime.now()
        )

        # Record a "no" decision
        decision_key = (freelancer.id, job.id)
        marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # With cooloff disabled, job should never become available again
        for round_num in [2, 5, 10, 100]:
            available_jobs = marketplace.get_jobs_for_freelancer(
                freelancer, [job], current_round=round_num
            )
            assert len(available_jobs) == 0, f"Job became available in round {round_num} despite disabled cooloff"

    def test_cooloff_boundary_conditions(self):
        """Test cooloff exactly at the boundary"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=1,
            bid_cooloff_rounds=3, enable_reflections=False, max_workers=1
        )
        
        freelancer = Freelancer(
            id="test_freelancer", name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        
        job = Job(
            id="test_job", client_id="test_client", title="Test Job",
            description="Test", skills_required=["Python"], budget_type="hourly",
            budget_amount=60.0, timeline="1 month", special_requirements=[],
            category="Software", posted_time=datetime.now()
        )

        # Record a "no" decision in round 1
        decision_key = (freelancer.id, job.id)
        marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # Test exact boundary: round 4 (4-1 = 3, which equals cooloff period)
        available_jobs = marketplace.get_jobs_for_freelancer(
            freelancer, [job], current_round=4
        )
        assert len(available_jobs) == 1, "Job should be available exactly at cooloff boundary"

        # Test one round before boundary: round 3 (3-1 = 2, which is less than cooloff period)
        available_jobs = marketplace.get_jobs_for_freelancer(
            freelancer, [job], current_round=3
        )
        assert len(available_jobs) == 0, "Job should not be available before cooloff expires"

    def test_negative_cooloff_rounds(self):
        """Test that negative cooloff rounds are handled gracefully"""
        # The marketplace should accept negative values but treat them as 0 or handle gracefully
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=1,
            bid_cooloff_rounds=-5, enable_reflections=False, max_workers=1
        )
        
        # Should not crash during initialization
        assert marketplace.bid_cooloff_rounds == -5
        
        freelancer = Freelancer(
            id="test_freelancer", name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        
        job = Job(
            id="test_job", client_id="test_client", title="Test Job",
            description="Test", skills_required=["Python"], budget_type="hourly",
            budget_amount=60.0, timeline="1 month", special_requirements=[],
            category="Software", posted_time=datetime.now()
        )

        # Record a "no" decision in round 1
        decision_key = (freelancer.id, job.id)
        marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # With negative cooloff, the job should become available immediately
        available_jobs = marketplace.get_jobs_for_freelancer(
            freelancer, [job], current_round=2
        )
        # Should either be available (negative treated as 0) or not crash
        assert isinstance(available_jobs, list)

    def test_large_cooloff_period(self):
        """Test behavior with very large cooloff periods"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=1,
            bid_cooloff_rounds=1000, enable_reflections=False, max_workers=1
        )
        
        freelancer = Freelancer(
            id="test_freelancer", name="Test",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        
        job = Job(
            id="test_job", client_id="test_client", title="Test Job",
            description="Test", skills_required=["Python"], budget_type="hourly",
            budget_amount=60.0, timeline="1 month", special_requirements=[],
            category="Software", posted_time=datetime.now()
        )

        # Record a "no" decision in round 1
        decision_key = (freelancer.id, job.id)
        marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # Job should not be available for a very long time
        for round_num in [2, 10, 100, 500]:
            available_jobs = marketplace.get_jobs_for_freelancer(
                freelancer, [job], current_round=round_num
            )
            assert len(available_jobs) == 0, f"Job should not be available in round {round_num} with large cooloff"

        # But should eventually become available
        available_jobs = marketplace.get_jobs_for_freelancer(
            freelancer, [job], current_round=1001
        )
        assert len(available_jobs) == 1, "Job should be available after large cooloff period"
