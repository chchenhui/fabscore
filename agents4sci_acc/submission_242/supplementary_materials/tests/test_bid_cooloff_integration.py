"""
Integration tests for bid cooloff system with real marketplace simulation.
Tests the full workflow including job generation, bidding, and cooloff behavior.
"""
import pytest
from unittest.mock import patch, Mock
import sys
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketplace.entities import Freelancer, Client, Job
from marketplace.true_gpt_marketplace import TrueGPTMarketplace


class TestBidCooloffIntegration:
    """Integration tests for bid cooloff system"""

    @pytest.fixture
    def minimal_marketplace(self):
        """Create a minimal marketplace for integration testing"""
        return TrueGPTMarketplace(
            num_freelancers=2,
            num_clients=1,
            rounds=6,
            bids_per_round=1,
            jobs_per_freelancer_per_round=2,
            job_selection_method="random",
            relevance_mode="moderate",
            enable_reflections=False,
            max_workers=1,
            bid_cooloff_rounds=3,
            use_cache=False  # Disable cache for predictable testing
        )

    def test_full_simulation_with_cooloff(self, minimal_marketplace):
        """Test a complete simulation run with bid cooloff enabled"""
        # Create test freelancers
        freelancer1 = Freelancer(
            id="freelancer_1", name="Alice",
            category="Web, Mobile & Software Dev",
            skills=["Python", "Testing"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        freelancer2 = Freelancer(
            id="freelancer_2", name="Bob",
            category="Web, Mobile & Software Dev",
            skills=["Java", "Testing"], min_hourly_rate=60.0,
            personality="Test", motivation="Test", background="Test"
        )
        minimal_marketplace.freelancers = [freelancer1, freelancer2]

        # Create test client
        client = Client(
            id="client_1", company_name="TestCorp", company_size="small",
            budget_philosophy="balanced", hiring_style="quick",
            background="Test company", business_category="SOFTWARE"
        )
        minimal_marketplace.clients = [client]

        # Create a persistent job that freelancers will decline initially
        job = Job(
            id="test_job_persistent", client_id="client_1",
            title="Low Budget Python Job", description="Python job with low budget",
            skills_required=["Python"], budget_type="hourly", budget_amount=30.0,  # Below both freelancers' rates
            timeline="1 month", special_requirements=[], category="Software",
            posted_time=None
        )
        minimal_marketplace.all_jobs = [job]
        minimal_marketplace.active_jobs = [job]

        # Mock GPT responses to consistently decline the job
        decline_response = Mock()
        decline_response.choices = [Mock()]
        decline_response.choices[0].message.content = json.dumps({
            "decision": "no",
            "reasoning": "Budget too low for my minimum rate"
        })

        # Mock the bidding decision method
        def mock_bidding_decision(freelancer, job_obj, current_round=0):
            return {
                "decision": "no",
                "reasoning": f"Budget ${job_obj.budget_amount} is below my minimum rate of ${freelancer.min_hourly_rate}"
            }

        with patch.object(minimal_marketplace, 'freelancer_bidding_decision', side_effect=mock_bidding_decision):
            # Round 1: Both freelancers should see and decline the job
            round_bids, round_decisions = minimal_marketplace._process_freelancer_bidding(1)
            
            # Verify both freelancers made decisions
            assert len(round_decisions) == 2
            assert all(d['decision'] == 'no' for d in round_decisions)
            assert len(round_bids) == 0  # No successful bids

            # Check that decisions are tracked with round info
            decision_key1 = (freelancer1.id, job.id)
            decision_key2 = (freelancer2.id, job.id)
            
            # The decisions should be in job_decisions after processing bidding task
            # We need to simulate the full task processing
            task1 = (freelancer1, [job], 1)
            task2 = (freelancer2, [job], 1)
            
            result1 = minimal_marketplace._process_freelancer_bidding_task(task1)
            result2 = minimal_marketplace._process_freelancer_bidding_task(task2)
            
            # Update marketplace state
            for decision_key, decision_value in result1['job_decisions'].items():
                minimal_marketplace.freelancer_job_decisions[decision_key] = decision_value
            for decision_key, decision_value in result2['job_decisions'].items():
                minimal_marketplace.freelancer_job_decisions[decision_key] = decision_value

            # Verify decisions are stored correctly
            assert decision_key1 in minimal_marketplace.freelancer_job_decisions
            assert decision_key2 in minimal_marketplace.freelancer_job_decisions
            
            decision1 = minimal_marketplace.freelancer_job_decisions[decision_key1]
            decision2 = minimal_marketplace.freelancer_job_decisions[decision_key2]
            
            assert decision1['decision'] == 'no'
            assert decision1['round'] == 1
            assert decision2['decision'] == 'no'
            assert decision2['round'] == 1

            # Round 2-3: Freelancers should not see the job (cooloff active)
            for round_num in [2, 3]:
                available_jobs1 = minimal_marketplace.get_jobs_for_freelancer(freelancer1, [job], round_num)
                available_jobs2 = minimal_marketplace.get_jobs_for_freelancer(freelancer2, [job], round_num)
                
                assert len(available_jobs1) == 0, f"Freelancer 1 should not see job in round {round_num}"
                assert len(available_jobs2) == 0, f"Freelancer 2 should not see job in round {round_num}"

            # Round 4: Cooloff should expire, job becomes available again
            available_jobs1 = minimal_marketplace.get_jobs_for_freelancer(freelancer1, [job], 4)
            available_jobs2 = minimal_marketplace.get_jobs_for_freelancer(freelancer2, [job], 4)
            
            assert len(available_jobs1) == 1, "Freelancer 1 should see job again after cooloff"
            assert len(available_jobs2) == 1, "Freelancer 2 should see job again after cooloff"
            assert available_jobs1[0].id == job.id
            assert available_jobs2[0].id == job.id

    def test_cooloff_with_job_budget_adjustment(self, minimal_marketplace):
        """Test cooloff system when job budgets are adjusted during cooloff period"""
        # Create freelancer
        freelancer = Freelancer(
            id="freelancer_1", name="Alice",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=70.0,
            personality="Test", motivation="Test", background="Test"
        )
        minimal_marketplace.freelancers = [freelancer]

        # Create job with low budget initially
        job = Job(
            id="test_job", client_id="client_1",
            title="Python Job", description="Python development",
            skills_required=["Python"], budget_type="hourly", budget_amount=50.0,  # Below freelancer's rate
            timeline="1 month", special_requirements=[], category="Software",
            posted_time=None
        )
        minimal_marketplace.active_jobs = [job]

        # Round 1: Freelancer declines due to low budget
        decision_key = (freelancer.id, job.id)
        minimal_marketplace.freelancer_job_decisions[decision_key] = {
            'decision': 'no',
            'round': 1
        }

        # During cooloff period: Simulate budget increase
        job.budget_amount = 80.0  # Increase budget above freelancer's rate

        # Round 2-3: Still in cooloff, should not see job despite budget increase
        for round_num in [2, 3]:
            available_jobs = minimal_marketplace.get_jobs_for_freelancer(freelancer, [job], round_num)
            assert len(available_jobs) == 0, f"Should not see job in round {round_num} despite budget increase"

        # Round 4: Cooloff expires, should see job with new higher budget
        available_jobs = minimal_marketplace.get_jobs_for_freelancer(freelancer, [job], 4)
        assert len(available_jobs) == 1
        assert available_jobs[0].budget_amount == 80.0

    def test_cooloff_mixed_decisions(self, minimal_marketplace):
        """Test cooloff with mixed accept/decline decisions across different jobs"""
        # Create freelancer
        freelancer = Freelancer(
            id="freelancer_1", name="Alice",
            category="Web, Mobile & Software Dev",
            skills=["Python", "Java"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        minimal_marketplace.freelancers = [freelancer]

        # Create multiple jobs
        job1 = Job(
            id="job_1", client_id="client_1", title="Low Budget Job",
            description="Low budget Python job", skills_required=["Python"],
            budget_type="hourly", budget_amount=40.0, timeline="1 month",
            special_requirements=[], category="Software", posted_time=None
        )
        job2 = Job(
            id="job_2", client_id="client_1", title="Good Budget Job",
            description="Good budget Java job", skills_required=["Java"],
            budget_type="hourly", budget_amount=70.0, timeline="1 month",
            special_requirements=[], category="Software", posted_time=None
        )
        job3 = Job(
            id="job_3", client_id="client_1", title="New Job",
            description="Brand new job", skills_required=["Python"],
            budget_type="hourly", budget_amount=60.0, timeline="1 month",
            special_requirements=[], category="Software", posted_time=None
        )

        # Round 1: Make different decisions
        minimal_marketplace.freelancer_job_decisions = {
            (freelancer.id, job1.id): {'decision': 'no', 'round': 1},   # Declined - should cooloff
            (freelancer.id, job2.id): {'decision': 'yes', 'round': 1},  # Accepted - permanently blocked
            # job3 has no decision - should always be available
        }

        # Round 2-3: Check availability during cooloff
        for round_num in [2, 3]:
            available_jobs = minimal_marketplace.get_jobs_for_freelancer(
                freelancer, [job1, job2, job3], round_num
            )
            # Only job3 should be available
            assert len(available_jobs) == 1
            assert available_jobs[0].id == job3.id

        # Round 4: Cooloff expires for job1
        available_jobs = minimal_marketplace.get_jobs_for_freelancer(
            freelancer, [job1, job2, job3], 4
        )
        # job1 and job3 should be available, job2 still blocked
        assert len(available_jobs) == 2
        available_ids = {job.id for job in available_jobs}
        assert "job_1" in available_ids
        assert "job_3" in available_ids
        assert "job_2" not in available_ids

    def test_cooloff_disabled_behavior(self):
        """Test marketplace behavior when cooloff is completely disabled"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=3,
            bids_per_round=1, jobs_per_freelancer_per_round=2,
            job_selection_method="random", relevance_mode="moderate",
            enable_reflections=False, max_workers=1,
            bid_cooloff_rounds=0,  # Disabled
            use_cache=False
        )

        freelancer = Freelancer(
            id="freelancer_1", name="Alice",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )

        job = Job(
            id="test_job", client_id="client_1", title="Test Job",
            description="Test job", skills_required=["Python"],
            budget_type="hourly", budget_amount=40.0, timeline="1 month",
            special_requirements=[], category="Software", posted_time=None
        )

        # Record a declined decision
        marketplace.freelancer_job_decisions[(freelancer.id, job.id)] = {
            'decision': 'no',
            'round': 1
        }

        # With cooloff disabled, job should never become available again
        for round_num in [2, 5, 10, 100]:
            available_jobs = marketplace.get_jobs_for_freelancer(freelancer, [job], round_num)
            assert len(available_jobs) == 0, f"Job should remain blocked in round {round_num} with disabled cooloff"

    def test_cooloff_with_multiple_freelancers(self, minimal_marketplace):
        """Test that cooloff works independently for different freelancers"""
        # Create two freelancers
        freelancer1 = Freelancer(
            id="freelancer_1", name="Alice",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=50.0,
            personality="Test", motivation="Test", background="Test"
        )
        freelancer2 = Freelancer(
            id="freelancer_2", name="Bob",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=60.0,
            personality="Test", motivation="Test", background="Test"
        )

        job = Job(
            id="test_job", client_id="client_1", title="Test Job",
            description="Test job", skills_required=["Python"],
            budget_type="hourly", budget_amount=40.0, timeline="1 month",
            special_requirements=[], category="Software", posted_time=None
        )

        # Freelancer 1 declines in round 1, Freelancer 2 declines in round 2
        minimal_marketplace.freelancer_job_decisions = {
            (freelancer1.id, job.id): {'decision': 'no', 'round': 1},
            (freelancer2.id, job.id): {'decision': 'no', 'round': 2},
        }

        # Round 3: Both should be in cooloff
        available_jobs1 = minimal_marketplace.get_jobs_for_freelancer(freelancer1, [job], 3)
        available_jobs2 = minimal_marketplace.get_jobs_for_freelancer(freelancer2, [job], 3)
        assert len(available_jobs1) == 0
        assert len(available_jobs2) == 0

        # Round 4: Freelancer 1's cooloff expires (4-1=3 >= 3), Freelancer 2 still in cooloff
        available_jobs1 = minimal_marketplace.get_jobs_for_freelancer(freelancer1, [job], 4)
        available_jobs2 = minimal_marketplace.get_jobs_for_freelancer(freelancer2, [job], 4)
        assert len(available_jobs1) == 1  # Cooloff expired
        assert len(available_jobs2) == 0   # Still in cooloff

        # Round 5: Both cooloffs should be expired
        available_jobs1 = minimal_marketplace.get_jobs_for_freelancer(freelancer1, [job], 5)
        available_jobs2 = minimal_marketplace.get_jobs_for_freelancer(freelancer2, [job], 5)
        assert len(available_jobs1) == 1
        assert len(available_jobs2) == 1


class TestBidCooloffRealWorldScenarios:
    """Test bid cooloff in realistic marketplace scenarios"""

    def test_freelancer_rate_adjustment_and_cooloff(self):
        """Test scenario where freelancer adjusts rate during cooloff period"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=6,
            bids_per_round=1, jobs_per_freelancer_per_round=1,
            enable_reflections=False, max_workers=1,
            bid_cooloff_rounds=3, use_cache=False
        )

        # Freelancer with high initial rate
        freelancer = Freelancer(
            id="freelancer_1", name="Alice",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=100.0,
            personality="Test", motivation="Test", background="Test"
        )

        # Job with moderate budget
        job = Job(
            id="test_job", client_id="client_1", title="Python Job",
            description="Python development", skills_required=["Python"],
            budget_type="hourly", budget_amount=70.0, timeline="1 month",
            special_requirements=[], category="Software", posted_time=None
        )

        # Round 1: Freelancer declines due to low budget relative to their rate
        marketplace.freelancer_job_decisions[(freelancer.id, job.id)] = {
            'decision': 'no',
            'round': 1
        }

        # Round 2: Freelancer adjusts rate downward (market learning)
        freelancer.min_hourly_rate = 60.0

        # Round 2-3: Still in cooloff despite rate adjustment
        for round_num in [2, 3]:
            available_jobs = marketplace.get_jobs_for_freelancer(freelancer, [job], round_num)
            assert len(available_jobs) == 0

        # Round 4: Cooloff expires, freelancer can now reconsider with new rate
        available_jobs = marketplace.get_jobs_for_freelancer(freelancer, [job], 4)
        assert len(available_jobs) == 1
        # The job budget (70.0) is now above the freelancer's adjusted rate (60.0)

    def test_job_reposting_and_cooloff(self):
        """Test scenario where similar jobs are reposted during cooloff"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=6,
            bids_per_round=1, jobs_per_freelancer_per_round=3,
            enable_reflections=False, max_workers=1,
            bid_cooloff_rounds=3, use_cache=False
        )

        freelancer = Freelancer(
            id="freelancer_1", name="Alice",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=60.0,
            personality="Test", motivation="Test", background="Test"
        )

        # Original job that freelancer declined
        job1 = Job(
            id="job_1", client_id="client_1", title="Python Job v1",
            description="Original Python job", skills_required=["Python"],
            budget_type="hourly", budget_amount=50.0, timeline="1 month",
            special_requirements=[], category="Software", posted_time=None
        )

        # Similar job posted later (different ID)
        job2 = Job(
            id="job_2", client_id="client_1", title="Python Job v2",
            description="Improved Python job", skills_required=["Python"],
            budget_type="hourly", budget_amount=65.0, timeline="1 month",
            special_requirements=[], category="Software", posted_time=None
        )

        # Round 1: Decline original job
        marketplace.freelancer_job_decisions[(freelancer.id, job1.id)] = {
            'decision': 'no',
            'round': 1
        }

        # Round 2: New similar job appears
        # Freelancer should see the new job (different ID, no cooloff)
        available_jobs = marketplace.get_jobs_for_freelancer(freelancer, [job1, job2], 2)
        assert len(available_jobs) == 1
        assert available_jobs[0].id == job2.id  # Only the new job

        # Round 4: Original job cooloff expires
        available_jobs = marketplace.get_jobs_for_freelancer(freelancer, [job1, job2], 4)
        assert len(available_jobs) == 2  # Both jobs available

    def test_cooloff_persistence_across_simulation(self):
        """Test that cooloff decisions persist properly throughout simulation"""
        marketplace = TrueGPTMarketplace(
            num_freelancers=1, num_clients=1, rounds=8,
            bids_per_round=1, jobs_per_freelancer_per_round=1,
            enable_reflections=False, max_workers=1,
            bid_cooloff_rounds=5, use_cache=False
        )

        freelancer = Freelancer(
            id="freelancer_1", name="Alice",
            category="Web, Mobile & Software Dev",
            skills=["Python"], min_hourly_rate=60.0,
            personality="Test", motivation="Test", background="Test"
        )

        job = Job(
            id="persistent_job", client_id="client_1", title="Long-running Job",
            description="Job that stays active", skills_required=["Python"],
            budget_type="hourly", budget_amount=50.0, timeline="6 months",
            special_requirements=[], category="Software", posted_time=None
        )

        # Simulate decisions over multiple rounds
        decisions_log = []

        # Round 1: Initial decline
        marketplace.freelancer_job_decisions[(freelancer.id, job.id)] = {
            'decision': 'no',
            'round': 1
        }
        decisions_log.append((1, 'declined'))

        # Rounds 2-5: Should be blocked
        for round_num in range(2, 6):
            available_jobs = marketplace.get_jobs_for_freelancer(freelancer, [job], round_num)
            decisions_log.append((round_num, f'blocked ({len(available_jobs)} jobs available)'))
            assert len(available_jobs) == 0

        # Round 6: Cooloff expires
        available_jobs = marketplace.get_jobs_for_freelancer(freelancer, [job], 6)
        decisions_log.append((6, f'available again ({len(available_jobs)} jobs available)'))
        assert len(available_jobs) == 1

        # Round 6: Decline again
        marketplace.freelancer_job_decisions[(freelancer.id, job.id)] = {
            'decision': 'no',
            'round': 6
        }
        decisions_log.append((6, 'declined again'))

        # Round 7-10: Should be blocked again
        for round_num in range(7, 11):
            available_jobs = marketplace.get_jobs_for_freelancer(freelancer, [job], round_num)
            decisions_log.append((round_num, f'blocked again ({len(available_jobs)} jobs available)'))
            assert len(available_jobs) == 0

        # Verify the complete decision timeline
        expected_pattern = [
            (1, 'declined'),
            (2, 'blocked (0 jobs available)'),
            (3, 'blocked (0 jobs available)'),
            (4, 'blocked (0 jobs available)'),
            (5, 'blocked (0 jobs available)'),
            (6, 'available again (1 jobs available)'),
            (6, 'declined again'),
            (7, 'blocked again (0 jobs available)'),
            (8, 'blocked again (0 jobs available)'),
            (9, 'blocked again (0 jobs available)'),
            (10, 'blocked again (0 jobs available)')
        ]
        
        assert decisions_log == expected_pattern
