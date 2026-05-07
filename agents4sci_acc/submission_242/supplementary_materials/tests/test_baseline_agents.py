#!/usr/bin/env python3
"""
Comprehensive tests for baseline agent implementations in TrueGPTMarketplace.

These tests ensure that all baseline agent types (random, greedy, no_reputation)
work correctly and produce deterministic, interpretable results for benchmarking.
"""

import pytest
import random
from datetime import datetime
from unittest.mock import patch

from src.marketplace.true_gpt_marketplace import TrueGPTMarketplace
from src.marketplace.entities import Freelancer, Client, Job, Bid, HiringDecision


class TestBaselineAgentConfiguration:
    """Test that baseline agent configuration works correctly."""

    def test_valid_agent_types_accepted(self):
        """Test that all valid agent types are accepted."""
        valid_types = ['llm', 'random', 'greedy', 'no_reputation']
        
        for freelancer_type in valid_types:
            for client_type in valid_types:
                marketplace = TrueGPTMarketplace(
                    freelancer_agent_type=freelancer_type,
                    client_agent_type=client_type,
                    rounds=1,
                    quiet_mode=True
                )
                assert marketplace.freelancer_agent_type == freelancer_type
                assert marketplace.client_agent_type == client_type

    def test_invalid_agent_types_rejected(self):
        """Test that invalid agent types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid freelancer_agent_type"):
            TrueGPTMarketplace(freelancer_agent_type='invalid')
            
        with pytest.raises(ValueError, match="Invalid client_agent_type"):
            TrueGPTMarketplace(client_agent_type='invalid')
    
    def test_invalid_random_bid_probability_rejected(self):
        """Test that invalid random freelancer bid probability raises ValueError."""
        with pytest.raises(ValueError, match="random_freelancer_bid_probability must be between 0.0 and 1.0"):
            TrueGPTMarketplace(random_freelancer_bid_probability=1.5)
            
        with pytest.raises(ValueError, match="random_freelancer_bid_probability must be between 0.0 and 1.0"):
            TrueGPTMarketplace(random_freelancer_bid_probability=-0.1)

    def test_baseline_parameters_stored_correctly(self):
        """Test that baseline configuration parameters are stored correctly."""
        marketplace = TrueGPTMarketplace(
            baseline_greedy_undercut=0.8,
            random_freelancer_bid_probability=0.3,
            quiet_mode=True
        )
        assert marketplace.baseline_greedy_undercut == 0.8
        assert marketplace.random_freelancer_bid_probability == 0.3


class TestFreelancerBiddingBaselines:
    """Test freelancer bidding behavior for each baseline type."""

    def setup_method(self):
        """Set up test fixtures."""
        self.freelancer = Freelancer(
            id="test_freelancer",
            name="Test Freelancer",
            category="Web, Mobile & Software Dev",
            skills=["python", "web development"],
            min_hourly_rate=50.0,
            personality="Professional and reliable",
            motivation="Build portfolio and gain experience",
            background="Software developer with web development experience"
        )
        
        self.job = Job(
            id="test_job",
            client_id="test_client",
            title="Test Job",
            description="A test job",
            category="Web, Mobile & Software Dev",
            skills_required=["python"],
            budget_type="fixed",
            budget_amount=100.0,
            timeline="1 month",
            posted_time=datetime.now()
        )

    def test_random_bidding_behavior(self):
        """Test that random bidding produces expected behavior."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='random',
            quiet_mode=True
        )
        
        # Test multiple decisions with fixed seeds for reproducibility
        decisions = []
        for seed in range(10):
            random.seed(seed)
            decision = marketplace._freelancer_bidding_decision_random(
                self.freelancer, self.job, 1
            )
            decisions.append(decision['decision'])
        
        # Should have a mix of 'yes' and 'no' decisions
        unique_decisions = set(decisions)
        assert len(unique_decisions) > 1, "Random bidding should produce both yes and no decisions"
        assert 'yes' in unique_decisions or 'no' in unique_decisions
        
        # Test specific behavior with known seed
        random.seed(42)  # Seed that should produce 'yes' decision
        decision = marketplace._freelancer_bidding_decision_random(
            self.freelancer, self.job, 1
        )
        
        # Verify structure regardless of decision
        assert decision['decision'] in ['yes', 'no']
        assert 'Random bidding baseline' in decision['reasoning']
        assert '50% probability' in decision['reasoning']  # Default probability
    
    def test_random_bidding_custom_probability(self):
        """Test that random bidding respects custom probability settings."""
        # Test with 10% probability (should bid less often)
        marketplace_low = TrueGPTMarketplace(
            freelancer_agent_type='random',
            random_freelancer_bid_probability=0.1,
            quiet_mode=True
        )
        
        # Test with 90% probability (should bid more often)
        marketplace_high = TrueGPTMarketplace(
            freelancer_agent_type='random',
            random_freelancer_bid_probability=0.9,
            quiet_mode=True
        )
        
        # Run multiple trials to test probability differences
        low_prob_bids = 0
        high_prob_bids = 0
        trials = 100
        
        for seed in range(trials):
            # Test low probability
            random.seed(seed)
            decision_low = marketplace_low._freelancer_bidding_decision_random(
                self.freelancer, self.job, 1
            )
            if decision_low['decision'] == 'yes':
                low_prob_bids += 1
                assert '10% probability' in decision_low['reasoning']
            else:
                assert '10% probability' in decision_low['reasoning']
                
            # Test high probability
            random.seed(seed)
            decision_high = marketplace_high._freelancer_bidding_decision_random(
                self.freelancer, self.job, 1
            )
            if decision_high['decision'] == 'yes':
                high_prob_bids += 1
                assert '90% probability' in decision_high['reasoning']
            else:
                assert '90% probability' in decision_high['reasoning']
        
        # High probability should result in significantly more bids than low probability
        assert high_prob_bids > low_prob_bids, f"High prob bids: {high_prob_bids}, Low prob bids: {low_prob_bids}"
        
        # With enough trials, should be roughly close to expected probabilities (with some variance)
        assert 0.03 <= low_prob_bids/trials <= 0.17, f"Low probability rate: {low_prob_bids/trials} should be around 0.1"
        assert 0.83 <= high_prob_bids/trials <= 0.97, f"High probability rate: {high_prob_bids/trials} should be around 0.9"

    def test_random_bidding_budget_constraint(self):
        """Test that random bidding respects bid budget constraints."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='random',
            bids_per_round=1,
            quiet_mode=True
        )
        
        # Exhaust freelancer's bid budget
        self.freelancer.bids_this_round = 1
        
        decision = marketplace._freelancer_bidding_decision_random(
            self.freelancer, self.job, 1
        )
        
        assert decision['decision'] == 'no'
        assert 'Already used up bid budget' in decision['reasoning']

    def test_greedy_bidding_behavior(self):
        """Test that greedy bidding always bids with correct pricing."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='greedy',
            baseline_greedy_undercut=0.9,
            quiet_mode=True
        )
        
        decision = marketplace._freelancer_bidding_decision_greedy(
            self.freelancer, self.job, 1
        )
        
        assert decision['decision'] == 'yes'
        assert 'Greedy baseline' in decision['reasoning']
        assert '90%' in decision['reasoning']  # Should mention undercut percentage

    def test_no_reputation_bidding_behavior(self):
        """Test that no_reputation bidding works like greedy."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='no_reputation',
            quiet_mode=True
        )
        
        decision = marketplace._freelancer_bidding_decision_no_reputation(
            self.freelancer, self.job, 1
        )
        
        # Should behave exactly like greedy
        greedy_decision = marketplace._freelancer_bidding_decision_greedy(
            self.freelancer, self.job, 1
        )
        
        assert decision['decision'] == greedy_decision['decision']
        assert decision['reasoning'] == greedy_decision['reasoning']


class TestClientHiringBaselines:
    """Test client hiring behavior for each baseline type."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = Client(
            id="test_client",
            company_name="Test Company",
            company_size="medium",
            budget_philosophy="balanced",
            hiring_style="professional",
            background="Software development company",
            business_category="SOFTWARE"
        )
        
        self.job = Job(
            id="test_job",
            client_id="test_client",
            title="Test Job",
            description="A test job",
            category="Web, Mobile & Software Dev",
            skills_required=["python"],
            budget_type="fixed",
            budget_amount=100.0,
            timeline="1 month",
            posted_time=datetime.now()
        )
        
        # Create some test bids
        self.bids = [
            Bid(
                freelancer_id="freelancer_1",
                job_id="test_job",
                proposed_rate=80.0,
                message="First bid",
                submission_time=datetime.now()
            ),
            Bid(
                freelancer_id="freelancer_2", 
                job_id="test_job",
                proposed_rate=90.0,
                message="Second bid",
                submission_time=datetime.now()
            ),
            Bid(
                freelancer_id="freelancer_3",
                job_id="test_job",
                proposed_rate=120.0,  # Over budget
                message="Third bid",
                submission_time=datetime.now()
            )
        ]

    def test_random_hiring_behavior(self):
        """Test that random hiring has 50% chance to hire, picks random when hiring."""
        marketplace = TrueGPTMarketplace(
            client_agent_type='random',
            quiet_mode=True
        )
        
        # Test with multiple seeds to verify randomness
        selected_freelancers = []
        hire_decisions = []
        
        for seed in range(50):  # More samples for better statistics
            random.seed(seed)
            decision = marketplace._client_hiring_decision_random(
                self.client, self.job, self.bids
            )
            selected_freelancers.append(decision.selected_freelancer)
            hire_decisions.append(decision.selected_freelancer is not None)
        
        # Should hire roughly 50% of the time (with some statistical variation)
        hire_rate = sum(hire_decisions) / len(hire_decisions)
        assert 0.3 <= hire_rate <= 0.7, f"Expected hire rate around 0.5, got {hire_rate}"
        
        # When hiring, should have variety in selections
        actual_hires = [f for f in selected_freelancers if f is not None]
        if len(actual_hires) > 1:  # Only test if we have multiple hires
            unique_selections = set(actual_hires)
            assert len(unique_selections) > 1, f"Expected variety in selections, got: {unique_selections}"
        
        # All selected freelancers should be from our bid list
        valid_freelancer_ids = {"freelancer_1", "freelancer_2", "freelancer_3"}
        assert all(sel in valid_freelancer_ids for sel in unique_selections)
        
        # Test specific behavior with known seed
        random.seed(42)
        decision = marketplace._client_hiring_decision_random(
            self.client, self.job, self.bids
        )
        
        # With random hiring, freelancer can be None (didn't hire) or a valid ID (hired)
        if decision.selected_freelancer is not None:
            assert decision.selected_freelancer in valid_freelancer_ids
            assert 'randomly selected from' in decision.reasoning
        else:
            assert 'randomly decided not to hire' in decision.reasoning
        assert 'Random hiring baseline' in decision.reasoning

    def test_random_hiring_no_bids(self):
        """Test random hiring with no bids."""
        marketplace = TrueGPTMarketplace(
            client_agent_type='random',
            quiet_mode=True
        )
        
        decision = marketplace._client_hiring_decision_random(
            self.client, self.job, []
        )
        
        assert decision.selected_freelancer is None
        assert decision.reasoning == "No bids received"

    def test_greedy_hiring_behavior(self):
        """Test that greedy hiring picks the lowest valid bid."""
        marketplace = TrueGPTMarketplace(
            client_agent_type='greedy',
            quiet_mode=True
        )
        
        decision = marketplace._client_hiring_decision_greedy(
            self.client, self.job, self.bids
        )
        
        # Should pick freelancer_1 with 80.0 rate (lowest within budget)
        assert decision.selected_freelancer == "freelancer_1"
        assert 'Greedy baseline' in decision.reasoning
        assert '$80.00' in decision.reasoning
        assert '2 valid bids' in decision.reasoning  # bids within budget

    def test_greedy_hiring_no_valid_bids(self):
        """Test greedy hiring when no bids are within budget."""
        marketplace = TrueGPTMarketplace(
            client_agent_type='greedy',
            quiet_mode=True
        )
        
        # Job with very low budget
        low_budget_job = Job(
            id="low_budget_job",
            client_id="test_client", 
            title="Low Budget Job",
            description="A low budget job",
            category="Web, Mobile & Software Dev",
            skills_required=["python"],
            budget_type="fixed",
            budget_amount=50.0,  # Lower than all bids
            timeline="1 month",
            posted_time=datetime.now()
        )
        
        decision = marketplace._client_hiring_decision_greedy(
            self.client, low_budget_job, self.bids
        )
        
        assert decision.selected_freelancer is None
        assert 'no bids within budget' in decision.reasoning
        assert '$50.00' in decision.reasoning

    def test_no_reputation_hiring_behavior(self):
        """Test that no_reputation hiring works like greedy."""
        marketplace = TrueGPTMarketplace(
            client_agent_type='no_reputation',
            quiet_mode=True
        )
        
        decision = marketplace._client_hiring_decision_no_reputation(
            self.client, self.job, self.bids
        )
        
        # Should behave exactly like greedy
        greedy_decision = marketplace._client_hiring_decision_greedy(
            self.client, self.job, self.bids
        )
        
        assert decision.selected_freelancer == greedy_decision.selected_freelancer
        assert decision.reasoning == greedy_decision.reasoning


class TestJobSelectionLogic:
    """Test job selection logic for different agent types."""

    def setup_method(self):
        """Set up test fixtures."""
        self.freelancer = Freelancer(
            id="test_freelancer",
            name="Test Freelancer",
            category="Web, Mobile & Software Dev",
            skills=["python", "web development"],
            min_hourly_rate=50.0,
            personality="Professional and reliable",
            motivation="Build portfolio and gain experience"
        )
        
        # Create jobs with different budgets
        self.jobs = [
            Job(id="job_1", client_id="client_1", title="Low Budget Job",
                 description="A low budget job", budget_type="fixed", 
                 budget_amount=50.0, timeline="1 month", category="Web, Mobile & Software Dev",
                 skills_required=["python"], posted_time=datetime.now()),
            Job(id="job_2", client_id="client_2", title="Medium Budget Job",
                 description="A medium budget job", budget_type="fixed", 
                 budget_amount=100.0, timeline="1 month", category="Web, Mobile & Software Dev", 
                 skills_required=["python"], posted_time=datetime.now()),
            Job(id="job_3", client_id="client_3", title="High Budget Job",
                 description="A high budget job", budget_type="fixed", 
                 budget_amount=150.0, timeline="1 month", category="Web, Mobile & Software Dev",
                 skills_required=["python"], posted_time=datetime.now())
        ]

    def test_greedy_job_selection(self):
        """Test that greedy agents see highest budget jobs first."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='greedy',
            jobs_per_freelancer_per_round=2,
            quiet_mode=True
        )
        
        selected_jobs = marketplace.select_jobs_for_freelancer(
            self.freelancer, self.jobs
        )
        
        # Should get the top 2 highest budget jobs
        assert len(selected_jobs) == 2
        assert selected_jobs[0].budget_amount == 150.0  # Highest budget first
        assert selected_jobs[1].budget_amount == 100.0  # Second highest

    def test_no_reputation_job_selection(self):
        """Test that no_reputation agents use same job selection as greedy."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='no_reputation',
            jobs_per_freelancer_per_round=2,
            quiet_mode=True
        )
        
        selected_jobs = marketplace.select_jobs_for_freelancer(
            self.freelancer, self.jobs
        )
        
        # Should behave like greedy
        assert len(selected_jobs) == 2
        assert selected_jobs[0].budget_amount == 150.0
        assert selected_jobs[1].budget_amount == 100.0

    def test_random_job_selection_unchanged(self):
        """Test that random/LLM agents still use original job selection logic."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='random',
            job_selection_method='random',
            jobs_per_freelancer_per_round=2,
            quiet_mode=True
        )
        
        # Test that random selection produces different results with different seeds
        selections = []
        for seed in range(10):
            random.seed(seed)
            selected_jobs = marketplace.select_jobs_for_freelancer(
                self.freelancer, self.jobs
            )
            selections.append(tuple(job.id for job in selected_jobs))
        
        # Should get 2 jobs each time
        assert all(len(selection) == 2 for selection in selections)
        
        # Should have some variation in selections
        unique_selections = set(selections)
        assert len(unique_selections) > 1, "Random job selection should produce different results"


class TestBidPriceStrategies:
    """Test different bid pricing strategies for baseline agents."""

    def setup_method(self):
        """Set up test fixtures."""
        self.freelancer = Freelancer(
            id="test_freelancer",
            name="Test Freelancer", 
            category="Web, Mobile & Software Dev",
            skills=["python"],
            min_hourly_rate=50.0,
            personality="Professional and reliable",
            motivation="Build portfolio and gain experience"
        )
        
        self.job = Job(
            id="test_job",
            client_id="test_client",
            title="Test Job",
            description="A test job",
            budget_type="fixed",
            budget_amount=100.0,
            timeline="1 month",
            category="Web, Mobile & Software Dev",
            skills_required=["python"],
            posted_time=datetime.now()
        )

    def test_uniform_pricing_strategy(self):
        """Test uniform pricing strategy generates bids in 50-150% range."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='random',
            quiet_mode=True
        )
        
        # Test multiple random bids to verify pricing is in expected range
        bid_amounts = []
        for seed in range(50):
            random.seed(seed)
            decision = marketplace._freelancer_bidding_decision_random(
                self.freelancer, self.job, 1
            )
            
            if decision['decision'] == 'yes':
                # Extract bid amount from message
                message = decision['message']
                # Look for pattern like "$125.00"
                import re
                match = re.search(r'\$(\d+\.\d+)', message)
                if match:
                    bid_amount = float(match.group(1))
                    bid_amounts.append(bid_amount)
        
        # Should have gotten some bids
        assert len(bid_amounts) > 0, "Should generate some bids with uniform pricing"
        
        # All bid amounts should be in 50-150% range of job budget (50-150)
        job_budget = self.job.budget_amount  # 100.0
        min_expected = job_budget * 0.5  # 50.0
        max_expected = job_budget * 1.5  # 150.0
        
        for amount in bid_amounts:
            assert min_expected <= amount <= max_expected, \
                f"Bid amount {amount} should be between {min_expected} and {max_expected}"

    def test_greedy_undercut_pricing(self):
        """Test greedy agents always use undercut pricing."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='greedy',
            baseline_greedy_undercut=0.8,
            quiet_mode=True
        )
        
        decision = marketplace._freelancer_bidding_decision_greedy(
            self.freelancer, self.job, 1
        )
        
        assert 'undercutting at 80%' in decision['reasoning']
        assert '$80.00' in decision['reasoning']

    def test_custom_undercut_percentage(self):
        """Test greedy agents use custom undercut percentage."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='greedy',
            baseline_greedy_undercut=0.85,  # 85%
            quiet_mode=True
        )
        
        decision = marketplace._freelancer_bidding_decision_greedy(
            self.freelancer, self.job, 1
        )
        
        assert 'undercutting at 85%' in decision['reasoning']
        assert '$85.00' in decision['reasoning']  # 100 * 0.85


class TestBaselineIntegration:
    """Test integration of baseline agents with the full marketplace."""

    def test_baseline_agents_work_with_main_decision_methods(self):
        """Test that baseline agents integrate correctly with main decision dispatch."""
        marketplace = TrueGPTMarketplace(
            freelancer_agent_type='greedy',
            client_agent_type='random',
            quiet_mode=True
        )
        
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test Freelancer",
            category="Web, Mobile & Software Dev", 
            skills=["python"],
            min_hourly_rate=50.0,
            personality="Professional and reliable",
            motivation="Build portfolio and gain experience"
        )
        
        job = Job(
            id="test_job",
            client_id="test_client",
            title="Test Job",
            description="A test job",
            budget_type="fixed",
            budget_amount=100.0,
            timeline="1 month",
            category="Web, Mobile & Software Dev",
            skills_required=["python"],
            posted_time=datetime.now()
        )
        
        # Test freelancer decision dispatch
        decision = marketplace.freelancer_bidding_decision(freelancer, job, 1)
        assert decision['decision'] == 'yes'  # Greedy always bids
        assert 'Greedy baseline' in decision['reasoning']
        
        # Test client decision dispatch
        client = Client(
            id="test_client",
            company_name="Test Company",
            company_size="medium",
            budget_philosophy="balanced",
            hiring_style="professional",
            background="Software development company",
            business_category="SOFTWARE"
        )
        
        bids = [
            Bid(freelancer_id="freelancer_1", job_id="test_job", 
                proposed_rate=80.0, message="Test bid", submission_time=datetime.now())
        ]
        
        # Set seed for reproducible behavior
        random.seed(42)
        
        hiring_decision = marketplace.client_hiring_decision(client, job, bids)
        # With 50% probability, might hire or might not hire
        if hiring_decision.selected_freelancer is not None:
            assert hiring_decision.selected_freelancer == "freelancer_1"  # Only one bid available
            assert 'randomly selected from' in hiring_decision.reasoning
        else:
            assert 'randomly decided not to hire' in hiring_decision.reasoning
        assert 'Random hiring baseline' in hiring_decision.reasoning

    def test_configuration_validation_in_main_class(self):
        """Test that the main class properly validates baseline configuration."""
        # Test invalid agent types are caught
        with pytest.raises(ValueError):
            TrueGPTMarketplace(freelancer_agent_type='invalid_type')
            
        with pytest.raises(ValueError): 
            TrueGPTMarketplace(client_agent_type='invalid_type')

    def test_baseline_agents_produce_deterministic_results(self):
        """Test that baseline agents produce deterministic results for reproducibility."""
        # Greedy agents should be completely deterministic
        marketplace1 = TrueGPTMarketplace(
            freelancer_agent_type='greedy',
            client_agent_type='greedy',
            baseline_greedy_undercut=0.9,
            quiet_mode=True
        )
        
        marketplace2 = TrueGPTMarketplace(
            freelancer_agent_type='greedy',
            client_agent_type='greedy',
            baseline_greedy_undercut=0.9,
            quiet_mode=True
        )
        
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test Freelancer",
            category="Web, Mobile & Software Dev",
            skills=["python"],
            min_hourly_rate=50.0,
            personality="Professional and reliable",
            motivation="Build portfolio and gain experience"
        )
        
        job = Job(
            id="test_job",
            client_id="test_client", 
            title="Test Job",
            description="A test job",
            budget_type="fixed",
            budget_amount=100.0,
            timeline="1 month",
            category="Web, Mobile & Software Dev",
            skills_required=["python"],
            posted_time=datetime.now()
        )
        
        decision1 = marketplace1.freelancer_bidding_decision(freelancer, job, 1)
        decision2 = marketplace2.freelancer_bidding_decision(freelancer, job, 1)
        
        # Should be identical
        assert decision1['decision'] == decision2['decision']
        assert decision1['reasoning'] == decision2['reasoning']


if __name__ == '__main__':
    pytest.main([__file__])
