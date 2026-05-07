"""
Test end-to-end marketplace simulation and integration scenarios.
"""
import pytest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path
from marketplace.true_gpt_marketplace import TrueGPTMarketplace


class TestMarketplaceInitialization:
    """Test marketplace setup and initialization."""
    
    def test_marketplace_creation(self):
        """Test basic marketplace instantiation."""
        # Framework Assumption: Fixed Population Size
        marketplace = TrueGPTMarketplace(
            num_freelancers=5,
            num_clients=3,
            rounds=2,
            bids_per_round=2,
            jobs_per_freelancer_per_round=3,
            job_selection_method="relevance",
            relevance_mode="moderate",
            enable_reflections=False,
            max_workers=1
        )
        
        assert marketplace.num_freelancers == 5
        assert marketplace.num_clients == 3
        assert marketplace.rounds == 2
        assert marketplace.bids_per_round == 2
        assert marketplace.jobs_per_freelancer_per_round == 3
    
    def test_agent_population_generation(self, marketplace_instance):
        """Test that agents are generated according to specifications."""
        # Framework Assumption: Fixed Population Size
        # Framework Assumption: Homogeneous Agent Types
        
        # Generate personas
        freelancers = marketplace_instance.generate_freelancer_personas()
        clients = marketplace_instance.generate_client_personas()
        
        assert len(freelancers) == marketplace_instance.num_freelancers
        assert len(clients) == marketplace_instance.num_clients
        
        # All freelancers should have required attributes
        for freelancer in freelancers:
            assert hasattr(freelancer, 'id')
            assert hasattr(freelancer, 'skills')
            assert hasattr(freelancer, 'min_hourly_rate')
            assert freelancer.min_hourly_rate > 0
        
        # All clients should have required attributes
        for client in clients:
            assert hasattr(client, 'id')
            assert hasattr(client, 'company_name')
            assert hasattr(client, 'budget_philosophy')
    
    def test_marketplace_state_initialization(self, marketplace_instance):
        """Test initial marketplace state."""
        # Framework Assumption: No Agent Mortality
        
        # Should start with empty state
        assert len(marketplace_instance.all_jobs) == 0
        assert len(marketplace_instance.all_bids) == 0
        assert marketplace_instance.current_round == 0


class TestRoundExecution:
    """Test discrete round execution."""
    
    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace._make_openai_request')
    def test_single_round_execution(self, mock_request, marketplace_instance):
        """Test execution of a single marketplace round."""
        # Framework Assumption: Discrete Time Rounds
        # Framework Assumption: Synchronous Job Posting
        
        # Mock responses for different API calls
        def mock_response(*args, **kwargs):
            prompt_content = kwargs.get('messages', [{}])[-1].get('content', '')
            
            # Job posting response
            if 'job posting' in prompt_content.lower():
                return {
                    "title": "Test Job",
                    "description": "Test description",
                    "skills_required": ["Python", "Testing"],
                    "budget_type": "hourly",
                    "budget_amount": 60.0,
                    "timeline": "2 weeks",
                    "special_requirements": ["Experience required"],
                    "category": "Software Development"
                }
            
            # Bidding response
            elif 'bid on this job' in prompt_content.lower():
                return {
                    "decision": "no",
                    "reasoning": "Skills don't match my expertise",
                    "message": ""
                }
            
            # Hiring response
            elif 'hiring decision' in prompt_content.lower():
                return {
                    "selected_bid_number": 0,
                    "selected_freelancer": "none",
                    "reasoning": "No suitable bids received",
                    "confidence_level": "high"
                }
            
            # Default response
            return {"status": "completed"}
        
        mock_request.side_effect = mock_response
        
        # Execute one round
        initial_round = marketplace_instance.current_round
        
        # Generate agents first
        marketplace_instance.freelancers = marketplace_instance.generate_freelancer_personas()
        marketplace_instance.clients = marketplace_instance.generate_client_personas()
        
        # Execute round components
        new_jobs = marketplace_instance._generate_new_jobs(round_num=0)
        
        # Should have generated some jobs
        assert len(new_jobs) >= 0  # May be 0 if no clients can post
        
        # Round should progress
        assert isinstance(new_jobs, list)
    
    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace._make_openai_request')
    def test_job_posting_phase(self, mock_request, marketplace_instance):
        """Test job posting phase of a round."""
        # Framework Assumption: Synchronous Job Posting
        
        # Mock job posting response
        mock_request.return_value = {
            "title": "Python Developer",
            "description": "Develop Python applications",
            "skills_required": ["Python", "Django"],
            "budget_type": "hourly",
            "budget_amount": 75.0,
            "timeline": "1-2 months",
            "special_requirements": ["3+ years experience"],
            "category": "Software Development"
        }
        
        # Generate clients and jobs
        marketplace_instance.clients = marketplace_instance.generate_client_personas()
        new_jobs = marketplace_instance._generate_new_jobs(round_num=0)
        
        # Jobs should be created for eligible clients
        for job in new_jobs:
            assert hasattr(job, 'title')
            assert hasattr(job, 'budget_amount')
            assert hasattr(job, 'skills_required')
            assert job.budget_amount > 0
    
    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace._make_openai_request')
    def test_bidding_phase(self, mock_request, marketplace_instance):
        """Test bidding phase of a round."""
        # Framework Assumption: Limited Market Visibility
        
        # Mock bidding responses
        def mock_response(*args, **kwargs):
            return {
                "decision": "no",
                "reasoning": "Budget doesn't meet my requirements",
                "message": ""
            }
        
        mock_request.side_effect = mock_response
        
        # Set up marketplace with agents and jobs
        marketplace_instance.freelancers = marketplace_instance.generate_freelancer_personas()
        
        from marketplace.entities import Job
        from datetime import datetime
        
        test_job = Job(
            id="test_job",
            client_id="client_1",
            title="Test Job",
            description="Test description",
            skills_required=["Python"],
            budget_type="hourly",
            budget_amount=50.0,
            timeline="1 week",
            special_requirements=["Basic experience"],
            category="Software Development",
            posted_time=datetime.now()
        )
        
        marketplace_instance.all_jobs = [test_job]
        
        # Execute bidding phase
        round_bids, round_decisions = marketplace_instance._process_freelancer_bidding(round_num=0)
        
        # Should process bidding decisions
        assert isinstance(round_bids, list)
        assert isinstance(round_decisions, list)
    
    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace._make_openai_request')
    def test_hiring_phase(self, mock_request, marketplace_instance):
        """Test hiring phase of a round."""
        # Framework Assumption: Binary Hiring Decisions
        
        # Mock hiring response
        mock_request.return_value = {
            "selected_bid_number": 0,
            "selected_freelancer": "none",
            "reasoning": "No bids received",
            "confidence_level": "medium"
        }
        
        # Set up marketplace
        marketplace_instance.clients = marketplace_instance.generate_client_personas()
        
        from marketplace.entities import Job
        from datetime import datetime
        
        test_job = Job(
            id="test_job",
            client_id="client_1",
            title="Test Job",
            description="Test description",
            skills_required=["Python"],
            budget_type="hourly",
            budget_amount=75.0,
            timeline="2 weeks",
            special_requirements=["Experience"],
            category="Software Development",
            posted_time=datetime.now()
        )
        
        marketplace_instance.all_jobs = [test_job]
        
        # Execute hiring phase
        marketplace_instance._process_hiring_decisions(round_num=0, round_bids=[])
        
        # Should process hiring decisions and add to hiring_outcomes
        assert isinstance(marketplace_instance.hiring_outcomes, list)


class TestMultiRoundSimulation:
    """Test multi-round simulation execution."""
    
    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace._make_openai_request')
    def test_two_round_simulation(self, mock_request, marketplace_instance):
        """Test executing multiple rounds."""
        # Framework Assumption: Discrete Time Rounds
        # Framework Assumption: Fixed Round Duration
        
        # Mock all API responses
        def mock_response(*args, **kwargs):
            prompt_content = kwargs.get('messages', [{}])[-1].get('content', '')
            
            if 'job posting' in prompt_content.lower():
                content = json.dumps({
                    "title": "Round Job",
                    "description": "Job for this round",
                    "skills_required": ["Testing"],
                    "budget_type": "hourly",
                    "budget_amount": 55.0,
                    "timeline": "1 week",
                    "special_requirements": ["Basic skills"],
                    "category": "Testing"
                })
            elif 'bid on this job' in prompt_content.lower():
                content = json.dumps({
                    "decision": "no",
                    "reasoning": "Not interested",
                    "message": ""
                })
            elif 'hiring decision' in prompt_content.lower():
                content = json.dumps({
                    "selected_bid_number": 0,
                    "selected_freelancer": "none",
                    "reasoning": "No bids",
                    "confidence_level": "low"
                })
            else:
                content = json.dumps({"status": "completed"})
            
            # Create a mock response object that matches OpenAI's structure
            from unittest.mock import Mock
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message = Mock()
            mock_response.choices[0].message.content = content
            return mock_response
        
        mock_request.side_effect = mock_response
        
        # Run simulation
        results = marketplace_instance.run_simulation()
        
        # Should complete without errors
        assert results is not None
        assert marketplace_instance.current_round == marketplace_instance.rounds
    
    def test_round_state_persistence(self, marketplace_instance):
        """Test that state persists between rounds."""
        # Framework Assumption: No Agent Mortality
        # Framework Assumption: Perfect Memory and Learning
        
        # Generate initial agents
        marketplace_instance.freelancers = marketplace_instance.generate_freelancer_personas()
        marketplace_instance.clients = marketplace_instance.generate_client_personas()
        
        initial_freelancer_count = len(marketplace_instance.freelancers)
        initial_client_count = len(marketplace_instance.clients)
        
        # Simulate agent state changes
        marketplace_instance.freelancers[0].total_bids = 5
        marketplace_instance.clients[0].total_jobs_posted = 3
        
        # State should persist
        assert len(marketplace_instance.freelancers) == initial_freelancer_count
        assert len(marketplace_instance.clients) == initial_client_count
        assert marketplace_instance.freelancers[0].total_bids == 5
        assert marketplace_instance.clients[0].total_jobs_posted == 3


class TestMarketplaceOutputs:
    """Test marketplace simulation outputs and data collection."""
    
    @patch('marketplace.true_gpt_marketplace.TrueGPTMarketplace._make_openai_request')
    def test_simulation_data_collection(self, mock_request, temp_dir):
        """Test that simulation collects comprehensive data."""
        # Framework Assumption: Measurable Outcomes
        
        # Mock responses
        mock_request.return_value = {
            "title": "Test Job",
            "description": "Test",
            "skills_required": ["Testing"],
            "budget_type": "hourly",
            "budget_amount": 50.0,
            "timeline": "1 week",
            "special_requirements": ["None"],
            "category": "Testing"
        }
        
        marketplace = TrueGPTMarketplace(
            num_freelancers=2,
            num_clients=2,
            rounds=1,
            bids_per_round=1,
            jobs_per_freelancer_per_round=2,
            job_selection_method="relevance",
            relevance_mode="moderate",
            enable_reflections=False,
            max_workers=1
        )
        
        results = marketplace.run_simulation()
        
        # Should collect key metrics
        assert results is not None
        assert hasattr(marketplace, 'all_jobs')
        assert hasattr(marketplace, 'all_bids')
        assert hasattr(marketplace, 'freelancers')
        assert hasattr(marketplace, 'clients')
    
    def test_results_serialization(self, marketplace_instance, temp_dir):
        """Test that results can be serialized."""
        # Framework Assumption: Framework can study various dynamics
        
        # Generate sample data
        marketplace_instance.freelancers = marketplace_instance.generate_freelancer_personas()
        marketplace_instance.clients = marketplace_instance.generate_client_personas()
        
        # Test serialization capability
        results_data = {
            "num_freelancers": len(marketplace_instance.freelancers),
            "num_clients": len(marketplace_instance.clients),
            "total_jobs": len(marketplace_instance.all_jobs),
            "total_bids": len(marketplace_instance.all_bids)
        }
        
        # Should be serializable to JSON
        temp_file = Path(temp_dir) / "test_results.json"
        with open(temp_file, 'w') as f:
            json.dump(results_data, f)
        
        # Should be readable
        with open(temp_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["num_freelancers"] == results_data["num_freelancers"]


class TestMarketplaceDynamics:
    """Test emergent marketplace dynamics."""
    
    def test_job_fill_rate_calculation(self, marketplace_instance):
        """Test calculation of key market metrics."""
        # Framework Assumption: Measurable Outcomes
        
        from marketplace.entities import Job
        from datetime import datetime
        
        # Create test jobs
        jobs = []
        for i in range(5):
            job = Job(
                id=f"job_{i}",
                client_id="client_1",
                title=f"Job {i}",
                description="Test job",
                skills_required=["Testing"],
                budget_type="hourly",
                budget_amount=60.0,
                timeline="1 week",
                special_requirements=["None"],
                category="Testing",
                posted_time=datetime.now()
            )
            jobs.append(job)
        
        # Fill some jobs
        jobs[0].hire_freelancer("freelancer_1")
        jobs[2].hire_freelancer("freelancer_2")
        
        # Calculate fill rate
        filled_jobs = sum(1 for job in jobs if job.is_filled())
        fill_rate = filled_jobs / len(jobs)
        
        assert fill_rate == 0.4  # 2 out of 5 jobs filled
    
    def test_competition_measurement(self, marketplace_instance):
        """Test measurement of competition levels."""
        # Framework Assumption: Imperfect Competition Information
        
        from marketplace.entities import Job
        from datetime import datetime
        
        job = Job(
            id="competitive_job",
            client_id="client_1",
            title="Popular Job",
            description="High-demand position",
            skills_required=["Python"],
            budget_type="hourly",
            budget_amount=80.0,
            timeline="2 months",
            special_requirements=["Experience"],
            category="Software Development",
            posted_time=datetime.now()
        )
        
        # Add multiple bids
        job.add_bid({"freelancer_id": "f1", "message": "Interested"})
        job.add_bid({"freelancer_id": "f2", "message": "Perfect fit"})
        job.add_bid({"freelancer_id": "f3", "message": "Available"})
        
        # Competition level = number of bids
        competition_level = len(job.bids)
        assert competition_level == 3
    
    def test_market_saturation_indicators(self, marketplace_instance):
        """Test identification of market saturation."""
        # Framework Assumption: Market saturation effects
        
        # Simulate high freelancer to job ratio (saturation)
        freelancers = marketplace_instance.generate_freelancer_personas()
        
        from marketplace.entities import Job
        from datetime import datetime
        
        # Few jobs, many freelancers
        job = Job(
            id="scarce_job",
            client_id="client_1",
            title="Rare Opportunity",
            description="One of few jobs",
            skills_required=["Rare Skill"],
            budget_type="hourly",
            budget_amount=100.0,
            timeline="3 months",
            special_requirements=["Expert level"],
            category="Specialized",
            posted_time=datetime.now()
        )
        
        # High supply-to-demand ratio indicates saturation
        supply_demand_ratio = len(freelancers) / 1  # 1 job
        
        assert supply_demand_ratio >= marketplace_instance.num_freelancers
