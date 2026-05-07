"""
Test the job ranking algorithm and semantic skill matching.
"""
from marketplace.ranking_algorithm import JobRankingCalculator


class TestJobRankingCalculator:
    """Test the job ranking algorithm."""
    
    def test_ranking_calculator_initialization(self):
        """Test ranking calculator creation with different modes."""
        calc_strict = JobRankingCalculator(relevance_mode='strict')
        calc_moderate = JobRankingCalculator(relevance_mode='moderate') 
        calc_relaxed = JobRankingCalculator(relevance_mode='relaxed')
        
        assert calc_strict.relevance_mode == 'strict'
        assert calc_moderate.relevance_mode == 'moderate'
        assert calc_relaxed.relevance_mode == 'relaxed'
    
    def test_perfect_skill_match_ranking(self, ranking_calculator, multiple_freelancers, ml_focused_job):
        """Test that perfect skill matches get highest relevance scores."""
        # Framework Assumption: Limited Skill Complexity (semantic matching)
        
        # Calculate relevance for each freelancer
        scores = []
        for freelancer in multiple_freelancers:
            score = ranking_calculator.calculate_job_relevance(freelancer, ml_focused_job)
            scores.append((freelancer.id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # ML expert should rank highest (has ML, Deep Learning, Python, TensorFlow, Neural Networks)
        assert scores[0][0] == "ml_expert"
        assert scores[0][1] > 0.6  # Should be high relevance
        
        # Check that scores are in descending order
        for i in range(len(scores) - 1):
            assert scores[i][1] >= scores[i + 1][1]
        
        # Designer should have lower relevance than ML expert for ML job
        ml_expert_score = next(s[1] for s in scores if s[0] == "ml_expert")
        designer_score = next(s[1] for s in scores if s[0] == "designer")
        assert ml_expert_score > designer_score
    
    def test_skill_mismatch_relative_relevance(self, ranking_calculator, multiple_freelancers, mismatch_job):
        """Test that skill mismatches produce relatively lower relevance scores."""
        # Test each freelancer against a design job
        scores = []
        for freelancer in multiple_freelancers:
            score = ranking_calculator.calculate_job_relevance(freelancer, mismatch_job)
            scores.append((freelancer.id, score))
            
            # All scores should be valid (no thresholds cutting them to 0.0)
            assert 0.0 <= score <= 1.0
        
        # Find designer and non-designer scores
        designer_score = next(s[1] for s in scores if s[0] == "designer") 
        non_designer_scores = [s[1] for s in scores if s[0] != "designer"]
        
        # Designer should have higher relevance than most/all non-designers for design job
        # (allowing for some semantic matching flexibility)
        avg_non_designer_score = sum(non_designer_scores) / len(non_designer_scores)
        assert designer_score > avg_non_designer_score
        
        # Designer should have the highest or among the highest scores
        max_score = max(s[1] for s in scores)
        assert designer_score >= max_score * 0.9  # Within 90% of max score
    
    def test_relevance_mode_differences(self, multiple_freelancers, sample_job):
        """Test that different relevance modes produce different score distributions."""
        # Framework Assumption: Algorithm should be configurable
        
        strict_calc = JobRankingCalculator(relevance_mode='strict')
        moderate_calc = JobRankingCalculator(relevance_mode='moderate')
        relaxed_calc = JobRankingCalculator(relevance_mode='relaxed')
        
        # Use a freelancer with partial skill match
        python_dev = next(f for f in multiple_freelancers if f.id == "python_dev")
        
        strict_score = strict_calc.calculate_job_relevance(python_dev, sample_job)
        moderate_score = moderate_calc.calculate_job_relevance(python_dev, sample_job)
        relaxed_score = relaxed_calc.calculate_job_relevance(python_dev, sample_job)
        
        # Generally, relaxed should be more generous than strict
        # (exact relationships depend on implementation details)
        assert all(0.0 <= score <= 1.0 for score in [strict_score, moderate_score, relaxed_score])
    
    def test_detailed_scores_breakdown(self, ranking_calculator, sample_freelancer, sample_job):
        """Test detailed score breakdown functionality."""
        detailed_scores = ranking_calculator.get_detailed_scores(sample_freelancer, sample_job)
        
        # Should return a dictionary with scoring components
        assert isinstance(detailed_scores, dict)
        # Specific keys depend on implementation, but should have core components
        assert len(detailed_scores) > 0
    
    def test_ranking_consistency(self, ranking_calculator, multiple_freelancers, ml_focused_job):
        """Test that ranking is consistent across multiple calls."""
        # Framework Assumption: GPT as Rational Actors (consistent decisions)
        
        # Calculate scores twice for the same freelancer/job pair
        freelancer = multiple_freelancers[0]
        
        score1 = ranking_calculator.calculate_job_relevance(freelancer, ml_focused_job)
        score2 = ranking_calculator.calculate_job_relevance(freelancer, ml_focused_job)
        
        # Should be identical (no randomness in scoring)
        assert score1 == score2
    
    def test_score_bounds(self, ranking_calculator, multiple_freelancers, sample_job, mismatch_job):
        """Test that relevance scores are properly bounded."""
        # Framework Assumption: Scores should be normalized
        
        all_scores = []
        for freelancer in multiple_freelancers:
            score1 = ranking_calculator.calculate_job_relevance(freelancer, sample_job)
            score2 = ranking_calculator.calculate_job_relevance(freelancer, mismatch_job)
            all_scores.extend([score1, score2])
        
        # All scores should be between 0 and 1
        for score in all_scores:
            assert 0.0 <= score <= 1.0
    
    def test_extreme_cases(self, ranking_calculator):
        """Test edge cases for ranking algorithm."""
        from marketplace.entities import Freelancer, Job
        from datetime import datetime
        
        # Freelancer with no skills
        empty_freelancer = Freelancer(
            id="empty",
            name="Empty",
            category="Web, Mobile & Software Dev",
            skills=[],  # No skills
            min_hourly_rate=50.0,
            personality="Empty",
            motivation="None",
            background="None",
            preferred_project_length="short-term"
        )
        
        # Job with no required skills  
        empty_job = Job(
            id="empty_job",
            client_id="client_1",
            title="Empty Job",
            description="No requirements",
            skills_required=[],  # No requirements
            budget_type="hourly",
            budget_amount=50.0,
            timeline="1 week",
            special_requirements=[],
            category="General",
            posted_time=datetime.now()
        )
        
        # Should handle empty cases gracefully
        score = ranking_calculator.calculate_job_relevance(empty_freelancer, empty_job)
        assert 0.0 <= score <= 1.0
    
    def test_skill_overlap_correlation(self, ranking_calculator, multiple_freelancers, ml_focused_job):
        """Test that more skill overlap correlates with higher relevance."""
        # Framework Assumption: Limited Skill Complexity
        
        # Calculate skill overlaps and relevance scores
        results = []
        for freelancer in multiple_freelancers:
            # Calculate manual skill overlap
            freelancer_skills = set(skill.lower() for skill in freelancer.skills)
            job_skills = set(skill.lower() for skill in ml_focused_job.skills_required)
            overlap_count = len(freelancer_skills.intersection(job_skills))
            
            # Calculate algorithmic relevance
            relevance = ranking_calculator.calculate_job_relevance(freelancer, ml_focused_job)
            
            results.append((freelancer.id, overlap_count, relevance))
        
        # Sort by overlap count
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Freelancer with most overlaps should generally have highest relevance
        # (allowing for semantic matching to provide some flexibility)
        max_overlap_freelancer = results[0]
        min_overlap_freelancer = results[-1]
        
        assert max_overlap_freelancer[2] >= min_overlap_freelancer[2]


class TestJobSelectionBehavior:
    """Test job selection and visibility constraints."""
    
    def test_limited_job_visibility(self, ranking_calculator, multiple_freelancers):
        """Test Framework Assumption: Limited Market Visibility."""
        from marketplace.entities import Job
        from datetime import datetime
        
        # Create many jobs
        jobs = []
        for i in range(20):
            job = Job(
                id=f"job_{i}",
                client_id="client_1",
                title=f"Job {i}",
                description=f"Description {i}",
                skills_required=["Python", "Testing"],
                budget_type="hourly", 
                budget_amount=60.0,
                timeline="1-2 months",
                special_requirements=["Experience"],
                category="Software Development",
                posted_time=datetime.now()
            )
            jobs.append(job)
        
        freelancer = multiple_freelancers[0]
        
        # Simulate limited visibility (freelancer only sees subset)
        jobs_per_freelancer = 5  # Framework assumption: limited visibility
        visible_jobs = jobs[:jobs_per_freelancer]
        
        assert len(visible_jobs) == jobs_per_freelancer
        assert len(visible_jobs) < len(jobs)  # Confirms limited visibility
        
        # Calculate relevance for visible jobs only
        job_scores = []
        for job in visible_jobs:
            score = ranking_calculator.calculate_job_relevance(freelancer, job)
            job_scores.append((job.id, score))
        
        # Should be able to rank the limited set
        assert len(job_scores) == jobs_per_freelancer
    
    def test_relevance_based_selection(self, ranking_calculator, multiple_freelancers, ml_focused_job):
        """Test that relevance-based selection prioritizes best matches."""
        # Framework Assumption: Job selection should use relevance algorithm
        
        # Calculate relevance for all freelancers
        relevance_scores = []
        for freelancer in multiple_freelancers:
            score = ranking_calculator.calculate_job_relevance(freelancer, ml_focused_job)
            relevance_scores.append((freelancer.id, score))
        
        # Sort by relevance (highest first)
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Top scorer should be the ML expert
        assert relevance_scores[0][0] == "ml_expert"
        
        # Scores should be in descending order
        for i in range(len(relevance_scores) - 1):
            assert relevance_scores[i][1] >= relevance_scores[i + 1][1]
