"""
Test the unified reflection-based adjustment system.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from marketplace.reflection_adjustments import ReflectionAdjustmentEngine
from marketplace.entities import Freelancer, Client


class TestReflectionAdjustmentEngine:
    """Test the modular reflection adjustment engine."""
    
    def test_engine_initialization(self):
        """Test engine initializes with correct bounds."""
        engine = ReflectionAdjustmentEngine()
        
        assert engine.min_hourly_rate == 5.0
        assert engine.max_hourly_rate == 500.0
        assert engine.min_budget_multiplier == 0.5
        assert engine.max_budget_multiplier == 2.0
    
    def test_freelancer_rate_increase(self):
        """Test freelancer rate adjustment - increase."""
        engine = ReflectionAdjustmentEngine()
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test Worker",
            category="Web, Mobile & Software Dev",
            skills=["Python"],
            min_hourly_rate=50.0,
            personality="test",
            motivation="test",
            background="test",
            preferred_project_length="medium"
        )
        
        rate_adjustment = {
            'change': 'increase',
            'new_rate': 75.0,
            'reasoning': 'Market demand is high'
        }
        
        result = engine.apply_freelancer_rate_adjustment(
            freelancer, rate_adjustment, round_num=5
        )
        
        assert result is not None
        assert result['old_rate'] == 50.0
        assert result['new_rate'] == 75.0
        assert freelancer.min_hourly_rate == 75.0
        assert freelancer.last_rate_adjustment_round == 5
    
    def test_freelancer_rate_bounds(self):
        """Test freelancer rate adjustment respects bounds."""
        engine = ReflectionAdjustmentEngine()
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test Worker",
            category="Web, Mobile & Software Dev",
            skills=["Python"],
            min_hourly_rate=25.0,
            personality="test",
            motivation="test",
            background="test",
            preferred_project_length="medium"
        )
        
        # Test minimum bound
        rate_adjustment = {
            'change': 'decrease',
            'new_rate': 1.0,  # Below minimum
            'reasoning': 'Testing minimum bound'
        }
        
        result = engine.apply_freelancer_rate_adjustment(
            freelancer, rate_adjustment, round_num=1
        )
        
        assert result is not None
        assert freelancer.min_hourly_rate == engine.min_hourly_rate
    
    def test_client_budget_increase(self):
        """Test client budget adjustment - increase."""
        engine = ReflectionAdjustmentEngine()
        client = Client(
            id="test_client",
            company_name="Test Corp",
            company_size="medium",
            budget_philosophy="balanced",
            hiring_style="thorough",
            background="Test company",
            business_category="SOFTWARE"
        )
        
        budget_adjustment = {
            'change': 'increase',
            'reasoning': 'Need better candidates'
        }
        
        old_multiplier = client.budget_multiplier
        result = engine.apply_client_budget_adjustment(
            client, budget_adjustment, round_num=3
        )
        
        assert result is not None
        assert result['old_multiplier'] == old_multiplier
        assert result['new_multiplier'] == old_multiplier * 1.15
        assert client.budget_multiplier == old_multiplier * 1.15
        assert client.last_budget_adjustment_round == 3
    
    def test_no_adjustment_when_keep(self):
        """Test no adjustment when change is 'keep'."""
        engine = ReflectionAdjustmentEngine()
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test Worker",
            category="Web, Mobile & Software Dev",
            skills=["Python"],
            min_hourly_rate=50.0,
            personality="test",
            motivation="test", 
            background="test",
            preferred_project_length="medium"
        )
        
        rate_adjustment = {
            'change': 'keep',
            'new_rate': 60.0,
            'reasoning': 'Rate is good'
        }
        
        result = engine.apply_freelancer_rate_adjustment(
            freelancer, rate_adjustment, round_num=1
        )
        
        assert result is None
        assert freelancer.min_hourly_rate == 50.0  # Unchanged
    
    def test_insignificant_change_ignored(self):
        """Test that very small changes are ignored."""
        engine = ReflectionAdjustmentEngine()
        freelancer = Freelancer(
            id="test_freelancer",
            name="Test Worker",
            category="Web, Mobile & Software Dev",
            skills=["Python"],
            min_hourly_rate=100.0,
            personality="test",
            motivation="test",
            background="test",
            preferred_project_length="medium"
        )
        
        # 2% change - below 5% threshold
        rate_adjustment = {
            'change': 'increase',
            'new_rate': 102.0,
            'reasoning': 'Tiny adjustment'
        }
        
        result = engine.apply_freelancer_rate_adjustment(
            freelancer, rate_adjustment, round_num=1
        )
        
        assert result is None
        assert freelancer.min_hourly_rate == 100.0  # Unchanged
