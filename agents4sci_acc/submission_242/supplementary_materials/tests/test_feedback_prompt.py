"""Tests for feedback prompt generation"""
import pytest
from datetime import datetime
from prompts.freelancer import generate_feedback_prompt
from marketplace.entities import HiringDecision


def test_generate_feedback_prompt_with_hiring_decision_dict():
    """Test feedback prompt generation with HiringDecision converted to dict"""
    # Arrange
    job_dict = {
        'title': 'Test Job', 
        'budget_amount': 50,
        'description': 'Test job description'
    }
    hired_freelancer_id = 'freelancer_123'
    job_bids_dict = [{
        'message': 'Test bid message', 
        'freelancer_id': 'freelancer_456',
        'amount': 45
    }]
    
    # Create HiringDecision and convert to dict (as done in the marketplace)
    decision = HiringDecision(
        job_id='test_job',
        client_id='test_client', 
        selected_freelancer='freelancer_123',
        reasoning='Selected based on experience and competitive pricing',
        timestamp=datetime.now()
    )
    decision_dict = decision.to_dict()
    
    # Use the first bid as the one receiving feedback
    freelancer_bid_dict = job_bids_dict[0]
    
    # Act
    prompt = generate_feedback_prompt(
        job_dict, decision_dict, freelancer_bid_dict
    )
    
    # Assert
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert 'Test Job' in prompt
    assert 'Selected based on experience and competitive pricing' in prompt
    assert 'Test bid message' in prompt
    assert 'JSON object' in prompt  # Should contain JSON format instructions


def test_generate_feedback_prompt_with_empty_reasoning():
    """Test feedback prompt generation when reasoning is empty"""
    job_dict = {'title': 'Test Job', 'budget_amount': 50}
    hired_freelancer_id = 'freelancer_123'
    job_bids_dict = [{'message': 'Test message', 'freelancer_id': 'freelancer_456'}]
    
    # Create decision dict with no reasoning
    decision_dict = {
        'job_id': 'test_job',
        'client_id': 'test_client', 
        'selected_freelancer': 'freelancer_123',
        'reasoning': '',
        'timestamp': '2025-08-21T16:00:00'
    }
    
    freelancer_bid_dict = job_bids_dict[0]
    
    # Act
    prompt = generate_feedback_prompt(
        job_dict, decision_dict, freelancer_bid_dict
    )
    
    # Assert
    assert 'No reasoning provided' in prompt


def test_generate_feedback_prompt_with_none_reasoning():
    """Test feedback prompt generation when reasoning is None"""
    job_dict = {'title': 'Test Job', 'budget_amount': 50}
    hired_freelancer_id = 'freelancer_123'
    job_bids_dict = [{'message': 'Test message', 'freelancer_id': 'freelancer_456'}]
    
    # Create decision dict with None reasoning
    decision_dict = {
        'job_id': 'test_job',
        'client_id': 'test_client', 
        'selected_freelancer': 'freelancer_123',
        'reasoning': None,
        'timestamp': '2025-08-21T16:00:00'
    }
    
    freelancer_bid_dict = job_bids_dict[0]
    
    # Act
    prompt = generate_feedback_prompt(
        job_dict, decision_dict, freelancer_bid_dict
    )
    
    # Assert
    assert 'No reasoning provided' in prompt


def test_generate_feedback_prompt_missing_reasoning_key():
    """Test feedback prompt generation when reasoning key is missing"""
    job_dict = {'title': 'Test Job', 'budget_amount': 50}
    hired_freelancer_id = 'freelancer_123'
    job_bids_dict = [{'message': 'Test message', 'freelancer_id': 'freelancer_456'}]
    
    # Create decision dict without reasoning key
    decision_dict = {
        'job_id': 'test_job',
        'client_id': 'test_client', 
        'selected_freelancer': 'freelancer_123',
        'timestamp': '2025-08-21T16:00:00'
        # No 'reasoning' key
    }
    
    freelancer_bid_dict = job_bids_dict[0]
    
    # Act
    prompt = generate_feedback_prompt(
        job_dict, decision_dict, freelancer_bid_dict
    )
    
    # Assert
    assert 'No reasoning provided' in prompt


def test_generate_feedback_prompt_with_no_message():
    """Test feedback prompt generation when bid has no message"""
    job_dict = {'title': 'Test Job', 'budget_amount': 50}
    hired_freelancer_id = 'freelancer_123'
    job_bids_dict = [{'freelancer_id': 'freelancer_456'}]  # No message
    
    decision_dict = {
        'job_id': 'test_job',
        'client_id': 'test_client', 
        'selected_freelancer': 'freelancer_123',
        'reasoning': 'Test reasoning',
        'timestamp': '2025-08-21T16:00:00'
    }
    
    freelancer_bid_dict = job_bids_dict[0]
    
    # Act
    prompt = generate_feedback_prompt(
        job_dict, decision_dict, freelancer_bid_dict
    )
    
    # Assert
    assert 'No message' in prompt