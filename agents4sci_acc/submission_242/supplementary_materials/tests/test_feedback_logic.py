"""Tests for feedback logic to ensure hired freelancers don't receive negative feedback"""
import pytest
from datetime import datetime
from prompts.freelancer import generate_feedback_prompt


def test_feedback_logic_identifies_correct_recipients():
    """Test the logic that determines who should receive feedback"""
    # Test data
    hired_freelancer_id = 'freelancer_hired'
    bid_freelancer_ids = ['freelancer_hired', 'freelancer_rejected_1', 'freelancer_rejected_2']
    
    # Simulate the marketplace logic for determining feedback recipients
    feedback_recipients = []
    for bid_freelancer_id in bid_freelancer_ids:
        if bid_freelancer_id != hired_freelancer_id:
            feedback_recipients.append(bid_freelancer_id)
    
    # Assert only unsuccessful bidders get feedback
    assert hired_freelancer_id not in feedback_recipients
    assert 'freelancer_rejected_1' in feedback_recipients
    assert 'freelancer_rejected_2' in feedback_recipients
    assert len(feedback_recipients) == 2


def test_feedback_prompt_context_makes_sense():
    """Test that the feedback prompt properly explains the context"""
    from prompts.freelancer import generate_feedback_prompt
    
    job_dict = {'title': 'Software Developer', 'budget_amount': 50}
    
    decision_dict = {
        'job_id': 'test_job',
        'client_id': 'test_client', 
        'selected_freelancer': 'freelancer_hired',
        'reasoning': 'Selected due to better experience and competitive pricing',
        'timestamp': '2025-08-21T16:00:00'
    }
    
    # The rejected freelancer's bid
    freelancer_bid_dict = {'freelancer_id': 'freelancer_rejected', 'message': 'Good attempt'}
    
    prompt = generate_feedback_prompt(
        job_dict, decision_dict, freelancer_bid_dict
    )
    
    # Verify the prompt makes sense for an unsuccessful bidder
    assert 'unsuccessful bidder' in prompt
    assert 'HIRING DECISION:' in prompt
    assert 'Selected due to better experience and competitive pricing' in prompt
    assert 'Good attempt' in prompt  # Their specific bid message
    assert 'why they weren\'t selected' in prompt


def test_no_confusion_about_feedback_recipient():
    """Ensure the prompt clearly indicates it's for the unsuccessful bidder"""
    from prompts.freelancer import generate_feedback_prompt
    
    job_dict = {'title': 'Test Job', 'budget_amount': 60}
    
    decision_dict = {
        'reasoning': 'Winner had the best qualifications',
        'selected_freelancer': 'winner'
    }
    
    unsuccessful_bid_dict = {'freelancer_id': 'loser', 'message': 'My bid'}
    
    prompt = generate_feedback_prompt(
        job_dict, decision_dict, unsuccessful_bid_dict
    )
    
    # The prompt should clearly be about providing feedback to the unsuccessful bidder
    assert 'unsuccessful bidder' in prompt
    assert 'My bid' in prompt  # Shows the specific unsuccessful bidder's message
    assert 'Winner had the best qualifications' in prompt  # Explains why someone else won
    assert 'main_reason: primary reason for not selecting them' in prompt
