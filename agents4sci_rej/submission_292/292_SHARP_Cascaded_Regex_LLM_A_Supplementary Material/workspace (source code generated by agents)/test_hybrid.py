#!/usr/bin/env python3
"""
Test script for the Hybrid LLM-Regex Detector
"""

from hybrid_llm_regex_detector import HybridLLMRegexDetector
import json

def test_hybrid_detector():
    """Test our hybrid detector with sample emails"""

    # Initialize detector
    detector = HybridLLMRegexDetector()

    # Test samples
    test_samples = [
        {
            'content': 'URGENT: Your account will be suspended! Click here to verify: http://192.168.1.1/paypal',
            'expected': 'phishing'
        },
        {
            'content': 'Dear Customer, your payment of $1000 has been processed. Thank you for your business.',
            'expected': 'legitimate'
        },
        {
            'content': 'Congratulations! You have won $1,000,000!!! Click here NOW to claim your prize!!!',
            'expected': 'phishing'
        },
        ['Your account needs immediate verification at paipal.com', 'phishing'],  # Test list format
        ['Meeting scheduled for tomorrow at 2pm', 'legitimate']  # Test list format
    ]

    # Train on dummy data
    train_data = [
        {'content': 'verify your account urgently', 'label': 'phishing'},
        {'content': 'thank you for your order', 'label': 'legitimate'},
        ['click here to update password', 'phishing'],
        ['your invoice is attached', 'legitimate']
    ]

    print("Training detector...")
    detector.train(train_data)

    print("\nTesting predictions:")
    print("-" * 80)

    for i, sample in enumerate(test_samples):
        if isinstance(sample, dict):
            content = sample['content']
            expected = sample.get('expected', 'unknown')
        else:
            content = sample[0]
            expected = sample[1] if len(sample) > 1 else 'unknown'

        # Make prediction
        result = detector.predict(sample)

        print(f"\nTest {i+1}:")
        print(f"Content: {content[:60]}...")
        print(f"Expected: {expected}")
        print(f"Predicted: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Method: {result['method']}")

        if 'details' in result:
            if 'regex_score' in result['details']:
                print(f"Regex Score: {result['details']['regex_score']:.2f}")
            if 'regex_matches' in result['details']:
                print(f"Patterns: {result['details']['regex_matches']}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")

if __name__ == "__main__":
    test_hybrid_detector()