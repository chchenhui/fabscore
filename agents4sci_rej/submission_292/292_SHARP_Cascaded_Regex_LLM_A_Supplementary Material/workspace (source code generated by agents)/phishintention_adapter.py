"""
PhishIntention Adapter
Based on "Inferring Phishing Intention via Webpage Appearance and Dynamics: A Deep Vision Based Approach"
USENIX Security 2022
GitHub: https://github.com/lindsey98/PhishIntention

This is a simplified adapter that implements the core concepts from the paper.
The full implementation requires browser automation and computer vision models.
"""

import logging
import re
from typing import Dict, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class PhishIntentionAdapter:
    """
    Simplified adapter for PhishIntention approach
    Analyzes both brand intention and credential-taking intention
    """

    def __init__(self):
        """Initialize the PhishIntention adapter"""
        self.brand_keywords = self._load_brand_keywords()
        self.credential_indicators = self._load_credential_indicators()
        self.suspicious_patterns = self._load_suspicious_patterns()

    def _load_brand_keywords(self) -> Dict[str, List[str]]:
        """Load brand-specific keywords for major targets"""
        return {
            'paypal': ['paypal', 'payment', 'transaction', 'account limited'],
            'amazon': ['amazon', 'prime', 'delivery', 'order', 'package'],
            'microsoft': ['microsoft', 'outlook', 'office', 'windows', 'azure'],
            'google': ['google', 'gmail', 'drive', 'account', 'security'],
            'apple': ['apple', 'icloud', 'itunes', 'iphone', 'app store'],
            'facebook': ['facebook', 'meta', 'messenger', 'instagram'],
            'netflix': ['netflix', 'streaming', 'subscription', 'viewing'],
            'bank': ['bank', 'banking', 'account', 'transfer', 'statement'],
            'ebay': ['ebay', 'auction', 'bid', 'seller', 'buyer'],
            'linkedin': ['linkedin', 'professional', 'network', 'connection']
        }

    def _load_credential_indicators(self) -> List[str]:
        """Load indicators of credential-taking intention"""
        return [
            'password', 'username', 'login', 'sign in', 'signin',
            'email', 'account', 'verify', 'confirm', 'update',
            'security', 'suspended', 'locked', 'expired',
            'credit card', 'card number', 'cvv', 'billing',
            'social security', 'ssn', 'date of birth', 'dob'
        ]

    def _load_suspicious_patterns(self) -> List[str]:
        """Load patterns commonly found in phishing emails"""
        return [
            r'urgent.{0,20}action',
            r'verify.{0,20}account',
            r'suspend.{0,20}account',
            r'click.{0,20}here',
            r'limited.{0,20}time',
            r'act.{0,20}now',
            r'confirm.{0,20}identity',
            r'update.{0,20}information',
            r'security.{0,20}alert',
            r'unusual.{0,20}activity'
        ]

    def extract_brand_intention(self, text: str) -> Tuple[str, float]:
        """
        Extract the brand that the email/webpage is trying to impersonate
        Returns: (brand_name, confidence_score)
        """
        text_lower = text.lower()
        brand_scores = {}

        for brand, keywords in self.brand_keywords.items():
            score = 0
            keyword_count = 0

            for keyword in keywords:
                if keyword in text_lower:
                    keyword_count += 1
                    # Weight based on keyword importance
                    if keyword == brand:  # Direct brand mention
                        score += 3
                    else:
                        score += 1

            if keyword_count > 0:
                # Normalize score
                brand_scores[brand] = score / len(keywords)

        if brand_scores:
            # Get the brand with highest score
            best_brand = max(brand_scores, key=brand_scores.get)
            confidence = min(brand_scores[best_brand], 1.0)
            return best_brand, confidence

        return 'unknown', 0.0

    def extract_credential_intention(self, text: str) -> float:
        """
        Extract credential-taking intention score
        Returns: score between 0 and 1
        """
        text_lower = text.lower()
        credential_count = 0

        for indicator in self.credential_indicators:
            if indicator in text_lower:
                credential_count += 1

        # Check for forms and input fields (simplified)
        form_indicators = ['<form', '<input', 'type="password"', 'type="text"']
        for indicator in form_indicators:
            if indicator in text_lower:
                credential_count += 2  # Higher weight for actual form elements

        # Normalize score
        max_possible = len(self.credential_indicators) + len(form_indicators) * 2
        score = min(credential_count / max_possible * 3, 1.0)  # Scale up for sensitivity

        return score

    def check_suspicious_patterns(self, text: str) -> float:
        """
        Check for suspicious patterns in the text
        Returns: suspicion score between 0 and 1
        """
        import re

        text_lower = text.lower()
        pattern_count = 0

        for pattern in self.suspicious_patterns:
            if re.search(pattern, text_lower):
                pattern_count += 1

        # Additional checks
        # Check for URL shorteners
        shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co']
        for shortener in shorteners:
            if shortener in text_lower:
                pattern_count += 2

        # Check for typos in common domains (typosquatting)
        typos = ['amazom', 'payp4l', 'mircosoft', 'gooogle', 'facebok']
        for typo in typos:
            if typo in text_lower:
                pattern_count += 3

        # Normalize score
        max_patterns = len(self.suspicious_patterns) + len(shorteners) * 2 + len(typos) * 3
        score = min(pattern_count / max_patterns * 2, 1.0)

        return score

    def analyze_domain_mismatch(self, text: str, claimed_brand: str) -> float:
        """
        Check if the domain in URLs matches the claimed brand
        Returns: mismatch score (higher = more suspicious)
        """
        # Extract URLs from text (simplified)
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text.lower())

        if not urls:
            return 0.0

        legitimate_domains = {
            'paypal': ['paypal.com', 'paypal.me'],
            'amazon': ['amazon.com', 'amazon.co.uk', 'amazon.de'],
            'microsoft': ['microsoft.com', 'outlook.com', 'office.com'],
            'google': ['google.com', 'gmail.com', 'youtube.com'],
            'apple': ['apple.com', 'icloud.com', 'itunes.com'],
            'facebook': ['facebook.com', 'fb.com', 'messenger.com'],
            'netflix': ['netflix.com'],
            'ebay': ['ebay.com', 'ebay.co.uk'],
            'linkedin': ['linkedin.com']
        }

        if claimed_brand not in legitimate_domains:
            return 0.3  # Unknown brand, moderate suspicion

        legit_domains = legitimate_domains[claimed_brand]
        mismatch_count = 0

        for url in urls:
            # Extract domain from URL
            domain_match = re.search(r'https?://([^/]+)', url)
            if domain_match:
                domain = domain_match.group(1)
                # Check if domain is legitimate
                is_legitimate = any(legit in domain for legit in legit_domains)
                if not is_legitimate:
                    mismatch_count += 1

        if len(urls) > 0:
            return mismatch_count / len(urls)

        return 0.0

    def predict(self, email_dict: Dict) -> str:
        """
        Main prediction method following PhishIntention approach
        """
        text = email_dict.get('text', '')

        # Step 1: Extract brand intention
        brand, brand_confidence = self.extract_brand_intention(text)

        # Step 2: Extract credential-taking intention
        credential_score = self.extract_credential_intention(text)

        # Step 3: Check suspicious patterns
        suspicion_score = self.check_suspicious_patterns(text)

        # Step 4: Check domain mismatch if brand is identified
        mismatch_score = 0.0
        if brand != 'unknown':
            mismatch_score = self.analyze_domain_mismatch(text, brand)

        # Combine scores for final decision
        # PhishIntention approach: both brand and credential intentions must be present
        phishing_score = 0.0

        if brand != 'unknown' and brand_confidence > 0.3:
            # Brand is identified with reasonable confidence
            if credential_score > 0.2:
                # Credential-taking intention is present
                phishing_score = (brand_confidence * 0.3 +
                                credential_score * 0.3 +
                                suspicion_score * 0.2 +
                                mismatch_score * 0.2)
            else:
                # Brand mentioned but no credential request - likely legitimate
                phishing_score = suspicion_score * 0.3 + mismatch_score * 0.3
        else:
            # No clear brand intention
            if credential_score > 0.4:
                # Generic phishing attempt
                phishing_score = credential_score * 0.5 + suspicion_score * 0.5
            else:
                phishing_score = suspicion_score * 0.5

        # Log details for debugging
        logger.debug(f"PhishIntention Analysis:")
        logger.debug(f"  Brand: {brand} (confidence: {brand_confidence:.2f})")
        logger.debug(f"  Credential Score: {credential_score:.2f}")
        logger.debug(f"  Suspicion Score: {suspicion_score:.2f}")
        logger.debug(f"  Domain Mismatch: {mismatch_score:.2f}")
        logger.debug(f"  Final Score: {phishing_score:.2f}")

        # Decision threshold
        if phishing_score > 0.5:
            return 'phishing'
        else:
            return 'legitimate'

    def __call__(self, email_dict: Dict) -> str:
        """Make the detector callable"""
        return self.predict(email_dict)

    def train(self, train_data: List[Dict], val_data: List[Dict] = None):
        """
        PhishIntention doesn't require training as it's rule-based
        This method is here for compatibility
        """
        logger.info("PhishIntention adapter initialized (no training required)")
        if val_data:
            # Could use validation data to tune thresholds
            correct = 0
            for item in val_data[:100]:  # Quick validation check
                pred = self.predict(item)
                if pred == item['label']:
                    correct += 1
            logger.info(f"Validation accuracy on sample: {correct/100:.2f}")