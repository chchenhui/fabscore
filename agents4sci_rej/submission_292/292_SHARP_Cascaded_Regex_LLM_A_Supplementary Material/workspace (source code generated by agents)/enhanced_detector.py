"""
Enhanced Phishing Detector with Advanced Features
Implements state-of-the-art techniques for phishing detection
"""

import re
import json
import logging
import hashlib
import base64
from urllib.parse import urlparse
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class EnhancedPhishingDetector:
    """Advanced phishing detector with multiple sophisticated techniques"""
    
    def __init__(self):
        # Initialize feature extractors
        self.url_analyzer = AdvancedURLAnalyzer()
        self.text_analyzer = TextFeatureExtractor()
        self.sender_analyzer = SenderReputationAnalyzer()
        self.structure_analyzer = EmailStructureAnalyzer()
        
        # Ensemble weights (can be optimized)
        self.weights = {
            'url_features': 0.25,
            'text_features': 0.25,
            'sender_features': 0.25,
            'structure_features': 0.25
        }
        
        # Known phishing indicators database
        self.phishing_db = PhishingIndicatorDatabase()
    
    def predict(self, emails):
        """Predict phishing emails using advanced features"""
        predictions = []
        
        for email in emails:
            # Extract all features
            features = self.extract_all_features(email)
            
            # Combine scores
            final_score = self.ensemble_predict(features)
            
            # Make prediction
            predictions.append(1 if final_score > 0.5 else 0)
        
        return np.array(predictions)
    
    def extract_all_features(self, email):
        """Extract comprehensive features from email"""
        features = {}
        
        # URL-based features
        features['url'] = self.url_analyzer.analyze(email)
        
        # Text-based features
        features['text'] = self.text_analyzer.extract(email)
        
        # Sender-based features
        features['sender'] = self.sender_analyzer.analyze(email)
        
        # Structure-based features
        features['structure'] = self.structure_analyzer.analyze(email)
        
        return features
    
    def ensemble_predict(self, features):
        """Combine multiple feature scores"""
        total_score = 0
        
        for feature_type, weight in self.weights.items():
            feature_key = feature_type.replace('_features', '')
            if feature_key in features:
                total_score += features[feature_key]['score'] * weight
        
        return total_score
    
    def fit(self, train_data):
        """Train/optimize the detector"""
        # Update phishing indicator database
        self.phishing_db.update_from_data(train_data)
        
        # Optimize ensemble weights if needed
        # (Could use validation data for this)
        
        logger.info("Enhanced detector trained")

class AdvancedURLAnalyzer:
    """Advanced URL analysis for phishing detection"""
    
    def __init__(self):
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download']
        self.shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co']
        self.legitimate_domains = self._load_legitimate_domains()
    
    def analyze(self, email):
        """Analyze URLs in email"""
        body = email.get('body', '')
        urls = self.extract_urls(body)
        
        if not urls:
            return {'score': 0.3, 'features': {}}
        
        features = {
            'num_urls': len(urls),
            'has_ip': False,
            'has_shortener': False,
            'has_suspicious_tld': False,
            'has_homograph': False,
            'has_misleading_subdomain': False,
            'avg_url_length': 0,
            'has_https': False,
            'has_port': False,
            'num_subdomains': 0,
            'entropy': 0
        }
        
        total_score = 0
        url_lengths = []
        
        for url in urls:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check for IP address
            if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain):
                features['has_ip'] = True
                total_score += 0.3
            
            # Check for URL shorteners
            if any(shortener in domain for shortener in self.shorteners):
                features['has_shortener'] = True
                total_score += 0.2
            
            # Check for suspicious TLDs
            if any(domain.endswith(tld) for tld in self.suspicious_tlds):
                features['has_suspicious_tld'] = True
                total_score += 0.25
            
            # Check for homograph attacks
            if self._has_homograph(domain):
                features['has_homograph'] = True
                total_score += 0.3
            
            # Check for misleading subdomains
            if self._has_misleading_subdomain(domain):
                features['has_misleading_subdomain'] = True
                total_score += 0.25
            
            # URL length
            url_lengths.append(len(url))
            
            # HTTPS check
            if parsed.scheme == 'https':
                features['has_https'] = True
                total_score -= 0.1  # HTTPS is good
            
            # Custom port
            if parsed.port and parsed.port not in [80, 443]:
                features['has_port'] = True
                total_score += 0.15
            
            # Count subdomains
            subdomains = domain.split('.')
            features['num_subdomains'] = max(features['num_subdomains'], len(subdomains) - 2)
            
            # Calculate entropy
            features['entropy'] = max(features['entropy'], self._calculate_entropy(domain))
        
        features['avg_url_length'] = np.mean(url_lengths) if url_lengths else 0
        
        # Adjust score based on features
        if features['avg_url_length'] > 100:
            total_score += 0.1
        
        if features['num_subdomains'] > 3:
            total_score += 0.15
        
        if features['entropy'] > 4.0:
            total_score += 0.1
        
        # Normalize score
        final_score = min(total_score / len(urls), 1.0)
        
        return {'score': final_score, 'features': features}
    
    def extract_urls(self, text):
        """Extract all URLs from text"""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, text)
    
    def _has_homograph(self, domain):
        """Check for homograph attacks using lookalike characters"""
        homographs = {
            'o': ['0', 'ο'],  # Latin o vs zero vs Greek omicron
            'i': ['1', 'l', 'ı'],  # Latin i vs one vs L vs Turkish i
            'a': ['а', '@'],  # Latin a vs Cyrillic a
            'e': ['е', '3'],  # Latin e vs Cyrillic e
        }
        
        for char, lookalikes in homographs.items():
            for lookalike in lookalikes:
                if lookalike in domain and char not in domain:
                    return True
        
        return False
    
    def _has_misleading_subdomain(self, domain):
        """Check for misleading subdomains"""
        parts = domain.split('.')
        if len(parts) > 2:
            # Check if legitimate domain names appear as subdomains
            legitimate = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook']
            subdomain = '.'.join(parts[:-2]).lower()
            
            for legit in legitimate:
                if legit in subdomain and f"{legit}.com" not in domain:
                    return True
        
        return False
    
    def _calculate_entropy(self, string):
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0
        
        prob = [float(string.count(c)) / len(string) for c in set(string)]
        entropy = -sum([p * np.log2(p) for p in prob])
        
        return entropy
    
    def _load_legitimate_domains(self):
        """Load list of legitimate domains"""
        return [
            'google.com', 'facebook.com', 'amazon.com', 'microsoft.com',
            'apple.com', 'paypal.com', 'ebay.com', 'netflix.com',
            'linkedin.com', 'twitter.com', 'instagram.com', 'youtube.com'
        ]

class TextFeatureExtractor:
    """Extract sophisticated text features"""
    
    def __init__(self):
        self.urgency_words = [
            'urgent', 'immediate', 'expire', 'suspend', 'terminate',
            'deadline', 'limited', 'act now', 'hurry', 'quick'
        ]
        
        self.action_words = [
            'click', 'verify', 'confirm', 'update', 'validate',
            'secure', 'restore', 'unlock', 'activate', 'claim'
        ]
        
        self.threat_words = [
            'suspend', 'close', 'terminate', 'disable', 'block',
            'restrict', 'freeze', 'lock', 'cancel', 'delete'
        ]
        
        self.reward_words = [
            'winner', 'prize', 'reward', 'bonus', 'gift',
            'congratulations', 'selected', 'chosen', 'lucky', 'free'
        ]
        
        self.credential_words = [
            'password', 'username', 'pin', 'ssn', 'account',
            'login', 'credential', 'security', 'verify', 'confirm'
        ]
    
    def extract(self, email):
        """Extract text features"""
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        full_text = subject + ' ' + body
        
        features = {
            'urgency_count': 0,
            'action_count': 0,
            'threat_count': 0,
            'reward_count': 0,
            'credential_count': 0,
            'exclamation_count': full_text.count('!'),
            'question_count': full_text.count('?'),
            'caps_ratio': 0,
            'grammar_errors': 0,
            'text_length': len(full_text),
            'avg_word_length': 0,
            'unique_word_ratio': 0
        }
        
        # Count keyword categories
        for word in self.urgency_words:
            features['urgency_count'] += full_text.count(word)
        
        for word in self.action_words:
            features['action_count'] += full_text.count(word)
        
        for word in self.threat_words:
            features['threat_count'] += full_text.count(word)
        
        for word in self.reward_words:
            features['reward_count'] += full_text.count(word)
        
        for word in self.credential_words:
            features['credential_count'] += full_text.count(word)
        
        # Calculate caps ratio
        if len(full_text) > 0:
            features['caps_ratio'] = sum(1 for c in full_text if c.isupper()) / len(full_text)
        
        # Check for common grammar errors
        grammar_errors = [
            r'\byou\'re account\b',
            r'\byour an?\b',
            r'\btheir is\b',
            r'\bthere account\b',
            r'\brecieve\b',
            r'\boccured\b',
            r'\brefered\b'
        ]
        
        for error_pattern in grammar_errors:
            if re.search(error_pattern, full_text):
                features['grammar_errors'] += 1
        
        # Word statistics
        words = full_text.split()
        if words:
            features['avg_word_length'] = np.mean([len(w) for w in words])
            features['unique_word_ratio'] = len(set(words)) / len(words)
        
        # Calculate phishing score
        score = 0
        
        # High urgency is suspicious
        if features['urgency_count'] > 2:
            score += 0.2
        
        # Many action words
        if features['action_count'] > 3:
            score += 0.15
        
        # Threats
        if features['threat_count'] > 1:
            score += 0.2
        
        # Too good to be true
        if features['reward_count'] > 2:
            score += 0.25
        
        # Credential requests
        if features['credential_count'] > 1:
            score += 0.3
        
        # Excessive punctuation
        if features['exclamation_count'] > 3:
            score += 0.1
        
        # High caps ratio
        if features['caps_ratio'] > 0.2:
            score += 0.15
        
        # Grammar errors
        if features['grammar_errors'] > 0:
            score += 0.1 * features['grammar_errors']
        
        # Very short or very long emails
        if features['text_length'] < 50 or features['text_length'] > 5000:
            score += 0.1
        
        # Low word diversity
        if features['unique_word_ratio'] < 0.5:
            score += 0.1
        
        return {'score': min(score, 1.0), 'features': features}

class SenderReputationAnalyzer:
    """Analyze sender reputation and authenticity"""
    
    def __init__(self):
        self.suspicious_patterns = [
            r'noreply',
            r'no-reply',
            r'donotreply',
            r'notification',
            r'alert',
            r'security',
            r'account',
            r'update',
            r'verify'
        ]
        
        self.legitimate_providers = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
            'icloud.com', 'protonmail.com', 'aol.com'
        ]
    
    def analyze(self, email):
        """Analyze sender information"""
        sender = email.get('sender', '').lower()
        
        if not sender:
            return {'score': 0.5, 'features': {}}
        
        features = {
            'has_numbers': bool(re.search(r'\d', sender)),
            'suspicious_pattern': False,
            'legitimate_provider': False,
            'domain_mismatch': False,
            'subdomain_count': 0,
            'sender_length': len(sender),
            'special_chars': 0
        }
        
        score = 0.5  # Start neutral
        
        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, sender):
                features['suspicious_pattern'] = True
                score += 0.1
                break
        
        # Check for legitimate provider
        for provider in self.legitimate_providers:
            if sender.endswith(f"@{provider}"):
                features['legitimate_provider'] = True
                score -= 0.2
                break
        
        # Extract domain
        if '@' in sender:
            local, domain = sender.split('@', 1)
            
            # Check for numbers in local part
            if re.search(r'\d{3,}', local):
                score += 0.15
            
            # Count subdomains
            features['subdomain_count'] = domain.count('.')
            if features['subdomain_count'] > 2:
                score += 0.1
            
            # Check for domain spoofing
            spoofed_domains = ['payp@l', 'amaz0n', 'g00gle', 'micr0soft']
            for spoofed in spoofed_domains:
                if spoofed in domain:
                    features['domain_mismatch'] = True
                    score += 0.3
            
            # Count special characters
            features['special_chars'] = sum(1 for c in sender if not c.isalnum() and c not in '@.-_')
            if features['special_chars'] > 2:
                score += 0.1
        
        # Long sender addresses are suspicious
        if features['sender_length'] > 50:
            score += 0.1
        
        return {'score': min(max(score, 0.0), 1.0), 'features': features}

class EmailStructureAnalyzer:
    """Analyze email structure and formatting"""
    
    def analyze(self, email):
        """Analyze email structure"""
        subject = email.get('subject', '')
        body = email.get('body', '')
        
        features = {
            'has_subject': bool(subject),
            'subject_length': len(subject),
            'body_length': len(body),
            'html_tags': 0,
            'hidden_text': False,
            'link_text_mismatch': False,
            'excessive_formatting': False,
            'attachment_mentioned': False
        }
        
        score = 0
        
        # Missing or very short subject
        if not features['has_subject'] or features['subject_length'] < 5:
            score += 0.1
        
        # Very long subject
        if features['subject_length'] > 100:
            score += 0.1
        
        # Check for HTML tags
        html_pattern = r'<[^>]+>'
        features['html_tags'] = len(re.findall(html_pattern, body))
        
        if features['html_tags'] > 20:
            features['excessive_formatting'] = True
            score += 0.15
        
        # Check for hidden text
        if re.search(r'display:\s*none|visibility:\s*hidden|color:\s*white', body, re.IGNORECASE):
            features['hidden_text'] = True
            score += 0.25
        
        # Check for link-text mismatch
        link_pattern = r'<a[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
        links = re.findall(link_pattern, body, re.IGNORECASE)
        
        for url, text in links:
            if 'paypal' in text.lower() and 'paypal.com' not in url.lower():
                features['link_text_mismatch'] = True
                score += 0.3
                break
        
        # Check for attachment mentions
        attachment_keywords = ['.exe', '.zip', '.rar', '.doc', '.pdf', 'attachment', 'attached']
        for keyword in attachment_keywords:
            if keyword in body.lower():
                features['attachment_mentioned'] = True
                score += 0.1
                break
        
        # Very short body
        if features['body_length'] < 50:
            score += 0.1
        
        return {'score': min(score, 1.0), 'features': features}

class PhishingIndicatorDatabase:
    """Database of known phishing indicators"""
    
    def __init__(self):
        self.known_phishing_domains = set()
        self.known_phishing_ips = set()
        self.known_phishing_subjects = []
        self.known_legitimate_senders = set()
    
    def update_from_data(self, train_data):
        """Update database from training data"""
        for email in train_data:
            if email['label'] == 1:  # Phishing
                # Extract and store phishing indicators
                sender = email.get('sender', '')
                if '@' in sender:
                    domain = sender.split('@')[1]
                    self.known_phishing_domains.add(domain)
                
                subject = email.get('subject', '')
                if subject:
                    self.known_phishing_subjects.append(subject.lower())
            
            else:  # Legitimate
                sender = email.get('sender', '')
                if sender:
                    self.known_legitimate_senders.add(sender.lower())
    
    def check_indicators(self, email):
        """Check email against known indicators"""
        sender = email.get('sender', '').lower()
        subject = email.get('subject', '').lower()
        
        # Check sender
        if sender in self.known_legitimate_senders:
            return -0.2  # Legitimate
        
        if '@' in sender:
            domain = sender.split('@')[1]
            if domain in self.known_phishing_domains:
                return 0.5  # Known phishing domain
        
        # Check subject similarity
        for known_subject in self.known_phishing_subjects:
            if self._similarity(subject, known_subject) > 0.8:
                return 0.3  # Similar to known phishing
        
        return 0  # No match
    
    def _similarity(self, str1, str2):
        """Calculate string similarity"""
        if not str1 or not str2:
            return 0
        
        # Simple character-based similarity
        common = sum(1 for c in str1 if c in str2)
        return common / max(len(str1), len(str2))