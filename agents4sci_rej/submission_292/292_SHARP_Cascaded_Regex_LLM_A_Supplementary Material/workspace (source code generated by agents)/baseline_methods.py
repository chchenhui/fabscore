"""
Baseline methods for phishing email detection
Implements traditional rule-based and ML methods
"""

import re
import logging
from typing import List, Dict, Any
import numpy as np

# Try to import sklearn, but continue if not available
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("sklearn not available. Some baseline methods will be disabled.")

logger = logging.getLogger(__name__)

class DummyDetector:
    """Dummy detector when sklearn is not available"""
    
    def __init__(self, name):
        self.name = name
        logger.warning(f"{name} not available due to missing sklearn")
    
    def fit(self, train_data):
        pass
    
    def predict(self, emails):
        # Return random predictions
        return np.random.randint(0, 2, len(emails))

class BaselineMethods:
    """Collection of baseline phishing detection methods"""
    
    def __init__(self):
        self.rule_based_detector = RuleBasedDetector()
        if SKLEARN_AVAILABLE:
            self.tfidf_svm_detector = TfidfSvmDetector()
        else:
            self.tfidf_svm_detector = DummyDetector("TF-IDF + SVM")
        self.regex_pattern_detector = RegexPatternDetector()

class RuleBasedDetector:
    """Simple rule-based phishing detector"""
    
    def __init__(self):
        # Suspicious keywords and phrases
        self.phishing_keywords = [
            'urgent', 'verify', 'suspend', 'click here', 'act now',
            'confirm', 'update', 'validate', 'secure', 'expire',
            'limited time', 'winner', 'congratulations', 'claim',
            'prize', 'reward', 'refund', 'billing', 'payment'
        ]
        
        self.phishing_domains = [
            'bit.ly', 'tinyurl', 'goo.gl', 'ow.ly',
            '.tk', '.ml', '.ga', '.cf'
        ]
        
        self.legitimate_domains = [
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com',
            'company.com', 'edu', 'gov', 'org'
        ]
    
    def predict(self, emails):
        """Predict phishing emails using rules"""
        predictions = []
        
        for email in emails:
            score = 0
            
            # Combine subject and body
            text = (email.get('subject', '') + ' ' + email.get('body', '')).lower()
            sender = email.get('sender', '').lower()
            
            # Check for phishing keywords
            for keyword in self.phishing_keywords:
                if keyword in text:
                    score += 2
            
            # Check for suspicious domains in sender
            for domain in self.phishing_domains:
                if domain in sender:
                    score += 5
            
            # Check for legitimate domains
            for domain in self.legitimate_domains:
                if domain in sender:
                    score -= 3
            
            # Check for URLs in body
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, email.get('body', ''))
            
            # Suspicious if many URLs
            if len(urls) > 2:
                score += 3
            
            # Check for urgency indicators
            urgency_patterns = [
                r'within \d+ hours?',
                r'expires? (today|tomorrow|soon)',
                r'immediate(ly)?',
                r'asap',
                r'urgent'
            ]
            
            for pattern in urgency_patterns:
                if re.search(pattern, text):
                    score += 2
            
            # Check for credential requests
            credential_patterns = [
                r'password',
                r'username',
                r'social security',
                r'ssn',
                r'credit card',
                r'bank account',
                r'pin'
            ]
            
            for pattern in credential_patterns:
                if re.search(pattern, text):
                    score += 3
            
            # Make prediction based on score
            predictions.append(1 if score > 5 else 0)
        
        return np.array(predictions)
    
    def fit(self, train_data):
        """No training needed for rule-based method"""
        pass

if SKLEARN_AVAILABLE:
    class TfidfSvmDetector:
        """TF-IDF with SVM classifier"""
        
        def __init__(self):
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.classifier = SVC(kernel='linear', probability=True)
            self.is_trained = False
        
        def fit(self, train_data):
            """Train the TF-IDF + SVM model"""
            texts = []
            labels = []
            
            for email in train_data:
                text = email.get('subject', '') + ' ' + email.get('body', '')
                texts.append(text)
                labels.append(email['label'])
            
            # Fit vectorizer and transform texts
            X = self.vectorizer.fit_transform(texts)
            
            # Train classifier
            self.classifier.fit(X, labels)
            self.is_trained = True
            
            logger.info(f"TF-IDF + SVM trained on {len(train_data)} samples")
        
        def predict(self, emails):
            """Predict using TF-IDF + SVM"""
            if not self.is_trained:
                # Return random predictions if not trained
                return np.random.randint(0, 2, len(emails))
            
            texts = []
            for email in emails:
                text = email.get('subject', '') + ' ' + email.get('body', '')
                texts.append(text)
            
            X = self.vectorizer.transform(texts)
            predictions = self.classifier.predict(X)
            
            return predictions

class RegexPatternDetector:
    """Advanced regex pattern-based detector"""
    
    def __init__(self):
        # Compile regex patterns for efficiency
        self.phishing_patterns = [
            # Suspicious URLs
            re.compile(r'http[s]?://[^\s]*\.(tk|ml|ga|cf)', re.IGNORECASE),
            re.compile(r'bit\.ly|tinyurl|goo\.gl', re.IGNORECASE),
            
            # IP addresses as URLs
            re.compile(r'http[s]?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'),
            
            # Misspelled domains
            re.compile(r'(payp[a@]l|amaz[o0]n|micr[o0]s[o0]ft|g[o0][o0]gle)', re.IGNORECASE),
            
            # Unicode homographs
            re.compile(r'[а-яА-Я]'),  # Cyrillic characters
            
            # Urgency patterns
            re.compile(r'(urgent|immediate|expire|suspend|terminat)', re.IGNORECASE),
            
            # Money patterns
            re.compile(r'\$[\d,]+(\.\d{2})?'),
            re.compile(r'(million|thousand) dollars?', re.IGNORECASE),
            
            # Personal info requests
            re.compile(r'(ssn|social security|password|pin|cvv)', re.IGNORECASE),
            
            # Fake sender patterns
            re.compile(r'noreply|no-reply|donotreply', re.IGNORECASE),
            
            # Hidden or misleading text
            re.compile(r'<[^>]*display:\s*none[^>]*>', re.IGNORECASE),
        ]
        
        self.legitimate_patterns = [
            # Company email patterns
            re.compile(r'@[a-zA-Z0-9-]+\.(com|org|edu|gov)$'),
            
            # Professional language
            re.compile(r'(sincerely|regards|best|thanks)', re.IGNORECASE),
            
            # Meeting/calendar patterns
            re.compile(r'(meeting|schedule|agenda|minutes)', re.IGNORECASE),
        ]
    
    def predict(self, emails):
        """Predict using regex patterns"""
        predictions = []
        
        for email in emails:
            phishing_score = 0
            legitimate_score = 0
            
            text = email.get('subject', '') + ' ' + email.get('body', '')
            sender = email.get('sender', '')
            
            # Check phishing patterns
            for pattern in self.phishing_patterns:
                matches = pattern.findall(text + ' ' + sender)
                phishing_score += len(matches)
            
            # Check legitimate patterns
            for pattern in self.legitimate_patterns:
                matches = pattern.findall(text + ' ' + sender)
                legitimate_score += len(matches)
            
            # Additional checks
            
            # Check for excessive capitalization
            caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            if caps_ratio > 0.3:
                phishing_score += 2
            
            # Check for multiple exclamation marks
            if text.count('!') > 3:
                phishing_score += 1
            
            # Check for suspicious attachment mentions
            attachment_patterns = ['.exe', '.zip', '.scr', '.vbs', '.bat']
            for pattern in attachment_patterns:
                if pattern in text.lower():
                    phishing_score += 3
            
            # Make prediction
            if phishing_score > legitimate_score + 2:
                predictions.append(1)
            else:
                predictions.append(0)
        
        return np.array(predictions)
    
    def fit(self, train_data):
        """No training needed for regex-based method"""
        pass

class NaiveBayesDetector:
    """Naive Bayes classifier for comparison"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.classifier = MultinomialNB()
        self.is_trained = False
    
    def fit(self, train_data):
        """Train the Naive Bayes model"""
        texts = []
        labels = []
        
        for email in train_data:
            text = email.get('subject', '') + ' ' + email.get('body', '')
            texts.append(text)
            labels.append(email['label'])
        
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict(self, emails):
        """Predict using Naive Bayes"""
        if not self.is_trained:
            return np.random.randint(0, 2, len(emails))
        
        texts = []
        for email in emails:
            text = email.get('subject', '') + ' ' + email.get('body', '')
            texts.append(text)
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

class RandomForestDetector:
    """Random Forest classifier"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
    
    def fit(self, train_data):
        """Train the Random Forest model"""
        texts = []
        labels = []
        
        for email in train_data:
            text = email.get('subject', '') + ' ' + email.get('body', '')
            texts.append(text)
            labels.append(email['label'])
        
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict(self, emails):
        """Predict using Random Forest"""
        if not self.is_trained:
            return np.random.randint(0, 2, len(emails))
        
        texts = []
        for email in emails:
            text = email.get('subject', '') + ' ' + email.get('body', '')
            texts.append(text)
        
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)