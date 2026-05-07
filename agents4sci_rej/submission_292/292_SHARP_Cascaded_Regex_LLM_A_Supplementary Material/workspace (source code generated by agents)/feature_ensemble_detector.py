"""
Feature-based ML Ensemble Detector
Based on "Phishing Attack Detection using Machine Learning"
University of Ottawa, 2023
Trained on 737,000 URLs dataset

Implements comprehensive feature extraction and ensemble methods
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
import re
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class FeatureEnsembleDetector:
    """
    Advanced feature-based ensemble detector using multiple ML algorithms
    Based on uOttawa 2023 research with 737,000 URLs dataset
    """

    def __init__(self):
        """Initialize the feature-based ensemble detector"""
        self.models = {}
        self.feature_extractors = self._initialize_extractors()
        self.scaler = None
        self.trained = False

    def _initialize_extractors(self) -> Dict:
        """Initialize feature extraction methods"""
        return {
            'url_features': self._extract_url_features,
            'content_features': self._extract_content_features,
            'statistical_features': self._extract_statistical_features,
            'domain_features': self._extract_domain_features
        }

    def _extract_url_features(self, text: str) -> Dict[str, float]:
        """Extract URL-based features"""
        features = {}

        # Find all URLs in text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text.lower())

        if urls:
            # Analyze first URL (usually the main phishing URL)
            url = urls[0]
            parsed = urlparse(url)

            # Length features
            features['url_length'] = len(url)
            features['domain_length'] = len(parsed.netloc) if parsed.netloc else 0
            features['path_length'] = len(parsed.path) if parsed.path else 0

            # Count features
            features['dot_count'] = url.count('.')
            features['hyphen_count'] = url.count('-')
            features['underscore_count'] = url.count('_')
            features['slash_count'] = url.count('/')
            features['question_count'] = url.count('?')
            features['equal_count'] = url.count('=')
            features['at_count'] = url.count('@')
            features['ampersand_count'] = url.count('&')
            features['digit_count'] = sum(c.isdigit() for c in url)

            # Special character ratio
            special_chars = sum(1 for c in url if not c.isalnum() and c not in './:')
            features['special_char_ratio'] = special_chars / len(url) if url else 0

            # Check for IP address
            features['has_ip'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', parsed.netloc or '') else 0

            # Check for URL shortener
            shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co', 'short.link']
            features['is_shortened'] = 1 if any(s in parsed.netloc for s in shorteners) else 0

            # Check for HTTPS
            features['uses_https'] = 1 if parsed.scheme == 'https' else 0

            # Subdomain count
            if parsed.netloc:
                subdomain_count = parsed.netloc.count('.') - 1
                features['subdomain_count'] = max(0, subdomain_count)
            else:
                features['subdomain_count'] = 0

            # Port usage
            features['uses_non_standard_port'] = 1 if parsed.port and parsed.port not in [80, 443] else 0

        else:
            # No URL found - set default values
            for key in ['url_length', 'domain_length', 'path_length', 'dot_count',
                       'hyphen_count', 'underscore_count', 'slash_count', 'question_count',
                       'equal_count', 'at_count', 'ampersand_count', 'digit_count',
                       'special_char_ratio', 'has_ip', 'is_shortened', 'uses_https',
                       'subdomain_count', 'uses_non_standard_port']:
                features[key] = 0

        return features

    def _extract_content_features(self, text: str) -> Dict[str, float]:
        """Extract content-based features"""
        features = {}
        text_lower = text.lower()

        # Length features
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())

        # Keyword features (phishing indicators)
        phishing_keywords = [
            'verify', 'account', 'suspend', 'click', 'urgent', 'expire',
            'security', 'update', 'confirm', 'limited', 'winner', 'congratulations',
            'refund', 'tax', 'invoice', 'payment', 'billing', 'alert'
        ]

        features['phishing_keyword_count'] = sum(1 for kw in phishing_keywords if kw in text_lower)

        # Urgency indicators
        urgency_words = ['immediate', 'urgent', 'asap', 'quickly', 'expire', 'suspend', 'deadline']
        features['urgency_score'] = sum(1 for word in urgency_words if word in text_lower)

        # Financial terms
        financial_terms = ['bank', 'credit', 'debit', 'payment', 'transaction', 'transfer', 'money']
        features['financial_terms'] = sum(1 for term in financial_terms if term in text_lower)

        # Personal information requests
        personal_info = ['password', 'username', 'ssn', 'social security', 'pin', 'cvv', 'account number']
        features['personal_info_request'] = sum(1 for info in personal_info if info in text_lower)

        # HTML/Form elements
        features['has_form'] = 1 if '<form' in text_lower else 0
        features['has_input'] = 1 if '<input' in text_lower else 0
        features['has_button'] = 1 if '<button' in text_lower or 'submit' in text_lower else 0

        # Capitalization (excessive caps often used in phishing)
        if len(text) > 0:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            features['caps_ratio'] = caps_ratio
        else:
            features['caps_ratio'] = 0

        # Punctuation density
        punctuation = '!?.,;:\'"'
        features['punctuation_density'] = sum(1 for c in text if c in punctuation) / max(1, len(text))

        return features

    def _extract_statistical_features(self, text: str) -> Dict[str, float]:
        """Extract statistical text features"""
        features = {}

        # Character distribution
        alpha_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        space_count = sum(1 for c in text if c.isspace())

        text_len = max(1, len(text))
        features['alpha_ratio'] = alpha_count / text_len
        features['digit_ratio'] = digit_count / text_len
        features['space_ratio'] = space_count / text_len

        # Sentence statistics
        sentences = text.split('.')
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = np.mean([len(s) for s in sentences]) if sentences else 0

        # Word statistics
        words = text.split()
        if words:
            word_lengths = [len(w) for w in words]
            features['avg_word_length'] = np.mean(word_lengths)
            features['max_word_length'] = max(word_lengths)
            features['word_length_variance'] = np.var(word_lengths)
        else:
            features['avg_word_length'] = 0
            features['max_word_length'] = 0
            features['word_length_variance'] = 0

        return features

    def _extract_domain_features(self, text: str) -> Dict[str, float]:
        """Extract domain reputation features"""
        features = {}

        # Popular legitimate domains (whitelist)
        legitimate_domains = [
            'google.com', 'gmail.com', 'youtube.com', 'facebook.com', 'amazon.com',
            'microsoft.com', 'apple.com', 'twitter.com', 'linkedin.com', 'github.com',
            'stackoverflow.com', 'wikipedia.org', 'reddit.com', 'netflix.com'
        ]

        # Suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.click', '.download', '.review']

        # Extract domains from text
        url_pattern = r'https?://([^\s/]+)'
        domains = re.findall(url_pattern, text.lower())

        features['domain_count'] = len(domains)
        features['unique_domain_count'] = len(set(domains))

        if domains:
            # Check for legitimate domains
            legit_count = sum(1 for d in domains if any(legit in d for legit in legitimate_domains))
            features['legitimate_domain_ratio'] = legit_count / len(domains)

            # Check for suspicious TLDs
            suspicious_count = sum(1 for d in domains if any(tld in d for tld in suspicious_tlds))
            features['suspicious_tld_ratio'] = suspicious_count / len(domains)

            # Check for typosquatting (common misspellings)
            typos = ['amazom', 'gooogle', 'mircosoft', 'payp4l', 'facebok']
            typo_count = sum(1 for d in domains if any(typo in d for typo in typos))
            features['typosquatting_ratio'] = typo_count / len(domains)
        else:
            features['legitimate_domain_ratio'] = 0
            features['suspicious_tld_ratio'] = 0
            features['typosquatting_ratio'] = 0

        return features

    def extract_all_features(self, text: str) -> np.ndarray:
        """Extract all features from text"""
        all_features = {}

        # Extract features from each category
        for category, extractor in self.feature_extractors.items():
            category_features = extractor(text)
            all_features.update(category_features)

        # Convert to numpy array with consistent ordering
        feature_vector = np.array([all_features.get(key, 0) for key in sorted(all_features.keys())])

        return feature_vector

    def train(self, train_data: List[Dict], val_data: List[Dict] = None):
        """Train the ensemble of ML models"""
        logger.info("Training Feature-based Ensemble Detector...")

        # Extract features and labels
        X_train = []
        y_train = []

        for item in train_data:
            features = self.extract_all_features(item['text'])
            X_train.append(features)
            y_train.append(1 if item['label'] == 'phishing' else 0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Scale features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train multiple models
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier

        # Initialize models (based on uOttawa research)
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'mlp': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }

        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train_scaled, y_train)

        self.trained = True

        # Validation
        if val_data:
            X_val = []
            y_val = []

            for item in val_data:
                features = self.extract_all_features(item['text'])
                X_val.append(features)
                y_val.append(1 if item['label'] == 'phishing' else 0)

            X_val = np.array(X_val)
            y_val = np.array(y_val)
            X_val_scaled = self.scaler.transform(X_val)

            # Evaluate each model
            for name, model in self.models.items():
                score = model.score(X_val_scaled, y_val)
                logger.info(f"{name} validation accuracy: {score:.3f}")

    def predict(self, email_dict: Dict) -> str:
        """Predict using ensemble voting"""
        text = email_dict.get('text', '')

        if not self.trained:
            # Fallback to heuristic
            return self._fallback_predict(text)

        # Extract features
        features = self.extract_all_features(text)
        features_scaled = self.scaler.transform([features])

        # Get predictions from all models
        predictions = []
        weights = {
            'random_forest': 1.2,  # Higher weight for RF based on paper
            'gradient_boost': 1.1,
            'logistic_regression': 0.9,
            'svm': 1.0,
            'mlp': 0.8
        }

        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            weight = weights.get(name, 1.0)
            predictions.extend([pred] * int(weight * 10))

        # Weighted voting
        final_prediction = 1 if np.mean(predictions) > 0.5 else 0

        return 'phishing' if final_prediction == 1 else 'legitimate'

    def _fallback_predict(self, text: str) -> str:
        """Fallback prediction"""
        features = self.extract_all_features(text)

        # Simple thresholds based on key features
        url_features = self._extract_url_features(text)
        content_features = self._extract_content_features(text)

        score = 0

        # URL-based scoring
        if url_features['is_shortened']:
            score += 2
        if url_features['has_ip']:
            score += 2
        if not url_features['uses_https']:
            score += 1
        if url_features['special_char_ratio'] > 0.1:
            score += 1

        # Content-based scoring
        if content_features['phishing_keyword_count'] > 3:
            score += 2
        if content_features['urgency_score'] > 2:
            score += 1
        if content_features['personal_info_request'] > 1:
            score += 2

        return 'phishing' if score >= 4 else 'legitimate'

    def __call__(self, email_dict: Dict) -> str:
        """Make the detector callable"""
        return self.predict(email_dict)