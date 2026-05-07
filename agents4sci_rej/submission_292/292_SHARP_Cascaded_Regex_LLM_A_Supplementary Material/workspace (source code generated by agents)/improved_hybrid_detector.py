"""
Improved Hybrid Phishing Detector with better LLM integration
Handles Docker container connectivity to host Ollama
"""

import re
import json
import logging
import subprocess
import requests
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import time

logger = logging.getLogger(__name__)

class ImprovedHybridDetector:
    """Enhanced hybrid detector with improved LLM integration and fallback mechanisms"""
    
    def __init__(self, model_name="dolphin3:latest", use_api=True):
        self.model_name = model_name
        self.use_api = use_api
        
        # Try different Ollama endpoints for Docker compatibility
        self.ollama_endpoints = [
            "http://host.docker.internal:11434",  # Docker on Mac/Windows
            "http://172.17.0.1:11434",            # Docker on Linux (default bridge)
            "http://localhost:11434",              # Direct local connection
            "http://127.0.0.1:11434"              # Alternative local
        ]
        
        self.ollama_url = None
        self.llm_available = False
        
        # Initialize components
        self.rule_engine = AdvancedRuleEngine()
        self.feature_extractor = FeatureExtractor()
        
        # Adaptive weights (will be optimized during training)
        self.weights = {
            'llm_score': 0.4,
            'rule_score': 0.2,
            'url_score': 0.15,
            'sender_score': 0.1,
            'content_score': 0.15
        }
        
        # Check LLM availability
        self._check_llm_availability()
        
        # Cache for LLM responses (to avoid duplicate calls)
        self.llm_cache = {}
        
    def _check_llm_availability(self):
        """Check if Ollama is available via API"""
        if self.use_api:
            for endpoint in self.ollama_endpoints:
                try:
                    response = requests.get(f"{endpoint}/api/tags", timeout=2)
                    if response.status_code == 200:
                        self.ollama_url = endpoint
                        self.llm_available = True
                        logger.info(f"Connected to Ollama at {endpoint}")
                        
                        # Check if model exists
                        models = response.json().get('models', [])
                        model_names = [m.get('name', '') for m in models]
                        
                        if not any(self.model_name in name for name in model_names):
                            logger.warning(f"Model {self.model_name} not found. Available models: {model_names}")
                            # Try to pull the model
                            self._pull_model()
                        return
                except Exception as e:
                    continue
            
            logger.warning("Could not connect to Ollama API. Using rule-based detection only.")
            self.llm_available = False
        else:
            # Fallback to subprocess method
            self._check_ollama_cli()
    
    def _pull_model(self):
        """Pull model via API"""
        if self.ollama_url:
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/pull",
                    json={"name": self.model_name},
                    timeout=300
                )
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model: {self.model_name}")
                else:
                    logger.warning(f"Failed to pull model: {response.text}")
                    self.llm_available = False
            except Exception as e:
                logger.warning(f"Error pulling model: {e}")
                self.llm_available = False
    
    def _check_ollama_cli(self):
        """Check Ollama via CLI (fallback method)"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.llm_available = True
                logger.info("Ollama CLI is available")
            else:
                self.llm_available = False
        except Exception as e:
            logger.warning(f"Ollama CLI not available: {e}")
            self.llm_available = False
    
    def train(self, train_data, val_data=None):
        """Train the hybrid detector"""
        logger.info("Training improved hybrid detector...")
        
        # Learn patterns from training data
        self.rule_engine.learn_from_data(train_data)
        self.feature_extractor.fit(train_data)
        
        # Optimize weights if validation data is available
        if val_data:
            self._optimize_weights(val_data)
        
        logger.info("Training complete")
    
    def predict(self, emails):
        """Predict phishing emails"""
        predictions = []
        
        # Batch process for efficiency
        batch_size = 10
        for i in range(0, len(emails), batch_size):
            batch = emails[i:i+batch_size]
            batch_predictions = self._predict_batch(batch)
            predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def _predict_batch(self, emails):
        """Predict a batch of emails"""
        predictions = []
        
        for email in emails:
            scores = self._compute_all_scores(email)
            final_score = self._combine_scores(scores)
            predictions.append(1 if final_score > 0.5 else 0)
        
        return predictions
    
    def _compute_all_scores(self, email):
        """Compute all detection scores for an email"""
        scores = {}
        
        # 1. LLM-based score (with caching)
        email_hash = hash(json.dumps(email, sort_keys=True))
        if email_hash in self.llm_cache:
            scores['llm_score'] = self.llm_cache[email_hash]
        elif self.llm_available:
            scores['llm_score'] = self._get_llm_score(email)
            self.llm_cache[email_hash] = scores['llm_score']
        else:
            scores['llm_score'] = 0.5  # Neutral if LLM not available
        
        # 2. Rule-based score
        scores['rule_score'] = self.rule_engine.analyze(email)
        
        # 3. URL analysis score
        scores['url_score'] = self.feature_extractor.analyze_urls(email)
        
        # 4. Sender analysis score
        scores['sender_score'] = self.feature_extractor.analyze_sender(email)
        
        # 5. Content analysis score
        scores['content_score'] = self.feature_extractor.analyze_content(email)
        
        return scores
    
    def _get_llm_score(self, email):
        """Get phishing probability from LLM"""
        if self.use_api and self.ollama_url:
            return self._get_llm_score_api(email)
        else:
            return self._get_llm_score_cli(email)
    
    def _get_llm_score_api(self, email):
        """Get LLM score via API"""
        try:
            prompt = self._create_enhanced_prompt(email)
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for consistency
                        "top_p": 0.9
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_response = result.get('response', '')
                return self._parse_llm_response(llm_response)
            else:
                logger.warning(f"LLM API error: {response.status_code}")
                return 0.5
                
        except Exception as e:
            logger.warning(f"LLM API error: {e}")
            return 0.5
    
    def _get_llm_score_cli(self, email):
        """Get LLM score via CLI (fallback)"""
        try:
            prompt = self._create_enhanced_prompt(email)
            
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                return self._parse_llm_response(result.stdout.strip())
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"LLM CLI error: {e}")
            return 0.5
    
    def _create_enhanced_prompt(self, email):
        """Create an enhanced prompt for better LLM analysis"""
        prompt = f"""Analyze this email for phishing indicators. Reply with ONLY a JSON object containing score and reasoning.

Email Details:
Subject: {email.get('subject', 'No subject')}
Sender: {email.get('sender', 'Unknown')}
Body: {email.get('body', '')[:500]}  # Limit body length

Analyze for:
1. Urgency and pressure tactics
2. Suspicious sender address
3. Grammar and spelling errors
4. Requests for sensitive information
5. Suspicious links or attachments
6. Too-good-to-be-true offers
7. Threats or fear tactics

Respond with JSON only:
{{"score": <0.0 to 1.0>, "reasoning": "<brief explanation>"}}
"""
        return prompt
    
    def _parse_llm_response(self, response):
        """Parse LLM response to extract score"""
        try:
            # Try to parse as JSON first
            if '{' in response and '}' in response:
                json_str = response[response.find('{'):response.rfind('}')+1]
                result = json.loads(json_str)
                score = float(result.get('score', 0.5))
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except:
            pass
        
        # Fallback: look for numeric score
        import re
        numbers = re.findall(r'0\.\d+|1\.0|1|0', response)
        if numbers:
            try:
                score = float(numbers[0])
                return max(0.0, min(1.0, score))
            except:
                pass
        
        # Fallback: keyword-based scoring
        response_lower = response.lower()
        if any(word in response_lower for word in ['phishing', 'suspicious', 'scam', 'fraudulent']):
            return 0.8
        elif any(word in response_lower for word in ['legitimate', 'safe', 'genuine', 'authentic']):
            return 0.2
        
        return 0.5  # Neutral if can't parse
    
    def _combine_scores(self, scores):
        """Combine scores using weighted average"""
        # Adjust weights if LLM is not available
        adjusted_weights = self.weights.copy()
        
        if scores['llm_score'] == 0.5 and not self.llm_available:
            # Redistribute LLM weight to other components
            llm_weight = adjusted_weights['llm_score']
            adjusted_weights['llm_score'] = 0
            
            # Redistribute proportionally
            remaining_keys = [k for k in adjusted_weights if k != 'llm_score']
            for key in remaining_keys:
                adjusted_weights[key] += llm_weight / len(remaining_keys)
        
        # Calculate weighted average
        total_weight = sum(adjusted_weights.values())
        final_score = sum(
            scores.get(key, 0.5) * adjusted_weights[key] 
            for key in adjusted_weights
        ) / total_weight
        
        return final_score
    
    def _optimize_weights(self, val_data):
        """Optimize weights using validation data"""
        logger.info("Optimizing detection weights...")
        
        best_weights = self.weights.copy()
        best_accuracy = 0
        
        # Grid search for optimal weights
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for llm_w in weight_options:
            for rule_w in weight_options:
                for url_w in weight_options:
                    for sender_w in weight_options:
                        for content_w in weight_options:
                            # Ensure weights sum to 1
                            total = llm_w + rule_w + url_w + sender_w + content_w
                            if abs(total - 1.0) > 0.1:
                                continue
                            
                            # Test these weights
                            test_weights = {
                                'llm_score': llm_w,
                                'rule_score': rule_w,
                                'url_score': url_w,
                                'sender_score': sender_w,
                                'content_score': content_w
                            }
                            
                            self.weights = test_weights
                            predictions = self.predict([e for e in val_data])
                            labels = np.array([e['label'] for e in val_data])
                            
                            accuracy = np.mean(predictions == labels)
                            
                            if accuracy > best_accuracy:
                                best_accuracy = accuracy
                                best_weights = test_weights.copy()
        
        self.weights = best_weights
        logger.info(f"Optimized weights: {self.weights}")


class AdvancedRuleEngine:
    """Advanced rule-based detection engine"""
    
    def __init__(self):
        self.phishing_indicators = {
            'urgency_words': ['urgent', 'immediate', 'expire', 'suspend', 'act now'],
            'threat_words': ['suspend', 'close', 'terminate', 'block', 'restrict'],
            'reward_words': ['winner', 'prize', 'congratulations', 'selected', 'free'],
            'action_words': ['click', 'verify', 'confirm', 'update', 'validate'],
            'credential_words': ['password', 'username', 'pin', 'account', 'login']
        }
        
        self.learned_patterns = []
    
    def learn_from_data(self, train_data):
        """Learn patterns from training data"""
        phishing_emails = [e for e in train_data if e['label'] == 1]
        
        # Extract common patterns
        for email in phishing_emails:
            subject = email.get('subject', '').lower()
            body = email.get('body', '').lower()
            
            # Learn new suspicious phrases
            if 'verify' in subject or 'verify' in body:
                self.learned_patterns.append('verification_request')
            if 'suspended' in subject or 'suspended' in body:
                self.learned_patterns.append('account_suspension')
    
    def analyze(self, email):
        """Analyze email using rules"""
        score = 0.0
        
        subject = email.get('subject', '').lower()
        body = email.get('body', '').lower()
        full_text = subject + ' ' + body
        
        # Check for urgency
        urgency_count = sum(1 for word in self.phishing_indicators['urgency_words'] if word in full_text)
        if urgency_count > 0:
            score += 0.2 * min(urgency_count, 3)
        
        # Check for threats
        threat_count = sum(1 for word in self.phishing_indicators['threat_words'] if word in full_text)
        if threat_count > 0:
            score += 0.25 * min(threat_count, 2)
        
        # Check for rewards
        reward_count = sum(1 for word in self.phishing_indicators['reward_words'] if word in full_text)
        if reward_count > 0:
            score += 0.3 * min(reward_count, 2)
        
        # Check for credential requests
        cred_count = sum(1 for word in self.phishing_indicators['credential_words'] if word in full_text)
        if cred_count > 1:
            score += 0.35
        
        # Check learned patterns
        for pattern in self.learned_patterns:
            if pattern == 'verification_request' and 'verify' in full_text:
                score += 0.1
            elif pattern == 'account_suspension' and 'suspended' in full_text:
                score += 0.15
        
        return min(score, 1.0)


class FeatureExtractor:
    """Extract various features from emails"""
    
    def __init__(self):
        self.suspicious_tlds = ['.tk', '.ml', '.ga', '.cf']
        self.url_shorteners = ['bit.ly', 'tinyurl.com', 'goo.gl']
    
    def fit(self, train_data):
        """Learn from training data"""
        # Could implement feature learning here
        pass
    
    def analyze_urls(self, email):
        """Analyze URLs in email"""
        body = email.get('body', '')
        
        # Simple URL detection
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, body)
        
        if not urls:
            return 0.3  # No URLs is slightly suspicious
        
        score = 0.0
        
        for url in urls:
            # Check for suspicious TLDs
            if any(tld in url for tld in self.suspicious_tlds):
                score += 0.3
            
            # Check for URL shorteners
            if any(shortener in url for shortener in self.url_shorteners):
                score += 0.2
            
            # Check for IP addresses
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                score += 0.4
        
        return min(score / len(urls), 1.0)
    
    def analyze_sender(self, email):
        """Analyze sender information"""
        sender = email.get('sender', '').lower()
        
        if not sender:
            return 0.5
        
        score = 0.5  # Start neutral
        
        # Check for suspicious patterns
        if 'noreply' in sender or 'no-reply' in sender:
            score += 0.1
        
        if re.search(r'\d{3,}', sender):  # Many numbers
            score += 0.15
        
        # Check for spoofing attempts
        if '@' in sender:
            domain = sender.split('@')[1]
            
            # Check for lookalike domains
            legitimate = ['paypal.com', 'amazon.com', 'google.com', 'microsoft.com']
            for legit in legitimate:
                base = legit.split('.')[0]
                if base in domain and legit != domain:
                    score += 0.3  # Likely spoofing
        
        return min(score, 1.0)
    
    def analyze_content(self, email):
        """Analyze email content"""
        subject = email.get('subject', '')
        body = email.get('body', '')
        
        score = 0.0
        
        # Check for all caps in subject
        if subject.isupper() and len(subject) > 5:
            score += 0.2
        
        # Check for excessive punctuation
        if subject.count('!') > 2 or body.count('!') > 5:
            score += 0.15
        
        # Check for grammar errors (simple heuristics)
        grammar_errors = [
            r'\byou\'re account\b',
            r'\byour an?\b',
            r'\brecieve\b',
            r'\boccured\b'
        ]
        
        full_text = (subject + ' ' + body).lower()
        for pattern in grammar_errors:
            if re.search(pattern, full_text):
                score += 0.1
        
        # Check for hidden text attempts
        if '<' in body and '>' in body:
            if 'display:none' in body or 'visibility:hidden' in body:
                score += 0.4
        
        return min(score, 1.0)