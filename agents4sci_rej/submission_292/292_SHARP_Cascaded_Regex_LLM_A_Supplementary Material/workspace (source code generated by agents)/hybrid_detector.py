"""
Hybrid Phishing Detector combining LLM and rule-based methods
Uses Ollama for LLM-based detection
"""

import re
import json
import logging
import subprocess
import numpy as np
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class HybridPhishingDetector:
    """Hybrid detector combining LLM analysis with rule-based methods"""
    
    def __init__(self, use_ollama=True, model_name="dolphin3:latest"):
        self.use_ollama = use_ollama
        self.model_name = model_name
        
        # Rule-based components
        self.rule_engine = EnhancedRuleEngine()
        
        # Feature weights for ensemble
        self.weights = {
            'llm_score': 0.5,
            'rule_score': 0.3,
            'url_analysis': 0.1,
            'sender_analysis': 0.1
        }
        
        # Check Ollama availability
        if self.use_ollama:
            self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"Ollama is available. Using model: {self.model_name}")
                # Check if model exists
                if self.model_name not in result.stdout:
                    logger.warning(f"Model {self.model_name} not found. Pulling model...")
                    self._pull_model()
            else:
                logger.warning("Ollama not available. Falling back to rule-based only.")
                self.use_ollama = False
        except Exception as e:
            logger.warning(f"Could not connect to Ollama: {e}. Using rule-based only.")
            self.use_ollama = False
    
    def _pull_model(self):
        """Pull the Ollama model if not available"""
        try:
            subprocess.run(
                ["ollama", "pull", self.model_name],
                timeout=300  # 5 minutes timeout
            )
            logger.info(f"Successfully pulled model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            self.use_ollama = False
    
    def train(self, train_data, val_data=None):
        """Train/optimize the hybrid detector"""
        logger.info("Training hybrid detector...")
        
        # Optimize weights using validation data if available
        if val_data:
            self._optimize_weights(val_data)
        
        # Train rule engine on patterns
        self.rule_engine.learn_patterns(train_data)
        
        logger.info("Hybrid detector training complete")
    
    def predict(self, emails):
        """Predict phishing emails using hybrid approach"""
        predictions = []
        
        for email in emails:
            # Get individual scores
            scores = {}
            
            # 1. LLM-based analysis
            if self.use_ollama:
                scores['llm_score'] = self._get_llm_score(email)
            else:
                scores['llm_score'] = 0.5  # Neutral if LLM not available
            
            # 2. Rule-based analysis
            scores['rule_score'] = self.rule_engine.analyze(email)
            
            # 3. URL analysis
            scores['url_analysis'] = self._analyze_urls(email)
            
            # 4. Sender analysis
            scores['sender_analysis'] = self._analyze_sender(email)
            
            # Combine scores using weighted average
            final_score = sum(
                scores[key] * self.weights[key] 
                for key in scores
            )
            
            # Make prediction (threshold at 0.5)
            predictions.append(1 if final_score > 0.5 else 0)
        
        return np.array(predictions)
    
    def _get_llm_score(self, email):
        """Get phishing probability from LLM"""
        try:
            # Prepare prompt for LLM
            prompt = self._create_llm_prompt(email)
            
            # Call Ollama
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                response = result.stdout.strip()
                
                # Parse LLM response
                score = self._parse_llm_response(response)
                return score
            else:
                logger.warning(f"Ollama error: {result.stderr}")
                return 0.5
                
        except subprocess.TimeoutExpired:
            logger.warning("Ollama timeout")
            return 0.5
        except Exception as e:
            logger.warning(f"LLM scoring error: {e}")
            return 0.5
    
    def _create_llm_prompt(self, email):
        """Create prompt for LLM analysis"""
        prompt = f"""Analyze this email for phishing indicators. Reply with ONLY a single number between 0 and 1 representing the probability this is a phishing email (0=legitimate, 1=phishing).

Subject: {email.get('subject', 'N/A')}
From: {email.get('sender', 'N/A')}
Body: {email.get('body', 'N/A')[:500]}

Consider: urgency, credential requests, suspicious links, sender authenticity, grammar errors, and typical phishing patterns.

Probability (0-1):"""
        
        return prompt
    
    def _parse_llm_response(self, response):
        """Parse LLM response to extract probability"""
        try:
            # Try to extract a number from the response
            numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
            if numbers:
                score = float(numbers[0])
                # Ensure score is between 0 and 1
                return min(max(score, 0.0), 1.0)
        except:
            pass
        
        # Fallback: look for keywords
        response_lower = response.lower()
        if 'phishing' in response_lower or 'suspicious' in response_lower:
            return 0.8
        elif 'legitimate' in response_lower or 'safe' in response_lower:
            return 0.2
        
        return 0.5
    
    def _analyze_urls(self, email):
        """Analyze URLs in the email"""
        body = email.get('body', '')
        
        # Extract URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, body)
        
        if not urls:
            return 0.3  # No URLs is slightly safer
        
        suspicious_count = 0
        for url in urls:
            # Check for suspicious patterns
            if any(pattern in url.lower() for pattern in [
                'bit.ly', 'tinyurl', 'goo.gl', '.tk', '.ml', '.ga'
            ]):
                suspicious_count += 1
            
            # Check for IP addresses
            if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
                suspicious_count += 1
            
            # Check for misleading domains
            if re.search(r'[0-9]+(paypal|amazon|google|microsoft)', url.lower()):
                suspicious_count += 1
        
        # Calculate score
        if suspicious_count > 0:
            return min(0.5 + (suspicious_count * 0.2), 1.0)
        return 0.2
    
    def _analyze_sender(self, email):
        """Analyze sender address for legitimacy"""
        sender = email.get('sender', '').lower()
        
        if not sender:
            return 0.5
        
        score = 0.5  # Start neutral
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'noreply',
            r'no-reply',
            r'donotreply',
            r'\d{3,}',  # Many numbers
            r'[0-9]+@',  # Numbers before @
            r'@.*\.(tk|ml|ga|cf)$',  # Suspicious TLDs
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, sender):
                score += 0.15
        
        # Check for legitimate patterns
        legitimate_patterns = [
            r'@(gmail|yahoo|outlook|hotmail)\.com$',
            r'@.*\.(edu|gov|org)$',
            r'^[a-zA-Z]+\.[a-zA-Z]+@',  # firstname.lastname format
        ]
        
        for pattern in legitimate_patterns:
            if re.search(pattern, sender):
                score -= 0.2
        
        # Domain mismatch check
        if 'paypal' in sender and 'paypal.com' not in sender:
            score += 0.3
        if 'amazon' in sender and 'amazon.com' not in sender:
            score += 0.3
        
        return min(max(score, 0.0), 1.0)
    
    def _optimize_weights(self, val_data):
        """Optimize ensemble weights using validation data"""
        # Simple grid search for optimal weights
        best_accuracy = 0
        best_weights = self.weights.copy()
        
        weight_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for llm_w in weight_options:
            for rule_w in weight_options:
                for url_w in weight_options:
                    sender_w = 1.0 - llm_w - rule_w - url_w
                    
                    if sender_w >= 0 and sender_w <= 1:
                        temp_weights = {
                            'llm_score': llm_w,
                            'rule_score': rule_w,
                            'url_analysis': url_w,
                            'sender_analysis': sender_w
                        }
                        
                        # Test these weights
                        self.weights = temp_weights
                        predictions = self.predict(val_data)
                        
                        labels = [email['label'] for email in val_data]
                        accuracy = np.mean(predictions == labels)
                        
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_weights = temp_weights.copy()
        
        self.weights = best_weights
        logger.info(f"Optimized weights: {self.weights}")

class EnhancedRuleEngine:
    """Enhanced rule engine with learning capabilities"""
    
    def __init__(self):
        self.phishing_indicators = {
            'urgent_language': {
                'patterns': [
                    r'\burgent\b', r'\bimmediate\b', r'\bact now\b',
                    r'\bexpire', r'\blimited time\b', r'\basap\b'
                ],
                'weight': 0.15
            },
            'credential_request': {
                'patterns': [
                    r'\bpassword\b', r'\busername\b', r'\bpin\b',
                    r'\bssn\b', r'\bsocial security\b', r'\bcredit card\b'
                ],
                'weight': 0.25
            },
            'money_mention': {
                'patterns': [
                    r'\$\d+', r'\bmillion\b', r'\bprize\b',
                    r'\bwinner\b', r'\breward\b', r'\brefund\b'
                ],
                'weight': 0.2
            },
            'grammatical_errors': {
                'patterns': [
                    r'\b(recieve|loose|there account|you\'re account)\b',
                    r'[.!?]{2,}',  # Multiple punctuation
                    r'\b[A-Z]{5,}\b'  # Excessive caps
                ],
                'weight': 0.1
            },
            'suspicious_attachments': {
                'patterns': [
                    r'\.exe\b', r'\.zip\b', r'\.scr\b',
                    r'\.bat\b', r'\.com\b', r'\.pif\b'
                ],
                'weight': 0.2
            },
            'impersonation': {
                'patterns': [
                    r'payp[a@]l', r'amaz[o0]n', r'micr[o0]s[o0]ft',
                    r'app1e', r'g[o0][o0]gle'
                ],
                'weight': 0.3
            }
        }
        
        self.learned_patterns = []
    
    def learn_patterns(self, train_data):
        """Learn new patterns from training data"""
        phishing_emails = [e for e in train_data if e['label'] == 1]
        
        # Extract common n-grams from phishing emails
        from collections import Counter
        
        all_text = ' '.join([
            e.get('subject', '') + ' ' + e.get('body', '')
            for e in phishing_emails
        ]).lower()
        
        # Find common 2-grams and 3-grams
        words = all_text.split()
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
        common_bigrams = Counter(bigrams).most_common(20)
        common_trigrams = Counter(trigrams).most_common(10)
        
        # Add as learned patterns
        for phrase, count in common_bigrams + common_trigrams:
            if count > 5:  # Appears in multiple emails
                self.learned_patterns.append(phrase)
    
    def analyze(self, email):
        """Analyze email using enhanced rules"""
        text = (email.get('subject', '') + ' ' + email.get('body', '')).lower()
        
        total_score = 0
        
        # Check predefined indicators
        for indicator_name, indicator_data in self.phishing_indicators.items():
            indicator_score = 0
            
            for pattern in indicator_data['patterns']:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    indicator_score += len(matches) * 0.1
            
            # Cap individual indicator contribution
            indicator_score = min(indicator_score, 1.0)
            total_score += indicator_score * indicator_data['weight']
        
        # Check learned patterns
        for pattern in self.learned_patterns:
            if pattern in text:
                total_score += 0.05
        
        # Additional heuristics
        
        # Check for hidden text
        if re.search(r'color:\s*white|display:\s*none', text):
            total_score += 0.2
        
        # Check for homograph attacks (Unicode lookalikes)
        if any(ord(c) > 127 for c in text):
            total_score += 0.1
        
        # Length-based heuristics
        if len(text) < 50:  # Very short emails
            total_score += 0.1
        
        # Normalize score to [0, 1]
        return min(total_score, 1.0)