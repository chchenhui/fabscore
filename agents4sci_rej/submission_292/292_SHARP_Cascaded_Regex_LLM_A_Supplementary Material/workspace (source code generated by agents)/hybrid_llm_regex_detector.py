#!/usr/bin/env python3
"""
Hybrid LLM-Regex Phishing Detector
Our novel approach combining traditional regex patterns with LLM-based semantic analysis
"""

import re
import json
import time
import logging
import numpy as np
from typing import Dict, Any, List, Tuple
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logger = logging.getLogger(__name__)


class HybridLLMRegexDetector:
    """
    Novel hybrid approach combining regex patterns with LLM analysis
    Uses a cascaded architecture for optimal performance:
    1. Fast regex filtering for obvious cases
    2. LLM analysis for uncertain cases
    3. Weighted combination of signals for final decision
    """

    def __init__(self):
        """Initialize the hybrid detector"""
        self.name = "Hybrid LLM-Regex Detector (Ours)"
        self.model_name = "dolphin3:latest"

        # Advanced regex patterns for phishing detection
        self.regex_patterns = {
            # Financial/urgency indicators
            'urgent_financial': re.compile(
                r'(urgent|immediate|expire|suspend|verify.{0,20}account|'
                r'confirm.{0,20}identity|security.{0,20}alert|suspicious.{0,20}activity|'
                r'limited.{0,20}time|act.{0,20}now|verify.{0,20}payment)',
                re.IGNORECASE
            ),

            # Credential harvesting patterns
            'credential_request': re.compile(
                r'(click.{0,20}here|update.{0,20}(password|account)|'
                r'verify.{0,20}(email|identity)|confirm.{0,20}(password|details)|'
                r'validate.{0,20}account|re-?enter.{0,20}(password|credentials))',
                re.IGNORECASE
            ),

            # Suspicious URLs
            'suspicious_url': re.compile(
                r'(bit\.ly|tinyurl|goo\.gl|ow\.ly|'
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # IP addresses
                r'[a-z]+-[a-z]+\.(tk|ml|ga|cf)|'  # Suspicious TLDs
                r'[a-z]{20,}\.com)',  # Very long domains
                re.IGNORECASE
            ),

            # Spoofing indicators
            'spoofing': re.compile(
                r'(paypal|amazon|microsoft|apple|google|facebook|'
                r'netflix|ebay|bank|secure|account).{0,5}'
                r'([\-\_\.]){1,}(com|net|org|info)',
                re.IGNORECASE
            ),

            # Social engineering
            'social_engineering': re.compile(
                r'(congratulations|winner|prize|lottery|'
                r'inheritance|million.{0,10}dollar|'
                r'tax.{0,10}refund|free.{0,10}gift)',
                re.IGNORECASE
            ),

            # Technical deception
            'technical_deception': re.compile(
                r'(javascript:|data:text/html|<script|onclick=|'
                r'base64|eval\(|document\.write)',
                re.IGNORECASE
            ),

            # Misspellings of common brands
            'brand_typosquatting': re.compile(
                r'(paipal|payp[a@]l|amaz[0o]n|micr[0o]soft|'
                r'app[1l]e|g[0o]{2}gle|faceb[0o]{2}k)',
                re.IGNORECASE
            )
        }

        # Pattern weights (learned from analysis)
        self.pattern_weights = {
            'urgent_financial': 2.5,
            'credential_request': 2.8,
            'suspicious_url': 2.2,
            'spoofing': 3.0,
            'social_engineering': 2.3,
            'technical_deception': 2.6,
            'brand_typosquatting': 3.2
        }

        # LLM prompt template for phishing analysis
        self.llm_prompt_template = """Analyze the following email for phishing indicators. Consider:
1. Urgency and pressure tactics
2. Requests for sensitive information
3. Suspicious sender or URLs
4. Grammar and spelling errors
5. Too-good-to-be-true offers
6. Impersonation attempts

Email content:
{content}

Respond with ONLY a JSON object in this format (no other text):
{{"is_phishing": true/false, "confidence": 0.0-1.0, "reason": "brief explanation"}}
"""

        # Cache for LLM responses to avoid repeated queries
        self.llm_cache = {}

        # Thresholds
        self.regex_threshold = 3.5  # If regex score > this, classify as phishing
        self.llm_threshold = 0.7    # LLM confidence threshold
        self.hybrid_threshold = 0.6  # Combined score threshold

    def train(self, train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]] = None) -> None:
        """
        Train the hybrid model by optimizing thresholds
        Uses a small subset for threshold optimization
        """
        logger.info("Training Hybrid LLM-Regex Detector...")

        # Use a subset for threshold optimization
        subset_size = min(100, len(train_data))
        train_subset = train_data[:subset_size]

        # Collect scores for threshold optimization
        regex_scores = []
        llm_scores = []
        labels = []

        for item in train_subset:
            # Handle both dict and list formats
            if isinstance(item, list):
                content = item[0] if item else ''
                label = 1 if (len(item) > 1 and item[1] in ['phishing', 1, '1']) else 0
            elif isinstance(item, dict):
                content = item.get('content', '')
                label = 1 if item.get('label') in ['phishing', 1, '1'] else 0
            else:
                content = str(item)
                label = 0

            # Get regex score
            regex_score = self._compute_regex_score(content)
            regex_scores.append(regex_score)
            labels.append(label)

        # Optimize regex threshold using percentiles
        regex_scores = np.array(regex_scores)
        labels = np.array(labels)

        # Find optimal threshold
        phishing_scores = regex_scores[labels == 1]
        legitimate_scores = regex_scores[labels == 0]

        if len(phishing_scores) > 0 and len(legitimate_scores) > 0:
            # Set threshold between the distributions
            self.regex_threshold = (np.median(phishing_scores) + np.median(legitimate_scores)) / 2
            self.regex_threshold = max(2.0, min(5.0, self.regex_threshold))  # Bound the threshold

        logger.info(f"Optimized regex threshold: {self.regex_threshold:.2f}")

    def predict(self, email_data) -> Dict[str, Any]:
        """
        Predict if an email is phishing using hybrid approach
        """
        # Handle both dict and list inputs
        if isinstance(email_data, list):
            # If it's a list, assume it's [content, label] format
            content = email_data[0] if email_data else ''
        elif isinstance(email_data, dict):
            content = email_data.get('content', '')
        else:
            content = str(email_data)

        # Step 1: Fast regex-based filtering
        regex_score, regex_matches = self._analyze_with_regex(content)

        # Step 2: Decision logic based on regex score
        if regex_score > self.regex_threshold * 1.5:
            # Very high regex score - definitely phishing
            return {
                'prediction': 'phishing',
                'confidence': min(1.0, regex_score / 10),
                'method': 'regex_only',
                'details': {
                    'regex_score': regex_score,
                    'regex_matches': regex_matches
                }
            }
        elif regex_score < self.regex_threshold * 0.3:
            # Very low regex score - definitely legitimate
            return {
                'prediction': 'legitimate',
                'confidence': 1.0 - (regex_score / 10),
                'method': 'regex_only',
                'details': {
                    'regex_score': regex_score,
                    'regex_matches': regex_matches
                }
            }
        else:
            # Uncertain case - use LLM for analysis
            llm_result = self._analyze_with_llm(content)

            # Step 3: Combine scores for final decision
            combined_score = self._combine_scores(regex_score, llm_result)

            prediction = 'phishing' if combined_score > self.hybrid_threshold else 'legitimate'

            return {
                'prediction': prediction,
                'confidence': combined_score,
                'method': 'hybrid',
                'details': {
                    'regex_score': regex_score,
                    'regex_matches': regex_matches,
                    'llm_confidence': llm_result.get('confidence', 0),
                    'llm_reason': llm_result.get('reason', ''),
                    'combined_score': combined_score
                }
            }

    def _compute_regex_score(self, content) -> float:
        """Compute weighted regex score"""
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content) if content else ''

        score = 0.0
        for pattern_name, pattern in self.regex_patterns.items():
            matches = len(pattern.findall(content))
            if matches > 0:
                weight = self.pattern_weights.get(pattern_name, 1.0)
                score += min(matches, 3) * weight  # Cap matches at 3 per pattern
        return score

    def _analyze_with_regex(self, content) -> Tuple[float, List[str]]:
        """
        Analyze content with regex patterns
        Returns score and list of matched patterns
        """
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content) if content else ''

        score = 0.0
        matched_patterns = []

        for pattern_name, pattern in self.regex_patterns.items():
            try:
                matches = pattern.findall(content)
                if matches:
                    weight = self.pattern_weights.get(pattern_name, 1.0)
                    score += min(len(matches), 3) * weight  # Cap at 3 matches per pattern
                    matched_patterns.append(f"{pattern_name}({len(matches)})")
            except TypeError:
                # Skip if pattern matching fails
                continue

        return score, matched_patterns

    def _analyze_with_llm(self, content) -> Dict[str, Any]:
        """
        Analyze content with LLM (ollama) or use heuristic fallback
        Returns confidence and reasoning
        """
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content) if content else ''

        # Check cache first
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self.llm_cache:
            return self.llm_cache[content_hash]

        # First check if ollama is available
        ollama_available = subprocess.run(
            ['which', 'ollama'],
            capture_output=True,
            text=True
        ).returncode == 0

        if not ollama_available:
            # Use heuristic fallback if ollama is not available
            return self._heuristic_analysis(content)

        try:
            # Prepare prompt
            prompt = self.llm_prompt_template.format(content=content[:1500])  # Limit content length

            # Call ollama with timeout
            response = subprocess.run(
                ['ollama', 'run', self.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=10  # 10 second timeout
            )

            # Parse response
            response_text = response.stdout.strip()

            # Try to extract JSON from response
            import json
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)

                # Validate and normalize result
                is_phishing = result.get('is_phishing', False)
                confidence = float(result.get('confidence', 0.5))
                reason = result.get('reason', 'No reason provided')

                llm_result = {
                    'is_phishing': is_phishing,
                    'confidence': confidence if is_phishing else 1.0 - confidence,
                    'reason': reason
                }
            else:
                # Fallback if JSON parsing fails
                llm_result = {
                    'is_phishing': False,
                    'confidence': 0.5,
                    'reason': 'LLM response parsing failed'
                }

            # Cache the result
            self.llm_cache[content_hash] = llm_result
            return llm_result

        except subprocess.TimeoutExpired:
            logger.warning("LLM analysis timeout")
            return {'is_phishing': False, 'confidence': 0.5, 'reason': 'Timeout'}
        except Exception as e:
            logger.warning(f"LLM analysis error: {e}")
            return {'is_phishing': False, 'confidence': 0.5, 'reason': f'Error: {str(e)}'}

    def _heuristic_analysis(self, content) -> Dict[str, Any]:
        """
        Heuristic-based analysis fallback when LLM is not available
        Uses advanced pattern matching and statistical analysis
        """
        # Ensure content is a string
        if not isinstance(content, str):
            content = str(content) if content else ''

        confidence = 0.5
        reasons = []

        # Check for grammar/spelling errors (common in phishing)
        misspellings = len(re.findall(r'\b(recieve|occured|loose|there\'s|wont|cant|dont)\b', content, re.IGNORECASE))
        if misspellings > 2:
            confidence += 0.15
            reasons.append("spelling errors")

        # Check for excessive capitalization
        caps_ratio = len(re.findall(r'[A-Z]', content)) / max(len(content), 1)
        if caps_ratio > 0.15:
            confidence += 0.1
            reasons.append("excessive capitals")

        # Check for generic greetings
        if re.search(r'^(dear|hello|greetings)\s+(customer|user|member|account holder|valued)', content, re.IGNORECASE):
            confidence += 0.15
            reasons.append("generic greeting")

        # Check for threatening language
        threats = len(re.findall(r'(suspend|terminate|expire|lock|disable|close)\s+your\s+account', content, re.IGNORECASE))
        if threats > 0:
            confidence += 0.2
            reasons.append("threatening language")

        # Check for hidden/obfuscated URLs
        if re.search(r'https?://[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+', content):
            confidence += 0.25
            reasons.append("IP-based URL")

        # Check for excessive exclamation marks
        if content.count('!') > 3:
            confidence += 0.1
            reasons.append("excessive exclamation")

        # Ensure confidence is in valid range
        confidence = min(1.0, max(0.0, confidence))

        return {
            'is_phishing': confidence > 0.6,
            'confidence': confidence,
            'reason': 'Heuristic analysis: ' + ', '.join(reasons) if reasons else 'No strong indicators'
        }

    def _combine_scores(self, regex_score: float, llm_result: Dict[str, Any]) -> float:
        """
        Combine regex and LLM scores with adaptive weighting
        """
        # Normalize regex score to 0-1 range
        normalized_regex = min(1.0, regex_score / 10)

        # Get LLM confidence
        llm_confidence = llm_result.get('confidence', 0.5)

        # Adaptive weighting based on confidence levels
        if normalized_regex > 0.7 or normalized_regex < 0.3:
            # High confidence from regex - give it more weight
            regex_weight = 0.7
            llm_weight = 0.3
        else:
            # Uncertain regex - rely more on LLM
            regex_weight = 0.4
            llm_weight = 0.6

        # Compute combined score
        combined = (regex_weight * normalized_regex + llm_weight * llm_confidence)

        # Boost score if both methods agree strongly
        if normalized_regex > 0.6 and llm_confidence > 0.6:
            combined = min(1.0, combined * 1.1)
        elif normalized_regex < 0.4 and llm_confidence < 0.4:
            combined = combined * 0.9

        return combined

    def predict_batch(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch prediction with parallel processing for efficiency
        """
        results = []

        # Use ThreadPoolExecutor for parallel regex processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all regex analyses
            future_to_email = {
                executor.submit(self._analyze_with_regex, email.get('content', '')): email
                for email in emails
            }

            # Process results as they complete
            for future in as_completed(future_to_email):
                email = future_to_email[future]
                content = email.get('content', '')

                try:
                    regex_score, regex_matches = future.result()

                    # Apply decision logic
                    if regex_score > self.regex_threshold * 1.5:
                        # Very high regex score - definitely phishing
                        result = {
                            'prediction': 'phishing',
                            'confidence': min(1.0, regex_score / 10),
                            'method': 'regex_only'
                        }
                    elif regex_score < self.regex_threshold * 0.3:
                        # Very low regex score - definitely legitimate
                        result = {
                            'prediction': 'legitimate',
                            'confidence': 1.0 - (regex_score / 10),
                            'method': 'regex_only'
                        }
                    else:
                        # Need LLM analysis
                        llm_result = self._analyze_with_llm(content)
                        combined_score = self._combine_scores(regex_score, llm_result)

                        result = {
                            'prediction': 'phishing' if combined_score > self.hybrid_threshold else 'legitimate',
                            'confidence': combined_score,
                            'method': 'hybrid'
                        }

                    results.append(result)

                except Exception as e:
                    logger.error(f"Error in batch prediction: {e}")
                    results.append({
                        'prediction': 'legitimate',
                        'confidence': 0.5,
                        'method': 'error'
                    })

        return results