"""
Statement of Purpose (SoP) rubric scoring and summarization module
Multi-label rubric classification with cited-span summary generation
"""

import re
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available, using rule-based scoring only")

from ocr_backends import Token

logger = logging.getLogger(__name__)

@dataclass
class RubricScore:
    """Represents a rubric dimension score with evidence"""
    dimension: str
    score: float  # 0-5 scale
    confidence: float
    evidence_spans: List[Tuple[int, int, str]]  # (start, end, text)
    reasoning: str

@dataclass
class SoPAnalysisResult:
    """Complete result of statement of purpose analysis"""
    rubric_scores: Dict[str, RubricScore]
    overall_score: float
    cited_summary: str
    rouge_score: float
    word_count: int
    readability_score: float
    key_themes: List[str]
    warnings: List[str]


class SoPRubricAnalyzer:
    """Analyzes statements of purpose using multi-dimensional rubric"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.rubric_dimensions = [
            "research_interest",
            "experience_relevance", 
            "writing_quality",
            "goal_clarity",
            "fit_assessment"
        ]
        
        # Initialize vocabularies and patterns
        self._initialize_scoring_vocabularies()
        self._initialize_patterns()
        
        # Initialize ML models if available
        self.models = {}
        if SKLEARN_AVAILABLE:
            self._initialize_ml_models()
    
    def _initialize_scoring_vocabularies(self):
        """Initialize vocabulary for rubric scoring"""
        
        # Research interest keywords
        self.research_keywords = {
            'machine_learning': ['machine learning', 'ml', 'deep learning', 'neural networks', 'ai', 'artificial intelligence'],
            'data_science': ['data science', 'data mining', 'analytics', 'big data', 'statistics'],
            'computer_vision': ['computer vision', 'image processing', 'cv', 'object detection', 'recognition'],
            'nlp': ['natural language processing', 'nlp', 'text mining', 'language models', 'linguistics'],
            'robotics': ['robotics', 'automation', 'control systems', 'mechatronics'],
            'systems': ['systems', 'distributed systems', 'databases', 'networks', 'architecture'],
            'security': ['security', 'cybersecurity', 'cryptography', 'privacy', 'blockchain'],
            'hci': ['human computer interaction', 'hci', 'user experience', 'interface', 'usability']
        }
        
        # Experience indicators
        self.experience_indicators = [
            'internship', 'intern', 'work experience', 'project', 'research', 'publication',
            'conference', 'workshop', 'competition', 'hackathon', 'open source',
            'github', 'portfolio', 'leadership', 'team', 'collaboration'
        ]
        
        # Goal clarity indicators
        self.goal_indicators = [
            'goal', 'objective', 'aim', 'plan', 'career', 'future', 'aspire',
            'pursue', 'specialize', 'focus', 'interested', 'passion'
        ]
        
        # Writing quality indicators (positive)
        self.quality_positive = [
            'furthermore', 'moreover', 'additionally', 'consequently', 'therefore',
            'specifically', 'particularly', 'especially', 'significantly'
        ]
        
        # Writing quality indicators (negative)  
        self.quality_negative = [
            'very very', 'really really', 'so so', 'definitely definitely'
        ]
        
        # Fit assessment indicators
        self.fit_indicators = [
            'university', 'program', 'department', 'faculty', 'professor',
            'advisor', 'research group', 'lab', 'facility', 'resource',
            'opportunity', 'align', 'match', 'fit', 'suitable'
        ]
    
    def _initialize_patterns(self):
        """Initialize regex patterns for text analysis"""
        
        # Sentence boundary detection
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        
        # Word counting
        self.word_pattern = re.compile(r'\b\w+\b')
        
        # Year patterns for timeline
        self.year_pattern = re.compile(r'\b(19|20)\d{2}\b')
        
        # First person indicators
        self.first_person_pattern = re.compile(r'\b(I|my|me|myself|we|our|us)\b', re.IGNORECASE)
        
        # Future tense indicators
        self.future_pattern = re.compile(r'\b(will|would|plan to|hope to|intend to|aim to)\b', re.IGNORECASE)
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for rubric scoring"""
        if not SKLEARN_AVAILABLE:
            return
            
        # Simple TF-IDF + Logistic Regression for each dimension
        for dimension in self.rubric_dimensions:
            self.models[dimension] = {
                'vectorizer': TfidfVectorizer(max_features=1000, stop_words='english'),
                'classifier': LogisticRegression(random_state=42)
            }
    
    def analyze_statement(self, tokens: List[Token], ground_truth_bullets: List[str] = None) -> SoPAnalysisResult:
        """Analyze statement of purpose and generate rubric scores"""
        # Convert tokens to text
        text = " ".join(token.text for token in tokens)
        
        try:
            # Basic text statistics
            word_count = len(self.word_pattern.findall(text))
            sentences = self.sentence_pattern.split(text)
            
            # Score each rubric dimension
            rubric_scores = {}
            for dimension in self.rubric_dimensions:
                score = self._score_dimension(dimension, text, sentences)
                rubric_scores[dimension] = score
            
            # Compute overall score
            overall_score = np.mean([score.score for score in rubric_scores.values()])
            
            # Generate cited summary
            cited_summary = self._generate_cited_summary(text, rubric_scores)
            
            # Compute ROUGE score if ground truth available
            rouge_score = self._compute_rouge_score(cited_summary, ground_truth_bullets) if ground_truth_bullets else 0.0
            
            # Compute readability
            readability_score = self._compute_readability(text, sentences, word_count)
            
            # Extract key themes
            key_themes = self._extract_key_themes(text)
            
            # Generate warnings
            warnings = self._generate_warnings(text, word_count, rubric_scores)
            
            result = SoPAnalysisResult(
                rubric_scores=rubric_scores,
                overall_score=overall_score,
                cited_summary=cited_summary,
                rouge_score=rouge_score,
                word_count=word_count,
                readability_score=readability_score,
                key_themes=key_themes,
                warnings=warnings
            )
            
            logger.info(f"Analyzed SoP: {word_count} words, overall score: {overall_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"SoP analysis failed: {e}")
            # Return minimal result on error
            return SoPAnalysisResult(
                rubric_scores={dim: RubricScore(dim, 0.0, 0.0, [], f"Analysis error: {e}") for dim in self.rubric_dimensions},
                overall_score=0.0,
                cited_summary="Analysis failed",
                rouge_score=0.0,
                word_count=len(text.split()),
                readability_score=0.0,
                key_themes=[],
                warnings=[f"Analysis error: {str(e)}"]
            )
    
    def _score_dimension(self, dimension: str, text: str, sentences: List[str]) -> RubricScore:
        """Score a single rubric dimension"""
        text_lower = text.lower()
        
        if dimension == "research_interest":
            return self._score_research_interest(text_lower, sentences)
        elif dimension == "experience_relevance":
            return self._score_experience_relevance(text_lower, sentences)
        elif dimension == "writing_quality":
            return self._score_writing_quality(text_lower, sentences)
        elif dimension == "goal_clarity":
            return self._score_goal_clarity(text_lower, sentences)
        elif dimension == "fit_assessment":
            return self._score_fit_assessment(text_lower, sentences)
        else:
            return RubricScore(dimension, 0.0, 0.0, [], "Unknown dimension")
    
    def _score_research_interest(self, text_lower: str, sentences: List[str]) -> RubricScore:
        """Score research interest articulation"""
        score = 0.0
        evidence_spans = []
        reasoning_parts = []
        
        # Check for research area keywords
        areas_mentioned = 0
        for area, keywords in self.research_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    areas_mentioned += 1
                    score += 0.5
                    # Find evidence span
                    start = text_lower.find(keyword)
                    if start != -1:
                        evidence_spans.append((start, start + len(keyword), keyword))
                    break  # Only count each area once
        
        if areas_mentioned > 0:
            reasoning_parts.append(f"Mentions {areas_mentioned} research areas")
        
        # Check for specificity (technical terms, methodologies)
        specific_terms = ['algorithm', 'model', 'framework', 'methodology', 'approach', 'technique']
        specificity_count = sum(1 for term in specific_terms if term in text_lower)
        if specificity_count >= 2:
            score += 1.0
            reasoning_parts.append("Shows technical specificity")
        
        # Check for research context
        research_context = ['literature', 'state of the art', 'current research', 'recent work', 'publications']
        if any(context in text_lower for context in research_context):
            score += 0.5
            reasoning_parts.append("References research context")
        
        # Cap at 5.0
        score = min(5.0, score)
        
        # Assess confidence based on evidence
        confidence = min(0.9, 0.3 + (areas_mentioned * 0.2) + (specificity_count * 0.1))
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Limited research interest articulation"
        
        return RubricScore("research_interest", score, confidence, evidence_spans, reasoning)
    
    def _score_experience_relevance(self, text_lower: str, sentences: List[str]) -> RubricScore:
        """Score experience relevance and presentation"""
        score = 0.0
        evidence_spans = []
        reasoning_parts = []
        
        # Count experience indicators
        exp_count = 0
        for indicator in self.experience_indicators:
            if indicator in text_lower:
                exp_count += 1
                score += 0.3
                # Find evidence
                start = text_lower.find(indicator)
                if start != -1:
                    evidence_spans.append((start, start + len(indicator), indicator))
        
        if exp_count > 0:
            reasoning_parts.append(f"Describes {exp_count} types of experience")
        
        # Check for quantitative details
        numbers = re.findall(r'\b\d+\b', text_lower)
        if len(numbers) >= 2:
            score += 0.5
            reasoning_parts.append("Includes quantitative details")
        
        # Check for impact/results language
        impact_words = ['impact', 'result', 'achievement', 'improvement', 'success', 'contribution']
        impact_count = sum(1 for word in impact_words if word in text_lower)
        if impact_count >= 2:
            score += 1.0
            reasoning_parts.append("Describes impact and results")
        
        # Check for learning/growth language
        growth_words = ['learned', 'developed', 'gained', 'improved', 'enhanced', 'acquired']
        if any(word in text_lower for word in growth_words):
            score += 0.5
            reasoning_parts.append("Shows learning and growth")
        
        score = min(5.0, score)
        confidence = min(0.9, 0.2 + (exp_count * 0.1) + (impact_count * 0.1))
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Limited experience description"
        
        return RubricScore("experience_relevance", score, confidence, evidence_spans, reasoning)
    
    def _score_writing_quality(self, text_lower: str, sentences: List[str]) -> RubricScore:
        """Score writing quality and clarity"""
        score = 3.0  # Start with baseline
        evidence_spans = []
        reasoning_parts = []
        
        # Check for transition words (positive)
        transition_count = sum(1 for word in self.quality_positive if word in text_lower)
        if transition_count >= 2:
            score += 0.5
            reasoning_parts.append("Uses appropriate transitions")
        
        # Check for repetitive language (negative)
        negative_count = sum(1 for phrase in self.quality_negative if phrase in text_lower)
        if negative_count > 0:
            score -= 0.5
            reasoning_parts.append("Contains repetitive language")
        
        # Sentence variety
        if sentences:
            sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
            if sentence_lengths:
                length_variance = np.var(sentence_lengths)
                if length_variance > 25:  # Good variety
                    score += 0.5
                    reasoning_parts.append("Shows sentence variety")
                elif length_variance < 5:  # Too uniform
                    score -= 0.3
                    reasoning_parts.append("Limited sentence variety")
        
        # Check vocabulary sophistication
        sophisticated_words = ['furthermore', 'consequently', 'nevertheless', 'specifically', 'particularly']
        soph_count = sum(1 for word in sophisticated_words if word in text_lower)
        if soph_count >= 2:
            score += 0.5
            reasoning_parts.append("Uses sophisticated vocabulary")
        
        # Penalize excessive first-person usage
        first_person_matches = self.first_person_pattern.findall(text_lower)
        if len(first_person_matches) > len(text_lower.split()) * 0.15:  # > 15% of words
            score -= 0.5
            reasoning_parts.append("Overuses first-person pronouns")
        
        score = max(1.0, min(5.0, score))
        confidence = 0.7  # Medium confidence for style assessment
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Standard writing quality"
        
        return RubricScore("writing_quality", score, confidence, evidence_spans, reasoning)
    
    def _score_goal_clarity(self, text_lower: str, sentences: List[str]) -> RubricScore:
        """Score clarity of goals and objectives"""
        score = 0.0
        evidence_spans = []
        reasoning_parts = []
        
        # Check for goal-related keywords
        goal_count = sum(1 for indicator in self.goal_indicators if indicator in text_lower)
        if goal_count >= 2:
            score += 1.5
            reasoning_parts.append("Clearly states goals and objectives")
        elif goal_count >= 1:
            score += 1.0
            reasoning_parts.append("Mentions goals")
        
        # Check for future tense (planning)
        future_matches = self.future_pattern.findall(text_lower)
        if len(future_matches) >= 2:
            score += 1.0
            reasoning_parts.append("Shows future planning")
        
        # Check for specific career mentions
        career_terms = ['career', 'profession', 'industry', 'field', 'specialization']
        career_count = sum(1 for term in career_terms if term in text_lower)
        if career_count >= 1:
            score += 0.5
            reasoning_parts.append("Discusses career direction")
        
        # Check for timeline/steps
        if any(word in text_lower for word in ['first', 'then', 'next', 'eventually', 'ultimately']):
            score += 0.5
            reasoning_parts.append("Provides timeline or steps")
        
        # Check for specificity vs vagueness
        vague_terms = ['good', 'nice', 'great', 'awesome', 'amazing', 'perfect']
        vague_count = sum(1 for term in vague_terms if term in text_lower)
        if vague_count > 3:
            score -= 0.5
            reasoning_parts.append("Uses vague language")
        
        score = max(0.0, min(5.0, score))
        confidence = min(0.9, 0.3 + (goal_count * 0.2))
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Goals not clearly articulated"
        
        return RubricScore("goal_clarity", score, confidence, evidence_spans, reasoning)
    
    def _score_fit_assessment(self, text_lower: str, sentences: List[str]) -> RubricScore:
        """Score program/university fit assessment"""
        score = 0.0
        evidence_spans = []
        reasoning_parts = []
        
        # Check for fit-related keywords
        fit_count = sum(1 for indicator in self.fit_indicators if indicator in text_lower)
        if fit_count >= 3:
            score += 2.0
            reasoning_parts.append("Demonstrates research into program")
        elif fit_count >= 1:
            score += 1.0
            reasoning_parts.append("Shows awareness of program")
        
        # Check for specific faculty mentions
        if 'professor' in text_lower or 'dr.' in text_lower:
            score += 1.0
            reasoning_parts.append("Mentions specific faculty")
        
        # Check for specific program features
        program_features = ['curriculum', 'courses', 'research areas', 'facilities', 'resources']
        feature_count = sum(1 for feature in program_features if feature in text_lower)
        if feature_count >= 2:
            score += 1.0
            reasoning_parts.append("Discusses program features")
        
        # Check for alignment language
        align_words = ['align', 'match', 'fit', 'suit', 'complement', 'correspond']
        if any(word in text_lower for word in align_words):
            score += 0.5
            reasoning_parts.append("Articulates alignment")
        
        score = min(5.0, score)
        confidence = min(0.9, 0.2 + (fit_count * 0.15))
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Limited program fit discussion"
        
        return RubricScore("fit_assessment", score, confidence, evidence_spans, reasoning)
    
    def _generate_cited_summary(self, text: str, rubric_scores: Dict[str, RubricScore]) -> str:
        """Generate a cited summary based on rubric analysis"""
        sentences = self.sentence_pattern.split(text)
        summary_parts = []
        
        # Select key sentences based on rubric evidence
        key_sentences = set()
        
        for dimension, score in rubric_scores.items():
            if score.score >= 3.0 and score.evidence_spans:
                # Find sentences containing evidence
                for start, end, evidence_text in score.evidence_spans:
                    for i, sentence in enumerate(sentences):
                        if evidence_text.lower() in sentence.lower():
                            key_sentences.add(i)
                            break
        
        # If no good evidence, take first and last sentences
        if not key_sentences and sentences:
            key_sentences.add(0)
            if len(sentences) > 1:
                key_sentences.add(len(sentences) - 1)
        
        # Build summary from key sentences
        for i in sorted(key_sentences):
            if i < len(sentences) and sentences[i].strip():
                sentence = sentences[i].strip()
                if len(sentence) > 20:  # Skip very short fragments
                    summary_parts.append(sentence)
        
        # Limit to 3-5 sentences
        if len(summary_parts) > 5:
            summary_parts = summary_parts[:5]
        
        summary = " ".join(summary_parts)
        return summary if summary else "Unable to generate meaningful summary."
    
    def _compute_rouge_score(self, generated_summary: str, ground_truth_bullets: List[str]) -> float:
        """Compute ROUGE-L score against ground truth"""
        if not ground_truth_bullets:
            return 0.0
        
        # Simple ROUGE-L approximation using token overlap
        gen_tokens = set(generated_summary.lower().split())
        
        max_overlap = 0.0
        for bullet in ground_truth_bullets:
            ref_tokens = set(bullet.lower().split())
            if ref_tokens:
                overlap = len(gen_tokens & ref_tokens) / len(ref_tokens)
                max_overlap = max(max_overlap, overlap)
        
        return max_overlap
    
    def _compute_readability(self, text: str, sentences: List[str], word_count: int) -> float:
        """Compute readability score (simplified Flesch-Kincaid approximation)"""
        if not sentences or word_count == 0:
            return 0.0
        
        # Count syllables (approximation: vowel groups)
        syllable_pattern = re.compile(r'[aeiouy]+', re.IGNORECASE)
        syllable_count = len(syllable_pattern.findall(text))
        
        # Average sentence length
        sentence_count = len([s for s in sentences if s.strip()])
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Average syllables per word
        avg_syllables = syllable_count / word_count if word_count > 0 else 0
        
        # Simplified readability score (higher = more readable)
        readability = max(0, 10 - (avg_sentence_length * 0.1) - (avg_syllables * 2))
        return min(10.0, readability)
    
    def _extract_key_themes(self, text: str) -> List[str]:
        """Extract key themes from the text"""
        text_lower = text.lower()
        themes = []
        
        # Check each research area
        for area, keywords in self.research_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                themes.append(area.replace('_', ' ').title())
        
        # Add general themes
        if any(word in text_lower for word in ['research', 'study', 'investigate']):
            themes.append('Research Focus')
        
        if any(word in text_lower for word in ['career', 'professional', 'industry']):
            themes.append('Career Goals')
        
        if any(word in text_lower for word in ['learn', 'education', 'knowledge']):
            themes.append('Learning Objectives')
        
        return themes[:5]  # Limit to top 5 themes
    
    def _generate_warnings(self, text: str, word_count: int, rubric_scores: Dict[str, RubricScore]) -> List[str]:
        """Generate warnings about statement quality"""
        warnings = []
        
        # Word count warnings
        if word_count < 300:
            warnings.append("Statement is quite short (< 300 words)")
        elif word_count > 1000:
            warnings.append("Statement is quite long (> 1000 words)")
        
        # Low scores warnings
        low_scores = [dim for dim, score in rubric_scores.items() if score.score < 2.0]
        if low_scores:
            warnings.append(f"Low scores in: {', '.join(low_scores)}")
        
        # Generic content warning
        generic_phrases = ['since childhood', 'always been interested', 'dream come true', 'passion for']
        generic_count = sum(1 for phrase in generic_phrases if phrase in text.lower())
        if generic_count >= 2:
            warnings.append("Contains generic/clichéd language")
        
        # Missing elements
        if 'research' not in text.lower():
            warnings.append("Limited discussion of research interests")
        
        if not any(word in text.lower() for word in ['university', 'program', 'department']):
            warnings.append("Limited discussion of program fit")
        
        return warnings


def analyze_statement_of_purpose(tokens: List[Token], config: Dict[str, Any] = None, 
                                ground_truth_bullets: List[str] = None) -> SoPAnalysisResult:
    """Main entry point for statement of purpose analysis"""
    analyzer = SoPRubricAnalyzer(config)
    return analyzer.analyze_statement(tokens, ground_truth_bullets)


if __name__ == "__main__":
    # Test the SoP analyzer
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from ocr_backends import SimulatedOCRBackend
    
    # Generate test statement tokens
    backend = SimulatedOCRBackend()
    test_tokens = backend._generate_statement_tokens()
    
    # Analyze statement
    result = analyze_statement_of_purpose(test_tokens)
    
    print(f"Word count: {result.word_count}")
    print(f"Overall score: {result.overall_score:.2f}")
    print(f"Readability score: {result.readability_score:.2f}")
    print(f"Key themes: {result.key_themes}")
    print(f"Warnings: {result.warnings}")
    
    print("\nRubric Scores:")
    for dimension, score in result.rubric_scores.items():
        print(f"  {dimension}: {score.score:.2f} (confidence: {score.confidence:.2f})")
        print(f"    Reasoning: {score.reasoning}")
    
    print(f"\nCited Summary:")
    print(f"  {result.cited_summary}")
    
    # Test ROUGE computation
    gt_bullets = [
        "Strong research interest in machine learning",
        "Good programming experience with Python", 
        "Clear career goals in data science"
    ]
    
    analyzer = SoPRubricAnalyzer()
    rouge = analyzer._compute_rouge_score(result.cited_summary, gt_bullets)
    print(f"\nROUGE-L score: {rouge:.3f}")