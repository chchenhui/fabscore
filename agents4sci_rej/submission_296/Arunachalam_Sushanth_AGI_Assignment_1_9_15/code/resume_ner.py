"""
Resume Named Entity Recognition (NER) module
Lightweight NER implementation using regex patterns and keyword matching
"""

import re
import logging
from typing import List, Dict, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict

from ocr_backends import Token

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Represents a named entity with type and position"""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "type": self.entity_type,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }


class ResumeNERExtractor:
    """Named Entity Recognition for resumes using pattern matching"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._initialize_patterns()
        self._load_vocabularies()
        
    def _initialize_patterns(self):
        """Initialize regex patterns for entity recognition"""
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone pattern
        self.phone_pattern = re.compile(
            r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b'
        )
        
        # Date patterns
        self.date_patterns = [
            re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b', re.IGNORECASE),
            re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b'),
            re.compile(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b'),
            re.compile(r'\b(19|20)\d{2}\b')
        ]
        
        # Education patterns
        self.degree_pattern = re.compile(
            r'\b(PhD|Ph\.D\.|Doctor|Masters?|Master|M\.S\.|M\.A\.|M\.Eng\.|Bachelor|B\.S\.|B\.A\.|B\.Eng\.|Associates?|A\.A\.|A\.S\.)\b',
            re.IGNORECASE
        )
        
        # Experience patterns
        self.experience_pattern = re.compile(
            r'\b(\d+)(?:\+)?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|exp)\b',
            re.IGNORECASE
        )
        
        # GPA pattern
        self.gpa_pattern = re.compile(
            r'\b(?:GPA|Grade\s+Point\s+Average):\s*(\d\.\d{1,2})\b',
            re.IGNORECASE
        )
        
    def _load_vocabularies(self):
        """Load predefined vocabularies for entity recognition"""
        
        # Programming languages and technologies
        self.programming_skills = {
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust',
            'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css',
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask',
            'spring', 'rails', 'laravel', 'tensorflow', 'pytorch', 'keras',
            'scikit-learn', 'pandas', 'numpy', 'docker', 'kubernetes', 'aws',
            'azure', 'gcp', 'git', 'linux', 'mongodb', 'postgresql', 'mysql',
            'redis', 'elasticsearch', 'jenkins', 'terraform', 'ansible'
        }
        
        # Universities and institutions
        self.universities = {
            'mit', 'stanford', 'berkeley', 'caltech', 'harvard', 'yale', 'princeton',
            'columbia', 'upenn', 'cornell', 'duke', 'johns hopkins', 'northwestern',
            'university of michigan', 'ucla', 'usc', 'nyu', 'boston university',
            'georgia tech', 'carnegie mellon', 'university of texas', 'virginia tech'
        }
        
        # Companies
        self.companies = {
            'google', 'microsoft', 'amazon', 'apple', 'facebook', 'meta', 'netflix',
            'tesla', 'nvidia', 'intel', 'ibm', 'oracle', 'salesforce', 'uber',
            'airbnb', 'spotify', 'twitter', 'linkedin', 'adobe', 'vmware',
            'cisco', 'qualcomm', 'broadcom', 'servicenow', 'workday'
        }
        
        # Locations
        self.locations = {
            'new york', 'san francisco', 'los angeles', 'chicago', 'boston',
            'seattle', 'austin', 'denver', 'atlanta', 'miami', 'philadelphia',
            'phoenix', 'san diego', 'dallas', 'houston', 'california', 'texas',
            'florida', 'new york', 'washington', 'oregon', 'massachusetts'
        }
        
        # Job titles
        self.job_titles = {
            'software engineer', 'software developer', 'data scientist', 'data analyst',
            'machine learning engineer', 'ai engineer', 'devops engineer', 'sre',
            'product manager', 'project manager', 'technical lead', 'architect',
            'senior engineer', 'principal engineer', 'staff engineer', 'director',
            'vp engineering', 'cto', 'researcher', 'scientist', 'consultant'
        }
    
    def extract_entities(self, tokens: List[Token]) -> List[Entity]:
        """Extract named entities from resume tokens"""
        # Convert tokens to text
        text = " ".join(token.text for token in tokens)
        entities = []
        
        # Extract different types of entities
        entities.extend(self._extract_contact_info(text))
        entities.extend(self._extract_education(text))
        entities.extend(self._extract_experience(text))
        entities.extend(self._extract_skills(text))
        entities.extend(self._extract_organizations(text))
        entities.extend(self._extract_locations(text))
        entities.extend(self._extract_dates(text))
        
        # Remove overlapping entities
        entities = self._remove_overlaps(entities)
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        logger.info(f"Extracted {len(entities)} entities from resume")
        return entities
    
    def _extract_contact_info(self, text: str) -> List[Entity]:
        """Extract email addresses and phone numbers"""
        entities = []
        
        # Email addresses
        for match in self.email_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(0),
                entity_type="EMAIL",
                start=match.start(),
                end=match.end(),
                confidence=0.95
            ))
        
        # Phone numbers
        for match in self.phone_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(0),
                entity_type="PHONE",
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
            
        return entities
    
    def _extract_education(self, text: str) -> List[Entity]:
        """Extract educational information"""
        entities = []
        
        # Degrees
        for match in self.degree_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(0),
                entity_type="EDUCATION",
                start=match.start(),
                end=match.end(),
                confidence=0.85
            ))
        
        # Universities (case insensitive matching)
        text_lower = text.lower()
        for university in self.universities:
            pattern = re.compile(re.escape(university), re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="ORG",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
        
        # GPA
        for match in self.gpa_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(0),
                entity_type="EDUCATION",
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
            
        return entities
    
    def _extract_experience(self, text: str) -> List[Entity]:
        """Extract experience-related information"""
        entities = []
        
        # Years of experience
        for match in self.experience_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(0),
                entity_type="EXPERIENCE",
                start=match.start(),
                end=match.end(),
                confidence=0.9
            ))
        
        # Job titles
        text_lower = text.lower()
        for title in self.job_titles:
            pattern = re.compile(re.escape(title), re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="EXPERIENCE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.75
                ))
                
        return entities
    
    def _extract_skills(self, text: str) -> List[Entity]:
        """Extract technical skills"""
        entities = []
        
        # Programming languages and technologies
        text_lower = text.lower()
        for skill in self.programming_skills:
            # Use word boundaries to avoid false positives
            pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="SKILL",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85
                ))
                
        return entities
    
    def _extract_organizations(self, text: str) -> List[Entity]:
        """Extract company and organization names"""
        entities = []
        
        # Known companies
        text_lower = text.lower()
        for company in self.companies:
            pattern = re.compile(re.escape(company), re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="ORG",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
                
        return entities
    
    def _extract_locations(self, text: str) -> List[Entity]:
        """Extract location information"""
        entities = []
        
        # Known locations
        text_lower = text.lower()
        for location in self.locations:
            pattern = re.compile(re.escape(location), re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="LOCATION",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7
                ))
                
        return entities
    
    def _extract_dates(self, text: str) -> List[Entity]:
        """Extract dates"""
        entities = []
        
        for pattern in self.date_patterns:
            for match in pattern.finditer(text):
                entities.append(Entity(
                    text=match.group(0),
                    entity_type="DATE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                ))
                
        return entities
    
    def _remove_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping highest confidence"""
        if not entities:
            return entities
            
        # Sort by start position, then by confidence (descending)
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        non_overlapping = []
        for entity in entities:
            # Check if this entity overlaps with any already selected
            overlaps = False
            for selected in non_overlapping:
                if (entity.start < selected.end and entity.end > selected.start):
                    overlaps = True
                    break
                    
            if not overlaps:
                non_overlapping.append(entity)
                
        return non_overlapping
    
    def compute_ner_metrics(self, predicted_entities: List[Entity], 
                          ground_truth_entities: List[Entity]) -> Dict[str, float]:
        """Compute NER evaluation metrics (precision, recall, F1)"""
        if not ground_truth_entities and not predicted_entities:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        
        if not ground_truth_entities:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        if not predicted_entities:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Convert to sets of (start, end, type) tuples for exact matching
        pred_set = {(e.start, e.end, e.entity_type) for e in predicted_entities}
        gold_set = {(e.start, e.end, e.entity_type) for e in ground_truth_entities}
        
        # Compute metrics
        true_positives = len(pred_set & gold_set)
        false_positives = len(pred_set - gold_set)
        false_negatives = len(gold_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def extract_structured_features(self, entities: List[Entity]) -> Dict[str, Any]:
        """Extract structured features for downstream processing"""
        features = {
            "skill_count": 0,
            "experience_years": 0,
            "education_level": "unknown",
            "has_contact_info": False,
            "organization_count": 0,
            "skills": [],
            "organizations": [],
            "locations": []
        }
        
        # Count entities by type
        entity_counts = defaultdict(int)
        for entity in entities:
            entity_counts[entity.entity_type] += 1
            
        features["skill_count"] = entity_counts.get("SKILL", 0)
        features["organization_count"] = entity_counts.get("ORG", 0)
        features["has_contact_info"] = entity_counts.get("EMAIL", 0) > 0 or entity_counts.get("PHONE", 0) > 0
        
        # Extract specific information
        for entity in entities:
            if entity.entity_type == "SKILL":
                features["skills"].append(entity.text)
            elif entity.entity_type == "ORG":
                features["organizations"].append(entity.text)
            elif entity.entity_type == "LOCATION":
                features["locations"].append(entity.text)
            elif entity.entity_type == "EXPERIENCE":
                # Extract years of experience
                exp_match = self.experience_pattern.search(entity.text)
                if exp_match:
                    years = int(exp_match.group(1))
                    features["experience_years"] = max(features["experience_years"], years)
            elif entity.entity_type == "EDUCATION":
                # Determine education level
                if any(degree in entity.text.lower() for degree in ["phd", "ph.d", "doctor"]):
                    features["education_level"] = "PhD"
                elif any(degree in entity.text.lower() for degree in ["master", "m.s", "m.a", "m.eng"]):
                    if features["education_level"] not in ["PhD"]:
                        features["education_level"] = "Masters"
                elif any(degree in entity.text.lower() for degree in ["bachelor", "b.s", "b.a", "b.eng"]):
                    if features["education_level"] not in ["PhD", "Masters"]:
                        features["education_level"] = "Bachelors"
                        
        return features


def extract_resume_entities(tokens: List[Token], config: Dict[str, Any] = None) -> Tuple[List[Entity], Dict[str, Any]]:
    """Main entry point for resume entity extraction"""
    extractor = ResumeNERExtractor(config)
    entities = extractor.extract_entities(tokens)
    features = extractor.extract_structured_features(entities)
    return entities, features


if __name__ == "__main__":
    # Test the NER extractor
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from ocr_backends import SimulatedOCRBackend
    
    # Generate test resume tokens
    backend = SimulatedOCRBackend()
    test_tokens = backend._generate_resume_tokens()
    
    # Extract entities
    entities, features = extract_resume_entities(test_tokens)
    
    print(f"Extracted {len(entities)} entities:")
    for entity in entities:
        print(f"  {entity.entity_type}: '{entity.text}' (confidence: {entity.confidence:.2f})")
        
    print(f"\nStructured features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
        
    # Test metrics computation
    extractor = ResumeNERExtractor()
    gt_entities = entities[:3]  # Use first 3 as ground truth
    pred_entities = entities[1:4]  # Use overlapping set as predictions
    
    metrics = extractor.compute_ner_metrics(pred_entities, gt_entities)
    print(f"\nNER Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")