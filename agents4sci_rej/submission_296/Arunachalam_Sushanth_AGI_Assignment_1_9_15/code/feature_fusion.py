"""
Feature fusion module for combining multi-modal features from transcripts, resumes, and statements
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FeatureVector:
    """Combined feature representation"""
    academic_features: np.ndarray
    experience_features: np.ndarray
    narrative_features: np.ndarray
    combined_features: np.ndarray
    feature_names: List[str]
    
    @property
    def dimension(self) -> int:
        return len(self.combined_features)


class FeatureFusionEngine:
    """Combines features from multiple document types"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Feature weights
        weights = self.config.get("features", {})
        self.academic_weight = weights.get("academic_weight", 0.6)
        self.experience_weight = weights.get("experience_weight", 0.25)
        self.narrative_weight = weights.get("narrative_weight", 0.15)
        
        # Normalization parameters (learned from training data)
        self.normalization_params = {}
        
        logger.info(f"Initialized feature fusion with weights: academic={self.academic_weight}, "
                   f"experience={self.experience_weight}, narrative={self.narrative_weight}")
    
    def fuse_features(self, 
                     transcript_data: Dict[str, Any] = None,
                     resume_data: Dict[str, Any] = None,
                     sop_data: Dict[str, Any] = None) -> FeatureVector:
        """Combine features from all available sources"""
        
        # Extract individual feature vectors
        academic_features = self._extract_academic_features(transcript_data)
        experience_features = self._extract_experience_features(resume_data)
        narrative_features = self._extract_narrative_features(sop_data)
        
        # Normalize features
        academic_features = self._normalize_features(academic_features, "academic")
        experience_features = self._normalize_features(experience_features, "experience")
        narrative_features = self._normalize_features(narrative_features, "narrative")
        
        # Weighted combination
        combined_features = np.concatenate([
            academic_features * self.academic_weight,
            experience_features * self.experience_weight,
            narrative_features * self.narrative_weight
        ])
        
        # Generate feature names
        feature_names = (
            self._get_academic_feature_names() +
            self._get_experience_feature_names() + 
            self._get_narrative_feature_names()
        )
        
        return FeatureVector(
            academic_features=academic_features,
            experience_features=experience_features,
            narrative_features=narrative_features,
            combined_features=combined_features,
            feature_names=feature_names
        )
    
    def _extract_academic_features(self, transcript_data: Dict[str, Any]) -> np.ndarray:
        """Extract normalized academic features"""
        if not transcript_data:
            return np.zeros(5)  # Default academic feature dimension
        
        gpa = transcript_data.get("gpa", 0.0)
        total_credits = transcript_data.get("total_credits", 0.0)
        parsing_confidence = transcript_data.get("parsing_confidence", 0.0)
        
        # Additional derived features
        course_count = len(transcript_data.get("courses", []))
        avg_grade_points = gpa  # GPA is already average grade points
        
        features = np.array([
            gpa / 4.0,  # Normalize to [0, 1]
            min(1.0, total_credits / 180.0),  # Normalize typical max credits
            parsing_confidence,
            min(1.0, course_count / 40.0),  # Normalize typical max courses
            avg_grade_points / 4.0
        ])
        
        return features
    
    def _extract_experience_features(self, resume_data: Dict[str, Any]) -> np.ndarray:
        """Extract experience-related features"""
        if not resume_data:
            return np.zeros(4)  # Default experience feature dimension
        
        structured_features = resume_data.get("structured_features", {})
        
        skill_count = structured_features.get("skill_count", 0)
        experience_years = structured_features.get("experience_years", 0)
        org_count = structured_features.get("organization_count", 0)
        has_contact = 1.0 if structured_features.get("has_contact_info", False) else 0.0
        
        features = np.array([
            min(1.0, skill_count / 10.0),  # Normalize to reasonable max
            min(1.0, experience_years / 8.0),  # Normalize to 8 years max
            min(1.0, org_count / 5.0),  # Normalize to 5 orgs max
            has_contact
        ])
        
        return features
    
    def _extract_narrative_features(self, sop_data: Dict[str, Any]) -> np.ndarray:
        """Extract statement of purpose features"""
        if not sop_data:
            return np.zeros(6)  # Default narrative feature dimension
        
        overall_score = sop_data.get("overall_score", 0.0) / 5.0  # Normalize to [0, 1]
        word_count = min(1.0, sop_data.get("word_count", 0) / 800.0)  # Normalize to typical max
        readability = sop_data.get("readability_score", 0.0) / 10.0  # Normalize to [0, 1]
        theme_count = min(1.0, len(sop_data.get("key_themes", [])) / 5.0)  # Max 5 themes
        
        # Rubric dimension scores (normalized)
        rubric_scores = sop_data.get("rubric_scores", {})
        research_score = rubric_scores.get("research_interest", {}).get("score", 0.0) / 5.0
        writing_quality = rubric_scores.get("writing_quality", {}).get("score", 0.0) / 5.0
        
        features = np.array([
            overall_score,
            word_count,
            readability,
            theme_count,
            research_score,
            writing_quality
        ])
        
        return features
    
    def _normalize_features(self, features: np.ndarray, feature_type: str) -> np.ndarray:
        """Apply feature normalization"""
        # For now, features are already normalized to [0, 1] range
        # In a real implementation, this would use learned statistics
        
        # Ensure features are in [0, 1] range
        normalized = np.clip(features, 0.0, 1.0)
        
        # Apply z-score normalization if we have learned parameters
        if feature_type in self.normalization_params:
            params = self.normalization_params[feature_type]
            mean = params.get("mean", 0.0)
            std = params.get("std", 1.0)
            normalized = (normalized - mean) / (std + 1e-8)  # Avoid division by zero
        
        return normalized
    
    def _get_academic_feature_names(self) -> List[str]:
        """Get academic feature names"""
        return [
            "gpa_normalized",
            "credits_normalized", 
            "parsing_confidence",
            "course_count_normalized",
            "avg_grade_points_normalized"
        ]
    
    def _get_experience_feature_names(self) -> List[str]:
        """Get experience feature names"""
        return [
            "skill_count_normalized",
            "experience_years_normalized",
            "organization_count_normalized",
            "has_contact_info"
        ]
    
    def _get_narrative_feature_names(self) -> List[str]:
        """Get narrative feature names"""
        return [
            "sop_overall_score",
            "sop_word_count_normalized",
            "sop_readability",
            "sop_theme_count",
            "sop_research_score",
            "sop_writing_quality"
        ]
    
    def fit_normalization(self, training_features: List[FeatureVector]):
        """Learn normalization parameters from training data"""
        if not training_features:
            return
        
        # Collect all features by type
        academic_features = np.array([fv.academic_features for fv in training_features])
        experience_features = np.array([fv.experience_features for fv in training_features])
        narrative_features = np.array([fv.narrative_features for fv in training_features])
        
        # Compute statistics
        self.normalization_params = {
            "academic": {
                "mean": np.mean(academic_features, axis=0),
                "std": np.std(academic_features, axis=0)
            },
            "experience": {
                "mean": np.mean(experience_features, axis=0),
                "std": np.std(experience_features, axis=0)
            },
            "narrative": {
                "mean": np.mean(narrative_features, axis=0),
                "std": np.std(narrative_features, axis=0)
            }
        }
        
        logger.info("Fitted normalization parameters from training data")
    
    def compute_readiness_score(self, feature_vector: FeatureVector, 
                              model_weights: np.ndarray = None) -> Tuple[float, float]:
        """Compute readiness score from fused features"""
        if model_weights is None:
            # Simple linear combination as default
            academic_contribution = np.mean(feature_vector.academic_features) * 0.6
            experience_contribution = np.mean(feature_vector.experience_features) * 0.25  
            narrative_contribution = np.mean(feature_vector.narrative_features) * 0.15
            
            readiness_score = academic_contribution + experience_contribution + narrative_contribution
            confidence = 0.7  # Default confidence
        else:
            # Use trained model weights
            readiness_score = np.dot(feature_vector.combined_features, model_weights)
            readiness_score = 1.0 / (1.0 + np.exp(-readiness_score))  # Sigmoid
            confidence = 0.8  # Higher confidence with trained model
        
        return float(readiness_score), float(confidence)


def fuse_application_features(transcript_data: Dict[str, Any] = None,
                            resume_data: Dict[str, Any] = None,
                            sop_data: Dict[str, Any] = None,
                            config: Dict[str, Any] = None) -> FeatureVector:
    """Main entry point for feature fusion"""
    engine = FeatureFusionEngine(config)
    return engine.fuse_features(transcript_data, resume_data, sop_data)


if __name__ == "__main__":
    # Test feature fusion
    logging.basicConfig(level=logging.INFO)
    
    # Sample data
    transcript_data = {
        "gpa": 3.5,
        "total_credits": 120,
        "parsing_confidence": 0.9,
        "courses": [{"course_code": f"CS{100+i}", "credits": 3} for i in range(12)]
    }
    
    resume_data = {
        "structured_features": {
            "skill_count": 8,
            "experience_years": 2,
            "organization_count": 2,
            "has_contact_info": True
        }
    }
    
    sop_data = {
        "overall_score": 4.2,
        "word_count": 650,
        "readability_score": 7.5,
        "key_themes": ["Machine Learning", "Research Focus", "Career Goals"],
        "rubric_scores": {
            "research_interest": {"score": 4.5},
            "writing_quality": {"score": 4.0}
        }
    }
    
    # Fuse features
    features = fuse_application_features(transcript_data, resume_data, sop_data)
    
    print(f"Combined feature dimension: {features.dimension}")
    print(f"Academic features: {features.academic_features}")
    print(f"Experience features: {features.experience_features}")
    print(f"Narrative features: {features.narrative_features}")
    print(f"Feature names: {features.feature_names}")
    
    # Compute readiness score
    engine = FeatureFusionEngine()
    readiness, confidence = engine.compute_readiness_score(features)
    print(f"Readiness score: {readiness:.3f} (confidence: {confidence:.3f})")