"""
Decision rules engine for academic pre-screening
Implements configurable threshold-based decisions with confidence estimation
"""

import logging
from typing import Dict, Tuple, Any, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class AcademicDecision(Enum):
    """Possible academic decisions"""
    ACCEPT_ACADEMIC = "ACCEPT_ACADEMIC"
    REVIEW = "REVIEW" 
    REJECT_ACADEMIC = "REJECT_ACADEMIC"
    ABSTAIN = "ABSTAIN"


@dataclass
class DecisionResult:
    """Result of academic decision making"""
    decision: AcademicDecision
    reason: str
    confidence: float
    details: Dict[str, Any]
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class DecisionEngine:
    """Engine for making academic pre-screening decisions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.thresholds = config.get("thresholds", {})
        self.program_rules = self.thresholds.get("program_rules", {})
        
        # Default thresholds
        self.default_gpa_threshold = self.thresholds.get("gpa_threshold", 3.0)
        self.default_min_credits = self.thresholds.get("min_credits", 90)
        self.abstain_threshold = self.thresholds.get("abstain_threshold", 0.7)
        self.completeness_threshold = self.thresholds.get("completeness_threshold", 0.8)
        
        logger.info(f"Initialized decision engine with GPA threshold: {self.default_gpa_threshold}, "
                   f"min credits: {self.default_min_credits}, abstain threshold: {self.abstain_threshold}")
    
    def make_decision(self, 
                     gpa: float, 
                     credits: float, 
                     program: str = "general", 
                     parsing_confidence: float = 1.0,
                     additional_features: Dict[str, Any] = None) -> DecisionResult:
        """
        Make academic decision based on GPA, credits, and program requirements
        
        Args:
            gpa: Computed cumulative GPA
            credits: Total credit hours
            program: Target program (used for program-specific thresholds)
            parsing_confidence: Confidence in transcript parsing
            additional_features: Additional features (resume, SoP scores, etc.)
            
        Returns:
            DecisionResult with decision, reason, and confidence
        """
        if additional_features is None:
            additional_features = {}
            
        try:
            # Get program-specific thresholds
            program_config = self.program_rules.get(program, {})
            gpa_threshold = program_config.get("gpa_threshold", self.default_gpa_threshold)
            min_credits = program_config.get("min_credits", self.default_min_credits)
            min_math_credits = program_config.get("min_math_credits", 0)
            
            # Validate inputs
            validation_warnings = self._validate_inputs(gpa, credits, parsing_confidence)
            
            # Compute base confidence
            base_confidence = self._compute_base_confidence(gpa, credits, gpa_threshold, min_credits, parsing_confidence)
            
            # Apply abstention rule first
            if base_confidence < self.abstain_threshold:
                return DecisionResult(
                    decision=AcademicDecision.ABSTAIN,
                    reason=f"Low confidence ({base_confidence:.2f} < {self.abstain_threshold}) - human review required",
                    confidence=base_confidence,
                    details={
                        "gpa": gpa,
                        "credits": credits,
                        "gpa_threshold": gpa_threshold,
                        "min_credits": min_credits,
                        "program": program,
                        "parsing_confidence": parsing_confidence
                    },
                    warnings=validation_warnings
                )
            
            # Apply primary decision logic
            primary_decision = self._apply_primary_rules(gpa, credits, gpa_threshold, min_credits)
            
            # Apply secondary considerations
            adjusted_decision, adjusted_confidence, reason = self._apply_secondary_rules(
                primary_decision, base_confidence, gpa, credits, additional_features, program_config
            )
            
            return DecisionResult(
                decision=adjusted_decision,
                reason=reason,
                confidence=adjusted_confidence,
                details={
                    "gpa": gpa,
                    "credits": credits,
                    "gpa_threshold": gpa_threshold,
                    "min_credits": min_credits,
                    "program": program,
                    "primary_decision": primary_decision.value,
                    "parsing_confidence": parsing_confidence,
                    "additional_features": additional_features
                },
                warnings=validation_warnings
            )
            
        except Exception as e:
            logger.error(f"Decision making failed: {e}")
            return DecisionResult(
                decision=AcademicDecision.ABSTAIN,
                reason=f"Decision error: {str(e)}",
                confidence=0.0,
                details={"error": str(e)},
                warnings=[f"Decision processing error: {str(e)}"]
            )
    
    def _validate_inputs(self, gpa: float, credits: float, parsing_confidence: float) -> list:
        """Validate input parameters and return warnings"""
        warnings = []
        
        if not (0 <= gpa <= 4.5):  # Allow some tolerance above 4.0
            warnings.append(f"GPA {gpa:.2f} is outside normal range [0, 4.0]")
            
        if credits < 0 or credits > 300:
            warnings.append(f"Credits {credits} is outside reasonable range [0, 300]")
            
        if not (0 <= parsing_confidence <= 1.0):
            warnings.append(f"Parsing confidence {parsing_confidence:.2f} is outside valid range [0, 1.0]")
            
        if credits > 0 and credits < 30:
            warnings.append("Low credit count - transcript may be incomplete")
            
        return warnings
    
    def _compute_base_confidence(self, gpa: float, credits: float, gpa_threshold: float, 
                                min_credits: float, parsing_confidence: float) -> float:
        """Compute base confidence score"""
        confidence_factors = []
        
        # Parsing confidence factor (most important)
        confidence_factors.append(parsing_confidence * 0.4)
        
        # GPA clarity factor - higher confidence when GPA is clearly above/below threshold
        gpa_distance = abs(gpa - gpa_threshold)
        gpa_clarity = min(1.0, gpa_distance / 0.5)  # Full confidence if 0.5+ GPA difference
        confidence_factors.append(gpa_clarity * 0.3)
        
        # Credit clarity factor
        credit_distance = abs(credits - min_credits)
        credit_clarity = min(1.0, credit_distance / 30)  # Full confidence if 30+ credit difference
        confidence_factors.append(credit_clarity * 0.2)
        
        # Reasonableness factor - penalize extreme values
        reasonableness = 1.0
        if gpa < 1.0 or gpa > 4.0:
            reasonableness *= 0.8
        if credits < 60 or credits > 200:
            reasonableness *= 0.9
        confidence_factors.append(reasonableness * 0.1)
        
        base_confidence = sum(confidence_factors)
        return max(0.1, min(0.95, base_confidence))  # Clamp to reasonable range
    
    def _apply_primary_rules(self, gpa: float, credits: float, gpa_threshold: float, 
                           min_credits: float) -> AcademicDecision:
        """Apply primary GPA and credit-based decision rules"""
        
        # Strong accept conditions
        if gpa >= gpa_threshold + 0.3 and credits >= min_credits + 20:
            return AcademicDecision.ACCEPT_ACADEMIC
            
        # Strong reject conditions  
        if gpa < gpa_threshold - 0.5 or credits < min_credits - 30:
            return AcademicDecision.REJECT_ACADEMIC
            
        # Clear accept conditions
        if gpa >= gpa_threshold and credits >= min_credits:
            return AcademicDecision.ACCEPT_ACADEMIC
            
        # Clear reject conditions
        if gpa < gpa_threshold - 0.2 or credits < min_credits - 15:
            return AcademicDecision.REJECT_ACADEMIC
            
        # Borderline cases require human review
        return AcademicDecision.REVIEW
    
    def _apply_secondary_rules(self, primary_decision: AcademicDecision, base_confidence: float,
                             gpa: float, credits: float, additional_features: Dict[str, Any],
                             program_config: Dict[str, Any]) -> Tuple[AcademicDecision, float, str]:
        """Apply secondary rules and generate reasoning"""
        
        decision = primary_decision
        confidence = base_confidence
        reasons = []
        
        # Generate primary reasoning
        if decision == AcademicDecision.ACCEPT_ACADEMIC:
            reasons.append(f"GPA {gpa:.2f} and {credits:.0f} credits meet academic standards")
        elif decision == AcademicDecision.REJECT_ACADEMIC:
            reasons.append(f"GPA {gpa:.2f} or credits {credits:.0f} below minimum requirements")
        else:
            reasons.append(f"GPA {gpa:.2f} and {credits:.0f} credits require human evaluation")
            
        # Check additional features if available
        if additional_features:
            decision, confidence, additional_reasons = self._evaluate_additional_features(
                decision, confidence, additional_features, program_config
            )
            reasons.extend(additional_reasons)
            
        # Check for program-specific requirements
        if program_config:
            program_reasons = self._check_program_requirements(gpa, credits, additional_features, program_config)
            if program_reasons:
                reasons.extend(program_reasons)
                # Slight confidence boost for meeting program requirements
                confidence = min(0.95, confidence + 0.05)
                
        reason = "; ".join(reasons)
        return decision, confidence, reason
    
    def _evaluate_additional_features(self, decision: AcademicDecision, confidence: float,
                                    additional_features: Dict[str, Any], 
                                    program_config: Dict[str, Any]) -> Tuple[AcademicDecision, float, list]:
        """Evaluate additional features like experience, skills, statement quality"""
        reasons = []
        
        # Experience evaluation
        years_exp = additional_features.get("experience_years", 0)
        if years_exp >= 2:
            reasons.append(f"{years_exp} years relevant experience")
            # Experience can push borderline REVIEW cases toward ACCEPT
            if decision == AcademicDecision.REVIEW and years_exp >= 3:
                decision = AcademicDecision.ACCEPT_ACADEMIC
                confidence = min(0.9, confidence + 0.1)
                
        # Skills assessment
        skill_count = additional_features.get("skill_count", 0)
        if skill_count >= 5:
            reasons.append(f"{skill_count} relevant technical skills")
            confidence = min(0.95, confidence + 0.05)
            
        # Statement of purpose quality
        sop_scores = additional_features.get("sop_rubric_scores", {})
        if sop_scores:
            avg_score = sum(sop_scores.values()) / len(sop_scores)
            if avg_score >= 4.0:
                reasons.append(f"Strong statement of purpose (avg score: {avg_score:.1f})")
                confidence = min(0.95, confidence + 0.08)
            elif avg_score <= 2.0:
                reasons.append(f"Weak statement of purpose (avg score: {avg_score:.1f})")
                confidence = max(0.1, confidence - 0.1)
                
        return decision, confidence, reasons
    
    def _check_program_requirements(self, gpa: float, credits: float, 
                                  additional_features: Dict[str, Any],
                                  program_config: Dict[str, Any]) -> list:
        """Check program-specific requirements"""
        reasons = []
        
        # Math credit requirements
        min_math_credits = program_config.get("min_math_credits", 0)
        if min_math_credits > 0:
            math_credits = additional_features.get("math_credits", 0)
            if math_credits >= min_math_credits:
                reasons.append(f"Meets math requirement ({math_credits} >= {min_math_credits} credits)")
            else:
                reasons.append(f"Below math requirement ({math_credits} < {min_math_credits} credits)")
                
        # Prerequisites check
        required_courses = program_config.get("required_courses", [])
        if required_courses:
            completed_courses = set(additional_features.get("completed_courses", []))
            missing = set(required_courses) - completed_courses
            if not missing:
                reasons.append("All prerequisite courses completed")
            else:
                reasons.append(f"Missing prerequisites: {', '.join(missing)}")
                
        return reasons
    
    def academically_good(self, gpa: float, credits: float, program: str = "general") -> Tuple[AcademicDecision, str, float]:
        """
        Simplified interface for academic readiness assessment
        Returns: (decision, reason, confidence)
        """
        result = self.make_decision(gpa, credits, program)
        return result.decision, result.reason, result.confidence
    
    def should_abstain(self, gpa: float, credits: float, parsing_confidence: float, 
                      program: str = "general") -> bool:
        """Check if system should abstain from making a decision"""
        result = self.make_decision(gpa, credits, program, parsing_confidence)
        return result.decision == AcademicDecision.ABSTAIN or result.confidence < self.abstain_threshold


def load_decision_engine(config_path: str = None, config_dict: Dict[str, Any] = None) -> DecisionEngine:
    """Factory function to load decision engine from config"""
    if config_dict:
        config = config_dict
    elif config_path:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "thresholds": {
                "gpa_threshold": 3.0,
                "min_credits": 90,
                "abstain_threshold": 0.7
            }
        }
    
    return DecisionEngine(config)


if __name__ == "__main__":
    # Test the decision engine
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with default config
    engine = load_decision_engine()
    
    test_cases = [
        (3.5, 120, "computer_science", 0.9),  # Should accept
        (2.8, 85, "general", 0.8),           # Should review
        (2.2, 60, "engineering", 0.9),       # Should reject
        (3.1, 95, "business", 0.4),          # Should abstain (low confidence)
    ]
    
    for gpa, credits, program, parsing_conf in test_cases:
        result = engine.make_decision(gpa, credits, program, parsing_conf)
        print(f"GPA {gpa}, Credits {credits}, Program {program}:")
        print(f"  Decision: {result.decision.value}")
        print(f"  Reason: {result.reason}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Warnings: {result.warnings}")
        print()