"""
Service client for interfacing with the IDP processing pipeline
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add code directory to path
sys.path.append(str(Path(__file__).parent.parent / "code"))

try:
    from ocr_backends import extract_tokens
    from transcript_parser import parse_transcript  
    from decision_rules import load_decision_engine
    from resume_ner import extract_resume_entities
    from sop_rubric import analyze_statement_of_purpose
    from feature_fusion import fuse_application_features
except ImportError as e:
    logging.warning(f"Could not import processing modules: {e}")

logger = logging.getLogger(__name__)

class ServiceClient:
    """Client for processing documents through the IDP pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize decision engine
        try:
            self.decision_engine = load_decision_engine(config_dict=self.config)
        except Exception as e:
            logger.error(f"Failed to initialize decision engine: {e}")
            self.decision_engine = None
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "thresholds": {
                "gpa_threshold": 3.0,
                "min_credits": 90,
                "abstain_threshold": 0.7
            },
            "ocr": {
                "backend": "simulated",
                "fallback_backend": "simulated"
            }
        }
    
    def process_file(self, file_path: str, ocr_backend: str = "auto", 
                    program: str = "general") -> Dict[str, Any]:
        """Process a single file through the IDP pipeline"""
        
        start_time = datetime.now()
        
        try:
            # Generate application ID
            app_id = f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{Path(file_path).stem}"
            
            # Determine document type
            doc_type = self._determine_document_type(file_path)
            
            # Extract tokens using OCR
            tokens = extract_tokens(file_path, ocr_backend, self.config)
            
            # Process based on document type
            if doc_type == "transcript":
                result = self._process_transcript(tokens, app_id, program)
            elif doc_type == "resume":
                result = self._process_resume(tokens, app_id)
            elif doc_type == "statement":
                result = self._process_statement(tokens, app_id)
            else:
                result = self._process_generic(tokens, app_id)
            
            # Add processing metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result.update({
                "processing_info": {
                    "ocr_backend": ocr_backend,
                    "processing_time_seconds": processing_time,
                    "documents_processed": [{
                        "filename": Path(file_path).name,
                        "document_type": doc_type,
                        "pages": 1,  # Simplified
                        "ocr_confidence": 0.9  # Mock value
                    }]
                },
                "timestamp": datetime.now().isoformat()
            })
            
            # Save result
            self._save_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Processing failed for {file_path}: {e}")
            return {
                "application_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "decision": "ABSTAIN",
                "reason": f"Processing error: {str(e)}",
                "gpa": 0.0,
                "credits": 0.0,
                "confidence": 0.0,
                "warnings": [f"Processing error: {str(e)}"],
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_document_type(self, file_path: str) -> str:
        """Determine document type from filename"""
        filename = Path(file_path).name.lower()
        
        if any(keyword in filename for keyword in ["transcript", "grades", "academic"]):
            return "transcript"
        elif any(keyword in filename for keyword in ["resume", "cv", "vitae"]):
            return "resume"
        elif any(keyword in filename for keyword in ["statement", "sop", "purpose"]):
            return "statement"
        else:
            return "transcript"  # Default assumption
    
    def _process_transcript(self, tokens, app_id: str, program: str) -> Dict[str, Any]:
        """Process transcript document"""
        
        # Parse transcript
        parse_result = parse_transcript(tokens, self.config)
        
        # Make academic decision
        if self.decision_engine:
            decision_result = self.decision_engine.make_decision(
                gpa=parse_result.gpa,
                credits=parse_result.total_credits,
                program=program,
                parsing_confidence=parse_result.parsing_confidence
            )
            
            return {
                "application_id": app_id,
                "decision": decision_result.decision.value,
                "reason": decision_result.reason,
                "gpa": parse_result.gpa,
                "credits": parse_result.total_credits,
                "confidence": decision_result.confidence,
                "warnings": parse_result.warnings + decision_result.warnings,
                "evidence": {
                    "transcript_spans": parse_result.evidence_spans
                },
                "details": decision_result.details,
                "program": program
            }
        else:
            # Fallback without decision engine
            return {
                "application_id": app_id,
                "decision": "ABSTAIN",
                "reason": "Decision engine not available",
                "gpa": parse_result.gpa,
                "credits": parse_result.total_credits,
                "confidence": 0.0,
                "warnings": parse_result.warnings + ["Decision engine not available"],
                "evidence": {
                    "transcript_spans": parse_result.evidence_spans
                },
                "program": program
            }
    
    def _process_resume(self, tokens, app_id: str) -> Dict[str, Any]:
        """Process resume document"""
        
        try:
            entities, features = extract_resume_entities(tokens, self.config)
            
            return {
                "application_id": app_id,
                "decision": "REVIEW",  # Resumes typically require review
                "reason": "Resume processed - requires holistic evaluation",
                "gpa": 0.0,  # Not applicable
                "credits": 0.0,  # Not applicable
                "confidence": 0.6,
                "warnings": [],
                "evidence": {
                    "resume_entities": [entity.to_dict() for entity in entities]
                },
                "structured_features": features
            }
            
        except Exception as e:
            return {
                "application_id": app_id,
                "decision": "ABSTAIN",
                "reason": f"Resume processing error: {str(e)}",
                "gpa": 0.0,
                "credits": 0.0,
                "confidence": 0.0,
                "warnings": [f"Resume processing error: {str(e)}"]
            }
    
    def _process_statement(self, tokens, app_id: str) -> Dict[str, Any]:
        """Process statement of purpose"""
        
        try:
            analysis_result = analyze_statement_of_purpose(tokens, self.config)
            
            return {
                "application_id": app_id,
                "decision": "REVIEW",  # Statements typically require review
                "reason": "Statement processed - requires holistic evaluation",
                "gpa": 0.0,  # Not applicable
                "credits": 0.0,  # Not applicable  
                "confidence": 0.7,
                "warnings": analysis_result.warnings,
                "evidence": {
                    "sop_spans": [
                        {
                            "start": 0, "end": len(analysis_result.cited_summary),
                            "text": analysis_result.cited_summary,
                            "rubric_dimension": "overall",
                            "score": analysis_result.overall_score
                        }
                    ]
                },
                "sop_analysis": {
                    "overall_score": analysis_result.overall_score,
                    "word_count": analysis_result.word_count,
                    "readability_score": analysis_result.readability_score,
                    "key_themes": analysis_result.key_themes,
                    "rubric_scores": {
                        dim: {
                            "score": score.score,
                            "reasoning": score.reasoning,
                            "confidence": score.confidence
                        }
                        for dim, score in analysis_result.rubric_scores.items()
                    }
                }
            }
            
        except Exception as e:
            return {
                "application_id": app_id,
                "decision": "ABSTAIN",
                "reason": f"Statement processing error: {str(e)}",
                "gpa": 0.0,
                "credits": 0.0,
                "confidence": 0.0,
                "warnings": [f"Statement processing error: {str(e)}"]
            }
    
    def _process_generic(self, tokens, app_id: str) -> Dict[str, Any]:
        """Process generic document"""
        
        return {
            "application_id": app_id,
            "decision": "ABSTAIN",
            "reason": "Document type could not be determined",
            "gpa": 0.0,
            "credits": 0.0,
            "confidence": 0.0,
            "warnings": ["Unknown document type"],
            "evidence": {}
        }
    
    def _save_result(self, result: Dict[str, Any]):
        """Save processing result to JSON file"""
        
        output_dir = Path("processed")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"{result['application_id']}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
            logger.info(f"Saved result to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save result: {e}")
    
    def get_processed_applications(self) -> list:
        """Get list of processed applications"""
        
        processed_dir = Path("processed")
        applications = []
        
        if processed_dir.exists():
            for json_file in processed_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        app_data = json.load(f)
                        applications.append(app_data)
                except Exception as e:
                    logger.error(f"Error loading {json_file}: {e}")
        
        return applications
    
    def get_application_by_id(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get specific application by ID"""
        
        processed_dir = Path("processed")
        json_file = processed_dir / f"{app_id}.json"
        
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading application {app_id}: {e}")
        
        return None
    
    def delete_application(self, app_id: str) -> bool:
        """Delete an application"""
        
        processed_dir = Path("processed")
        json_file = processed_dir / f"{app_id}.json"
        
        if json_file.exists():
            try:
                json_file.unlink()
                logger.info(f"Deleted application {app_id}")
                return True
            except Exception as e:
                logger.error(f"Error deleting application {app_id}: {e}")
        
        return False


if __name__ == "__main__":
    # Test the service client
    logging.basicConfig(level=logging.INFO)
    
    client = ServiceClient()
    
    # Test with a mock file (would need actual file in real usage)
    print("ServiceClient initialized successfully")
    print(f"Config loaded: {client.config is not None}")
    print(f"Decision engine available: {client.decision_engine is not None}")