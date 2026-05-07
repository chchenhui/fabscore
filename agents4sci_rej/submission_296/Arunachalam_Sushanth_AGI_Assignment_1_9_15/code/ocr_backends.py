"""
OCR Backend implementations for document text extraction
Supports multiple backends: pdfminer, simulated OCR, and optional pytesseract
"""

import os
import logging
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import random

# Import available backends
try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextBox, LTTextLine, LTChar
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False
    logging.warning("pdfminer.six not available, falling back to simulated OCR")

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class Token:
    """Represents a text token with bounding box coordinates"""
    text: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float = 1.0
    page: int = 0
    
    def __post_init__(self):
        """Validate token data"""
        if not isinstance(self.text, str):
            raise ValueError("Token text must be string")
        if len(self.bbox) != 4:
            raise ValueError("Bounding box must have 4 coordinates")


class OCRBackend(ABC):
    """Abstract base class for OCR backends"""
    
    @abstractmethod
    def extract_tokens(self, file_path: str, config: Dict[str, Any]) -> List[Token]:
        """Extract tokens from document with spatial information"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available for use"""
        pass


class PDFMinerBackend(OCRBackend):
    """PDF text extraction using pdfminer.six with layout analysis"""
    
    def is_available(self) -> bool:
        return PDFMINER_AVAILABLE
    
    def extract_tokens(self, file_path: str, config: Dict[str, Any]) -> List[Token]:
        """Extract text tokens with bounding boxes using pdfminer"""
        if not self.is_available():
            raise RuntimeError("pdfminer.six not available")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        tokens = []
        
        try:
            # Configure layout analysis parameters
            laparams = LAParams(
                boxes_flow=0.5,
                word_margin=0.1,
                char_margin=2.0,
                line_margin=0.5,
                all_texts=False
            )
            
            # Extract pages with layout information
            for page_num, page in enumerate(extract_pages(file_path, laparams=laparams)):
                page_tokens = self._extract_page_tokens(page, page_num)
                tokens.extend(page_tokens)
                
        except Exception as e:
            logger.error(f"PDFMiner extraction failed for {file_path}: {e}")
            raise RuntimeError(f"PDFMiner extraction error: {e}")
            
        logger.info(f"Extracted {len(tokens)} tokens from {file_path}")
        return tokens
    
    def _extract_page_tokens(self, page, page_num: int) -> List[Token]:
        """Extract tokens from a single page"""
        tokens = []
        
        for element in page:
            if isinstance(element, LTTextBox):
                # Process text boxes
                for line in element:
                    if isinstance(line, LTTextLine):
                        # Split line into words
                        words = self._split_line_to_words(line)
                        tokens.extend(words)
                        
        return tokens
    
    def _split_line_to_words(self, line) -> List[Token]:
        """Split text line into word tokens with bounding boxes"""
        words = []
        text = line.get_text().strip()
        
        if not text:
            return words
            
        # Simple word splitting
        word_texts = text.split()
        if not word_texts:
            return words
            
        # Estimate word bounding boxes (simplified)
        bbox = line.bbox
        word_width = (bbox[2] - bbox[0]) / len(word_texts)
        
        for i, word_text in enumerate(word_texts):
            word_bbox = (
                bbox[0] + i * word_width,
                bbox[1],
                bbox[0] + (i + 1) * word_width,
                bbox[3]
            )
            
            words.append(Token(
                text=word_text,
                bbox=word_bbox,
                confidence=0.95,  # High confidence for PDF text
                page=0  # pdfminer page numbers start at 0
            ))
            
        return words


class SimulatedOCRBackend(OCRBackend):
    """Simulated OCR for testing and synthetic data"""
    
    def __init__(self, noise_level: float = 0.05):
        """Initialize with configurable noise level"""
        self.noise_level = noise_level
        
    def is_available(self) -> bool:
        return True
    
    def extract_tokens(self, file_path: str, config: Dict[str, Any]) -> List[Token]:
        """Generate simulated OCR tokens based on document type"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Determine document type from filename/extension
        doc_type = self._determine_document_type(file_path)
        
        if doc_type == "transcript":
            return self._generate_transcript_tokens()
        elif doc_type == "resume":
            return self._generate_resume_tokens()
        elif doc_type == "statement":
            return self._generate_statement_tokens()
        else:
            return self._generate_generic_tokens()
    
    def _determine_document_type(self, file_path: str) -> str:
        """Determine document type from filename patterns"""
        filename = os.path.basename(file_path).lower()
        
        if any(keyword in filename for keyword in ["transcript", "grades", "academic"]):
            return "transcript"
        elif any(keyword in filename for keyword in ["resume", "cv", "vitae"]):
            return "resume"
        elif any(keyword in filename for keyword in ["statement", "sop", "purpose"]):
            return "statement"
        else:
            return "generic"
    
    def _generate_transcript_tokens(self) -> List[Token]:
        """Generate simulated transcript tokens"""
        tokens = []
        y_pos = 800  # Start from top of page
        
        # Header
        header_tokens = [
            Token("UNIVERSITY", (100, y_pos, 200, y_pos+20), 0.98, 0),
            Token("TRANSCRIPT", (220, y_pos, 320, y_pos+20), 0.98, 0)
        ]
        tokens.extend(header_tokens)
        y_pos -= 40
        
        # Student info
        info_tokens = [
            Token("Student", (100, y_pos, 160, y_pos+15), 0.95, 0),
            Token("ID:", (170, y_pos, 190, y_pos+15), 0.95, 0),
            Token("123456789", (200, y_pos, 280, y_pos+15), 0.95, 0)
        ]
        tokens.extend(info_tokens)
        y_pos -= 60
        
        # Course data
        courses = [
            ("CS101", "3", "A"),
            ("CS102", "4", "A-"),
            ("MATH201", "4", "B+"),
            ("PHYS101", "3", "B"),
            ("ENG101", "3", "A"),
            ("CS201", "3", "B+"),
            ("CS202", "4", "A-"),
            ("MATH202", "4", "B")
        ]
        
        for course, credits, grade in courses:
            course_tokens = [
                Token(course, (100, y_pos, 160, y_pos+12), 0.92, 0),
                Token(credits, (300, y_pos, 320, y_pos+12), 0.94, 0),
                Token(grade, (400, y_pos, 430, y_pos+12), 0.93, 0)
            ]
            tokens.extend(course_tokens)
            y_pos -= 20
            
        # Add some OCR noise
        if self.noise_level > 0:
            tokens = self._add_ocr_noise(tokens)
            
        return tokens
    
    def _generate_resume_tokens(self) -> List[Token]:
        """Generate simulated resume tokens"""
        tokens = []
        y_pos = 800
        
        # Personal section
        personal_data = [
            "John", "Doe", "Software", "Engineer",
            "johndoe@email.com", "(555)", "123-4567"
        ]
        
        x_pos = 100
        for text in personal_data:
            tokens.append(Token(text, (x_pos, y_pos, x_pos+len(text)*8, y_pos+15), 0.96, 0))
            x_pos += len(text) * 10 + 20
            
        y_pos -= 40
        
        # Experience section
        exp_data = [
            "EXPERIENCE", "Software", "Developer", "TechCorp",
            "2020-2023", "Python", "JavaScript", "React",
            "Machine", "Learning", "Data", "Analysis"
        ]
        
        x_pos = 100
        for i, text in enumerate(exp_data):
            if i % 6 == 0:  # New line every 6 words
                y_pos -= 20
                x_pos = 100
                
            tokens.append(Token(text, (x_pos, y_pos, x_pos+len(text)*8, y_pos+12), 0.94, 0))
            x_pos += len(text) * 10 + 15
            
        return tokens
    
    def _generate_statement_tokens(self) -> List[Token]:
        """Generate simulated statement of purpose tokens"""
        tokens = []
        y_pos = 800
        
        statement_text = """
        My passion for computer science began during undergraduate studies at State University.
        Through coursework in algorithms and data structures, I developed strong analytical skills.
        My research experience in machine learning has prepared me for graduate study.
        I am particularly interested in artificial intelligence and its applications to healthcare.
        The PhD program at your university aligns perfectly with my career goals.
        """.strip().split()
        
        x_pos = 100
        line_width = 500
        
        for word in statement_text:
            word_width = len(word) * 8 + 10
            
            if x_pos + word_width > line_width:
                y_pos -= 20
                x_pos = 100
                
            tokens.append(Token(word, (x_pos, y_pos, x_pos+word_width-10, y_pos+12), 0.93, 0))
            x_pos += word_width
            
        return tokens
    
    def _generate_generic_tokens(self) -> List[Token]:
        """Generate generic document tokens"""
        tokens = []
        generic_words = ["Document", "Content", "Text", "Information", "Data"]
        
        for i, word in enumerate(generic_words):
            tokens.append(Token(
                word, 
                (100 + i*100, 400, 180 + i*100, 415), 
                0.90, 
                0
            ))
            
        return tokens
    
    def _add_ocr_noise(self, tokens: List[Token]) -> List[Token]:
        """Add realistic OCR noise to tokens"""
        noisy_tokens = []
        
        for token in tokens:
            if random.random() < self.noise_level:
                # Add common OCR errors
                noisy_text = self._apply_ocr_errors(token.text)
                confidence = max(0.3, token.confidence - random.uniform(0.1, 0.3))
                
                noisy_token = Token(
                    text=noisy_text,
                    bbox=token.bbox,
                    confidence=confidence,
                    page=token.page
                )
                noisy_tokens.append(noisy_token)
            else:
                noisy_tokens.append(token)
                
        return noisy_tokens
    
    def _apply_ocr_errors(self, text: str) -> str:
        """Apply common OCR character substitution errors"""
        error_map = {
            'o': '0', '0': 'o',
            'l': '1', '1': 'l',
            'm': 'n', 'n': 'm',
            'c': 'e', 'e': 'c',
            'u': 'v', 'v': 'u'
        }
        
        if random.random() < 0.5 and text and text.lower() in error_map:
            return error_map[text.lower()]
        return text


class TesseractBackend(OCRBackend):
    """Tesseract OCR backend (optional)"""
    
    def is_available(self) -> bool:
        return TESSERACT_AVAILABLE
    
    def extract_tokens(self, file_path: str, config: Dict[str, Any]) -> List[Token]:
        """Extract tokens using Tesseract OCR"""
        if not self.is_available():
            raise RuntimeError("pytesseract not available")
            
        # This would implement actual Tesseract OCR
        # For now, fall back to simulated OCR
        logger.warning("Tesseract backend not fully implemented, using simulated OCR")
        simulated = SimulatedOCRBackend()
        return simulated.extract_tokens(file_path, config)


def get_backend(backend_name: str, config: Dict[str, Any]) -> OCRBackend:
    """Factory function to get OCR backend by name"""
    backends = {
        "pdfminer": PDFMinerBackend(),
        "simulated": SimulatedOCRBackend(),
        "tesseract": TesseractBackend()
    }
    
    if backend_name == "auto":
        # Try backends in order of preference
        for name in ["pdfminer", "tesseract", "simulated"]:
            backend = backends[name]
            if backend.is_available():
                logger.info(f"Auto-selected OCR backend: {name}")
                return backend
        
        # Fallback to simulated
        logger.warning("No OCR backends available, using simulated")
        return backends["simulated"]
    
    if backend_name not in backends:
        raise ValueError(f"Unknown backend: {backend_name}")
        
    backend = backends[backend_name]
    if not backend.is_available():
        raise RuntimeError(f"Backend {backend_name} not available")
        
    return backend


def extract_tokens(file_path: str, backend: str, config: Dict[str, Any]) -> List[Token]:
    """Main entry point for token extraction"""
    ocr_backend = get_backend(backend, config)
    
    try:
        tokens = ocr_backend.extract_tokens(file_path, config)
        logger.info(f"Successfully extracted {len(tokens)} tokens using {backend} backend")
        return tokens
        
    except Exception as e:
        logger.error(f"Token extraction failed: {e}")
        
        # Try fallback backend if specified
        fallback = config.get("ocr", {}).get("fallback_backend", "simulated")
        if fallback != backend:
            logger.info(f"Trying fallback backend: {fallback}")
            return extract_tokens(file_path, fallback, config)
        else:
            raise


if __name__ == "__main__":
    # Test the backends
    logging.basicConfig(level=logging.INFO)
    
    config = {"ocr": {"fallback_backend": "simulated"}}
    
    # Test simulated backend
    simulated_backend = SimulatedOCRBackend()
    test_tokens = simulated_backend._generate_transcript_tokens()
    print(f"Generated {len(test_tokens)} test tokens")
    
    for i, token in enumerate(test_tokens[:5]):
        print(f"Token {i+1}: '{token.text}' at {token.bbox}")