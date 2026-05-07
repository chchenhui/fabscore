"""
Transcript parsing module for extracting academic records from OCR tokens
Handles course identification, credit extraction, and GPA computation
"""

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

from ocr_backends import Token

logger = logging.getLogger(__name__)

@dataclass
class CourseRecord:
    """Represents a single course record from a transcript"""
    course_code: str
    course_name: str
    credits: float
    grade: str
    grade_points: float
    semester: str = ""
    year: str = ""
    evidence_span: Tuple[int, int] = (0, 0)  # Character positions in original text
    
    def __post_init__(self):
        """Validate course record data"""
        if self.credits < 0:
            raise ValueError("Credits cannot be negative")
        if self.grade_points < 0 or self.grade_points > 4.0:
            raise ValueError("Grade points must be between 0 and 4.0")


@dataclass
class TranscriptParseResult:
    """Complete result of transcript parsing"""
    courses: List[CourseRecord]
    total_credits: float
    gpa: float
    evidence_spans: List[Dict[str, Any]]
    warnings: List[str]
    parsing_confidence: float
    
    @property
    def is_valid(self) -> bool:
        """Check if parsing result is valid"""
        return (
            len(self.courses) > 0 and
            self.total_credits > 0 and
            0 <= self.gpa <= 4.0 and
            self.parsing_confidence > 0.5
        )


class TranscriptParser:
    """Parses academic transcripts from OCR tokens"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.grade_points_map = config.get("grade_points", self._default_grade_points())
        
        # Compile regex patterns
        self._compile_patterns()
        
    def _default_grade_points(self) -> Dict[str, float]:
        """Default grade to points mapping"""
        return {
            "A+": 4.0, "A": 4.0, "A-": 3.7,
            "B+": 3.3, "B": 3.0, "B-": 2.7,
            "C+": 2.3, "C": 2.0, "C-": 1.7,
            "D+": 1.3, "D": 1.0, "D-": 0.7,
            "F": 0.0, "I": 0.0, "W": 0.0, "AU": 0.0
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for parsing"""
        # Course code patterns (e.g., CS101, MATH 201, ENG-101)
        self.course_code_pattern = re.compile(
            r'\b([A-Z]{2,5})\s*[-\s]?\s*(\d{3,4}[A-Z]?)\b',
            re.IGNORECASE
        )
        
        # Grade patterns
        self.grade_pattern = re.compile(
            r'\b([A-F][+-]?|I|W|AU)\b'
        )
        
        # Credit patterns (handle decimal credits)
        self.credit_pattern = re.compile(
            r'\b(\d{1,2}(?:\.\d{1,2})?)\s*(?:cr|credit|hour)s?\b',
            re.IGNORECASE
        )
        
        # Semester/term patterns
        self.semester_pattern = re.compile(
            r'\b(Fall|Spring|Summer|Winter)\s+(\d{4})\b',
            re.IGNORECASE
        )
        
        # GPA patterns
        self.gpa_pattern = re.compile(
            r'(?:GPA|Grade\s+Point\s+Average):\s*(\d\.\d{2,3})',
            re.IGNORECASE
        )
    
    def parse_transcript(self, tokens: List[Token]) -> TranscriptParseResult:
        """Main method to parse transcript from OCR tokens"""
        try:
            # Convert tokens to text representation
            text, token_map = self._tokens_to_text(tokens)
            
            # Extract course records
            courses = self._extract_courses(text, token_map, tokens)
            
            # Compute GPA and credits
            total_credits = sum(course.credits for course in courses)
            gpa = self._compute_gpa(courses)
            
            # Generate evidence spans
            evidence_spans = self._generate_evidence_spans(courses, text)
            
            # Assess parsing quality and generate warnings
            warnings = self._generate_warnings(courses, text)
            confidence = self._assess_parsing_confidence(courses, text, len(tokens))
            
            result = TranscriptParseResult(
                courses=courses,
                total_credits=total_credits,
                gpa=gpa,
                evidence_spans=evidence_spans,
                warnings=warnings,
                parsing_confidence=confidence
            )
            
            logger.info(f"Parsed {len(courses)} courses, GPA: {gpa:.2f}, Credits: {total_credits}")
            return result
            
        except Exception as e:
            logger.error(f"Transcript parsing failed: {e}")
            # Return empty result with error
            return TranscriptParseResult(
                courses=[],
                total_credits=0.0,
                gpa=0.0,
                evidence_spans=[],
                warnings=[f"Parsing error: {str(e)}"],
                parsing_confidence=0.0
            )
    
    def _tokens_to_text(self, tokens: List[Token]) -> Tuple[str, Dict[int, Token]]:
        """Convert tokens to continuous text with position mapping"""
        # Sort tokens by position (top to bottom, left to right)
        sorted_tokens = sorted(tokens, key=lambda t: (-t.bbox[1], t.bbox[0]))
        
        text_parts = []
        token_map = {}
        current_pos = 0
        
        for token in sorted_tokens:
            text_parts.append(token.text)
            token_map[current_pos] = token
            current_pos += len(token.text) + 1  # +1 for space
            
        text = " ".join(text_parts)
        return text, token_map
    
    def _extract_courses(self, text: str, token_map: Dict[int, Token], tokens: List[Token]) -> List[CourseRecord]:
        """Extract course records using pattern matching"""
        courses = []
        lines = text.split('\n')
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Try to parse as a course record
            course = self._parse_course_line(line, line_idx)
            if course:
                courses.append(course)
                
        # If pattern matching failed, try table-based extraction
        if not courses:
            courses = self._extract_table_courses(tokens)
            
        # Filter out invalid courses
        valid_courses = [c for c in courses if self._validate_course(c)]
        
        return valid_courses
    
    def _parse_course_line(self, line: str, line_idx: int) -> Optional[CourseRecord]:
        """Parse a single line that might contain course information"""
        # Look for course code
        course_match = self.course_code_pattern.search(line)
        if not course_match:
            return None
            
        course_code = f"{course_match.group(1)}{course_match.group(2)}"
        
        # Extract credits
        credit_match = self.credit_pattern.search(line)
        credits = float(credit_match.group(1)) if credit_match else 3.0  # Default 3 credits
        
        # Extract grade
        grade_match = self.grade_pattern.search(line)
        if not grade_match:
            return None
            
        grade = grade_match.group(1).upper()
        grade_points = self.grade_points_map.get(grade, 0.0)
        
        # Extract course name (heuristic)
        course_name = self._extract_course_name(line, course_match, grade_match, credit_match)
        
        # Extract semester/year if present
        semester_match = self.semester_pattern.search(line)
        semester = f"{semester_match.group(1)} {semester_match.group(2)}" if semester_match else ""
        
        return CourseRecord(
            course_code=course_code,
            course_name=course_name,
            credits=credits,
            grade=grade,
            grade_points=grade_points,
            semester=semester,
            evidence_span=(0, len(line))  # Approximate span
        )
    
    def _extract_course_name(self, line: str, course_match, grade_match, credit_match) -> str:
        """Extract course name from line by removing known patterns"""
        # Start with full line
        name = line
        
        # Remove course code
        name = name.replace(course_match.group(0), "", 1)
        
        # Remove grade
        name = name.replace(grade_match.group(0), "", 1)
        
        # Remove credits
        if credit_match:
            name = name.replace(credit_match.group(0), "", 1)
            
        # Clean up
        name = re.sub(r'\s+', ' ', name).strip()
        return name[:50]  # Limit length
    
    def _extract_table_courses(self, tokens: List[Token]) -> List[CourseRecord]:
        """Extract courses using table structure detection"""
        # Group tokens by rows (similar Y coordinates)
        rows = self._group_tokens_by_rows(tokens)
        
        courses = []
        for row_tokens in rows:
            if len(row_tokens) >= 3:  # Minimum: course, credits, grade
                course = self._parse_token_row(row_tokens)
                if course:
                    courses.append(course)
                    
        return courses
    
    def _group_tokens_by_rows(self, tokens: List[Token], tolerance: float = 5.0) -> List[List[Token]]:
        """Group tokens into rows based on Y coordinates"""
        # Sort by Y coordinate (top to bottom)
        sorted_tokens = sorted(tokens, key=lambda t: -t.bbox[1])
        
        rows = []
        current_row = []
        current_y = None
        
        for token in sorted_tokens:
            token_y = token.bbox[1]
            
            if current_y is None or abs(token_y - current_y) <= tolerance:
                current_row.append(token)
                current_y = token_y
            else:
                if current_row:
                    # Sort row by X coordinate (left to right)
                    current_row.sort(key=lambda t: t.bbox[0])
                    rows.append(current_row)
                current_row = [token]
                current_y = token_y
                
        # Add last row
        if current_row:
            current_row.sort(key=lambda t: t.bbox[0])
            rows.append(current_row)
            
        return rows
    
    def _parse_token_row(self, tokens: List[Token]) -> Optional[CourseRecord]:
        """Parse a row of tokens as a course record"""
        # Combine tokens into text
        text = " ".join(token.text for token in tokens)
        
        # Try to identify course components
        course_match = self.course_code_pattern.search(text)
        grade_match = self.grade_pattern.search(text)
        
        if not course_match or not grade_match:
            return None
            
        # Look for numeric values (likely credits)
        numbers = re.findall(r'\b(\d{1,2}(?:\.\d)?)\b', text)
        credits = 3.0  # Default
        
        for num_str in numbers:
            num = float(num_str)
            if 0.5 <= num <= 20:  # Reasonable credit range
                credits = num
                break
                
        course_code = f"{course_match.group(1)}{course_match.group(2)}"
        grade = grade_match.group(1).upper()
        grade_points = self.grade_points_map.get(grade, 0.0)
        
        # Extract course name (everything except course code, grade, and credits)
        course_name = self._clean_course_name(text, course_code, grade, str(credits))
        
        return CourseRecord(
            course_code=course_code,
            course_name=course_name,
            credits=credits,
            grade=grade,
            grade_points=grade_points
        )
    
    def _clean_course_name(self, text: str, course_code: str, grade: str, credits: str) -> str:
        """Clean course name by removing known components"""
        name = text
        
        # Remove components
        name = name.replace(course_code, "")
        name = name.replace(grade, "")
        name = name.replace(credits, "")
        
        # Clean up whitespace and common artifacts
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'[^\w\s]', ' ', name)
        name = name.strip()
        
        return name[:50] if name else "Unknown Course"
    
    def _validate_course(self, course: CourseRecord) -> bool:
        """Validate that course record is reasonable"""
        return (
            len(course.course_code) >= 5 and
            0.5 <= course.credits <= 20 and
            course.grade in self.grade_points_map and
            len(course.course_name) > 0
        )
    
    def _compute_gpa(self, courses: List[CourseRecord]) -> float:
        """Compute cumulative GPA from course records"""
        if not courses:
            return 0.0
            
        total_grade_points = 0.0
        total_credits = 0.0
        
        for course in courses:
            # Only include courses with grades in GPA calculation
            if course.grade not in ['I', 'W', 'AU']:
                total_grade_points += course.grade_points * course.credits
                total_credits += course.credits
                
        return total_grade_points / total_credits if total_credits > 0 else 0.0
    
    def _generate_evidence_spans(self, courses: List[CourseRecord], text: str) -> List[Dict[str, Any]]:
        """Generate evidence spans for transparency"""
        spans = []
        
        for course in courses:
            # Find course in text and create evidence
            course_pattern = re.escape(course.course_code)
            match = re.search(course_pattern, text, re.IGNORECASE)
            
            if match:
                spans.append({
                    "start": match.start(),
                    "end": match.end(),
                    "text": course.course_code,
                    "type": "course",
                    "metadata": {
                        "credits": course.credits,
                        "grade": course.grade,
                        "grade_points": course.grade_points
                    }
                })
                
        return spans
    
    def _generate_warnings(self, courses: List[CourseRecord], text: str) -> List[str]:
        """Generate warnings about parsing quality"""
        warnings = []
        
        if not courses:
            warnings.append("No courses found in transcript")
            
        if len(courses) < 8:
            warnings.append("Fewer than 8 courses found - transcript may be incomplete")
            
        # Check for unusual GPA
        if courses:
            gpa = self._compute_gpa(courses)
            if gpa < 1.0:
                warnings.append("Computed GPA is unusually low")
            elif gpa > 4.0:
                warnings.append("Computed GPA exceeds 4.0 scale")
                
        # Check for missing grades
        missing_grades = sum(1 for c in courses if c.grade in ['I', 'W'])
        if missing_grades > len(courses) * 0.2:
            warnings.append("High number of incomplete/withdrawn courses")
            
        return warnings
    
    def _assess_parsing_confidence(self, courses: List[CourseRecord], text: str, num_tokens: int) -> float:
        """Assess confidence in parsing results"""
        if not courses:
            return 0.0
            
        confidence_factors = []
        
        # Factor 1: Number of courses found vs. expected
        expected_courses = max(8, num_tokens // 50)  # Rough estimate
        course_factor = min(1.0, len(courses) / expected_courses)
        confidence_factors.append(course_factor)
        
        # Factor 2: Grade distribution reasonableness
        grades = [c.grade for c in courses]
        unique_grades = set(grades)
        grade_factor = min(1.0, len(unique_grades) / 6.0)  # Expect variety
        confidence_factors.append(grade_factor)
        
        # Factor 3: Credit distribution
        credits = [c.credits for c in courses]
        if all(1 <= c <= 6 for c in credits):  # Reasonable range
            credit_factor = 1.0
        else:
            credit_factor = 0.7
        confidence_factors.append(credit_factor)
        
        # Factor 4: GPA reasonableness
        gpa = self._compute_gpa(courses)
        if 2.0 <= gpa <= 4.0:
            gpa_factor = 1.0
        else:
            gpa_factor = 0.5
        confidence_factors.append(gpa_factor)
        
        # Combined confidence
        return sum(confidence_factors) / len(confidence_factors)


def parse_transcript(tokens: List[Token], config: Dict[str, Any]) -> TranscriptParseResult:
    """Main entry point for transcript parsing"""
    parser = TranscriptParser(config)
    return parser.parse_transcript(tokens)


if __name__ == "__main__":
    # Test the parser
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from ocr_backends import SimulatedOCRBackend
    
    # Generate test tokens
    backend = SimulatedOCRBackend()
    test_tokens = backend._generate_transcript_tokens()
    
    # Parse transcript
    config = {}
    result = parse_transcript(test_tokens, config)
    
    print(f"Parsed {len(result.courses)} courses")
    print(f"GPA: {result.gpa:.2f}")
    print(f"Total credits: {result.total_credits}")
    print(f"Confidence: {result.parsing_confidence:.2f}")
    print(f"Warnings: {result.warnings}")
    
    for course in result.courses:
        print(f"  {course.course_code}: {course.credits} credits, {course.grade} ({course.grade_points} pts)")