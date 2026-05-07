"""
Synthetic data generator for creating realistic but privacy-safe academic documents
"""

import random
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class SyntheticTranscript:
    id: str
    true_gpa: float
    true_credits: float
    courses: List[Dict[str, Any]]
    university: str
    template: str

@dataclass 
class SyntheticResume:
    id: str
    experience_years: int
    skills: List[str]
    education_level: str
    organizations: List[str]

@dataclass
class SyntheticStatement:
    id: str
    word_count: int
    research_areas: List[str]
    quality_score: float

class SyntheticDataGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        synthetic_config = config.get("synthetic", {})
        self.random_seed = synthetic_config.get("random_seed", 42)
        
        # Set seeds for reproducibility
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        self._initialize_vocabularies()
    
    def _initialize_vocabularies(self):
        """Initialize data vocabularies"""
        self.courses = [
            "CS101", "CS102", "CS201", "CS202", "CS301", "CS302",
            "MATH101", "MATH201", "MATH202", "MATH301",
            "PHYS101", "PHYS102", "ENG101", "ENG102",
            "STAT201", "STAT202", "ECON101", "HIST101"
        ]
        
        self.universities = [
            "State University", "Tech Institute", "Community College",
            "Metropolitan University", "Regional College"
        ]
        
        self.skills = [
            "Python", "Java", "JavaScript", "C++", "SQL", "React",
            "Machine Learning", "Data Analysis", "Git", "Linux"
        ]
        
        self.companies = [
            "TechCorp", "DataSystems", "InnovativeSoft", "CloudTech", "StartupXYZ"
        ]
    
    def generate_transcripts(self, count: int) -> List[Dict[str, Any]]:
        """Generate synthetic transcript data"""
        transcripts = []
        
        for i in range(count):
            # Generate GPA from normal distribution
            gpa = np.clip(np.random.normal(3.2, 0.6), 2.0, 4.0)
            
            # Generate number of courses
            num_courses = random.randint(8, 24)
            
            # Generate courses with consistent GPA
            courses = []
            total_credits = 0
            total_grade_points = 0
            
            for j in range(num_courses):
                course_code = random.choice(self.courses)
                credits = random.choice([3, 4])  # Most common credit values
                
                # Generate grade consistent with target GPA
                grade_points = np.clip(np.random.normal(gpa, 0.3), 0, 4)
                grade = self._grade_points_to_letter(grade_points)
                
                courses.append({
                    "course_code": f"{course_code}_{j}",
                    "credits": credits,
                    "grade": grade,
                    "grade_points": grade_points
                })
                
                total_credits += credits
                total_grade_points += grade_points * credits
            
            # Ensure computed GPA matches target
            actual_gpa = total_grade_points / total_credits if total_credits > 0 else 0
            
            transcript_data = {
                "id": f"transcript_{i}",
                "true_gpa": actual_gpa,
                "true_credits": total_credits,
                "courses": courses,
                "university": random.choice(self.universities),
                "template": random.choice(["table", "list", "mixed"])
            }
            
            transcripts.append(transcript_data)
        
        return transcripts
    
    def generate_resumes(self, count: int) -> List[Dict[str, Any]]:
        """Generate synthetic resume data"""
        resumes = []
        
        for i in range(count):
            experience_years = np.random.exponential(2)  # Exponential distribution
            experience_years = int(np.clip(experience_years, 0, 8))
            
            # Generate skills based on experience
            skill_count = min(10, max(2, int(np.random.normal(5 + experience_years, 2))))
            selected_skills = random.sample(self.skills, min(skill_count, len(self.skills)))
            
            # Education level
            education_probs = {"Bachelors": 0.8, "Masters": 0.15, "PhD": 0.05}
            education_level = np.random.choice(
                list(education_probs.keys()), 
                p=list(education_probs.values())
            )
            
            # Organizations
            org_count = min(3, max(1, experience_years // 2 + 1))
            organizations = random.sample(self.companies, min(org_count, len(self.companies)))
            
            resume_data = {
                "id": f"resume_{i}",
                "experience_years": experience_years,
                "skills": selected_skills,
                "education_level": education_level,
                "organizations": organizations
            }
            
            resumes.append(resume_data)
        
        return resumes
    
    def generate_statements(self, count: int) -> List[Dict[str, Any]]:
        """Generate synthetic statement of purpose data"""
        statements = []
        
        research_areas = ["Machine Learning", "Data Science", "Computer Vision", 
                         "Natural Language Processing", "Systems", "Security"]
        
        for i in range(count):
            # Word count from normal distribution
            word_count = int(np.clip(np.random.normal(500, 120), 300, 800))
            
            # Research areas (1-3 areas)
            num_areas = random.randint(1, 3)
            selected_areas = random.sample(research_areas, num_areas)
            
            # Quality score (bimodal distribution)
            if random.random() < 0.6:
                quality_score = np.clip(np.random.normal(2.5, 0.5), 1, 3.5)  # Lower mode
            else:
                quality_score = np.clip(np.random.normal(4.0, 0.4), 3.5, 5)  # Higher mode
            
            statement_data = {
                "id": f"statement_{i}",
                "word_count": word_count,
                "research_areas": selected_areas,
                "quality_score": quality_score
            }
            
            statements.append(statement_data)
        
        return statements
    
    def _grade_points_to_letter(self, grade_points: float) -> str:
        """Convert grade points to letter grade"""
        if grade_points >= 3.85:
            return "A"
        elif grade_points >= 3.5:
            return "A-"
        elif grade_points >= 3.15:
            return "B+"
        elif grade_points >= 2.85:
            return "B"
        elif grade_points >= 2.5:
            return "B-"
        elif grade_points >= 2.15:
            return "C+"
        elif grade_points >= 1.85:
            return "C"
        elif grade_points >= 1.5:
            return "C-"
        elif grade_points >= 1.15:
            return "D+"
        elif grade_points >= 0.85:
            return "D"
        else:
            return "F"