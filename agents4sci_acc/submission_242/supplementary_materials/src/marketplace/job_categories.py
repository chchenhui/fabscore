"""
Job Categories Module

Simple enumeration of job categories for the marketplace.
Each client is assigned one primary category and generates jobs within that category.
"""

from enum import Enum
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class CategoryMetrics:
    """Metrics for a job category"""
    avg_hourly_rate: float
    competition_level: float  # 0-1 scale
    success_rate: float  # 0-1 scale

@dataclass
class CategoryDefinition:
    """Complete definition of a job category"""
    name: str
    description: str
    core_skills: List[str]
    related_skills: List[str]
    common_projects: List[str]
    certification_value: float  # 0-1 scale
    portfolio_value: float  # 0-1 scale
    metrics: CategoryMetrics
    tools: List[str]
    specializations: List[str]

class JobCategory(Enum):
    """Standard job categories for the marketplace with typical hourly rates"""
    ACCOUNTING = ("Accounting & Consulting", 45)
    ADMIN = ("Admin Support", 20)
    CUSTOMER_SERVICE = ("Customer Service", 25)
    DATA_SCIENCE = ("Data Science & Analytics", 65)
    DESIGN = ("Design & Creative", 45)
    ENGINEERING = ("Engineering & Architecture", 70)
    IT = ("IT & Networking", 50)
    LEGAL = ("Legal", 85)
    SALES = ("Sales & Marketing", 45)
    TRANSLATION = ("Translation", 35)
    SOFTWARE = ("Web, Mobile & Software Dev", 55)
    WRITING = ("Writing", 40)
    
    def __init__(self, title: str, avg_rate: float):
        self.title = title
        self.avg_rate = avg_rate

class JobCategoryManager:
    """Manages job category definitions and market dynamics"""
    
    def __init__(self):
        self._categories: Dict[JobCategory, CategoryDefinition] = self._initialize_categories()
        
    def get_category(self, category: JobCategory) -> CategoryDefinition:
        """Get the complete definition for a category"""
        return self._categories[category]
    
    def get_all_categories(self) -> Dict[JobCategory, CategoryDefinition]:
        """Get all category definitions"""
        return self._categories
    

    
    def get_category_for_job(self, job_title: str, job_skills: List[str]) -> Optional[JobCategory]:
        """Determine the most likely category for a job based on title and skills"""
        best_match = None
        best_score = 0
        
        job_text = (job_title + " " + " ".join(job_skills)).lower()
        
        for category, definition in self._categories.items():
            # Check for category name in job text
            if definition.name.lower() in job_text:
                return category
            
            # Calculate skill match score
            skill_matches = sum(1 for skill in definition.core_skills 
                              if skill.lower() in job_text)
            skill_matches += sum(1 for skill in definition.related_skills 
                               if skill.lower() in job_text) * 0.5
            
            # Check for common project types
            project_matches = sum(1 for project in definition.common_projects 
                                if project.lower() in job_text) * 0.3
            
            total_score = skill_matches + project_matches
            
            if total_score > best_score:
                best_score = total_score
                best_match = category
        
        return best_match if best_score > 0.5 else None
    
    def _initialize_categories(self) -> Dict[JobCategory, CategoryDefinition]:
        """Initialize all category definitions"""
        return {
            JobCategory.ACCOUNTING: CategoryDefinition(
                name="Accounting & Consulting",
                description="Financial services, accounting, and business consulting",
                core_skills=[
                    "Bookkeeping", "Financial Analysis", "Tax Preparation",
                    "QuickBooks", "Business Consulting", "Financial Reporting"
                ],
                related_skills=[
                    "Budgeting", "Forecasting", "Risk Assessment",
                    "Audit", "Business Planning", "Cost Accounting"
                ],
                common_projects=[
                    "Monthly Bookkeeping", "Tax Return Preparation",
                    "Financial Statement Analysis", "Business Plan Development",
                    "Audit Support", "Budget Planning"
                ],
                certification_value=0.9,  # High value on certifications
                portfolio_value=0.4,  # Medium-low portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=45.0,
                    competition_level=0.7,
                    success_rate=0.6,
                ),
                tools=["QuickBooks", "Xero", "SAP", "Excel", "Sage"],
                specializations=[
                    "Tax Accounting", "Management Consulting",
                    "Forensic Accounting", "Risk Management"
                ]
            ),
            
            JobCategory.ADMIN: CategoryDefinition(
                name="Admin Support",
                description="Administrative and virtual assistance services",
                core_skills=[
                    "Data Entry", "Email Management", "Calendar Management",
                    "Customer Support", "Document Preparation", "Virtual Assistance"
                ],
                related_skills=[
                    "MS Office", "Google Workspace", "CRM Management",
                    "Project Coordination", "Travel Planning", "Transcription"
                ],
                common_projects=[
                    "Virtual Assistant", "Data Entry Projects",
                    "Email Management", "Calendar Organization",
                    "Document Processing", "Customer Service Support"
                ],
                certification_value=0.3,  # Lower value on certifications
                portfolio_value=0.5,  # Medium portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=20.0,
                    competition_level=0.8,
                    success_rate=0.7,
                ),
                tools=["MS Office", "Google Workspace", "Asana", "Trello", "Zoom"],
                specializations=[
                    "Executive Assistance", "Medical Administration",
                    "Real Estate Administration", "Legal Administration"
                ]
            ),
            
            JobCategory.DATA_SCIENCE: CategoryDefinition(
                name="Data Science & Analytics",
                description="Data analysis, machine learning, and statistical modeling",
                core_skills=[
                    "Python", "R", "SQL", "Machine Learning",
                    "Statistical Analysis", "Data Visualization"
                ],
                related_skills=[
                    "Deep Learning", "Natural Language Processing",
                    "Big Data", "A/B Testing", "Data Mining", "Tableau"
                ],
                common_projects=[
                    "Predictive Modeling", "Data Analysis",
                    "Dashboard Creation", "Machine Learning Models",
                    "Business Intelligence", "Statistical Research"
                ],
                certification_value=0.7,  # High value on certifications
                portfolio_value=0.8,  # High portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=65.0,
                    competition_level=0.6,
                    success_rate=0.5,
                ),
                tools=["Python", "R", "TensorFlow", "PyTorch", "Tableau", "Power BI"],
                specializations=[
                    "Machine Learning", "NLP", "Computer Vision",
                    "Business Analytics", "Bioinformatics"
                ]
            ),
            
            JobCategory.SOFTWARE: CategoryDefinition(
                name="Web, Mobile & Software Dev",
                description="Software development across web, mobile, and desktop platforms",
                core_skills=[
                    "JavaScript", "Python", "Java", "React",
                    "Node.js", "SQL", "Mobile Development"
                ],
                related_skills=[
                    "TypeScript", "Angular", "Vue.js", "AWS",
                    "DevOps", "Docker", "Kubernetes"
                ],
                common_projects=[
                    "Web Application", "Mobile App Development",
                    "E-commerce Platform", "API Development",
                    "Software Architecture", "System Integration"
                ],
                certification_value=0.5,  # Medium value on certifications
                portfolio_value=0.9,  # Very high portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=55.0,
                    competition_level=0.9,
                    success_rate=0.6,
                ),
                tools=["VS Code", "Git", "Docker", "AWS", "Jira"],
                specializations=[
                    "Frontend Development", "Backend Development",
                    "Mobile Development", "DevOps", "Cloud Architecture"
                ]
            ),
            
            JobCategory.DESIGN: CategoryDefinition(
                name="Design & Creative",
                description="Visual design, creative services, and multimedia production",
                core_skills=[
                    "Graphic Design", "UI/UX Design", "Adobe Creative Suite",
                    "Logo Design", "Illustration", "Brand Identity"
                ],
                related_skills=[
                    "Motion Graphics", "Video Editing", "3D Modeling",
                    "Photography", "Animation", "Web Design"
                ],
                common_projects=[
                    "Logo Design", "Brand Identity", "Website Design",
                    "Marketing Materials", "Social Media Graphics",
                    "Product Packaging"
                ],
                certification_value=0.4,  # Lower value on certifications
                portfolio_value=1.0,  # Extremely high portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=45.0,
                    competition_level=0.8,
                    success_rate=0.6,
                ),
                tools=["Adobe Creative Suite", "Figma", "Sketch", "Cinema 4D", "After Effects"],
                specializations=[
                    "Brand Design", "UI/UX Design", "Motion Design",
                    "Print Design", "Packaging Design"
                ]
            ),
            
            JobCategory.ENGINEERING: CategoryDefinition(
                name="Engineering & Architecture",
                description="Engineering services and architectural design",
                core_skills=[
                    "CAD", "3D Modeling", "Structural Engineering",
                    "Mechanical Engineering", "Architecture", "Technical Drawing"
                ],
                related_skills=[
                    "AutoCAD", "Revit", "Civil Engineering",
                    "Product Design", "Construction Planning", "BIM"
                ],
                common_projects=[
                    "Architectural Design", "Product Design",
                    "Structural Analysis", "MEP Design",
                    "Construction Documents", "3D Modeling"
                ],
                certification_value=0.9,  # Very high value on certifications
                portfolio_value=0.8,  # High portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=70.0,
                    competition_level=0.5,
                    success_rate=0.5,
                ),
                tools=["AutoCAD", "Revit", "SolidWorks", "SketchUp", "MATLAB"],
                specializations=[
                    "Structural Engineering", "Mechanical Engineering",
                    "Civil Engineering", "Architecture", "MEP Engineering"
                ]
            ),
            
            JobCategory.IT: CategoryDefinition(
                name="IT & Networking",
                description="Information technology infrastructure and networking services",
                core_skills=[
                    "Network Administration", "System Administration",
                    "Cloud Infrastructure", "Cybersecurity", "IT Support"
                ],
                related_skills=[
                    "AWS", "Azure", "Linux", "Windows Server",
                    "Virtualization", "Network Security"
                ],
                common_projects=[
                    "Network Setup", "Cloud Migration",
                    "Security Implementation", "System Maintenance",
                    "IT Infrastructure", "Technical Support"
                ],
                certification_value=0.9,  # Very high value on certifications
                portfolio_value=0.5,  # Medium portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=50.0,
                    competition_level=0.7,
                    success_rate=0.6,
                ),
                tools=["AWS", "Azure", "VMware", "Cisco", "Linux"],
                specializations=[
                    "Network Security", "Cloud Architecture",
                    "System Administration", "DevOps", "IT Support"
                ]
            ),
            
            JobCategory.LEGAL: CategoryDefinition(
                name="Legal",
                description="Legal services and documentation",
                core_skills=[
                    "Legal Writing", "Contract Law", "Legal Research",
                    "Intellectual Property", "Corporate Law"
                ],
                related_skills=[
                    "Paralegal Services", "Document Review",
                    "Compliance", "Patent Law", "Immigration Law"
                ],
                common_projects=[
                    "Contract Review", "Legal Documentation",
                    "Patent Applications", "Legal Research",
                    "Compliance Review", "Terms of Service"
                ],
                certification_value=1.0,  # Extremely high value on certifications
                portfolio_value=0.6,  # Medium-high portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=85.0,
                    competition_level=0.4,
                    success_rate=0.5,
                ),
                tools=["LexisNexis", "Westlaw", "Legal Case Management Software"],
                specializations=[
                    "Contract Law", "Intellectual Property",
                    "Corporate Law", "Immigration Law", "Patent Law"
                ]
            ),
            
            JobCategory.SALES: CategoryDefinition(
                name="Sales & Marketing",
                description="Sales, marketing, and business development services",
                core_skills=[
                    "Digital Marketing", "Social Media Marketing",
                    "Content Marketing", "SEO", "Email Marketing"
                ],
                related_skills=[
                    "Market Research", "Analytics", "CRM",
                    "Lead Generation", "Brand Strategy", "PPC"
                ],
                common_projects=[
                    "Marketing Strategy", "Social Media Campaign",
                    "SEO Optimization", "Email Marketing",
                    "Content Strategy", "Lead Generation"
                ],
                certification_value=0.5,  # Medium value on certifications
                portfolio_value=0.8,  # High portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=45.0,
                    competition_level=0.8,
                    success_rate=0.7,
                ),
                tools=["Google Analytics", "HubSpot", "Mailchimp", "SEMrush", "Facebook Ads"],
                specializations=[
                    "Digital Marketing", "Content Marketing",
                    "Social Media Marketing", "SEO", "Email Marketing"
                ]
            ),
            
            JobCategory.TRANSLATION: CategoryDefinition(
                name="Translation",
                description="Language translation and localization services",
                core_skills=[
                    "Translation", "Proofreading", "Localization",
                    "Editing", "Language Expertise"
                ],
                related_skills=[
                    "Subtitling", "Technical Translation",
                    "Medical Translation", "Legal Translation"
                ],
                common_projects=[
                    "Document Translation", "Website Localization",
                    "Technical Manual Translation", "Subtitle Creation",
                    "Marketing Translation", "Legal Document Translation"
                ],
                certification_value=0.8,  # High value on certifications
                portfolio_value=0.6,  # Medium-high portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=35.0,
                    competition_level=0.7,
                    success_rate=0.7,
                ),
                tools=["CAT Tools", "SDL Trados", "MemoQ", "Wordfast"],
                specializations=[
                    "Technical Translation", "Legal Translation",
                    "Medical Translation", "Literary Translation",
                    "Marketing Translation"
                ]
            ),
            
            JobCategory.WRITING: CategoryDefinition(
                name="Writing",
                description="Content writing and copywriting services",
                core_skills=[
                    "Content Writing", "Copywriting", "Editing",
                    "Blog Writing", "Technical Writing"
                ],
                related_skills=[
                    "SEO Writing", "Creative Writing", "Proofreading",
                    "Research", "Journalism", "Grant Writing"
                ],
                common_projects=[
                    "Blog Posts", "Website Content", "Product Descriptions",
                    "Technical Documentation", "Marketing Copy",
                    "Article Writing"
                ],
                certification_value=0.3,  # Lower value on certifications
                portfolio_value=0.9,  # Very high portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=40.0,
                    competition_level=0.9,
                    success_rate=0.7,
                ),
                tools=["Grammarly", "Hemingway", "Google Docs", "WordPress"],
                specializations=[
                    "Technical Writing", "Creative Writing",
                    "Copywriting", "Content Marketing",
                    "Academic Writing"
                ]
            ),
            
            JobCategory.CUSTOMER_SERVICE: CategoryDefinition(
                name="Customer Service",
                description="Customer support and service management",
                core_skills=[
                    "Customer Support", "Phone Support", "Email Support",
                    "Live Chat", "Ticket Management", "CRM"
                ],
                related_skills=[
                    "Problem Resolution", "Technical Support",
                    "Quality Assurance", "Customer Experience",
                    "Help Desk"
                ],
                common_projects=[
                    "Customer Support", "Help Desk Management",
                    "Technical Support", "Customer Service Training",
                    "Support Documentation", "Quality Monitoring"
                ],
                certification_value=0.4,  # Lower value on certifications
                portfolio_value=0.5,  # Medium portfolio importance
                metrics=CategoryMetrics(
                    avg_hourly_rate=25.0,
                    competition_level=0.8,
                    success_rate=0.8,
                ),
                tools=["Zendesk", "Freshdesk", "Intercom", "Salesforce", "LiveChat"],
                specializations=[
                    "Technical Support", "Email Support",
                    "Phone Support", "Live Chat Support",
                    "Help Desk Management"
                ]
            )
        }

# Global instance for easy access
category_manager = JobCategoryManager()
