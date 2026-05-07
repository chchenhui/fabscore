#!/usr/bin/env python3
"""
Test script for the new semantic skill matching system.

Run this to see how the modern embedding-based matching works!
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from marketplace.skill_matcher import SemanticSkillMatcher


def test_skill_matching():
    """Test the semantic skill matching with real examples"""
    print("🚀 Testing Modern Semantic Skill Matching")
    print("=" * 60)
    
    # Initialize the matcher
    matcher = SemanticSkillMatcher()
    
    # Test cases: realistic freelancer skills vs job requirements
    test_cases = [
        {
            "name": "JavaScript Developer",
            "freelancer_skills": ["JavaScript", "React", "Node.js", "CSS", "HTML"],
            "job_skills": ["JS", "Frontend Development", "React.js", "Web Development"],
            "job_description": "We need a frontend developer to build our e-commerce website. Must be experienced with modern React development, responsive design, and have strong JavaScript fundamentals. The project involves creating interactive product pages and a shopping cart system."
        },
        {
            "name": "Data Scientist", 
            "freelancer_skills": ["Python", "Machine Learning", "Pandas", "Scikit-learn"],
            "job_skills": ["ML", "Data Analysis", "AI", "Statistical Modeling"],
            "job_description": "Looking for an AI specialist to develop predictive models for customer behavior analysis. Need someone with deep machine learning expertise, Python programming skills, and experience with large datasets. The role involves building recommendation systems and forecasting models."
        },
        {
            "name": "Designer",
            "freelancer_skills": ["Photoshop", "UI Design", "Figma", "Logo Design"],
            "job_skills": ["Graphic Design", "User Interface", "Adobe Creative Suite", "Branding"],
            "job_description": "Seeking a creative designer for rebranding project. Need someone who can create modern logos, design marketing materials, and develop a cohesive brand identity. Must be proficient with Adobe Creative Suite and have portfolio of brand design work."
        },
        {
            "name": "Marketing Specialist",
            "freelancer_skills": ["Digital Marketing", "SEO", "Content Writing", "Social Media"],
            "job_skills": ["Online Marketing", "Search Engine Optimization", "Blog Writing", "SMM"],
            "job_description": "Digital marketing expert needed for growing SaaS startup. Looking for someone to manage our content strategy, optimize our website for search engines, and grow our social media presence. Must have experience with B2B marketing and content creation."
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📋 Test Case: {test_case['name']}")
        print(f"   Freelancer skills: {test_case['freelancer_skills']}")
        print(f"   Job requirements:  {test_case['job_skills']}")
        print(f"   Job description: {test_case['job_description'][:100]}...")
        
        # Show the rich text representations
        freelancer_profile = matcher._create_freelancer_profile(test_case['freelancer_skills'])
        job_context = matcher._create_job_context(test_case['job_skills'], test_case['job_description'])
        print(f"   📝 Freelancer profile: '{freelancer_profile}'")
        print(f"   📝 Job context: '{job_context[:150]}...'")
        
        # Calculate overall match score (now with job description)
        overall_score = matcher.calculate_skill_match_score(
            test_case['freelancer_skills'], 
            test_case['job_skills'],
            test_case['job_description']
        )
        
        # Compare with skills-only matching
        skills_only_score = matcher.calculate_skill_match_score(
            test_case['freelancer_skills'], 
            test_case['job_skills'],
            ""  # No job description
        )
        
        print(f"   📊 SCORING COMPARISON:")
        print(f"      Skills only: {skills_only_score:.3f}")
        print(f"      With job description: {overall_score:.3f}")
        print(f"      📈 IMPROVEMENT: {overall_score - skills_only_score:+.3f}")
        
        # Show exact skill boost
        exact_boost = matcher._calculate_exact_skill_boost(
            test_case['freelancer_skills'], test_case['job_skills']
        )
        print(f"      Exact skill boost: {exact_boost:.3f}")
        
        print(f"   ✅ FINAL MATCH SCORE: {overall_score:.3f}")
    
    # Show cache statistics
    cache_stats = matcher.get_cache_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"   Cached embeddings: {cache_stats['cached_embeddings']}")
    print(f"   Cache size: {cache_stats['cache_size_mb']:.2f} MB")
    
    print("\n✅ Testing complete!")
    
    if matcher.use_openai:
        print("using openai")
    elif matcher.model is not None:
        print("using sentence-transformers")
    # Show installation note if using fallback
    elif matcher.model is None and not matcher.use_openai:
        print("\n💡 Note: Install sentence-transformers for better semantic matching:")
        print("   pip install sentence-transformers")


if __name__ == "__main__":
    test_skill_matching()
