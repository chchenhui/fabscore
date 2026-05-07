"""
Semantic Skill Matching Module

Uses modern NLP embeddings for intelligent skill matching. 
Because it's 2025, not 1995! 🚀
"""

from typing import List, Dict, Optional
import re
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class SkillMatch:
    """Represents a match between freelancer skill and job requirement"""
    freelancer_skill: str
    job_skill: str
    confidence: float  # 0.0 to 1.0
    match_type: str  # 'exact', 'semantic', 'related', 'partial'


class SemanticSkillMatcher:
    """
    Modern semantic skill matching using embeddings.
    
    Uses sentence transformers or OpenAI embeddings to understand
    skill relationships through vector similarity.
    """
    
    def __init__(self, use_openai: bool = False, model_name: str = "all-MiniLM-L6-v2"):
        self.use_openai = use_openai and OPENAI_AVAILABLE
        self.model_name = model_name
        self.model = None
        self._skill_cache = {}  # Cache embeddings to avoid recomputation
        
        if self.use_openai:
            logger.info("Using OpenAI embeddings for skill matching")
        elif SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence transformer model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.model = None
        else:
            logger.warning("No embedding models available, falling back to basic matching")
    
    def calculate_skill_match_score(self, freelancer_skills: List[str], job_skills: List[str], 
                                   job_description: str = "") -> float:
        """
        Calculate skill match score using job description context.
        
        Much more efficient approach:
        1. Create rich job context (skills + description)
        2. Create freelancer profile text
        3. Single embedding comparison
        4. Fallback to basic matching if embeddings unavailable
        
        Returns:
            float: Score from 0.0 to 1.0 representing skill compatibility
        """
        if not job_skills and not job_description:
            return 0.5  # Neutral score for jobs with no requirements
        
        if not freelancer_skills:
            return 0.0  # No skills to match
        
        # Create rich text representations
        freelancer_profile = self._create_freelancer_profile(freelancer_skills)
        job_context = self._create_job_context(job_skills, job_description)
        
        # Use single embedding comparison (much more efficient)
        if self.model is not None or self.use_openai:
            semantic_score = self._calculate_semantic_similarity(freelancer_profile, job_context)
            # Boost score slightly if there are exact skill matches
            exact_boost = self._calculate_exact_skill_boost(freelancer_skills, job_skills)
            return min(semantic_score + exact_boost, 1.0)
        else:
            # Fallback to basic text similarity when no embeddings available
            return self._calculate_basic_similarity(freelancer_profile, job_context)
    

    
    def _create_freelancer_profile(self, skills: List[str]) -> str:
        """Create rich freelancer profile text"""
        if not skills:
            return "general freelancer"
        
        cleaned_skills = [self._clean_skill(skill) for skill in skills]
        
        # Create natural language profile
        if len(cleaned_skills) == 1:
            return f"experienced freelancer specializing in {cleaned_skills[0]}"
        elif len(cleaned_skills) <= 3:
            return f"experienced freelancer with skills in {', '.join(cleaned_skills[:-1])} and {cleaned_skills[-1]}"
        else:
            # For many skills, group them more naturally
            main_skills = cleaned_skills[:3]
            return f"experienced freelancer specializing in {', '.join(main_skills)} and {len(cleaned_skills) - 3} other technologies"
    
    def _create_job_context(self, job_skills: List[str], job_description: str) -> str:
        """Create rich job context combining skills and description"""
        parts = []
        
        # Add skills context if available
        if job_skills:
            cleaned_skills = [self._clean_skill(skill) for skill in job_skills]
            if len(cleaned_skills) == 1:
                parts.append(f"requires expertise in {cleaned_skills[0]}")
            else:
                parts.append(f"requires skills in {', '.join(cleaned_skills)}")
        
        # Add job description context (extract key info)
        if job_description:
            # Clean and truncate description to avoid massive embeddings
            cleaned_desc = self._extract_key_job_info(job_description)
            if cleaned_desc:
                parts.append(cleaned_desc)
        
        if not parts:
            return "general project opportunity"
        
        return ". ".join(parts)
    
    def _extract_key_job_info(self, description: str) -> str:
        """Extract key information from job description"""
        if not description:
            return ""
        
        # Clean the description
        desc = description.lower().strip()
        
        # Limit length to avoid huge embeddings (first 200 chars + important keywords)
        if len(desc) <= 200:
            return desc
        
        # Take first 150 chars and add important keywords found later
        main_part = desc[:150]
        remaining = desc[150:]
        
        # Look for important keywords in remaining text
        important_keywords = [
            'react', 'python', 'javascript', 'ml', 'ai', 'design', 'mobile', 'web',
            'frontend', 'backend', 'fullstack', 'data', 'analytics', 'marketing',
            'urgent', 'asap', 'long-term', 'short-term', 'remote', 'onsite'
        ]
        
        found_keywords = []
        for keyword in important_keywords:
            if keyword in remaining and keyword not in main_part:
                found_keywords.append(keyword)
        
        if found_keywords:
            return f"{main_part}. Key requirements: {', '.join(found_keywords[:5])}"
        else:
            return main_part
    
    def _calculate_exact_skill_boost(self, freelancer_skills: List[str], job_skills: List[str]) -> float:
        """Calculate small boost for exact skill matches"""
        if not job_skills or not freelancer_skills:
            return 0.0
        
        # Simple exact matches (case-insensitive, cleaned)
        freelancer_clean = {self._clean_skill(skill) for skill in freelancer_skills}
        job_clean = {self._clean_skill(skill) for skill in job_skills}
        
        exact_matches = len(freelancer_clean & job_clean)
        max_possible = len(job_clean)
        
        # Small boost (max 0.1) for exact matches
        return min(0.1 * (exact_matches / max_possible), 0.1) if max_possible > 0 else 0.0
    
    def find_skill_matches(self, freelancer_skills: List[str], job_skills: List[str]) -> List[SkillMatch]:
        """Find skill matches using semantic similarity"""
        matches = []
        matched_job_skills = set()
        
        for job_skill in job_skills:
            if job_skill in matched_job_skills:
                continue
                
            best_match = self._find_best_semantic_match(job_skill, freelancer_skills)
            if best_match:
                matches.append(best_match)
                matched_job_skills.add(job_skill)
        
        return matches
    
    def _find_best_semantic_match(self, job_skill: str, freelancer_skills: List[str]) -> Optional[SkillMatch]:
        """Find the best semantic match for a job skill among freelancer skills"""
        job_skill_clean = self._clean_skill(job_skill)
        best_match = None
        best_confidence = 0.0
        
        # First check for exact matches (fast path)
        for freelancer_skill in freelancer_skills:
            freelancer_skill_clean = self._clean_skill(freelancer_skill)
            if freelancer_skill_clean.lower() == job_skill_clean.lower():
                return SkillMatch(
                    freelancer_skill=freelancer_skill,
                    job_skill=job_skill,
                    confidence=1.0,
                    match_type='exact'
                )
        
        # Use semantic similarity for non-exact matches
        for freelancer_skill in freelancer_skills:
            freelancer_skill_clean = self._clean_skill(freelancer_skill)
            similarity = self._calculate_semantic_similarity(freelancer_skill_clean, job_skill_clean)
            
            if similarity > best_confidence:
                best_confidence = similarity
                match_type = self._determine_match_type(similarity)
                best_match = SkillMatch(
                    freelancer_skill=freelancer_skill,
                    job_skill=job_skill,
                    confidence=similarity,
                    match_type=match_type
                )
        
        # Only return matches above minimum threshold
        return best_match if best_confidence > 0.4 else None
    
    def _calculate_semantic_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate semantic similarity using embeddings"""
        try:
            if self.use_openai:
                return self._calculate_openai_similarity(skill1, skill2)
            elif self.model is not None:
                return self._calculate_transformer_similarity(skill1, skill2)
            else:
                # Fallback to basic text similarity
                return self._calculate_basic_similarity(skill1, skill2)
        except Exception as e:
            logger.exception(f"Error calculating semantic similarity: {e}")
            return self._calculate_basic_similarity(skill1, skill2)
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text, with caching"""
        if text in self._skill_cache:
            return self._skill_cache[text]
        
        try:
            if self.use_openai:
                # Use OpenAI embeddings
                response = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = np.array(response.data[0].embedding)
            elif self.model is not None:
                # Use sentence transformer
                embedding = self.model.encode(text)
            else:
                return None
            
            self._skill_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.exception(f"Error getting embedding for '{text}': {e}")
            return None
    
    def _calculate_transformer_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity using sentence transformers"""
        emb1 = self._get_embedding(skill1)
        emb2 = self._get_embedding(skill2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # Convert from [-1, 1] to [0, 1] and apply some scaling
        return max(0.0, (similarity + 1) / 2)
    
    def _calculate_openai_similarity(self, skill1: str, skill2: str) -> float:
        """Calculate similarity using OpenAI embeddings"""
        emb1 = self._get_embedding(skill1)
        emb2 = self._get_embedding(skill2)
        
        if emb1 is None or emb2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        # OpenAI embeddings are typically well-normalized, so we can use similarity more directly
        return max(0.0, similarity)
    
    def _calculate_basic_similarity(self, text1: str, text2: str) -> float:
        """Simple fallback similarity when no embeddings available"""
        text1, text2 = text1.lower(), text2.lower()
        
        # Exact match
        if text1 == text2:
            return 1.0
        
        # Substring match
        if text1 in text2 or text2 in text1:
            longer = max(len(text1), len(text2))
            shorter = min(len(text1), len(text2))
            return 0.7 * (shorter / longer)
        
        # Simple word overlap (Jaccard similarity)
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        jaccard = intersection / union
        return jaccard * 0.6 if jaccard > 0.2 else 0.0
    
    def _clean_skill(self, skill: str) -> str:
        """Clean and normalize skill name"""
        # Remove extra whitespace, convert to lowercase
        cleaned = re.sub(r'\s+', ' ', skill.strip().lower())
        
        # Remove common prefixes/suffixes that don't affect meaning
        cleaned = re.sub(r'\b(expert|advanced|intermediate|beginner|junior|senior)\s+', '', cleaned)
        cleaned = re.sub(r'\s+(expert|advanced|intermediate|beginner|junior|senior)\b', '', cleaned)
        
        # Remove punctuation that doesn't add semantic meaning
        # "React.js" -> "react js", "Node.js" -> "node js", "C#" -> "c#" (keep # for C#)
        cleaned = re.sub(r'\.(?!net|js)', ' ', cleaned)  # Remove dots except for .NET, .js
        cleaned = re.sub(r'[,;()\[\]{}]', ' ', cleaned)  # Remove other punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Clean up extra spaces
        
        return cleaned
    
    def _determine_match_type(self, similarity: float) -> str:
        """Determine match type based on similarity score"""
        if similarity >= 0.9:
            return 'exact'
        elif similarity >= 0.7:
            return 'semantic'
        elif similarity >= 0.5:
            return 'related'
        else:
            return 'partial'
    
    def clear_cache(self):
        """Clear the embedding cache"""
        self._skill_cache.clear()
        
    def get_cache_stats(self) -> Dict[str, any]:
        """Get statistics about the embedding cache"""
        cache_size_bytes = sum(
            emb.nbytes for emb in self._skill_cache.values() 
            if hasattr(emb, 'nbytes')
        ) if self._skill_cache else 0
        
        return {
            'cached_embeddings': len(self._skill_cache),
            'cache_size_mb': cache_size_bytes / 1024 / 1024
        }