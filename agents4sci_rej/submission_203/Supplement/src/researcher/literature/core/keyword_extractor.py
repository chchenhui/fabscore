"""
LLM-powered keyword extraction for academic literature search.

This module provides intelligent keyword extraction using Large Language Models
to analyze research queries and extract relevant academic keywords for ArXiv searches.
"""

import asyncio
import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import asdict
from loguru import logger

from ..models.query_models import KeywordResult, SearchMode
from ..config.literature_config import get_config
from ..utils.text_utils import clean_text, extract_technical_terms, normalize_keywords
from ..utils.validation_utils import validate_keyword_list, validate_confidence_score

# OneSim ModelManager integration
from onesim.models import get_model_manager, get_model


class KeywordExtractor:
    """
    LLM-powered keyword extraction for academic literature search.
    
    Uses Large Language Models to intelligently extract and categorize keywords
    from research queries, enabling more effective literature searches.
    """
    
    def __init__(self, model_config_name: str = None):
        """
        Initialize keyword extractor.
        
        Args:
            model_config_name: OneSim model configuration name (optional)
        """
        self.model_config_name = model_config_name
        self.model = None
        if model_config_name:
            try:
                self.model = get_model(config_name=model_config_name)
            except Exception as e:
                logger.warning(f"Failed to initialize model {model_config_name}: {e}")
        
        self.config = get_config()
        self._fallback_enabled = True
    
    async def extract_keywords(
        self,
        text: str,
        domain_hint: Optional[str] = None,
        mode: SearchMode = SearchMode.COMPREHENSIVE
    ) -> KeywordResult:
        """
        Extract research keywords using LLM reasoning.
        
        Args:
            text: Research question or topic description
            domain_hint: Optional research domain context
            mode: Extraction strategy (focused/comprehensive/exploratory)
            
        Returns:
            KeywordResult with extracted keywords and search strategies
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        text = clean_text(text)
        
        logger.info(f"Extracting keywords from: '{text[:100]}...' (mode: {mode.value})")
        
        try:
            # Try LLM extraction first
            if self.model is not None:
                result = await self._llm_extract_keywords(text, domain_hint, mode)
                
                # Validate LLM result
                if self._validate_extraction_result(result):
                    logger.info(f"LLM extraction successful (confidence: {result.confidence_score:.2f})")
                    return result
                else:
                    logger.warning("LLM extraction validation failed, falling back to rule-based")
            
            # Fallback to rule-based extraction
            if self._fallback_enabled:
                result = await self._fallback_extraction(text, domain_hint, mode)
                logger.info("Using fallback keyword extraction")
                return result
            else:
                raise RuntimeError("LLM extraction failed and fallback is disabled")
                
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            
            if self._fallback_enabled:
                logger.info("Attempting emergency fallback extraction")
                return await self._emergency_fallback(text, domain_hint)
            else:
                raise
    
    async def _llm_extract_keywords(
        self,
        text: str,
        domain_hint: Optional[str],
        mode: SearchMode
    ) -> KeywordResult:
        """Extract keywords using LLM."""
        # Build domain-aware prompt
        prompt = self._build_extraction_prompt(text, domain_hint, mode)
        
        try:
            # Generate response using OneSim's Model
            from onesim.models import SystemMessage, UserMessage
            response = self.model(self.model.format(
                SystemMessage(content="You are an expert research assistant specializing in academic literature search."),
                UserMessage(content=prompt)
            ))
            
            # Parse structured response
            result = self._parse_extraction_response(response.text, domain_hint)
            
            # Post-process and validate
            result = await self._post_process_keywords(result, domain_hint, text)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            raise
    
    def _build_extraction_prompt(
        self,
        text: str,
        domain_hint: Optional[str],
        mode: SearchMode
    ) -> str:
        """Build context-aware extraction prompt."""
        
        # Get base prompt template
        base_prompt = self.config.get_prompt_template(mode.value, text, domain_hint or "")
        
        # Add domain-specific context
        domain_context = ""
        if domain_hint:
            categories = self.config.get_domain_categories(domain_hint)
            if categories:
                # Add enhanced domain-specific keyword suggestions
                domain_keywords = self._get_domain_specific_keywords(domain_hint)
                keyword_suggestions = ""
                if domain_keywords:
                    keyword_suggestions = f"""
Suggested {domain_hint} keywords: {', '.join(domain_keywords)}
"""
                
                domain_context = f"""
Research Domain: {domain_hint}
Relevant ArXiv Categories: {', '.join(categories)}{keyword_suggestions}

When extracting keywords, prioritize terms that would commonly appear in {domain_hint} papers and consider the domain-specific terminology above.
"""
        
        # Mode-specific instructions
        mode_instructions = {
            SearchMode.FOCUSED: """
Extract 5-8 highly specific, technical keywords that precisely target the core research question.
Focus on:
- Technical terminology that would appear in paper titles
- Specific methods, algorithms, or approaches  
- Domain-specific concepts and jargon
- Exact technical terms researchers would use
""",
            SearchMode.COMPREHENSIVE: """
Extract 10-15 keywords for comprehensive literature coverage.
Include:
- Core technical terms (primary keywords: 5-8 terms)
- Related concepts and methods (secondary keywords: 5-7 terms)
- Alternative phrasings and synonyms
- Broader conceptual terms
- Interdisciplinary connections
""",
            SearchMode.EXPLORATORY: """
Extract 15-20 keywords for exploratory literature discovery.
Include diverse terms for discovery:
- Core technical terms
- Emerging and cutting-edge terminology  
- Cross-disciplinary connections
- Alternative approaches and methodologies
- Related applications and use cases
- Broader research contexts
"""
        }
        
        # Construct full prompt
        full_prompt = f"""You are an expert research assistant specializing in academic literature search. Your task is to extract keywords for searching ArXiv papers.

{domain_context}

Research Query: "{text}"

{mode_instructions[mode]}

IMPORTANT GUIDELINES:
1. Keywords should be terms that would appear in academic paper titles, abstracts, or keyword lists
2. Prioritize technical terminology over common language
3. Include both specific terms and broader conceptual terms
4. Consider synonyms and alternative phrasings
5. Think about how researchers in this field would describe their work

Generate 2-3 optimized ArXiv search strings using Boolean operators (AND, OR) and field prefixes:
- ti: for title searches
- abs: for abstract searches  
- all: for general searches
- cat: for category filters

Respond ONLY with valid JSON in this exact format:
{{
    "primary_keywords": ["keyword1", "keyword2", "keyword3"],
    "secondary_keywords": ["keyword4", "keyword5", "keyword6"],
    "domain_category": "specific_domain",
    "search_queries": ["ti:(query1)", "abs:(query2)", "all:query3"],
    "confidence_score": 0.85,
    "reasoning": "Brief explanation of keyword selection strategy",
    "suggested_fields": ["title", "abstract"]
}}

Do not include any text before or after the JSON response."""
        
        return full_prompt
    
    def _get_domain_specific_keywords(self, domain_hint: str) -> List[str]:
        """Get domain-specific keyword suggestions to enhance LLM extraction."""
        domain_lower = domain_hint.lower().strip()
        
        # Domain-specific keyword banks
        domain_keyword_banks = {
            # Agent-based modeling and simulation
            "agent-based modeling": [
                "multi-agent systems", "agent-based simulation", "complex systems", 
                "emergent behavior", "agent interactions", "cellular automata",
                "NetLogo", "MASON", "social simulation", "computational modeling"
            ],
            "llm-based agent simulation": [
                "large language models", "LLM agents", "conversational agents",
                "chatbot simulation", "natural language agents", "GPT agents",
                "language model reasoning", "agent communication", "dialogue systems",
                "cognitive architectures", "AI assistants"
            ],
            "multi-agent systems": [
                "distributed AI", "cooperative agents", "agent coordination",
                "swarm intelligence", "consensus algorithms", "game theory",
                "mechanism design", "auction theory", "multi-robot systems"
            ],
            "social simulation": [
                "computational social science", "opinion dynamics", "cultural evolution",
                "social networks", "collective behavior", "social influence",
                "population dynamics", "demographic modeling", "epidemiological modeling"
            ],
            "agent-based social simulation": [
                "social agents", "cultural dynamics", "opinion formation",
                "social influence networks", "cultural transmission", "norm emergence",
                "social evolution", "collective decision making"
            ],
            "cultural dynamics": [
                "cultural evolution", "cultural transmission", "social learning",
                "cultural diffusion", "norm dynamics", "belief propagation",
                "cultural diversity", "language evolution", "memetics"
            ],
            "social dynamics modeling": [
                "social network analysis", "influence maximization", "viral spreading",
                "information cascades", "echo chambers", "polarization dynamics",
                "social contagion", "behavioral modeling"
            ],
            "computational social science": [
                "digital sociology", "computational sociology", "social computing",
                "behavioral analytics", "social media analysis", "crowd behavior",
                "collective intelligence", "social data mining"
            ],
            # General AI and ML terms that overlap
            "artificial intelligence": [
                "machine learning", "deep learning", "neural networks",
                "reinforcement learning", "natural language processing"
            ],
            "machine learning": [
                "supervised learning", "unsupervised learning", "deep learning",
                "neural networks", "feature learning", "representation learning"
            ]
        }
        
        # Try exact match first
        if domain_lower in domain_keyword_banks:
            return domain_keyword_banks[domain_lower]
        
        # Try partial matches for compound domains
        for domain_key, keywords in domain_keyword_banks.items():
            if any(term in domain_lower for term in domain_key.split()) or \
               any(term in domain_key for term in domain_lower.split()):
                return keywords
        
        return []
    
    def _parse_extraction_response(
        self,
        response: str,
        domain_hint: Optional[str]
    ) -> KeywordResult:
        """Parse LLM response into structured KeywordResult."""
        
        try:
            # Clean response text
            response = response.strip()
            
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            
            json_str = json_match.group()
            
            # Parse JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                json_str = self._fix_common_json_issues(json_str)
                data = json.loads(json_str)
            
            # Extract fields with defaults
            result = KeywordResult(
                primary_keywords=data.get("primary_keywords", []),
                secondary_keywords=data.get("secondary_keywords", []),
                domain_category=data.get("domain_category"),
                search_queries=data.get("search_queries", []),
                confidence_score=float(data.get("confidence_score", 0.5)),
                extraction_reasoning=data.get("reasoning"),
                suggested_fields=data.get("suggested_fields", ["title", "abstract"])
            )
            
            return result
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response content: {response[:500]}")
            raise ValueError(f"Invalid LLM response format: {e}")
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues in LLM responses."""
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix unescaped quotes in strings
        json_str = re.sub(r'(?<!\\)"(?=[^,\]\}]*[^",\]\}]")', r'\\"', json_str)
        
        # Remove comments (sometimes LLMs add them)
        json_str = re.sub(r'//.*?\n', '\n', json_str)
        
        return json_str
    
    async def _post_process_keywords(
        self,
        result: KeywordResult,
        domain_hint: Optional[str],
        original_text: str
    ) -> KeywordResult:
        """Refine and validate extracted keywords."""
        
        # Normalize keywords
        result.primary_keywords = normalize_keywords(result.primary_keywords)
        result.secondary_keywords = normalize_keywords(result.secondary_keywords)
        
        # Remove duplicates while preserving order
        result.primary_keywords = list(dict.fromkeys(result.primary_keywords))
        result.secondary_keywords = list(dict.fromkeys(result.secondary_keywords))
        
        # Remove secondary keywords that are already in primary
        result.secondary_keywords = [
            kw for kw in result.secondary_keywords 
            if kw.lower() not in [pk.lower() for pk in result.primary_keywords]
        ]
        
        # Limit keyword counts based on configuration
        max_primary = self.config.llm.max_keywords // 2
        max_secondary = self.config.llm.max_keywords - len(result.primary_keywords[:max_primary])
        
        result.primary_keywords = result.primary_keywords[:max_primary]
        result.secondary_keywords = result.secondary_keywords[:max_secondary]
        
        # Validate search queries or generate defaults
        if not result.search_queries or not all(result.search_queries):
            result.search_queries = self._generate_default_queries(result)
        
        # Set domain category if not provided
        if not result.domain_category and domain_hint:
            result.domain_category = domain_hint
        
        # Adjust confidence score based on quality indicators
        result.confidence_score = self._adjust_confidence_score(
            result, original_text
        )
        
        return result
    
    def _generate_default_queries(self, result: KeywordResult) -> List[str]:
        """Generate fallback search queries from keywords."""
        queries = []
        
        if result.primary_keywords:
            # Primary keyword query (title focus)
            primary_query = " AND ".join(f'"{kw}"' for kw in result.primary_keywords[:3])
            queries.append(f"ti:({primary_query})")
            
            # Broader primary query (all fields)
            broader_query = " OR ".join(f'"{kw}"' for kw in result.primary_keywords[:5])
            queries.append(f"all:({broader_query})")
        
        if result.secondary_keywords:
            # Secondary keyword query (abstract focus)
            secondary_query = " OR ".join(f'"{kw}"' for kw in result.secondary_keywords[:5])
            queries.append(f"abs:({secondary_query})")
        
        # Ensure we have at least one query
        if not queries and result.all_keywords:
            fallback_query = " OR ".join(f'"{kw}"' for kw in result.all_keywords[:3])
            queries.append(f"all:({fallback_query})")
        
        return queries
    
    def _adjust_confidence_score(
        self,
        result: KeywordResult,
        original_text: str
    ) -> float:
        """Adjust confidence score based on extraction quality indicators."""
        
        base_confidence = result.confidence_score
        
        # Factors that increase confidence
        confidence_boosts = 0.0
        
        # Good keyword count
        total_keywords = len(result.primary_keywords) + len(result.secondary_keywords)
        if 8 <= total_keywords <= 15:
            confidence_boosts += 0.1
        
        # Has technical terms
        technical_terms = extract_technical_terms(original_text)
        if technical_terms and any(
            term.lower() in kw.lower() 
            for kw in result.all_keywords 
            for term, _ in technical_terms[:5]
        ):
            confidence_boosts += 0.15
        
        # Has domain category
        if result.domain_category:
            confidence_boosts += 0.1
        
        # Has good search queries
        if result.search_queries and len(result.search_queries) >= 2:
            confidence_boosts += 0.1
        
        # Factors that decrease confidence
        confidence_penalties = 0.0
        
        # Too few keywords
        if total_keywords < 5:
            confidence_penalties += 0.2
        
        # Too many very short keywords
        short_keywords = [kw for kw in result.all_keywords if len(kw) < 4]
        if len(short_keywords) > len(result.all_keywords) * 0.5:
            confidence_penalties += 0.1
        
        # Adjust confidence
        adjusted_confidence = base_confidence + confidence_boosts - confidence_penalties
        
        # Clamp to valid range
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _validate_extraction_result(self, result: KeywordResult) -> bool:
        """Validate extraction result quality."""
        
        # Check basic structure
        if not result or not result.primary_keywords:
            return False
        
        # Check confidence score
        if not validate_confidence_score(result.confidence_score)[0]:
            return False
        
        # Check keyword list validity
        all_keywords = result.primary_keywords + result.secondary_keywords
        is_valid, _ = validate_keyword_list(all_keywords, min_count=3, max_count=25)
        if not is_valid:
            return False
        
        # Check for too many very short or generic keywords
        generic_terms = {'paper', 'study', 'research', 'analysis', 'method', 'approach'}
        generic_count = sum(1 for kw in all_keywords if kw.lower() in generic_terms)
        if generic_count > len(all_keywords) * 0.4:
            return False
        
        # Check if confidence is reasonable
        if result.confidence_score < self.config.llm.confidence_threshold:
            return False
        
        return True
    
    async def _fallback_extraction(
        self,
        text: str,
        domain_hint: Optional[str],
        mode: SearchMode
    ) -> KeywordResult:
        """Rule-based fallback keyword extraction with domain-specific enhancements."""
        
        logger.info("Using rule-based keyword extraction")
        
        # Extract technical terms using pattern matching
        technical_terms = extract_technical_terms(text, min_freq=1)
        
        # Extract phrases
        from ..utils.text_utils import extract_phrases
        phrases = extract_phrases(text, min_length=2, max_length=4)
        
        # Get domain-specific keywords if available
        domain_keywords = []
        if domain_hint:
            domain_keywords = self._get_domain_specific_keywords(domain_hint)
        
        # Combine and rank terms
        all_terms = []
        
        # Add technical terms with high priority
        for term, freq in technical_terms:
            all_terms.append((term, freq * 2))  # Boost technical terms
        
        # Add domain-specific terms found in text (with higher priority)
        for domain_kw in domain_keywords:
            if domain_kw.lower() in text.lower():
                all_terms.append((domain_kw, 3))  # Boost domain-specific terms even more
        
        # Add phrases
        for phrase in phrases:
            if phrase not in [term for term, _ in technical_terms]:
                all_terms.append((phrase, 1))
        
        # Sort by score
        all_terms.sort(key=lambda x: x[1], reverse=True)
        
        # Split into primary and secondary based on mode
        if mode == SearchMode.FOCUSED:
            max_primary, max_secondary = 5, 3
        elif mode == SearchMode.COMPREHENSIVE:
            max_primary, max_secondary = 7, 8
        else:  # EXPLORATORY
            max_primary, max_secondary = 8, 12
        
        # Extract keywords
        primary_keywords = [term for term, _ in all_terms[:max_primary]]
        secondary_keywords = [term for term, _ in all_terms[max_primary:max_primary + max_secondary]]
        
        # Generate search queries
        search_queries = self._generate_default_queries(KeywordResult(
            primary_keywords=primary_keywords,
            secondary_keywords=secondary_keywords,
            search_queries=[]
        ))
        
        return KeywordResult(
            primary_keywords=primary_keywords,
            secondary_keywords=secondary_keywords,
            domain_category=domain_hint,
            search_queries=search_queries,
            confidence_score=0.4,  # Lower confidence for fallback
            extraction_reasoning="Rule-based fallback extraction using technical term detection",
            suggested_fields=["title", "abstract"]
        )
    
    async def _emergency_fallback(
        self,
        text: str,
        domain_hint: Optional[str]
    ) -> KeywordResult:
        """Emergency fallback with minimal keyword extraction."""
        
        logger.warning("Using emergency fallback keyword extraction")
        
        # Simple word extraction
        words = clean_text(text).split()
        
        # Filter out common stop words and short words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'can', 'this', 'that', 'these', 'those', 'how', 'what', 'when', 'where'
        }
        
        keywords = []
        for word in words:
            word = word.strip('.,!?;:"()[]{}').lower()
            if len(word) > 3 and word not in stop_words:
                keywords.append(word)
        
        # Remove duplicates and limit
        keywords = list(dict.fromkeys(keywords))[:8]
        
        if not keywords:
            keywords = ["machine learning"]  # Ultimate fallback
        
        return KeywordResult(
            primary_keywords=keywords[:4],
            secondary_keywords=keywords[4:],
            domain_category=domain_hint,
            search_queries=[f'all:({" OR ".join(keywords[:3])})'],
            confidence_score=0.2,  # Very low confidence
            extraction_reasoning="Emergency fallback - simple word extraction",
            suggested_fields=["all"]
        )
    
    async def extract_keywords_batch(
        self,
        texts: List[str],
        domain_hints: Optional[List[str]] = None,
        mode: SearchMode = SearchMode.COMPREHENSIVE
    ) -> List[KeywordResult]:
        """
        Extract keywords from multiple texts in batch.
        
        Args:
            texts: List of research queries
            domain_hints: Optional list of domain hints (same length as texts)
            mode: Extraction mode to use for all texts
            
        Returns:
            List of KeywordResult objects
        """
        if not texts:
            return []
        
        if domain_hints and len(domain_hints) != len(texts):
            raise ValueError("domain_hints must have same length as texts")
        
        tasks = []
        for i, text in enumerate(texts):
            domain_hint = domain_hints[i] if domain_hints else None
            task = self.extract_keywords(text, domain_hint, mode)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to emergency fallback results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch extraction failed for text {i}: {result}")
                domain_hint = domain_hints[i] if domain_hints else None
                fallback_result = await self._emergency_fallback(texts[i], domain_hint)
                final_results.append(fallback_result)
            else:
                final_results.append(result)
        
        return final_results


# Mock ModelManager for testing when OneSim is not available
class MockModelManager:
    """Mock ModelManager for testing purposes."""
    
    async def generate_response(
        self,
        prompt: str,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """Generate mock response."""
        # Note: parameters are unused in mock, but preserved for interface compatibility
        _ = model_name, temperature, max_tokens
        
        # Simple mock response based on prompt content
        if "machine learning" in prompt.lower():
            return '''
{
    "primary_keywords": ["machine learning", "deep learning", "neural networks"],
    "secondary_keywords": ["artificial intelligence", "pattern recognition", "data mining"],
    "domain_category": "computer science",
    "search_queries": ["ti:(machine learning)", "abs:(deep learning OR neural networks)"],
    "confidence_score": 0.8,
    "reasoning": "Extracted core ML terms and related concepts",
    "suggested_fields": ["title", "abstract"]
}
'''
        else:
            return '''
{
    "primary_keywords": ["research", "analysis", "study"],
    "secondary_keywords": ["method", "approach", "technique"],
    "domain_category": "general",
    "search_queries": ["all:(research analysis)"],
    "confidence_score": 0.6,
    "reasoning": "Generic research terms extracted",
    "suggested_fields": ["all"]
}
'''


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Use real model for testing
        extractor = KeywordExtractor("chat_load_balancer")
        
        test_queries = [
            "How can transformers be applied to time series forecasting?",
            "Deep learning approaches for natural language processing",
            "Quantum machine learning algorithms for optimization"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                result = await extractor.extract_keywords(
                    query,
                    domain_hint="machine learning",
                    mode=SearchMode.COMPREHENSIVE
                )
                
                print(f"Primary keywords: {result.primary_keywords}")
                print(f"Secondary keywords: {result.secondary_keywords}")
                print(f"Search queries: {result.search_queries}")
                print(f"Confidence: {result.confidence_score:.2f}")
                
            except Exception as e:
                print(f"Error: {e}")
    
    # Run the example
    asyncio.run(main())