"""
Evidence Grounding System for Pharmaceutical AI Agent
Links every AI decision to authoritative sources for anti-hallucination

Following Linus principle: "Good code has no special cases"
Data structures: GroundedClaim is the core concept
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json
import os
from datetime import datetime

# Multi-LLM integration for source analysis
try:
    from .model_router import get_router, TaskType
except ImportError:
    try:
        from model_router import get_router, TaskType
    except ImportError:
        print("Warning: ModelRouter not available, evidence grounding will use fallback mode")
        get_router = None

class SourceType(Enum):
    """Types of pharmaceutical evidence sources"""
    FDA_LABEL = "fda_label"
    SEC_FILING = "sec_filing" 
    CLINICAL_TRIAL = "clinical_trial"
    EPIDEMIOLOGY = "epidemiology_study"
    MARKET_RESEARCH = "market_research"
    REGULATORY_GUIDANCE = "regulatory_guidance"
    EXPERT_OPINION = "expert_opinion"

@dataclass
class EvidenceSource:
    """Single piece of supporting evidence"""
    source_id: str
    source_type: SourceType
    title: str
    url: Optional[str]
    excerpt: str
    confidence: float  # 0.0-1.0
    date_accessed: str
    relevance_score: float  # How relevant to the claim

@dataclass 
class GroundedClaim:
    """AI claim with supporting evidence sources"""
    claim: str
    claim_type: str  # e.g., "market_size", "ptrs_estimate", "commercial_forecast"
    sources: List[EvidenceSource]
    confidence: float  # Overall confidence after grounding
    reasoning: str     # AI explanation of how sources support claim
    audit_trail: List[str]  # Step-by-step reasoning process
    grounding_quality: str  # "strong", "moderate", "weak"

class EvidenceGroundingAgent:
    """
    Core evidence grounding system for pharmaceutical AI decisions
    
    Principles:
    - Every quantitative claim must have supporting sources
    - Sources are ranked by confidence and relevance
    - AI explains how sources support each claim
    - Complete audit trail for every decision
    """
    
    def __init__(self):
        """Initialize with evidence databases and AI router"""
        self.router = None
        self.evidence_databases = {}
        self.grounding_stats = {
            "claims_grounded": 0,
            "sources_found": 0,
            "average_confidence": 0.0
        }
        
        # Initialize multi-LLM router for source analysis
        if get_router:
            try:
                self.router = get_router()
                print(f"[EVIDENCE GROUNDING] Initialized with AI providers: {list(self.router.providers.keys())}")
            except Exception as e:
                print(f"Warning: Evidence grounding AI failed to initialize: {e}")
        
        # Load pharmaceutical evidence databases
        self._initialize_evidence_databases()
    
    def _initialize_evidence_databases(self):
        """Initialize pharmaceutical evidence sources"""
        
        # Mock pharmaceutical evidence database (in real implementation, would connect to APIs)
        self.evidence_databases = {
            "fda_labels": {
                "dupixent": {
                    "source_id": "FDA_DUPIXENT_2017",
                    "title": "DUPIXENT (dupilumab) injection for subcutaneous use - FDA Label", 
                    "url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2017/761055lbl.pdf",
                    "indication": "atopic dermatitis, asthma, chronic rhinosinusitis",
                    "population": "adults and adolescents",
                    "efficacy": "significant improvement in EASI scores",
                    "safety": "conjunctivitis, injection site reactions"
                },
                "tezspire": {
                    "source_id": "FDA_TEZSPIRE_2021", 
                    "title": "TEZSPIRE (tezepelumab-ekko) injection - FDA Label",
                    "url": "https://www.accessdata.fda.gov/drugsatfda_docs/label/2021/761224lbl.pdf",
                    "indication": "severe asthma",
                    "population": "adults and adolescents 12 years and older", 
                    "efficacy": "reduction in exacerbation rates",
                    "mechanism": "TSLP pathway antagonist"
                }
            },
            "clinical_trials": {
                "CT.GOV.ASTHMA.SEVERE": {
                    "source_id": "CLINTRIALS_SEVERE_ASTHMA_2023",
                    "title": "Severe Asthma Population Estimates from Clinical Trials Registry",
                    "population_size": "~5% of all asthma patients",
                    "geographic_coverage": "US, EU, developed markets",
                    "uncontrolled_rate": "60-70% despite Step 4-5 treatment"
                }
            },
            "epidemiology": {
                "CDC_ASTHMA_2023": {
                    "source_id": "CDC_ASTHMA_SURVEILLANCE_2023",
                    "title": "Asthma Surveillance Data - CDC",
                    "url": "https://www.cdc.gov/asthma/data-visualizations/",
                    "prevalence": "25.2 million adults, 4.6 million children",
                    "severity_distribution": "5.7% severe, 23.1% moderate, 71.2% mild",
                    "geographic_variation": "higher in Northeast, lower in West"
                }
            },
            "market_research": {
                "IQVIA_RESPIRATORY_2024": {
                    "source_id": "IQVIA_RESPIRATORY_FORECAST_2024",
                    "title": "IQVIA Respiratory Market Forecast 2024-2030", 
                    "market_size": "$47.2B global respiratory therapeutics",
                    "growth_rate": "6.8% CAGR through 2030",
                    "biosimilar_impact": "15-25% price erosion for biologics"
                }
            }
        }
        
        print(f"[EVIDENCE DATABASE] Loaded {len(self.evidence_databases)} source categories")
        for db_name, sources in self.evidence_databases.items():
            print(f"  {db_name}: {len(sources)} sources")
    
    def ground_pharmaceutical_claim(self, claim: str, claim_type: str, context: Dict[str, Any] = None) -> GroundedClaim:
        """
        Ground a pharmaceutical AI claim in authoritative sources
        
        Args:
            claim: The claim to be grounded (e.g., "Severe asthma affects 5% of asthma patients")
            claim_type: Type of claim (e.g., "population_estimate", "market_size")
            context: Additional context for source selection
            
        Returns:
            GroundedClaim with supporting evidence and confidence assessment
        """
        
        # Step 1: Find relevant sources
        candidate_sources = self._find_relevant_sources(claim, claim_type, context)
        
        # Step 2: AI analysis of source relevance and confidence
        if self.router and candidate_sources:
            grounded_claim = self._ai_analyze_sources(claim, claim_type, candidate_sources)
        else:
            # Fallback: rule-based source grounding
            grounded_claim = self._fallback_source_grounding(claim, claim_type, candidate_sources)
        
        # Step 3: Update statistics
        self.grounding_stats["claims_grounded"] += 1
        self.grounding_stats["sources_found"] += len(grounded_claim.sources)
        if grounded_claim.sources:
            avg_confidence = sum(s.confidence for s in grounded_claim.sources) / len(grounded_claim.sources)
            self.grounding_stats["average_confidence"] = (
                self.grounding_stats["average_confidence"] * (self.grounding_stats["claims_grounded"] - 1) + 
                avg_confidence
            ) / self.grounding_stats["claims_grounded"]
        
        return grounded_claim
    
    def _find_relevant_sources(self, claim: str, claim_type: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find potentially relevant sources for a claim"""
        
        relevant_sources = []
        
        # Search strategy based on claim type
        search_databases = []
        if claim_type in ["population_estimate", "prevalence"]:
            search_databases = ["epidemiology", "clinical_trials"]
        elif claim_type in ["market_size", "commercial_forecast"]:
            search_databases = ["market_research", "sec_filings"]
        elif claim_type in ["drug_efficacy", "safety_profile"]:
            search_databases = ["fda_labels", "clinical_trials"]
        else:
            # Default: search all databases
            search_databases = list(self.evidence_databases.keys())
        
        # Search for relevant sources (simplified keyword matching)
        claim_keywords = self._extract_keywords(claim)
        
        for db_name in search_databases:
            if db_name in self.evidence_databases:
                for source_key, source_data in self.evidence_databases[db_name].items():
                    relevance_score = self._calculate_relevance(claim_keywords, source_data)
                    if relevance_score > 0.3:  # Minimum relevance threshold
                        relevant_sources.append({
                            "database": db_name,
                            "key": source_key,
                            "data": source_data,
                            "relevance": relevance_score
                        })
        
        # Sort by relevance score
        relevant_sources.sort(key=lambda x: x["relevance"], reverse=True)
        
        return relevant_sources[:5]  # Top 5 most relevant sources
    
    def _extract_keywords(self, claim: str) -> List[str]:
        """Extract key pharmaceutical terms from claim"""
        # Simplified keyword extraction (in production, would use NLP)
        pharma_keywords = [
            "asthma", "severe", "moderate", "mild", "pediatric", "adult",
            "dupixent", "tezspire", "biologic", "efficacy", "safety",
            "population", "prevalence", "market", "forecast", "PTRS"
        ]
        
        claim_lower = claim.lower()
        found_keywords = [kw for kw in pharma_keywords if kw in claim_lower]
        
        return found_keywords
    
    def _calculate_relevance(self, claim_keywords: List[str], source_data: Dict[str, Any]) -> float:
        """Calculate how relevant a source is to the claim"""
        
        # Convert source data to searchable text
        source_text = " ".join(str(v) for v in source_data.values()).lower()
        
        # Count keyword matches
        matches = sum(1 for keyword in claim_keywords if keyword in source_text)
        
        # Normalize by number of keywords
        if claim_keywords:
            relevance = matches / len(claim_keywords)
        else:
            relevance = 0.0
        
        return min(relevance, 1.0)
    
    def _ai_analyze_sources(self, claim: str, claim_type: str, candidate_sources: List[Dict[str, Any]]) -> GroundedClaim:
        """Use AI to analyze source relevance and create grounded claim"""
        
        # Prepare source information for AI analysis
        sources_text = ""
        for i, source in enumerate(candidate_sources, 1):
            source_data = source["data"]
            sources_text += f"\n{i}. [{source['database'].upper()}] {source_data.get('title', 'Unnamed source')}\n"
            sources_text += f"   Data: {json.dumps(source_data, indent=2)}\n"
            sources_text += f"   Relevance: {source['relevance']:.2f}\n"
        
        analysis_prompt = f"""
You are a pharmaceutical research expert analyzing evidence for AI claims.

CLAIM TO EVALUATE: "{claim}"
CLAIM TYPE: {claim_type}

CANDIDATE SOURCES:
{sources_text}

TASK:
1. Assess which sources best support or contradict the claim
2. Rate confidence in each source (0.0-1.0)
3. Determine overall grounding quality (strong/moderate/weak)
4. Provide reasoning explaining how sources relate to claim

OUTPUT FORMAT (JSON):
{{
  "supporting_sources": [
    {{
      "source_index": 1,
      "confidence": 0.85,
      "relevance_score": 0.92,
      "excerpt": "Key data point from source",
      "supports_claim": true
    }}
  ],
  "overall_confidence": 0.78,
  "grounding_quality": "strong",
  "reasoning": "Detailed explanation of how sources support/contradict claim",
  "audit_trail": [
    "Step 1: Identified relevant population data",
    "Step 2: Cross-verified with clinical trial registry",
    "Step 3: Confirmed consistency across sources"
  ]
}}
"""

        try:
            # Try structured JSON output first
            try:
                response = self.router.generate(
                    prompt=analysis_prompt,
                    task_type=TaskType.OBJECTIVE_REVIEW,
                    system_prompt="You are an expert pharmaceutical researcher performing rigorous evidence analysis. ALWAYS return valid JSON.",
                    max_tokens=1500,
                    temperature=1.0,  # GPT-5 compatibility
                    response_format={"type": "json_object"}  # Force structured JSON output
                )
            except Exception as e:
                print(f"[WARNING] Structured output failed ({e}), trying without response_format...")
                # Fallback without structured output
                response = self.router.generate(
                    prompt=analysis_prompt + "\n\nIMPORTANT: Return ONLY valid JSON, no other text.",
                    task_type=TaskType.OBJECTIVE_REVIEW,
                    system_prompt="You are an expert pharmaceutical researcher performing rigorous evidence analysis. ALWAYS return valid JSON.",
                    max_tokens=1500,
                    temperature=1.0
                )
            
            print(f"[DEBUG] Model used: {response.provider} ({response.model_used})")
            print(f"[DEBUG] Tokens: {response.input_tokens}→{response.output_tokens}")
            print(f"[DEBUG] Cost: ${response.cost_cents/100:.4f}")
            print(f"[DEBUG] Response length: {len(response.content)} chars")
            if len(response.content) > 0:
                print(f"[DEBUG] Response preview: '{response.content[:300]}...'")
            else:
                print(f"[DEBUG] EMPTY RESPONSE - Model returned no content!")
                print(f"[DEBUG] Full response object: {vars(response)}")
            
            # Parse AI analysis with better error handling
            try:
                if not response.content or response.content.strip() == "":
                    raise json.JSONDecodeError("Empty response from AI model", "", 0)
                analysis = json.loads(response.content)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON parsing failed: {e}")
                print(f"[ERROR] Raw response: '{response.content}'")
                # Try to extract JSON from response if it's wrapped in markdown
                content = response.content.strip()
                if content.startswith("```json") and content.endswith("```"):
                    content = content[7:-3].strip()
                    analysis = json.loads(content)
                elif content.startswith("```") and content.endswith("```"):
                    content = content[3:-3].strip()
                    analysis = json.loads(content)
                else:
                    raise e
            
            # Create evidence sources from AI analysis
            evidence_sources = []
            for source_analysis in analysis.get("supporting_sources", []):
                idx = source_analysis["source_index"] - 1  # Convert to 0-based
                if 0 <= idx < len(candidate_sources):
                    candidate = candidate_sources[idx]
                    
                    evidence_source = EvidenceSource(
                        source_id=candidate["data"].get("source_id", f"SOURCE_{idx+1}"),
                        source_type=self._determine_source_type(candidate["database"]),
                        title=candidate["data"].get("title", "Unnamed source"),
                        url=candidate["data"].get("url"),
                        excerpt=source_analysis.get("excerpt", ""),
                        confidence=source_analysis["confidence"],
                        date_accessed=datetime.now().strftime("%Y-%m-%d"),
                        relevance_score=source_analysis["relevance_score"]
                    )
                    evidence_sources.append(evidence_source)
            
            # Create grounded claim
            grounded_claim = GroundedClaim(
                claim=claim,
                claim_type=claim_type,
                sources=evidence_sources,
                confidence=analysis["overall_confidence"],
                reasoning=analysis["reasoning"],
                audit_trail=analysis.get("audit_trail", []),
                grounding_quality=analysis["grounding_quality"]
            )
            
            print(f"[AI GROUNDING] {claim_type}: {grounded_claim.grounding_quality} grounding with {len(evidence_sources)} sources")
            return grounded_claim
            
        except Exception as e:
            print(f"[ERROR] AI source analysis failed: {e}")
            # Fallback to rule-based grounding
            return self._fallback_source_grounding(claim, claim_type, candidate_sources)
    
    def _fallback_source_grounding(self, claim: str, claim_type: str, candidate_sources: List[Dict[str, Any]]) -> GroundedClaim:
        """Fallback rule-based source grounding when AI fails"""
        
        evidence_sources = []
        
        # Simple rule-based grounding
        for source in candidate_sources[:3]:  # Use top 3 sources
            evidence_source = EvidenceSource(
                source_id=source["data"].get("source_id", "FALLBACK_SOURCE"),
                source_type=self._determine_source_type(source["database"]),
                title=source["data"].get("title", "Pharmaceutical data source"),
                url=source["data"].get("url"),
                excerpt=f"Relevant data from {source['database']}",
                confidence=min(0.6, source["relevance"]),  # Conservative confidence
                date_accessed=datetime.now().strftime("%Y-%m-%d"),
                relevance_score=source["relevance"]
            )
            evidence_sources.append(evidence_source)
        
        # Determine grounding quality based on source count and relevance
        if evidence_sources and evidence_sources[0].relevance_score > 0.7:
            grounding_quality = "moderate"
            confidence = 0.65
        elif evidence_sources:
            grounding_quality = "weak"
            confidence = 0.45
        else:
            grounding_quality = "ungrounded"
            confidence = 0.2
        
        grounded_claim = GroundedClaim(
            claim=claim,
            claim_type=claim_type,
            sources=evidence_sources,
            confidence=confidence,
            reasoning=f"Fallback grounding based on {len(evidence_sources)} sources",
            audit_trail=[f"Found {len(candidate_sources)} candidate sources", "Applied rule-based grounding"],
            grounding_quality=grounding_quality
        )
        
        print(f"[FALLBACK GROUNDING] {claim_type}: {grounding_quality} with {len(evidence_sources)} sources")
        return grounded_claim
    
    def _determine_source_type(self, database_name: str) -> SourceType:
        """Map database name to SourceType enum"""
        mapping = {
            "fda_labels": SourceType.FDA_LABEL,
            "sec_filings": SourceType.SEC_FILING,
            "clinical_trials": SourceType.CLINICAL_TRIAL,
            "epidemiology": SourceType.EPIDEMIOLOGY,
            "market_research": SourceType.MARKET_RESEARCH
        }
        return mapping.get(database_name, SourceType.EXPERT_OPINION)
    
    def get_grounding_report(self) -> Dict[str, Any]:
        """Generate evidence grounding statistics report"""
        return {
            "claims_processed": self.grounding_stats["claims_grounded"],
            "total_sources_found": self.grounding_stats["sources_found"],
            "average_confidence": round(self.grounding_stats["average_confidence"], 3),
            "available_databases": list(self.evidence_databases.keys()),
            "source_count_by_database": {
                db_name: len(sources) 
                for db_name, sources in self.evidence_databases.items()
            }
        }

def test_evidence_grounding():
    """Test the evidence grounding system"""
    
    agent = EvidenceGroundingAgent()
    
    # Test pharmaceutical claims
    test_claims = [
        ("Severe asthma affects approximately 5% of all asthma patients", "population_estimate"),
        ("DUPIXENT shows significant efficacy in atopic dermatitis", "drug_efficacy"),
        ("The respiratory therapeutics market is valued at $47B globally", "market_size")
    ]
    
    print("=== EVIDENCE GROUNDING TEST ===")
    
    for claim, claim_type in test_claims:
        print(f"\nTesting: {claim}")
        grounded = agent.ground_pharmaceutical_claim(claim, claim_type)
        
        print(f"  Grounding quality: {grounded.grounding_quality}")
        print(f"  Confidence: {grounded.confidence:.2f}")
        print(f"  Sources found: {len(grounded.sources)}")
        
        for i, source in enumerate(grounded.sources, 1):
            print(f"    {i}. {source.title} (confidence: {source.confidence:.2f})")
    
    # Show final report
    print(f"\n=== GROUNDING REPORT ===")
    report = agent.get_grounding_report()
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    test_evidence_grounding()