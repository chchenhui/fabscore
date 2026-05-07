"""
AI Planning Agent for Commercial Forecast System.

This module implements an intelligent planning agent that can:
1. Parse natural language queries about pharmaceutical opportunities
2. Create structured execution plans  
3. Reason about appropriate parameters based on drug characteristics
4. Explain its decision-making process
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


class DrugType(Enum):
    """Supported drug types for parameter estimation."""
    BIOLOGIC = "biologic"
    SMALL_MOLECULE = "small_molecule" 
    BIOSIMILAR = "biosimilar"
    GENE_THERAPY = "gene_therapy"


class IndicationArea(Enum):
    """Supported therapeutic areas."""
    RESPIRATORY = "respiratory"
    ONCOLOGY = "oncology"
    IMMUNOLOGY = "immunology" 
    RARE_DISEASE = "rare_disease"
    CNS = "cns"


@dataclass
class DrugCharacteristics:
    """Structured representation of drug characteristics."""
    name: str
    drug_type: DrugType
    indication: str
    indication_area: IndicationArea
    severity: str  # "mild", "moderate", "severe" 
    patient_population: str  # "adult", "pediatric", "all"
    competitive_position: str  # "first_in_class", "best_in_class", "me_too"


@dataclass
class AnalysisPlan:
    """Structured analysis plan created by AI."""
    query: str
    drug_characteristics: DrugCharacteristics
    steps: List[str]
    reasoning: List[str]
    estimated_parameters: Dict[str, Any]
    confidence_level: str  # "high", "medium", "low"


class CommercialForecastPlanner:
    """AI Planning Agent for pharmaceutical commercial forecasting."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the planning agent."""
        self.llm = ChatAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-sonnet-20240229",
            temperature=0.1  # Low temperature for consistent reasoning
        )
        
        # Knowledge base for parameter estimation
        self.parameter_knowledge = {
            "biologic": {
                "bass_p": (0.04, 0.08, "Biologics typically have moderate early adopter rates"),
                "bass_q": (0.40, 0.60, "Strong word-of-mouth effect due to efficacy"),
                "pricing_multiple": (3.0, 8.0, "Premium pricing vs conventional therapy"),
                "cogs_pct": (0.10, 0.15, "Lower manufacturing costs than small molecules")
            },
            "respiratory": {
                "market_growth": (0.05, 0.08, "Aging population drives growth"),
                "regulatory_timeline": (8, 12, "Standard FDA approval timeline in years"),
                "access_tier": ("PA", "Typically requires prior authorization")
            },
            "severe": {
                "market_size_factor": (0.05, "Severe = ~5% of total indication population"),
                "adherence_rate": (0.80, 0.90, "Better adherence in severe disease"),
                "price_tolerance": ("high", "Higher willingness to pay for severe conditions")
            }
        }

    def parse_query(self, query: str) -> DrugCharacteristics:
        """Extract drug characteristics from natural language query."""
        
        system_prompt = """You are an expert pharmaceutical analyst. Extract key drug characteristics from the user's query.

        Look for:
        - Drug type (biologic, small molecule, biosimilar, gene therapy)
        - Indication area (respiratory, oncology, immunology, rare disease, CNS)
        - Severity (mild, moderate, severe)
        - Patient population (adult, pediatric, all)
        - Competitive position (first-in-class, best-in-class, me-too)
        
        If not explicitly stated, make reasonable inferences based on context.
        
        Respond in this exact format:
        NAME: [drug name or "Unknown Drug"]
        TYPE: [biologic/small_molecule/biosimilar/gene_therapy]
        INDICATION: [specific indication]  
        AREA: [respiratory/oncology/immunology/rare_disease/cns]
        SEVERITY: [mild/moderate/severe]
        POPULATION: [adult/pediatric/all]
        POSITION: [first_in_class/best_in_class/me_too]
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]
        
        response = self.llm.invoke(messages)
        return self._parse_characteristics_response(response.content, query)
    
    def _parse_characteristics_response(self, response: str, original_query: str) -> DrugCharacteristics:
        """Parse LLM response into DrugCharacteristics object."""
        
        # Extract fields using regex
        def extract_field(field_name: str) -> str:
            pattern = f"{field_name}:\\s*(.+)"
            match = re.search(pattern, response, re.IGNORECASE)
            return match.group(1).strip() if match else "unknown"
        
        return DrugCharacteristics(
            name=extract_field("NAME"),
            drug_type=DrugType(extract_field("TYPE").lower()),
            indication=extract_field("INDICATION"),
            indication_area=IndicationArea(extract_field("AREA").lower()),
            severity=extract_field("SEVERITY").lower(),
            patient_population=extract_field("POPULATION").lower(),
            competitive_position=extract_field("POSITION").lower()
        )
    
    def estimate_parameters(self, characteristics: DrugCharacteristics) -> Dict[str, Any]:
        """Estimate model parameters based on drug characteristics."""
        
        params = {}
        reasoning = []
        
        # Bass diffusion parameters
        drug_type_key = characteristics.drug_type.value
        if drug_type_key in self.parameter_knowledge:
            bass_p_range = self.parameter_knowledge[drug_type_key]["bass_p"]
            bass_q_range = self.parameter_knowledge[drug_type_key]["bass_q"]
            
            # Use middle of range as estimate
            params["bass_p"] = (bass_p_range[0] + bass_p_range[1]) / 2
            params["bass_q"] = (bass_q_range[0] + bass_q_range[1]) / 2
            
            reasoning.append(f"Bass p={params['bass_p']:.3f}: {bass_p_range[2]}")
            reasoning.append(f"Bass q={params['bass_q']:.3f}: {bass_q_range[2]}")
        
        # Market sizing based on severity
        if characteristics.severity == "severe":
            params["market_size_factor"] = self.parameter_knowledge["severe"]["market_size_factor"]
            reasoning.append(f"Market size factor: {params['market_size_factor']:.1%} - {self.parameter_knowledge['severe']['market_size_factor']}")
        
        # Pricing estimation
        if characteristics.indication_area == IndicationArea.RESPIRATORY and characteristics.drug_type == DrugType.BIOLOGIC:
            # Base on Tezspire-like pricing
            if characteristics.patient_population == "pediatric":
                params["list_price"] = 4000  # Slightly lower for pediatric
                reasoning.append("Pricing: $4,000/month - pediatric typically 10% below adult pricing")
            else:
                params["list_price"] = 4369  # Tezspire actual
                reasoning.append("Pricing: $4,369/month - based on Tezspire benchmark")
        
        # Market size estimation
        if characteristics.indication_area == IndicationArea.RESPIRATORY:
            base_asthma_population = 27_000_000  # US total asthma
            if characteristics.severity == "severe":
                if characteristics.patient_population == "pediatric":
                    params["market_size"] = int(base_asthma_population * 0.05 * 0.1)  # 5% severe, 10% pediatric
                    reasoning.append("Market size: ~135K patients (27M asthma × 5% severe × 10% pediatric)")
                else:
                    params["market_size"] = int(base_asthma_population * 0.05 * 0.9)  # 5% severe, 90% adult  
                    reasoning.append("Market size: ~1.2M patients (27M asthma × 5% severe × 90% adult)")
        
        params["reasoning"] = reasoning
        return params
    
    def create_analysis_plan(self, query: str) -> AnalysisPlan:
        """Create a comprehensive analysis plan from natural language query."""
        
        # Step 1: Parse drug characteristics
        characteristics = self.parse_query(query)
        
        # Step 2: Estimate parameters
        estimated_params = self.estimate_parameters(characteristics)
        reasoning = estimated_params.pop("reasoning", [])
        
        # Step 3: Create execution steps
        steps = [
            "🎯 Market Sizing: Estimate addressable patient population",
            "📈 Adoption Modeling: Calibrate Bass diffusion parameters", 
            "💰 Pricing Strategy: Determine optimal price point and access tier",
            "🧮 Financial Analysis: Calculate NPV with Monte Carlo uncertainty",
            "⚖️ Investment Decision: Synthesize results into go/no-go recommendation"
        ]
        
        # Step 4: Assess confidence
        confidence = self._assess_confidence(characteristics)
        
        return AnalysisPlan(
            query=query,
            drug_characteristics=characteristics,
            steps=steps,
            reasoning=reasoning,
            estimated_parameters=estimated_params,
            confidence_level=confidence
        )
    
    def _assess_confidence(self, characteristics: DrugCharacteristics) -> str:
        """Assess confidence level based on available data and precedents."""
        
        confidence_score = 0
        
        # Higher confidence for well-established areas
        if characteristics.indication_area == IndicationArea.RESPIRATORY:
            confidence_score += 2
        
        # Higher confidence for biologics (more precedents)
        if characteristics.drug_type == DrugType.BIOLOGIC:
            confidence_score += 2
            
        # Lower confidence for pediatric (less data)
        if characteristics.patient_population == "pediatric":
            confidence_score -= 1
        
        # Higher confidence for severe diseases (clearer value prop)
        if characteristics.severity == "severe":
            confidence_score += 1
            
        if confidence_score >= 3:
            return "high"
        elif confidence_score >= 1:
            return "medium"  
        else:
            return "low"


# Example usage for testing
if __name__ == "__main__":
    planner = CommercialForecastPlanner()
    
    # Test query
    query = "Should we develop a Tezspire competitor for pediatric severe asthma?"
    plan = planner.create_analysis_plan(query)
    
    print("=== AI PLANNING AGENT DEMO ===")
    print(f"Query: {plan.query}")
    print(f"Drug: {plan.drug_characteristics.name}")
    print(f"Confidence: {plan.confidence_level}")
    print("\nEstimated Parameters:")
    for key, value in plan.estimated_parameters.items():
        print(f"  {key}: {value}")
    print("\nReasoning:")
    for reason in plan.reasoning:
        print(f"  • {reason}")
    print("\nExecution Steps:")
    for i, step in enumerate(plan.steps, 1):
        print(f"  {i}. {step}")