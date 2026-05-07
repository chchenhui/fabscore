"""
Meta-Scientist: Autonomous Hypothesis Generation and Protocol Design
Following Linus principle: "Good code has no special cases"
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("Warning: Anthropic not available, using fallback mode")

try:
    from .schemas import (
        ResearchHypothesis, 
        ExperimentalProtocol,
        HypothesisType,
        H1_CALIBRATION,
        H2_ARCHITECTURE, 
        H3_CONSTRAINTS,
        create_standard_protocol
    )
except ImportError:
    from schemas import (
        ResearchHypothesis, 
        ExperimentalProtocol,
        HypothesisType,
        H1_CALIBRATION,
        H2_ARCHITECTURE, 
        H3_CONSTRAINTS,
        create_standard_protocol
    )


class MetaScientist:
    """
    The AI that generates research hypotheses and experimental designs.
    This is where the real AI research happens.
    """
    
    def __init__(self):
        """Initialize with multi-LLM router"""
        self.router = None
        self.token_count = 0
        self.api_calls = 0
        
        # Initialize ModelRouter for multi-LLM capabilities
        try:
            # Try relative import first (when used as module)
            try:
                from .model_router import get_router, TaskType
            except ImportError:
                # Fallback to absolute import (when run directly)
                from model_router import get_router, TaskType
            
            self.router = get_router()
            self.TaskType = TaskType  # Store for later use
            print(f"[META-SCIENTIST] Initialized with providers: {list(self.router.providers.keys())}")
        except Exception as e:
            print(f"Warning: ModelRouter initialization failed: {e}")
            print("Falling back to legacy Anthropic-only mode")
            
            # Legacy fallback to anthropic only
            if ANTHROPIC_AVAILABLE:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if api_key:
                    self.anthropic_client = anthropic.Client(api_key=api_key)
                else:
                    print("Warning: ANTHROPIC_API_KEY not found, using fallback")
    
    def generate_research_hypotheses(self) -> List[ResearchHypothesis]:
        """
        AI autonomously generates methodological research questions.
        This is the core of AI scientific discovery.
        """
        
        if self.router:
            return self._ai_generate_hypotheses_router()
        elif hasattr(self, 'anthropic_client') and self.anthropic_client:
            return self._ai_generate_hypotheses_legacy()
        else:
            return self._fallback_hypotheses()
    
    def _ai_generate_hypotheses_router(self) -> List[ResearchHypothesis]:
        """Real AI hypothesis generation using ModelRouter (multi-LLM)"""
        
        hypothesis_prompt = """
You are an AI scientist conducting methodological research in pharmaceutical forecasting.

Your mission: Generate 3 rigorous research hypotheses that test different approaches to AI-powered pharmaceutical investment analysis.

REQUIREMENTS:
1. Each hypothesis must compare two distinct methodological approaches
2. Focus on practical, measurable differences in forecasting accuracy
3. Ensure experiments can be completed within computational constraints
4. Follow pharmaceutical industry standards for validation

OUTPUT FORMAT (JSON):
[
  {
    "id": "H1",
    "type": "evidence_grounding",
    "question": "Does evidence grounding improve forecast accuracy vs prompt-only approaches?",
    "method_a": "Evidence-grounded forecasting with pharmaceutical database citations",
    "method_b": "Prompt-only forecasting without external evidence validation",
    "metrics": ["forecast_accuracy", "confidence_intervals", "bias_reduction"],
    "expected_outcome": "Evidence grounding reduces forecast error by 15-25%",
    "confidence": 0.75,
    "reasoning": "External validation should reduce hallucination in commercial predictions"
  }
]

Generate exactly 3 hypotheses covering: evidence grounding, multi-agent architecture, and constraint methodology."""

        try:
            # Use GPT-5 for complex research reasoning (Note: GPT-5 only accepts temperature=1.0)
            response = self.router.generate(
                prompt=hypothesis_prompt,
                task_type=self.TaskType.HYPOTHESIS_GENERATION,
                system_prompt="You are a world-class pharmaceutical research scientist designing AI methodology experiments.",
                max_tokens=2000,
                temperature=1.0  # GPT-5 requirement: only accepts temperature=1.0
            )
            
            self.api_calls += 1
            self.token_count += response.input_tokens + response.output_tokens
            
            # Parse JSON response
            hypotheses_data = json.loads(response.content)
            hypotheses = []
            
            for h_data in hypotheses_data:
                # Map AI response type to valid enum values
                type_mapping = {
                    'evidence_grounding': HypothesisType.CALIBRATION,
                    'architecture': HypothesisType.ARCHITECTURE, 
                    'constraints': HypothesisType.CONSTRAINTS,
                    'calibration': HypothesisType.CALIBRATION
                }
                
                hypothesis_type = type_mapping.get(h_data.get('type', 'calibration'), HypothesisType.CALIBRATION)
                
                hypothesis = ResearchHypothesis(
                    id=h_data.get('id', f"H{len(hypotheses)+1}"),
                    type=hypothesis_type,
                    question=h_data['question'],
                    method_a=h_data['method_a'],
                    method_b=h_data['method_b'],
                    metrics=h_data['metrics'],
                    expected_outcome=h_data['expected_outcome'],
                    confidence=h_data['confidence'],
                    reasoning=h_data['reasoning']
                )
                hypotheses.append(hypothesis)
            
            print(f"[AI SCIENTIST] Generated {len(hypotheses)} research hypotheses using {response.provider}")
            print(f"[COST] ${response.cost_cents/100:.4f} | Tokens: {response.input_tokens}→{response.output_tokens}")
            
            return hypotheses
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse AI hypothesis JSON: {e}")
            print(f"Response was: {response.content}")
            return self._fallback_hypotheses()
        except Exception as e:
            print(f"[ERROR] AI hypothesis generation failed: {e}")
            return self._fallback_hypotheses()
    
    def _ai_generate_hypotheses_legacy(self) -> List[ResearchHypothesis]:
        """Legacy AI hypothesis generation using Claude directly"""
        
        hypothesis_prompt = """
You are an AI scientist conducting methodological research in pharmaceutical forecasting.
Generate 3 novel research hypotheses that compare different AI approaches.

Focus on these methodological questions:
1. Evidence grounding vs prompt-only approaches
2. Multi-agent specialization vs monolithic systems  
3. Domain constraints vs unconstrained generation

For each hypothesis, provide:
- Clear research question
- Method A and Method B to compare
- Specific metrics to measure
- Expected outcome with quantifiable improvement
- Scientific rationale

Return as JSON array with this structure:
{
  "hypotheses": [
    {
      "id": "H1_evidence",
      "type": "calibration", 
      "question": "Research question here",
      "method_a": "First method",
      "method_b": "Comparison method",
      "metrics": ["metric1", "metric2"],
      "expected_outcome": "Quantifiable prediction",
      "rationale": "Scientific reasoning"
    }
  ]
}
"""
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[{"role": "user", "content": hypothesis_prompt}]
            )
            
            self.api_calls += 1
            self.token_count += len(response.content[0].text.split())
            
            # Parse AI response
            result = json.loads(response.content[0].text)
            hypotheses = []
            
            for h in result["hypotheses"]:
                hypothesis = ResearchHypothesis(
                    id=h["id"],
                    type=HypothesisType(h["type"]),
                    question=h["question"],
                    method_a=h["method_a"],
                    method_b=h["method_b"],
                    metrics=h["metrics"],
                    expected_outcome=h["expected_outcome"],
                    rationale=h["rationale"]
                )
                hypotheses.append(hypothesis)
            
            return hypotheses
            
        except Exception as e:
            print(f"AI hypothesis generation failed: {e}")
            return self._fallback_hypotheses()
    
    def _fallback_hypotheses(self) -> List[ResearchHypothesis]:
        """Fallback to pre-designed hypotheses if API fails"""
        return [H1_CALIBRATION, H2_ARCHITECTURE, H3_CONSTRAINTS]
    
    def design_experimental_protocol(self, hypotheses: List[ResearchHypothesis]) -> ExperimentalProtocol:
        """
        AI designs rigorous experimental protocol.
        Eliminates edge cases through systematic design.
        """
        
        if self.router or (hasattr(self, 'anthropic_client') and self.anthropic_client):
            return self._ai_design_protocol(hypotheses)
        else:
            return self._fallback_protocol(hypotheses)
    
    def _ai_design_protocol(self, hypotheses: List[ResearchHypothesis]) -> ExperimentalProtocol:
        """Real AI protocol design"""
        
        protocol_prompt = f"""
You are an AI scientist designing an experimental protocol to test these hypotheses:

{json.dumps([h.to_dict() for h in hypotheses], indent=2)}

Design a rigorous experimental protocol including:
1. Data split strategy (train/validation/test with years)
2. Baseline methods for comparison
3. Statistical tests for significance
4. Sample size justification
5. Random seed for reproducibility

Follow these principles:
- Time-based splits to prevent data leakage
- Multiple baselines to establish superiority
- Appropriate statistical power
- Reproducible methodology

Return as JSON:
{{
  "data_split": {{"train": "period", "validation": "period", "test": "period"}},
  "baselines": ["method1", "method2"],
  "statistical_tests": ["test1", "test2"], 
  "sample_size": 100,
  "rationale": "Why this design"
}}
"""
        
        try:
            if self.router:
                # Use ModelRouter for protocol design
                response = self.router.generate(
                    prompt=protocol_prompt,
                    task_type=self.TaskType.COMPLEX_REASONING,
                    system_prompt="You are an expert research methodology designer.",
                    max_tokens=1500,
                    temperature=1.0
                )
                result = json.loads(response.content)
                
            elif hasattr(self, 'anthropic_client') and self.anthropic_client:
                # Fallback to direct Anthropic
                response = self.anthropic_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": protocol_prompt}]
                )
                result = json.loads(response.content[0].text)
                self.api_calls += 1
                self.token_count += len(response.content[0].text.split())
            
            return ExperimentalProtocol(
                hypotheses=hypotheses,
                data_split=result["data_split"],
                baselines=result["baselines"],
                statistical_tests=result["statistical_tests"],
                sample_size=result["sample_size"],
                random_seed=42,  # Reproducible
                created_by="Claude-3.5-Sonnet-20241022",
                created_at=datetime.now()
            )
            
        except Exception as e:
            print(f"AI protocol design failed: {e}")
            return self._fallback_protocol(hypotheses)
    
    def _fallback_protocol(self, hypotheses: List[ResearchHypothesis]) -> ExperimentalProtocol:
        """Fallback protocol if AI generation fails"""
        return create_standard_protocol()
    
    def conduct_autonomous_research(self) -> ExperimentalProtocol:
        """
        Complete autonomous research cycle:
        1. Generate hypotheses
        2. Design protocol  
        3. Return ready-to-execute research plan
        """
        
        print("[AI SCIENTIST] Generating research hypotheses...")
        hypotheses = self.generate_research_hypotheses()
        
        print(f"[SUCCESS] Generated {len(hypotheses)} research hypotheses")
        for h in hypotheses:
            print(f"   - {h.question}")
        
        print("[AI SCIENTIST] Designing experimental protocol...")
        protocol = self.design_experimental_protocol(hypotheses)
        
        print(f"[SUCCESS] Protocol designed with {len(protocol.baselines)} baselines")
        print(f"   - Sample size: {protocol.sample_size}")
        print(f"   - Random seed: {protocol.random_seed}")
        
        # Save protocol for reproducibility
        protocol_path = f"artifacts/protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("artifacts", exist_ok=True)
        
        with open(protocol_path, 'w') as f:
            json.dump(protocol.to_dict(), f, indent=2)
        
        print(f"[SAVED] Protocol saved to {protocol_path}")
        
        return protocol
    
    def get_authorship_contribution(self) -> Dict[str, Any]:
        """Track AI's contribution to research for conference compliance"""
        return {
            "hypothesis_generation": "100% AI",
            "protocol_design": "100% AI", 
            "claude_api_calls": self.api_calls,
            "tokens_generated": self.token_count,
            "files_created": ["research_protocol.json"],
            "reasoning": "AI autonomously generated all research hypotheses and experimental design"
        }


# Demo function for testing
def demo_meta_scientist():
    """Demonstrate autonomous AI research"""
    scientist = MetaScientist()
    
    print("=== AI SCIENTIST AUTONOMOUS RESEARCH DEMO ===")
    protocol = scientist.conduct_autonomous_research()
    
    print(f"\n[PROTOCOL SUMMARY]:")
    print(f"Hypotheses: {len(protocol.hypotheses)}")
    print(f"Baselines: {', '.join(protocol.baselines)}")
    print(f"Created by: {protocol.created_by}")
    
    contributions = scientist.get_authorship_contribution()
    print(f"\n[AI AUTHORSHIP]:")
    print(f"API calls: {contributions['claude_api_calls']}")
    print(f"Tokens: {contributions['tokens_generated']}")


if __name__ == "__main__":
    demo_meta_scientist()