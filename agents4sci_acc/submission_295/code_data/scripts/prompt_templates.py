#!/usr/bin/env python3
"""
Prompt Engineering Templates for Catalyst Generation
Structured prompts for LLM-based catalyst hypothesis generation
"""

import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class GenerationStrategy(Enum):
    """Different strategies for catalyst generation"""
    CONSTRAINT_BASED = "constraint_based"
    ANALOGY_BASED = "analogy_based"
    SUBSTITUTION_BASED = "substitution_based"
    COMBINATORIAL = "combinatorial"
    MECHANISM_GUIDED = "mechanism_guided"


@dataclass
class CatalystConstraints:
    """Constraints for catalyst generation"""
    allowed_elements: List[str] = field(default_factory=list)
    forbidden_elements: List[str] = field(default_factory=list)
    max_elements: int = 5
    min_elements: int = 1
    structure_type: Optional[str] = None
    stability_threshold: float = 0.1  # eV above hull
    earth_abundant_only: bool = True
    cost_limit: Optional[float] = None
    target_properties: Dict[str, Any] = field(default_factory=dict)


class PromptTemplates:
    """Collection of prompt templates for catalyst generation"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.earth_abundant_elements = [
            "Fe", "Cu", "Ni", "Co", "Mn", "Ti", "V", "Cr", "Mo", "W",
            "Al", "Si", "Zn", "Mg", "Ca", "Na", "K"
        ]
        self.noble_metals = ["Pt", "Pd", "Rh", "Ru", "Ir", "Au", "Ag"]
    
    def _load_templates(self) -> Dict[str, str]:
        """Load base prompt templates"""
        return {
            "system": """You are an expert in materials science and catalysis. Your task is to generate novel catalyst candidates based on scientific principles, existing knowledge, and specified constraints. Always provide scientifically grounded suggestions with clear reasoning.""",
            
            "constraint_based": """Generate {num_candidates} catalyst candidates for {reaction} that satisfy the following constraints:

Constraints:
- Allowed elements: {allowed_elements}
- Maximum number of elements: {max_elements}
- Stability requirement: {stability_requirement}
- Target properties: {target_properties}

Retrieved Context:
{retrieved_context}

For each candidate, provide:
1. Chemical formula
2. Proposed crystal structure
3. Expected catalytic properties
4. Scientific rationale
5. Similarity to known catalysts

Format your response as a JSON array with the structure:
[{{"formula": "...", "structure": "...", "properties": {{...}}, "rationale": "...", "similar_to": ["..."]}}]""",
            
            "analogy_based": """Based on the known catalyst {reference_catalyst} for {reference_reaction}, suggest {num_candidates} analogous catalysts for {target_reaction}.

Reference catalyst properties:
{reference_properties}

Target reaction requirements:
{target_requirements}

Retrieved similar materials:
{retrieved_context}

Apply the following reasoning strategies:
1. d-band theory for transition metal substitutions
2. Sabatier principle for optimal binding
3. Periodic trends for element selection
4. Structure-activity relationships

Provide candidates in JSON format with explanations of the analogies used.""",
            
            "substitution_based": """Given the base catalyst {base_catalyst}, generate {num_candidates} variants through strategic element substitution.

Substitution rules:
1. Maintain similar atomic radii (±15%)
2. Preserve oxidation state compatibility
3. Consider d-electron count for transition metals
4. Ensure phase stability

Available elements for substitution: {allowed_elements}

Retrieved context on similar substitutions:
{retrieved_context}

For each variant:
- Specify which element(s) are substituted
- Explain the chemical rationale
- Predict changes in catalytic properties
- Assess stability risks

Format as JSON array.""",
            
            "combinatorial": """Design {num_candidates} high-entropy alloy (HEA) catalysts for {reaction}.

HEA design principles:
1. Equimolar or near-equimolar compositions
2. High configurational entropy (ΔSmix > 1.5R)
3. Solid solution formation tendency
4. Avoid strong compound formers

Element pool: {element_pool}
Number of components: {num_components}

Retrieved HEA examples:
{retrieved_context}

Consider:
- Atomic size mismatch (δ < 6.6%)
- Electronegativity difference
- Valence electron concentration
- Mixing enthalpy

Provide compositions with stability analysis.""",
            
            "mechanism_guided": """Design catalysts for {reaction} based on the reaction mechanism:

Reaction steps:
{reaction_steps}

Key intermediates and their optimal binding energies:
{intermediates}

Rate-determining step: {rds}

Retrieved catalysts with known binding energies:
{retrieved_context}

Generate {num_candidates} catalysts that:
1. Optimize binding of the rate-determining intermediate
2. Follow the Brønsted-Evans-Polanyi relationship
3. Balance all intermediates near optimal values
4. Consider scaling relations

Include predicted binding energies and volcano plot position."""
        }
    
    def build_generation_prompt(self,
                               strategy: GenerationStrategy,
                               constraints: CatalystConstraints,
                               reaction: str,
                               retrieved_context: str,
                               num_candidates: int = 5,
                               **kwargs) -> str:
        """Build a complete generation prompt"""
        
        # Get base template
        template = self.templates.get(strategy.value, self.templates["constraint_based"])
        
        # Prepare template variables
        template_vars = {
            "num_candidates": num_candidates,
            "reaction": reaction,
            "retrieved_context": retrieved_context,
            "allowed_elements": ", ".join(constraints.allowed_elements) if constraints.allowed_elements 
                               else ", ".join(self.earth_abundant_elements),
            "max_elements": constraints.max_elements,
            "stability_requirement": f"< {constraints.stability_threshold} eV above hull",
            "target_properties": json.dumps(constraints.target_properties, indent=2)
        }
        
        # Add strategy-specific variables
        template_vars.update(kwargs)
        
        # Fill template
        prompt = template.format(**template_vars)
        
        # Add constraints section if not already included
        if strategy != GenerationStrategy.CONSTRAINT_BASED:
            prompt += self._build_constraints_section(constraints)
        
        return prompt
    
    def _build_constraints_section(self, constraints: CatalystConstraints) -> str:
        """Build additional constraints section"""
        section = "\n\nAdditional Constraints:\n"
        
        if constraints.forbidden_elements:
            section += f"- Forbidden elements: {', '.join(constraints.forbidden_elements)}\n"
        
        if constraints.earth_abundant_only:
            section += "- Use only earth-abundant elements (no noble metals)\n"
        
        if constraints.cost_limit:
            section += f"- Material cost limit: ${constraints.cost_limit}/kg\n"
        
        if constraints.structure_type:
            section += f"- Preferred structure type: {constraints.structure_type}\n"
        
        return section
    
    def build_refinement_prompt(self,
                               initial_candidates: List[Dict],
                               feedback: Dict[str, Any],
                               constraints: CatalystConstraints) -> str:
        """Build prompt for refining candidates based on feedback"""
        
        prompt = """Based on the computational screening results, refine the catalyst candidates:

Initial candidates and their issues:
"""
        
        for i, (candidate, issues) in enumerate(zip(initial_candidates, feedback.get("issues", [])), 1):
            prompt += f"\n{i}. {candidate['formula']}:\n"
            for issue in issues:
                prompt += f"   - {issue}\n"
        
        prompt += f"""
Screening criteria that failed:
{json.dumps(feedback.get("failed_criteria", {}), indent=2)}

Generate {len(initial_candidates)} improved candidates that address these issues while maintaining the original design intent.

Consider:
1. Stability improvements through composition adjustment
2. Electronic structure tuning via dopants
3. Alternative structure types
4. Strain engineering possibilities

Format as JSON array with explanations of improvements made."""
        
        return prompt
    
    def build_descriptor_prompt(self, 
                               catalyst_class: str,
                               reaction: str,
                               known_descriptors: List[str]) -> str:
        """Build prompt for descriptor identification"""
        
        prompt = f"""Identify key descriptors for {catalyst_class} catalysts in {reaction}.

Known descriptors in catalysis:
{json.dumps(known_descriptors, indent=2)}

Analyze which descriptors are most relevant for this system and suggest:
1. Primary descriptor (most correlated with activity)
2. Secondary descriptors
3. Descriptor relationships (e.g., scaling relations)
4. Optimal descriptor values
5. How to tune these descriptors

Provide scientific justification based on:
- Electronic structure considerations
- Reaction mechanism
- Computational/experimental evidence"""
        
        return prompt
    
    def build_validation_prompt(self,
                               candidate: Dict,
                               dft_results: Dict) -> str:
        """Build prompt for interpreting DFT validation results"""
        
        prompt = f"""Interpret the DFT calculation results for catalyst candidate {candidate['formula']}:

DFT Results:
- Formation energy: {dft_results.get('formation_energy', 'N/A')} eV/atom
- Energy above hull: {dft_results.get('energy_above_hull', 'N/A')} eV/atom
- Band gap: {dft_results.get('band_gap', 'N/A')} eV
- d-band center: {dft_results.get('d_band_center', 'N/A')} eV
- Work function: {dft_results.get('work_function', 'N/A')} eV

Adsorption energies:
{json.dumps(dft_results.get('adsorption_energies', {}), indent=2)}

Assess:
1. Thermodynamic stability
2. Electronic structure suitability
3. Binding strength optimization
4. Comparison to known catalysts
5. Recommendations for improvement

Provide a clear verdict on viability and next steps."""
        
        return prompt
    
    def get_prompt_for_stage(self,
                            stage: str,
                            **kwargs) -> str:
        """Get appropriate prompt for pipeline stage"""
        
        stage_mapping = {
            "initial_generation": self.build_generation_prompt,
            "refinement": self.build_refinement_prompt,
            "descriptor_analysis": self.build_descriptor_prompt,
            "validation_interpretation": self.build_validation_prompt
        }
        
        if stage not in stage_mapping:
            raise ValueError(f"Unknown stage: {stage}")
        
        return stage_mapping[stage](**kwargs)


class PromptOptimizer:
    """Optimize prompts based on generation success rates"""
    
    def __init__(self):
        self.generation_history = []
        self.success_patterns = {}
    
    def record_generation(self,
                         prompt: str,
                         strategy: GenerationStrategy,
                         candidates: List[Dict],
                         validation_results: Dict):
        """Record generation attempt for learning"""
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy.value,
            "prompt_length": len(prompt),
            "num_candidates": len(candidates),
            "success_rate": self._calculate_success_rate(validation_results),
            "prompt_features": self._extract_prompt_features(prompt)
        }
        
        self.generation_history.append(record)
        self._update_success_patterns()
    
    def _calculate_success_rate(self, validation_results: Dict) -> float:
        """Calculate success rate from validation results"""
        if not validation_results:
            return 0.0
        
        passed = sum(1 for r in validation_results.values() if r.get("viable", False))
        total = len(validation_results)
        
        return passed / total if total > 0 else 0.0
    
    def _extract_prompt_features(self, prompt: str) -> Dict:
        """Extract features from prompt for analysis"""
        return {
            "has_examples": "example" in prompt.lower(),
            "has_context": "retrieved context" in prompt.lower(),
            "num_constraints": prompt.lower().count("constraint"),
            "specificity": len(prompt.split()) / 100  # Normalized word count
        }
    
    def _update_success_patterns(self):
        """Update patterns of successful prompts"""
        if len(self.generation_history) < 10:
            return
        
        # Analyze recent history
        recent = self.generation_history[-50:]
        
        # Find patterns in successful generations
        successful = [r for r in recent if r["success_rate"] > 0.5]
        
        if successful:
            # Update success patterns
            for feature in ["has_examples", "has_context"]:
                success_with_feature = [r for r in successful 
                                      if r["prompt_features"].get(feature, False)]
                
                if len(success_with_feature) > len(successful) * 0.7:
                    self.success_patterns[feature] = True
    
    def suggest_improvements(self, current_prompt: str) -> List[str]:
        """Suggest prompt improvements based on history"""
        suggestions = []
        
        features = self._extract_prompt_features(current_prompt)
        
        # Check against success patterns
        for feature, should_have in self.success_patterns.items():
            if should_have and not features.get(feature, False):
                if feature == "has_examples":
                    suggestions.append("Add specific examples of successful catalysts")
                elif feature == "has_context":
                    suggestions.append("Include more retrieved context from similar materials")
        
        # Check prompt length
        avg_success_length = np.mean([r["prompt_length"] for r in self.generation_history 
                                     if r["success_rate"] > 0.5])
        
        if len(current_prompt) < avg_success_length * 0.8:
            suggestions.append("Expand prompt with more detailed constraints and guidance")
        
        return suggestions


def main():
    """Example usage"""
    # Initialize templates
    templates = PromptTemplates()
    
    # Example constraints
    constraints = CatalystConstraints(
        allowed_elements=["Fe", "Cu", "Ni", "Co", "Mn"],
        max_elements=3,
        earth_abundant_only=True,
        target_properties={
            "co2_reduction_activity": "high",
            "h2_evolution_suppression": "high",
            "stability_in_aqueous": "high"
        }
    )
    
    # Example generation prompt
    prompt = templates.build_generation_prompt(
        strategy=GenerationStrategy.CONSTRAINT_BASED,
        constraints=constraints,
        reaction="CO2 reduction to CO",
        retrieved_context="[Example retrieved materials...]",
        num_candidates=5
    )
    
    print("Generated Prompt:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)


if __name__ == "__main__":
    main()