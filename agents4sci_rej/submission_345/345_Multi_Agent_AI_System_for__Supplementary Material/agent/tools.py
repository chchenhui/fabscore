"""
AI Tools for Commercial Forecast System.

This module creates AI-aware wrappers around existing analytical functions,
allowing the AI agent to call Bass diffusion, NPV analysis, and Monte Carlo 
simulation with intelligent parameter reasoning.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Add src to Python path to import existing modules
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

try:
    from models.bass import bass_adopters, bass_cumulative, bass_peak_time
    from econ.npv import calculate_cashflows, npv, monte_carlo_npv
    from access.pricing_sim import tier_from_price, gtn_from_tier, adoption_ceiling_from_tier
    from data.etl import load_config
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"[ERROR] Required modules not available: {e}")
    print("[ERROR] Please ensure all dependencies are installed")
    IMPORTS_AVAILABLE = False
    raise ImportError(f"Cannot proceed without required modules: {e}")


class AnalysisState:
    """Tracks the state of analysis and AI reasoning."""
    
    def __init__(self):
        self.reasoning_log = []
        self.parameters_used = {}
        self.results = {}
        self.confidence_scores = {}
    
    def log_reasoning(self, step: str, reasoning: str, confidence: str = "medium"):
        """Log AI reasoning for a particular step."""
        self.reasoning_log.append({
            "step": step,
            "reasoning": reasoning,
            "confidence": confidence
        })
    
    def set_parameter(self, name: str, value: Any, reasoning: str):
        """Set a parameter with AI reasoning."""
        self.parameters_used[name] = {
            "value": value,
            "reasoning": reasoning
        }
    
    def get_reasoning_summary(self) -> List[str]:
        """Get a summary of all AI reasoning steps."""
        return [f"**{entry['step']}**: {entry['reasoning']}" for entry in self.reasoning_log]


class CommercialForecastTools:
    """AI-enhanced tools for commercial forecasting."""
    
    def __init__(self):
        """Initialize tools with default configuration."""
        try:
            self.config = load_config()
        except:
            # Fallback configuration if loading fails
            self.config = {
                'market': {'eligible_patients': 500000},
                'bass': {'p': 0.05, 'q': 0.45},
                'price_access': {'list_price_month_usd': 4369},
                'economics': {'wacc_annual': 0.10, 'cogs_pct': 0.12, 'sga_launch_annual': 350000000}
            }
        
        self.state = AnalysisState()
    
    def intelligent_market_sizing(self, characteristics: Dict[str, Any], 
                                state: AnalysisState) -> int:
        """AI-driven market sizing based on drug characteristics."""
        
        # Default US respiratory market
        base_population = 27_000_000  # Total US asthma population
        
        if characteristics.get("indication_area") == "respiratory":
            if characteristics.get("severity") == "severe":
                if characteristics.get("patient_population") == "pediatric":
                    # Pediatric severe asthma
                    market_size = int(base_population * 0.05 * 0.10)  # 5% severe, 10% pediatric
                    reasoning = f"Pediatric severe asthma: {base_population:,} total asthma × 5% severe × 10% pediatric = {market_size:,} patients"
                    confidence = "medium"
                else:
                    # Adult severe asthma  
                    market_size = int(base_population * 0.05 * 0.90)  # 5% severe, 90% adult
                    reasoning = f"Adult severe asthma: {base_population:,} total asthma × 5% severe × 90% adult = {market_size:,} patients"
                    confidence = "high"
            else:
                # All asthma
                market_size = base_population
                reasoning = f"Total asthma market: {market_size:,} patients"
                confidence = "high"
        else:
            # Default fallback
            market_size = 500_000
            reasoning = "Using default market size due to limited indication data"
            confidence = "low"
        
        state.log_reasoning("Market Sizing", reasoning, confidence)
        state.set_parameter("market_size", market_size, reasoning)
        
        return market_size
    
    def intelligent_adoption_parameters(self, characteristics: Dict[str, Any],
                                     state: AnalysisState) -> Tuple[float, float]:
        """AI-driven Bass diffusion parameter estimation."""
        
        drug_type = characteristics.get("drug_type", "unknown")
        indication_area = characteristics.get("indication_area", "unknown")
        severity = characteristics.get("severity", "unknown")
        
        # Base parameters
        if drug_type == "biologic":
            if indication_area == "respiratory":
                # Respiratory biologics (Tezspire, Dupixent precedent)
                p = 0.055  # Moderate early adoption
                q = 0.50   # Strong word-of-mouth
                reasoning = "Respiratory biologics: p=5.5% (moderate early adoption), q=50% (strong efficacy word-of-mouth)"
                confidence = "high"
            else:
                # Other biologics
                p = 0.045
                q = 0.45
                reasoning = "General biologics: p=4.5% (typical early adoption), q=45% (good word-of-mouth)"
                confidence = "medium"
        else:
            # Small molecules or unknown
            p = 0.035
            q = 0.35
            reasoning = "Small molecule default: p=3.5% (conservative early adoption), q=35% (moderate word-of-mouth)"
            confidence = "low"
        
        # Adjust for severity
        if severity == "severe":
            p *= 1.2  # Faster early adoption for severe disease
            q *= 1.1  # Slightly better word-of-mouth
            reasoning += f" | Severe disease adjustment: +20% early adoption, +10% word-of-mouth"
        
        state.log_reasoning("Adoption Parameters", reasoning, confidence)
        state.set_parameter("bass_p", p, f"Early adopter rate: {p:.1%}")
        state.set_parameter("bass_q", q, f"Word-of-mouth coefficient: {q:.1%}")
        
        return p, q
    
    def intelligent_pricing(self, characteristics: Dict[str, Any],
                          state: AnalysisState) -> Dict[str, Any]:
        """AI-driven pricing estimation."""
        
        drug_type = characteristics.get("drug_type", "unknown")
        indication_area = characteristics.get("indication_area", "unknown") 
        patient_population = characteristics.get("patient_population", "adult")
        severity = characteristics.get("severity", "unknown")
        
        if drug_type == "biologic" and indication_area == "respiratory":
            if patient_population == "pediatric":
                # Pediatric often priced slightly lower
                list_price = 4000
                reasoning = "Pediatric respiratory biologic: $4,000/month (10% discount vs adult Tezspire pricing)"
            else:
                # Adult respiratory biologic (Tezspire benchmark)  
                list_price = 4369
                reasoning = "Adult respiratory biologic: $4,369/month (Tezspire benchmark)"
            confidence = "high"
        elif drug_type == "biologic":
            # Other biologics
            list_price = 3500
            reasoning = "General biologic: $3,500/month (typical biologic pricing range)"
            confidence = "medium"
        else:
            # Small molecules
            list_price = 2000  
            reasoning = "Small molecule: $2,000/month (conservative estimate)"
            confidence = "low"
        
        # Determine access tier
        access_tier = tier_from_price(list_price)
        gtn_pct = gtn_from_tier(access_tier)
        adoption_ceiling = adoption_ceiling_from_tier(access_tier)
        
        pricing_info = {
            "list_price": list_price,
            "access_tier": access_tier,
            "gtn_pct": gtn_pct,
            "adoption_ceiling": adoption_ceiling
        }
        
        state.log_reasoning("Pricing Strategy", reasoning, confidence)
        state.set_parameter("pricing", pricing_info, reasoning)
        
        return pricing_info
    
    def run_bass_analysis(self, market_size: int, p: float, q: float, 
                         adoption_ceiling: float, time_horizon: int = 5) -> Dict[str, Any]:
        """Run Bass diffusion analysis with AI parameter reasoning."""
        
        T = time_horizon * 4  # Convert to quarters
        effective_market = market_size * adoption_ceiling
        
        # Calculate adoption curves
        adopters = bass_adopters(T, effective_market, p, q)
        cumulative = bass_cumulative(T, effective_market, p, q)
        peak_quarter = np.argmax(adopters) + 1
        
        # Key insights
        total_adoption = cumulative[-1] 
        penetration_rate = total_adoption / market_size
        peak_patients = adopters[peak_quarter - 1]
        
        results = {
            "adopters": adopters,
            "cumulative": cumulative,
            "quarters": np.arange(1, T + 1),
            "peak_quarter": peak_quarter,
            "peak_patients": peak_patients,
            "total_adoption": total_adoption,
            "penetration_rate": penetration_rate,
            "effective_market": effective_market
        }
        
        self.state.results["bass_analysis"] = results
        
        reasoning = f"Peak adoption: Q{peak_quarter} with {peak_patients:,.0f} new patients. " \
                   f"Total penetration: {penetration_rate:.1%} of {market_size:,} eligible patients."
        
        self.state.log_reasoning("Bass Analysis", reasoning, "high")
        
        return results
    
    def run_financial_analysis(self, adopters: np.ndarray, pricing_info: Dict[str, Any],
                             wacc: float = 0.10) -> Dict[str, Any]:
        """Run NPV analysis with AI reasoning."""
        
        # Calculate cashflows using existing function
        cashflow_params = {
            'adopters': adopters,
            'list_price_monthly': pricing_info['list_price'],
            'gtn_pct': pricing_info['gtn_pct'], 
            'cogs_pct': self.config['economics']['cogs_pct'],
            'sga_launch': self.config['economics']['sga_launch_annual'] / 4,  # Quarterly
            'sga_decay_to_pct': 0.5,
            'adherence_rate': 0.85
        }
        
        cashflows = calculate_cashflows(**cashflow_params)
        net_cf = cashflows['net_cashflows']
        npv_value = npv(net_cf, wacc)
        
        results = {
            "npv": npv_value,
            "cashflows": cashflows,
            "total_revenue": cashflows['revenue'].sum(),
            "peak_revenue": cashflows['revenue'].max()
        }
        
        self.state.results["financial_analysis"] = results
        
        # AI reasoning about results
        if npv_value > 0:
            decision = "POSITIVE NPV - Attractive Investment" 
            confidence = "high" if npv_value > 1e9 else "medium"
        else:
            decision = "NEGATIVE NPV - Poor Investment"
            confidence = "high"
        
        reasoning = f"{decision}. NPV = ${npv_value/1e9:.1f}B. " \
                   f"Peak revenue: ${results['peak_revenue']/1e6:.0f}M/quarter."
        
        self.state.log_reasoning("Financial Analysis", reasoning, confidence)
        
        return results
    
    def run_monte_carlo(self, base_params: Dict[str, Any], n_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation with realistic uncertainty."""
        
        # Enhanced uncertainty parameters for realistic risk assessment
        uncertainty_params = {
            'gtn_pct': 0.15,           # Payer negotiation variance
            'list_price_monthly': base_params['list_price_monthly'] * 0.25,  # Competition/pricing pressure  
            'adherence_rate': 0.18,     # Real-world dropout variance
            'sga_launch': base_params['sga_launch'] * 0.40   # Launch execution risk
        }
        
        # Add WACC to base params
        base_params['wacc_annual'] = 0.10
        
        try:
            mc_results = monte_carlo_npv(
                base_params=base_params,
                uncertainty_params=uncertainty_params,
                n_simulations=n_simulations,
                random_seed=42
            )
            
            success_rate = mc_results['success_rate']
            
            # AI interpretation of risk
            if success_rate > 0.75:
                risk_assessment = "LOW RISK - Strong probability of success"
                recommendation = "GO - Proceed with investment"
            elif success_rate > 0.50:
                risk_assessment = "MEDIUM RISK - Moderate uncertainty"
                recommendation = "MAYBE - Require risk mitigation strategies"
            else:
                risk_assessment = "HIGH RISK - Significant downside probability" 
                recommendation = "NO-GO - Too risky vs alternatives"
            
            reasoning = f"{risk_assessment}. Success rate: {success_rate:.0%}. {recommendation}."
            
            self.state.log_reasoning("Risk Analysis", reasoning, "high")
            self.state.results["monte_carlo"] = mc_results
            
            return mc_results
            
        except Exception as e:
            self.state.log_reasoning("Risk Analysis", f"Monte Carlo failed: {str(e)}", "low")
            return {"error": str(e)}
    
    def generate_recommendation(self, characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final AI investment recommendation."""
        
        # Gather key results
        bass_results = self.state.results.get("bass_analysis", {})
        financial_results = self.state.results.get("financial_analysis", {})
        mc_results = self.state.results.get("monte_carlo", {})
        
        npv = financial_results.get("npv", 0)
        success_rate = mc_results.get("success_rate", 0) if mc_results else 0
        
        # AI decision logic
        if npv > 1e9 and success_rate > 0.75:
            decision = "STRONG GO"
            rationale = "High NPV with low risk profile"
            confidence = "high"
        elif npv > 0 and success_rate > 0.60:
            decision = "GO"
            rationale = "Positive NPV with acceptable risk"
            confidence = "medium"  
        elif npv > 0 and success_rate > 0.40:
            decision = "CONDITIONAL GO"
            rationale = "Positive NPV but higher risk - consider mitigation"
            confidence = "medium"
        else:
            decision = "NO-GO"  
            rationale = "Negative NPV or excessive risk"
            confidence = "high"
        
        # Key insights
        drug_name = characteristics.get("name", "Unknown Drug")
        market_size = self.state.parameters_used.get("market_size", {}).get("value", 0)
        
        recommendation = {
            "decision": decision,
            "rationale": rationale,
            "confidence": confidence,
            "key_metrics": {
                "npv_billions": npv / 1e9,
                "success_rate": success_rate,
                "market_size": market_size,
                "peak_quarter": bass_results.get("peak_quarter", 0)
            },
            "reasoning_summary": self.state.get_reasoning_summary()
        }
        
        final_reasoning = f"RECOMMENDATION: {decision} for {drug_name}. {rationale}. " \
                         f"NPV: ${npv/1e9:.1f}B, Risk: {success_rate:.0%} success rate."
        
        self.state.log_reasoning("Final Decision", final_reasoning, confidence)
        
        return recommendation


# Test the tools
if __name__ == "__main__":
    print("=== AI TOOLS DEMO ===")
    
    tools = CommercialForecastTools()
    
    # Test characteristics (pediatric severe asthma biologic)
    test_characteristics = {
        "name": "Tezspire Competitor",
        "drug_type": "biologic",
        "indication_area": "respiratory", 
        "severity": "severe",
        "patient_population": "pediatric"
    }
    
    # Run AI-driven analysis
    market_size = tools.intelligent_market_sizing(test_characteristics, tools.state)
    p, q = tools.intelligent_adoption_parameters(test_characteristics, tools.state)
    pricing_info = tools.intelligent_pricing(test_characteristics, tools.state)
    
    print(f"Market Size: {market_size:,} patients")
    print(f"Bass Parameters: p={p:.3f}, q={q:.3f}")
    print(f"Pricing: ${pricing_info['list_price']:,}/month ({pricing_info['access_tier']} tier)")
    
    print("\nAI Reasoning:")
    for reasoning in tools.state.get_reasoning_summary():
        print(f"  • {reasoning}")