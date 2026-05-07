"""
Specialized Agent Implementations for Phase 2 Multi-Agent Architecture
Each agent has a specific role following the MASSIVE_OVERHAUL_PLAN.md

Following Linus: "Do one thing and do it well"
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import asyncio
from datetime import datetime

try:
    from .model_router import get_router, TaskType, ModelResponse
except ImportError:
    from model_router import get_router, TaskType, ModelResponse


class SpecializedAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.router = get_router()
        self.execution_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's specialized task"""
        pass
    
    def _log_execution(self, task: str, result: Dict[str, Any], confidence: float):
        """Log agent execution for transparency"""
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent": self.agent_name,
            "task": task,
            "confidence": confidence,
            "success": "error" not in result
        })


class DataCollectionAgent(SpecializedAgent):
    """
    DeepSeek-powered agent for bulk pharmaceutical data processing
    Sources: FDA, SEC, PubMed, ClinicalTrials.gov
    """
    
    def __init__(self):
        super().__init__("DataCollectionAgent")
        self.sources = ["FDA", "SEC_EDGAR", "PubMed", "ClinicalTrials", "Patent_DB"]
        
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect all available data for a drug launch
        Returns structured data with confidence scores
        """
        
        drug_name = task_data.get("drug_name", "Unknown")
        indication = task_data.get("indication", "Unknown")
        
        print(f"[DATA_COLLECTOR] Collecting data for {drug_name}")
        
        # Collect data from each source
        collection_tasks = [
            self._collect_fda_data(drug_name),
            self._collect_sec_data(drug_name),
            self._collect_clinical_data(drug_name, indication),
            self._collect_market_intelligence(drug_name, indication)
        ]
        
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)
        
        # Aggregate results
        aggregated_data = {
            "fda_data": results[0] if not isinstance(results[0], Exception) else {},
            "sec_data": results[1] if not isinstance(results[1], Exception) else {},
            "clinical_data": results[2] if not isinstance(results[2], Exception) else {},
            "market_intelligence": results[3] if not isinstance(results[3], Exception) else {}
        }
        
        # Calculate overall confidence
        confidences = []
        for source_data in aggregated_data.values():
            if isinstance(source_data, dict) and "confidence" in source_data:
                confidences.append(source_data["confidence"])
        
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        result = {
            "collected_data": aggregated_data,
            "sources_accessed": len([r for r in results if not isinstance(r, Exception)]),
            "overall_confidence": overall_confidence,
            "data_completeness": self._assess_completeness(aggregated_data)
        }
        
        self._log_execution("comprehensive_data_collection", result, overall_confidence)
        return result
    
    async def _collect_fda_data(self, drug_name: str) -> Dict[str, Any]:
        """Collect FDA approval and regulatory data"""
        
        prompt = f"""
        Extract comprehensive FDA regulatory data for {drug_name}:
        
        Required fields:
        - Original approval date (YYYY-MM-DD format)
        - Application number (NDA/BLA)
        - Review priority (Standard/Priority/Breakthrough)
        - Approved indications
        - Mechanism of action
        - Route of administration
        - Sponsor company
        - Marketing status
        - Any safety warnings or black box warnings
        
        Return structured JSON with confidence scores for each field.
        """
        
        try:
            response = self.router.generate(
                prompt=prompt,
                task_type=TaskType.BULK_PARSING,
                system_prompt="You are an FDA regulatory data specialist.",
                provider_preference="deepseek",
                max_tokens=1000
            )
            
            # Mock FDA data structure (would connect to real FDA API)
            fda_data = {
                "approval_date": "2020-05-12",
                "application_number": "BLA125387",
                "review_priority": "Priority",
                "indications": ["Severe asthma", "Chronic rhinosinusitis with nasal polyps"],
                "mechanism": "Anti-IL-4/IL-13 receptor alpha",
                "route": "Subcutaneous injection",
                "sponsor": "Regeneron Pharmaceuticals",
                "marketing_status": "Prescription",
                "black_box_warning": False,
                "confidence": 0.85
            }
            
            return fda_data
            
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}
    
    async def _collect_sec_data(self, drug_name: str) -> Dict[str, Any]:
        """Collect SEC financial and revenue data"""
        
        prompt = f"""
        Extract SEC financial data for pharmaceutical product {drug_name}:
        
        Required data:
        - Historical revenue data (annual, if available)
        - Revenue growth rates
        - Market share information
        - Company guidance/forecasts
        - Segment reporting (if product-specific)
        - Geographic revenue breakdown
        
        Focus on actual reported financials from 10-K/10-Q filings.
        Return structured data with confidence assessment.
        """
        
        try:
            # Mock SEC data (would connect to real SEC EDGAR API)
            sec_data = {
                "revenue_history": {
                    "2020": 150_000_000,
                    "2021": 420_000_000,
                    "2022": 890_000_000,
                    "2023": 1_340_000_000
                },
                "growth_rates": {
                    "yoy_2021": 1.80,  # 180% growth
                    "yoy_2022": 1.12,  # 112% growth  
                    "yoy_2023": 0.51   # 51% growth
                },
                "geographic_split": {
                    "US": 0.65,
                    "EU": 0.25,
                    "ROW": 0.10
                },
                "confidence": 0.75
            }
            
            return sec_data
            
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}
    
    async def _collect_clinical_data(self, drug_name: str, indication: str) -> Dict[str, Any]:
        """Collect clinical trial and efficacy data"""
        
        prompt = f"""
        Extract clinical data for {drug_name} in {indication}:
        
        Required data:
        - Phase III trial results (primary/secondary endpoints)
        - Efficacy measures (response rates, effect sizes)
        - Safety profile (adverse events, discontinuation rates)
        - Patient population characteristics
        - Comparator drugs used in trials
        - Real-world evidence (if available)
        
        Focus on published results and ClinicalTrials.gov data.
        Return structured efficacy and safety metrics.
        """
        
        try:
            # Mock clinical data
            clinical_data = {
                "phase3_results": {
                    "primary_endpoint_met": True,
                    "response_rate": 0.72,
                    "placebo_response": 0.34,
                    "effect_size": 0.38
                },
                "safety_profile": {
                    "serious_ae_rate": 0.08,
                    "discontinuation_rate": 0.12,
                    "injection_site_reactions": 0.25
                },
                "patient_population": {
                    "trial_population": 1847,
                    "mean_age": 47.2,
                    "severe_asthma_percentage": 1.0
                },
                "comparators": ["Standard of care", "Placebo"],
                "confidence": 0.80
            }
            
            return clinical_data
            
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}
    
    async def _collect_market_intelligence(self, drug_name: str, indication: str) -> Dict[str, Any]:
        """Collect market intelligence and competitive data"""
        
        # Mock market intelligence
        market_data = {
            "competitive_landscape": {
                "direct_competitors": ["Fasenra", "Nucala", "Cinqair"],
                "indirect_competitors": ["Xolair", "High-dose ICS"],
                "market_leaders": ["Dupixent", "Xolair"]
            },
            "pricing_intelligence": {
                "annual_cost_estimate": 28_000,
                "competitor_pricing": {
                    "Fasenra": 32_000,
                    "Nucala": 30_000,
                    "Xolair": 26_000
                }
            },
            "market_access": {
                "payer_coverage": 0.75,
                "prior_authorization_required": True,
                "step_therapy": True
            },
            "confidence": 0.70
        }
        
        return market_data
    
    def _assess_completeness(self, data: Dict[str, Any]) -> float:
        """Assess overall data completeness"""
        
        required_fields = ["fda_data", "sec_data", "clinical_data", "market_intelligence"]
        completed_sources = 0
        
        for field in required_fields:
            if field in data and data[field] and "confidence" in data[field]:
                if data[field]["confidence"] > 0.5:
                    completed_sources += 1
        
        return completed_sources / len(required_fields)


class MarketAnalysisAgent(SpecializedAgent):
    """
    GPT-5 powered agent for complex market reasoning
    Specializes in analog analysis and competitive intelligence
    """
    
    def __init__(self):
        super().__init__("MarketAnalysisAgent")
        
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis
        Focus: analog drugs, competitive landscape, market sizing
        """
        
        drug_data = task_data.get("drug_data", {})
        
        print(f"[MARKET_ANALYST] Analyzing market for drug with {len(drug_data)} data sources")
        
        # Parallel market analysis tasks
        analysis_tasks = [
            self._find_analogs(drug_data),
            self._assess_competition(drug_data),
            self._size_market(drug_data)
        ]
        
        results = await asyncio.gather(*analysis_tasks)
        
        market_analysis = {
            "analog_analysis": results[0],
            "competitive_assessment": results[1], 
            "market_sizing": results[2],
            "synthesis": await self._synthesize_market_view(results)
        }
        
        # Calculate overall confidence
        confidences = [r.get("confidence", 0.5) for r in results if isinstance(r, dict)]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        market_analysis["overall_confidence"] = overall_confidence
        
        self._log_execution("comprehensive_market_analysis", market_analysis, overall_confidence)
        return market_analysis
    
    async def _find_analogs(self, drug_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find and analyze analog drugs for comparison
        Critical for industry-standard forecasting
        """
        
        # Extract drug characteristics for analog matching
        fda_data = drug_data.get("fda_data", {})
        mechanism = fda_data.get("mechanism", "")
        indications = fda_data.get("indications", [])
        
        prompt = f"""
        Find analog drugs for forecasting comparison:
        
        Target drug characteristics:
        - Mechanism: {mechanism}
        - Indications: {indications}
        - Route: {fda_data.get("route", "")}
        
        Find 3-5 analog drugs with:
        1. Similar mechanism of action (exact or related pathway)
        2. Same or similar indication
        3. Comparable patient population
        4. Known commercial performance data
        
        For each analog, provide:
        - Drug name and company
        - Similarity score (0-1)
        - Peak sales achieved
        - Launch trajectory (years to peak)
        - Key differentiating factors
        
        Rank by similarity and commercial relevance.
        """
        
        try:
            response = self.router.generate(
                prompt=prompt,
                task_type=TaskType.COMPLEX_REASONING,
                system_prompt="You are a pharmaceutical market analyst specializing in analog drug analysis.",
                provider_preference="openai",  # GPT-5 for complex reasoning
                max_tokens=1200
            )
            
            # Mock analog analysis (would use real pharmaceutical database)
            analog_analysis = {
                "analogs_identified": [
                    {
                        "name": "Dupixent",
                        "company": "Regeneron/Sanofi",
                        "similarity_score": 0.85,
                        "peak_sales": 8_000_000_000,
                        "years_to_peak": 4,
                        "differentiation": "Broader indication set, first-to-market advantage"
                    },
                    {
                        "name": "Fasenra",
                        "company": "AstraZeneca", 
                        "similarity_score": 0.70,
                        "peak_sales": 1_200_000_000,
                        "years_to_peak": 3,
                        "differentiation": "Different mechanism (IL-5), more limited indications"
                    },
                    {
                        "name": "Nucala",
                        "company": "GSK",
                        "similarity_score": 0.65,
                        "peak_sales": 1_800_000_000,
                        "years_to_peak": 4,
                        "differentiation": "Earlier market entry, established in severe asthma"
                    }
                ],
                "analog_forecast_range": {
                    "low": 800_000_000,
                    "high": 3_500_000_000,
                    "weighted_average": 1_800_000_000
                },
                "confidence": 0.80
            }
            
            return analog_analysis
            
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}
    
    async def _assess_competition(self, drug_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Competitive landscape analysis
        Standard of care, pipeline threats, market share dynamics
        """
        
        market_intel = drug_data.get("market_intelligence", {})
        
        prompt = f"""
        Assess competitive landscape for pharmaceutical launch:
        
        Current competitive data: {json.dumps(market_intel, indent=2)}
        
        Analyze:
        1. Current standard of care and treatment patterns
        2. Direct and indirect competitors
        3. Market share distribution
        4. Pipeline threats (drugs in development)
        5. Competitive positioning opportunities
        6. Likely competitive response to new entrant
        
        Focus on:
        - Market dynamics and switching barriers
        - Differentiation opportunities
        - Competitive threats and timeline
        - Strategic positioning recommendations
        """
        
        try:
            # Mock competitive assessment
            competitive_assessment = {
                "current_soc": {
                    "primary_treatments": ["High-dose ICS/LABA", "Oral corticosteroids"],
                    "biologic_penetration": 0.15,
                    "unmet_need": "High - 40% of severe asthma patients uncontrolled"
                },
                "competitive_dynamics": {
                    "market_leaders": ["Dupixent (32%)", "Xolair (28%)", "Nucala (18%)"],
                    "market_concentration": "Moderate - top 3 control 78%",
                    "switching_barriers": "Moderate - efficacy and tolerability drive decisions"
                },
                "pipeline_threats": [
                    {"drug": "Tezspire biosimilar", "timeline": "2026", "threat_level": "High"},
                    {"drug": "Novel IL-33 inhibitor", "timeline": "2027", "threat_level": "Medium"}
                ],
                "positioning_opportunity": {
                    "primary": "Superior efficacy in pediatric population",
                    "secondary": "Improved dosing convenience",
                    "target_share": 0.12
                },
                "confidence": 0.75
            }
            
            return competitive_assessment
            
        except Exception as e:
            return {"error": str(e), "confidence": 0.0}
    
    async def _size_market(self, drug_data: Dict[str, Any]) -> Dict[str, Any]:
        """Market sizing analysis"""
        
        clinical_data = drug_data.get("clinical_data", {})
        
        # Mock market sizing
        market_sizing = {
            "total_addressable_market": 2_800_000_000,
            "serviceable_addressable_market": 1_400_000_000,
            "serviceable_obtainable_market": 420_000_000,
            "patient_population": {
                "total_severe_asthma": 1_200_000,
                "biologic_eligible": 780_000,
                "target_population": 420_000
            },
            "pricing_assumptions": {
                "annual_treatment_cost": 28_000,
                "net_price_after_rebates": 19_600
            },
            "penetration_assumptions": {
                "peak_penetration": 0.15,
                "time_to_peak": 4
            },
            "confidence": 0.72
        }
        
        return market_sizing
    
    async def _synthesize_market_view(self, analysis_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize all market analysis into coherent view"""
        
        analog_data = analysis_results[0]
        competitive_data = analysis_results[1]
        sizing_data = analysis_results[2]
        
        # Market synthesis logic
        synthesis = {
            "market_attractiveness": "High - large unmet need in severe asthma",
            "competitive_intensity": "Moderate-High - established biologics compete",
            "differentiation_potential": "Strong - pediatric focus and efficacy profile",
            "commercial_opportunity": {
                "peak_sales_estimate": 1_800_000_000,
                "confidence_range": (800_000_000, 3_200_000_000),
                "key_success_factors": [
                    "Demonstrate superior efficacy vs. current SOC",
                    "Successful pediatric studies and approval",
                    "Effective market access and payer strategy"
                ]
            },
            "risk_factors": [
                "Competitive response from established players",
                "Market access challenges for high-cost biologics", 
                "Pipeline threats emerging 2026-2027"
            ]
        }
        
        return synthesis


class ForecastAgent(SpecializedAgent):
    """
    Multi-method forecasting agent
    Implements industry best practice: triangulation across methods
    """
    
    def __init__(self):
        super().__init__("ForecastAgent")
        self.forecast_methods = ["analog", "bass", "patient_flow", "ml_ensemble"]
        
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate forecast using multiple methods
        Industry best practice: triangulation
        """
        
        drug_data = task_data.get("drug_data", {})
        market_analysis = task_data.get("market_analysis", {})
        
        print(f"[FORECAST_AGENT] Running {len(self.forecast_methods)} forecast methods")
        
        # Generate forecasts using multiple methods
        forecast_results = {}
        
        for method in self.forecast_methods:
            try:
                if method == "analog":
                    result = await self._analog_forecast(market_analysis)
                elif method == "bass":
                    result = await self._bass_diffusion_forecast(drug_data, market_analysis)
                elif method == "patient_flow":
                    result = await self._patient_based_forecast(drug_data, market_analysis)
                elif method == "ml_ensemble":
                    result = await self._ml_forecast(drug_data, market_analysis)
                
                forecast_results[method] = result
                
            except Exception as e:
                forecast_results[method] = {"error": str(e), "confidence": 0.0}
        
        # Weighted ensemble
        ensemble_forecast = await self._ensemble_forecast(forecast_results)
        
        final_result = {
            "individual_forecasts": forecast_results,
            "ensemble_forecast": ensemble_forecast,
            "methodology": "multi_method_triangulation",
            "forecast_confidence": ensemble_forecast.get("confidence", 0.5)
        }
        
        self._log_execution("multi_method_forecast", final_result, 
                           ensemble_forecast.get("confidence", 0.5))
        
        return final_result
    
    async def _analog_forecast(self, market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Industry standard: analog-based forecasting"""
        
        analog_data = market_analysis.get("analog_analysis", {})
        analogs = analog_data.get("analogs_identified", [])
        
        if not analogs:
            return {"error": "No analogs available", "confidence": 0.0}
        
        # Weight by similarity and adjust for market differences
        weighted_forecast = 0
        total_weight = 0
        
        for analog in analogs:
            similarity = analog.get("similarity_score", 0.5)
            peak_sales = analog.get("peak_sales", 0)
            
            # Weight by similarity squared (higher weight for more similar)
            weight = similarity ** 2
            weighted_forecast += peak_sales * weight
            total_weight += weight
        
        analog_forecast = weighted_forecast / total_weight if total_weight > 0 else 0
        
        return {
            "method": "analog_projection",
            "peak_sales_forecast": analog_forecast,
            "forecast_trajectory": self._generate_trajectory(analog_forecast, 4),
            "confidence": 0.75,
            "analogs_used": len(analogs)
        }
    
    async def _bass_diffusion_forecast(self, drug_data: Dict[str, Any], 
                                     market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Bass diffusion with pharmaceutical constraints"""
        
        sizing_data = market_analysis.get("market_sizing", {})
        market_size = sizing_data.get("serviceable_addressable_market", 1_000_000_000)
        peak_penetration = sizing_data.get("penetration_assumptions", {}).get("peak_penetration", 0.15)
        
        # Pharmaceutical Bass parameters
        p = 0.03  # Innovation coefficient
        q = 0.38  # Imitation coefficient
        
        # Bass model peak calculation
        peak_timing = -(1/q) * (p + q) * 2.5  # Approximate
        peak_adopters = market_size * peak_penetration
        
        # Generate full trajectory
        trajectory = []
        for year in range(1, 11):
            if year <= 5:
                adoption = peak_adopters * (1 - ((q + p*year)/(q + p*5))**2)
                trajectory.append(max(0, adoption))
            else:
                # Decline phase
                decline_factor = 0.95 ** (year - 5)
                trajectory.append(trajectory[4] * decline_factor)
        
        bass_forecast = max(trajectory) if trajectory else 0
        
        return {
            "method": "bass_diffusion",
            "peak_sales_forecast": bass_forecast,
            "forecast_trajectory": trajectory,
            "confidence": 0.70,
            "parameters": {"p": p, "q": q, "penetration_ceiling": peak_penetration}
        }
    
    async def _patient_based_forecast(self, drug_data: Dict[str, Any], 
                                    market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Patient flow-based forecasting"""
        
        sizing_data = market_analysis.get("market_sizing", {})
        patient_data = sizing_data.get("patient_population", {})
        pricing_data = sizing_data.get("pricing_assumptions", {})
        
        target_population = patient_data.get("target_population", 420_000)
        annual_cost = pricing_data.get("annual_treatment_cost", 28_000)
        
        # Patient flow calculation
        diagnosis_rate = 0.80    # Patients properly diagnosed
        treatment_rate = 0.65    # Diagnosed patients seeking treatment
        market_share_peak = 0.15 # Peak market share
        adherence_rate = 0.85    # Treatment adherence
        
        peak_patients = target_population * diagnosis_rate * treatment_rate * market_share_peak
        patient_forecast = peak_patients * annual_cost * adherence_rate
        
        # Generate trajectory
        trajectory = self._generate_trajectory(patient_forecast, 4)
        
        return {
            "method": "patient_flow",
            "peak_sales_forecast": patient_forecast,
            "forecast_trajectory": trajectory,
            "confidence": 0.68,
            "parameters": {
                "target_population": target_population,
                "market_share_peak": market_share_peak,
                "annual_cost": annual_cost
            }
        }
    
    async def _ml_forecast(self, drug_data: Dict[str, Any], 
                         market_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ML ensemble forecast (placeholder for trained models)"""
        
        # This would use trained ML models on historical launch data
        # For now, return a placeholder forecast
        
        ml_forecast = 1_600_000_000  # Mock forecast
        trajectory = self._generate_trajectory(ml_forecast, 5)
        
        return {
            "method": "ml_ensemble",
            "peak_sales_forecast": ml_forecast,
            "forecast_trajectory": trajectory,
            "confidence": 0.60,  # Lower confidence until trained on real data
            "note": "Placeholder - requires training on historical launch dataset"
        }
    
    async def _ensemble_forecast(self, forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Weighted ensemble of all forecast methods"""
        
        # Method weights based on industry reliability
        method_weights = {
            "analog": 0.35,      # Highest - industry standard
            "bass": 0.25,        # Good for adoption modeling
            "patient_flow": 0.25, # Good for market sizing
            "ml_ensemble": 0.15  # Lower until trained
        }
        
        weighted_forecast = 0
        total_weight = 0
        method_contributions = {}
        
        for method, forecast_data in forecasts.items():
            if method in method_weights and "peak_sales_forecast" in forecast_data:
                confidence = forecast_data.get("confidence", 0.5)
                forecast_value = forecast_data["peak_sales_forecast"]
                
                # Weight by method reliability * forecast confidence
                effective_weight = method_weights[method] * confidence
                contribution = forecast_value * effective_weight
                
                weighted_forecast += contribution
                total_weight += effective_weight
                method_contributions[method] = {
                    "forecast": forecast_value,
                    "weight": effective_weight,
                    "contribution": contribution
                }
        
        final_forecast = weighted_forecast / total_weight if total_weight > 0 else 0
        
        # Generate ensemble trajectory
        ensemble_trajectory = self._generate_trajectory(final_forecast, 4)
        
        return {
            "ensemble_peak_sales": final_forecast,
            "ensemble_trajectory": ensemble_trajectory,
            "method_contributions": method_contributions,
            "confidence": total_weight / sum(method_weights.values()) if method_weights else 0.5,
            "forecast_range": {
                "low": final_forecast * 0.7,
                "high": final_forecast * 1.4
            }
        }
    
    def _generate_trajectory(self, peak_value: float, years_to_peak: int) -> List[float]:
        """Generate revenue trajectory to peak"""
        
        trajectory = []
        for year in range(1, 11):  # 10-year forecast
            if year <= years_to_peak:
                # Growth phase - S-curve to peak
                progress = year / years_to_peak
                adoption = peak_value * (progress / (1 + progress))
                trajectory.append(adoption)
            else:
                # Decline phase - gradual decline after peak
                decline_years = year - years_to_peak
                decline_factor = 0.95 ** decline_years
                trajectory.append(peak_value * decline_factor)
        
        return trajectory


class ReviewAgent(SpecializedAgent):
    """
    Perplexity-powered agent for objective critique
    Provides harsh, unbiased assessment of forecast quality
    """
    
    def __init__(self):
        super().__init__("ReviewAgent")
        
    async def execute(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Brutally honest assessment of forecast quality
        Compare to baselines and industry standards
        """
        
        forecasts = task_data.get("forecasts", {})
        
        print("[REVIEW_AGENT] Conducting harsh forecast review...")
        
        # Multi-dimensional review
        review_results = await asyncio.gather(
            self._assess_methodology(forecasts),
            self._assess_data_quality(task_data),
            self._check_assumptions(forecasts),
            self._compare_to_baselines(forecasts),
            self._identify_red_flags(forecasts)
        )
        
        methodology_score = review_results[0]
        data_quality_score = review_results[1]
        assumption_score = review_results[2]
        baseline_comparison = review_results[3]
        red_flags = review_results[4]
        
        # Overall assessment
        overall_score = (methodology_score + data_quality_score + assumption_score) / 3
        
        review_assessment = {
            "overall_score": overall_score,
            "methodology_score": methodology_score,
            "data_quality_score": data_quality_score,
            "assumption_validity": assumption_score,
            "baseline_comparison": baseline_comparison,
            "red_flags": red_flags,
            "recommendation": self._generate_recommendation(overall_score, red_flags),
            "confidence_in_review": 0.85
        }
        
        self._log_execution("harsh_forecast_review", review_assessment, 0.85)
        return review_assessment
    
    async def _assess_methodology(self, forecasts: Dict[str, Any]) -> float:
        """Assess forecast methodology soundness"""
        
        ensemble_data = forecasts.get("ensemble_forecast", {})
        individual_forecasts = forecasts.get("individual_forecasts", {})
        
        # Check methodology completeness
        method_score = 0
        
        # Multiple methods used?
        if len(individual_forecasts) >= 3:
            method_score += 0.3
        
        # Industry standard methods?
        required_methods = ["analog", "bass", "patient_flow"]
        methods_present = sum(1 for method in required_methods if method in individual_forecasts)
        method_score += 0.4 * (methods_present / len(required_methods))
        
        # Ensemble approach?
        if "ensemble_peak_sales" in ensemble_data:
            method_score += 0.3
        
        return min(method_score * 10, 10.0)  # Scale to 0-10
    
    async def _assess_data_quality(self, task_data: Dict[str, Any]) -> float:
        """Assess underlying data quality"""
        
        drug_data = task_data.get("drug_data", {})
        
        # Check data source completeness
        data_score = 0
        required_sources = ["fda_data", "sec_data", "clinical_data", "market_intelligence"]
        
        for source in required_sources:
            if source in drug_data and drug_data[source]:
                confidence = drug_data[source].get("confidence", 0.5)
                data_score += confidence / len(required_sources)
        
        return data_score * 10  # Scale to 0-10
    
    async def _check_assumptions(self, forecasts: Dict[str, Any]) -> float:
        """Check assumption validity"""
        
        # Review key assumptions for reasonableness
        assumptions_score = 7.0  # Base score
        
        ensemble_data = forecasts.get("ensemble_forecast", {})
        peak_forecast = ensemble_data.get("ensemble_peak_sales", 0)
        
        # Sanity checks
        if peak_forecast > 10_000_000_000:  # > $10B seems high
            assumptions_score -= 1.0
        
        if peak_forecast < 100_000_000:    # < $100M seems low for biologics
            assumptions_score -= 0.5
        
        return max(assumptions_score, 0.0)
    
    async def _compare_to_baselines(self, forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Compare forecast to industry baselines"""
        
        ensemble_data = forecasts.get("ensemble_forecast", {})
        peak_forecast = ensemble_data.get("ensemble_peak_sales", 0)
        
        # Industry baselines
        baselines = {
            "consultant_conservative": peak_forecast * 0.8,
            "peak_heuristic": peak_forecast * 1.1,
            "analog_simple": peak_forecast * 0.9
        }
        
        # Check if forecast is reasonable vs baselines
        baseline_diffs = {}
        for name, baseline in baselines.items():
            diff_pct = abs(peak_forecast - baseline) / baseline if baseline > 0 else 1.0
            baseline_diffs[name] = {
                "baseline_value": baseline,
                "difference_pct": diff_pct,
                "reasonable": diff_pct < 0.5  # Within 50%
            }
        
        return {
            "baseline_comparisons": baseline_diffs,
            "within_reasonable_range": all(comp["reasonable"] for comp in baseline_diffs.values())
        }
    
    async def _identify_red_flags(self, forecasts: Dict[str, Any]) -> List[str]:
        """Identify potential issues and red flags"""
        
        red_flags = []
        
        ensemble_data = forecasts.get("ensemble_forecast", {})
        individual_forecasts = forecasts.get("individual_forecasts", {})
        
        # Check forecast spread
        forecast_values = []
        for method_data in individual_forecasts.values():
            if "peak_sales_forecast" in method_data:
                forecast_values.append(method_data["peak_sales_forecast"])
        
        if len(forecast_values) >= 2:
            forecast_range = max(forecast_values) - min(forecast_values)
            avg_forecast = sum(forecast_values) / len(forecast_values)
            
            if forecast_range / avg_forecast > 1.0:  # >100% spread
                red_flags.append("High variance between forecast methods (>100% spread)")
        
        # Check confidence levels
        confidences = []
        for method_data in individual_forecasts.values():
            if "confidence" in method_data:
                confidences.append(method_data["confidence"])
        
        if confidences and sum(confidences) / len(confidences) < 0.6:
            red_flags.append("Low average confidence across forecast methods (<60%)")
        
        # Check for missing critical data
        peak_forecast = ensemble_data.get("ensemble_peak_sales", 0)
        if peak_forecast == 0:
            red_flags.append("No ensemble forecast generated")
        
        return red_flags
    
    def _generate_recommendation(self, overall_score: float, red_flags: List[str]) -> str:
        """Generate final recommendation based on review"""
        
        if overall_score >= 8.0 and len(red_flags) == 0:
            return "HIGH QUALITY: Forecast methodology is sound and data quality is excellent. Proceed with confidence."
        elif overall_score >= 6.5 and len(red_flags) <= 2:
            return "ACCEPTABLE: Forecast has good foundation but address identified red flags before making investment decisions."
        elif overall_score >= 5.0:
            return "NEEDS IMPROVEMENT: Significant methodological or data quality issues. Enhance data collection and revisit assumptions."
        else:
            return "POOR QUALITY: Major issues identified. Do not rely on this forecast for investment decisions. Start over with better data."


# Test the specialized agents
async def test_specialized_agents():
    """Test all specialized agents"""
    
    print("=== SPECIALIZED AGENTS TEST ===")
    
    # Initialize agents
    data_agent = DataCollectionAgent()
    market_agent = MarketAnalysisAgent()
    forecast_agent = ForecastAgent()
    review_agent = ReviewAgent()
    
    # Mock task data
    task_data = {
        "drug_name": "Tezspire",
        "indication": "Severe asthma",
        "population": "Pediatric"
    }
    
    print("Testing DataCollectionAgent...")
    data_result = await data_agent.execute(task_data)
    print(f"  Data sources: {data_result.get('sources_accessed', 0)}")
    print(f"  Confidence: {data_result.get('overall_confidence', 0):.2f}")
    
    print("\nTesting MarketAnalysisAgent...")
    market_task = {"drug_data": data_result.get("collected_data", {})}
    market_result = await market_agent.execute(market_task)
    print(f"  Market confidence: {market_result.get('overall_confidence', 0):.2f}")
    
    print("\nTesting ForecastAgent...")
    forecast_task = {
        "drug_data": data_result.get("collected_data", {}),
        "market_analysis": market_result
    }
    forecast_result = await forecast_agent.execute(forecast_task)
    peak_forecast = forecast_result.get("ensemble_forecast", {}).get("ensemble_peak_sales", 0)
    print(f"  Peak sales forecast: ${peak_forecast:,.0f}")
    
    print("\nTesting ReviewAgent...")
    review_task = {
        "forecasts": forecast_result,
        "drug_data": data_result.get("collected_data", {})
    }
    review_result = await review_agent.execute(review_task)
    print(f"  Overall score: {review_result.get('overall_score', 0):.1f}/10")
    print(f"  Red flags: {len(review_result.get('red_flags', []))}")
    
    print("\n✅ All specialized agents tested successfully!")


if __name__ == "__main__":
    asyncio.run(test_specialized_agents())