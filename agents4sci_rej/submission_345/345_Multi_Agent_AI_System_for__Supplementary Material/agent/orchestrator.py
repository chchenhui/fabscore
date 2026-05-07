"""
AI Agent Orchestrator for Commercial Forecast System.

This module implements the main LangGraph workflow that orchestrates
the AI planning, tool execution, and reasoning for pharmaceutical 
commercial forecasting.
"""

import os
from typing import Dict, Any, List, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

from .planner import CommercialForecastPlanner, AnalysisPlan
from .tools import CommercialForecastTools, AnalysisState


class AgentState(TypedDict):
    """State object that flows through the LangGraph workflow."""
    query: str
    analysis_plan: AnalysisPlan
    tools_state: AnalysisState
    results: Dict[str, Any]
    recommendation: Dict[str, Any]
    error: str
    reasoning_trace: List[str]


class CommercialForecastAgent:
    """Main AI Agent orchestrator for commercial forecasting."""
    
    def __init__(self, api_key: str = None):
        """Initialize the agent with all components."""
        self.planner = CommercialForecastPlanner(api_key)
        self.tools = CommercialForecastTools()
        self.llm = ChatAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-sonnet-20240229"
        )
        
        # Build the LangGraph workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow for the agent."""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planning_node)
        workflow.add_node("market_analysis", self._market_analysis_node)
        workflow.add_node("financial_modeling", self._financial_modeling_node)
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("recommendation", self._recommendation_node)
        
        # Define the flow
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "market_analysis")
        workflow.add_edge("market_analysis", "financial_modeling")
        workflow.add_edge("financial_modeling", "risk_assessment")
        workflow.add_edge("risk_assessment", "recommendation")
        workflow.add_edge("recommendation", END)
        
        return workflow.compile()
    
    def _planning_node(self, state: AgentState) -> AgentState:
        """AI Planning node - creates analysis plan from natural language query."""
        
        try:
            query = state["query"]
            state["reasoning_trace"] = [f"🤖 **AI PLANNING**: Analyzing query: '{query}'"]
            
            # Create analysis plan
            plan = self.planner.create_analysis_plan(query)
            state["analysis_plan"] = plan
            
            # Initialize tools state
            state["tools_state"] = AnalysisState()
            
            # Add planning reasoning to trace
            state["reasoning_trace"].extend([
                f"📋 **PLAN CREATED**: {len(plan.steps)} steps identified",
                f"🎯 **TARGET**: {plan.drug_characteristics.name} ({plan.drug_characteristics.drug_type.value})",
                f"📈 **CONFIDENCE**: {plan.confidence_level.upper()}",
                ""
            ])
            
            # Add parameter reasoning
            state["reasoning_trace"].append("🧠 **PARAMETER REASONING**:")
            for reason in plan.reasoning:
                state["reasoning_trace"].append(f"   • {reason}")
            state["reasoning_trace"].append("")
            
        except Exception as e:
            state["error"] = f"Planning failed: {str(e)}"
            
        return state
    
    def _market_analysis_node(self, state: AgentState) -> AgentState:
        """Market Analysis node - sizes market and estimates adoption parameters."""
        
        try:
            plan = state["analysis_plan"]
            tools_state = state["tools_state"]
            
            state["reasoning_trace"].append("📊 **STEP 1: MARKET ANALYSIS**")
            
            # Convert plan characteristics to dict
            char_dict = {
                "name": plan.drug_characteristics.name,
                "drug_type": plan.drug_characteristics.drug_type.value,
                "indication_area": plan.drug_characteristics.indication_area.value,
                "severity": plan.drug_characteristics.severity,
                "patient_population": plan.drug_characteristics.patient_population
            }
            
            # Run intelligent market sizing
            market_size = self.tools.intelligent_market_sizing(char_dict, tools_state)
            
            # Run adoption parameter estimation
            bass_p, bass_q = self.tools.intelligent_adoption_parameters(char_dict, tools_state)
            
            # Run Bass diffusion analysis
            pricing_info = self.tools.intelligent_pricing(char_dict, tools_state)
            bass_results = self.tools.run_bass_analysis(
                market_size, bass_p, bass_q, pricing_info["adoption_ceiling"]
            )
            
            # Update state
            state["tools_state"] = tools_state
            
            # Add reasoning to trace
            for reasoning in tools_state.get_reasoning_summary()[-3:]:  # Last 3 steps
                state["reasoning_trace"].append(f"   {reasoning}")
            state["reasoning_trace"].append("")
            
        except Exception as e:
            state["error"] = f"Market analysis failed: {str(e)}"
            
        return state
    
    def _financial_modeling_node(self, state: AgentState) -> AgentState:
        """Financial Modeling node - runs NPV analysis."""
        
        try:
            tools_state = state["tools_state"]
            
            state["reasoning_trace"].append("💰 **STEP 2: FINANCIAL MODELING**")
            
            # Get results from previous step
            bass_results = tools_state.results["bass_analysis"]
            pricing_info = tools_state.parameters_used["pricing"]["value"]
            
            # Run financial analysis
            financial_results = self.tools.run_financial_analysis(
                bass_results["adopters"], pricing_info
            )
            
            # Update state
            state["tools_state"] = tools_state
            
            # Add reasoning to trace
            latest_reasoning = tools_state.get_reasoning_summary()[-1]  # Latest reasoning
            state["reasoning_trace"].append(f"   {latest_reasoning}")
            state["reasoning_trace"].append("")
            
        except Exception as e:
            state["error"] = f"Financial modeling failed: {str(e)}"
            
        return state
    
    def _risk_assessment_node(self, state: AgentState) -> AgentState:
        """Risk Assessment node - runs Monte Carlo simulation."""
        
        try:
            tools_state = state["tools_state"] 
            
            state["reasoning_trace"].append("🎲 **STEP 3: RISK ASSESSMENT**")
            
            # Prepare Monte Carlo parameters
            bass_results = tools_state.results["bass_analysis"]
            pricing_info = tools_state.parameters_used["pricing"]["value"]
            
            base_params = {
                'adopters': bass_results["adopters"],
                'list_price_monthly': pricing_info['list_price'],
                'gtn_pct': pricing_info['gtn_pct'],
                'cogs_pct': self.tools.config['economics']['cogs_pct'],
                'sga_launch': self.tools.config['economics']['sga_launch_annual'] / 4,
                'sga_decay_to_pct': 0.5,
                'adherence_rate': 0.85
            }
            
            # Run Monte Carlo
            mc_results = self.tools.run_monte_carlo(base_params, n_simulations=1000)
            
            # Update state
            state["tools_state"] = tools_state
            
            # Add reasoning to trace
            if "error" not in mc_results:
                latest_reasoning = tools_state.get_reasoning_summary()[-1]
                state["reasoning_trace"].append(f"   {latest_reasoning}")
            else:
                state["reasoning_trace"].append(f"   ⚠️ Monte Carlo simulation failed: {mc_results['error']}")
            state["reasoning_trace"].append("")
            
        except Exception as e:
            state["error"] = f"Risk assessment failed: {str(e)}"
            
        return state
    
    def _recommendation_node(self, state: AgentState) -> AgentState:
        """Recommendation node - generates final investment decision."""
        
        try:
            plan = state["analysis_plan"]
            tools_state = state["tools_state"]
            
            state["reasoning_trace"].append("⚖️ **STEP 4: INVESTMENT DECISION**")
            
            # Convert characteristics for recommendation
            char_dict = {
                "name": plan.drug_characteristics.name,
                "drug_type": plan.drug_characteristics.drug_type.value,
                "indication_area": plan.drug_characteristics.indication_area.value,
                "severity": plan.drug_characteristics.severity,
                "patient_population": plan.drug_characteristics.patient_population
            }
            
            # Generate final recommendation
            recommendation = self.tools.generate_recommendation(char_dict)
            
            state["recommendation"] = recommendation
            state["tools_state"] = tools_state
            state["results"] = tools_state.results
            
            # Add final reasoning
            final_reasoning = tools_state.get_reasoning_summary()[-1]
            state["reasoning_trace"].append(f"   {final_reasoning}")
            state["reasoning_trace"].append("")
            
            # Add summary
            decision = recommendation["decision"]
            npv = recommendation["key_metrics"]["npv_billions"]
            success_rate = recommendation["key_metrics"]["success_rate"]
            
            state["reasoning_trace"].extend([
                "🎯 **FINAL RECOMMENDATION**:",
                f"   **Decision**: {decision}",
                f"   **NPV**: ${npv:.1f}B",
                f"   **Success Rate**: {success_rate:.0%}",
                f"   **Rationale**: {recommendation['rationale']}"
            ])
            
        except Exception as e:
            state["error"] = f"Recommendation generation failed: {str(e)}"
            
        return state
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """Main method to run the complete AI analysis."""
        
        # Initialize state
        initial_state = AgentState(
            query=query,
            analysis_plan=None,
            tools_state=None,
            results={},
            recommendation={},
            error="",
            reasoning_trace=[]
        )
        
        # Run the workflow
        result = self.workflow.invoke(initial_state)
        
        # Format the output
        output = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "success": not bool(result.get("error")),
            "error": result.get("error", ""),
            "recommendation": result.get("recommendation", {}),
            "results": result.get("results", {}),
            "reasoning_trace": result.get("reasoning_trace", []),
            "analysis_plan": result.get("analysis_plan")
        }
        
        return output


# Demo function for testing
def demo_ai_agent():
    """Demonstrate the AI agent with a sample query."""
    
    print("=== AI COMMERCIAL FORECAST AGENT DEMO ===")
    
    # Initialize agent
    agent = CommercialForecastAgent()
    
    # Test query
    query = "Should we develop a Tezspire competitor for pediatric severe asthma?"
    
    print(f"🎯 Query: {query}")
    print("🤖 AI Agent analyzing...")
    print()
    
    # Run analysis
    result = agent.analyze(query)
    
    # Display results
    if result["success"]:
        print("✅ Analysis Complete!")
        print()
        
        # Show AI reasoning trace
        print("🧠 AI REASONING TRACE:")
        for step in result["reasoning_trace"]:
            print(step)
        print()
        
        # Show recommendation
        rec = result["recommendation"]
        print("🎯 FINAL RECOMMENDATION:")
        print(f"   Decision: {rec['decision']}")  
        print(f"   NPV: ${rec['key_metrics']['npv_billions']:.1f}B")
        print(f"   Success Rate: {rec['key_metrics']['success_rate']:.0%}")
        print(f"   Rationale: {rec['rationale']}")
        
    else:
        print(f"❌ Analysis Failed: {result['error']}")
    
    return result


if __name__ == "__main__":
    demo_ai_agent()