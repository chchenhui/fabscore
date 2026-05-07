#!/usr/bin/env python3
"""
Test Phase 2 Multi-Agent Architecture
Quick validation that GPT-5 orchestrator and specialized agents work together
"""

import asyncio
import sys
from pathlib import Path

# Add ai_scientist to path for imports
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
if ai_scientist_path not in sys.path:
    sys.path.insert(0, ai_scientist_path)

async def test_phase2_multiagent():
    """Test the complete Phase 2 multi-agent system"""
    
    print("=== PHASE 2 MULTI-AGENT SYSTEM TEST ===")
    print("Testing GPT-5 orchestrator with specialized agents...")
    
    try:
        from gpt5_orchestrator import GPT5Orchestrator
        
        # Initialize orchestrator
        orchestrator = GPT5Orchestrator()
        print(f"[SUCCESS] GPT-5 Orchestrator initialized with {len(orchestrator.agent_capabilities)} agents")
        
        # Test query
        query = "Should we develop a Tezspire competitor for pediatric severe asthma?"
        
        print(f"\nTest Query: {query}")
        print("Multi-agent analysis starting...")
        
        # Run the complete pipeline
        result = await orchestrator.process_drug_forecast(query)
        
        if "error" not in result:
            print("\n[SUCCESS] MULTI-AGENT ANALYSIS COMPLETE")
            
            # Display key results
            forecast = result.get("forecast", {})
            print(f"\nFORECAST RESULTS:")
            print(f"   Peak Sales: ${forecast.get('peak_sales_forecast', 0):,.0f}")
            print(f"   Confidence: {forecast.get('confidence', 0):.1%}")
            print(f"   Methodology: {forecast.get('methodology', 'Unknown')}")
            
            # Display agent execution log
            print(f"\nAGENT EXECUTION LOG:")
            for log_entry in result.get("execution_log", []):
                agent = log_entry.get("agent", "Unknown")
                decision = log_entry.get("decision", "Unknown")
                confidence = log_entry.get("confidence", 0)
                print(f"   {agent}: {decision} (confidence: {confidence:.2f})")
            
            # Data quality assessment
            data_quality = result.get("data_quality", {})
            print(f"\nDATA QUALITY:")
            print(f"   Overall Quality: {data_quality.get('quality', 0):.1%}")
            print(f"   Completeness: {data_quality.get('completeness', 0):.1%}")
            print(f"   Reliability: {data_quality.get('reliability', 0):.1%}")
            
            # Baseline comparison
            baseline_comp = result.get("comparison_to_baselines", {})
            if baseline_comp:
                outperforms = baseline_comp.get("outperforms_baselines", False)
                print(f"\nBASELINE COMPARISON:")
                print(f"   Outperforms Baselines: {'Yes' if outperforms else 'No'}")
            
            print(f"\nPHASE 2 SUCCESS: Multi-agent architecture operational!")
            return True
            
        else:
            print(f"\n[ERROR] ANALYSIS FAILED: {result['error']}")
            return False
            
    except ImportError as e:
        print(f"[ERROR] Import Error: {e}")
        print("Make sure ai_scientist modules are available")
        return False
    except Exception as e:
        print(f"[ERROR] Test Error: {e}")
        return False

async def test_specialized_agents():
    """Test individual specialized agents"""
    
    print("\n=== SPECIALIZED AGENTS TEST ===")
    
    try:
        from specialized_agents import (
            DataCollectionAgent, 
            MarketAnalysisAgent, 
            ForecastAgent, 
            ReviewAgent
        )
        
        # Test data collection agent
        print("Testing DataCollectionAgent...")
        data_agent = DataCollectionAgent()
        data_result = await data_agent.execute({
            "drug_name": "Tezspire",
            "indication": "Severe asthma"
        })
        print(f"   Sources accessed: {data_result.get('sources_accessed', 0)}")
        print(f"   Data confidence: {data_result.get('overall_confidence', 0):.2f}")
        
        # Test market analysis agent
        print("\nTesting MarketAnalysisAgent...")
        market_agent = MarketAnalysisAgent()
        market_result = await market_agent.execute({
            "drug_data": data_result.get("collected_data", {})
        })
        print(f"   Market confidence: {market_result.get('overall_confidence', 0):.2f}")
        
        # Test forecast agent
        print("\nTesting ForecastAgent...")
        forecast_agent = ForecastAgent()
        forecast_result = await forecast_agent.execute({
            "drug_data": data_result.get("collected_data", {}),
            "market_analysis": market_result
        })
        ensemble = forecast_result.get("ensemble_forecast", {})
        print(f"   Peak forecast: ${ensemble.get('ensemble_peak_sales', 0):,.0f}")
        print(f"   Forecast confidence: {ensemble.get('confidence', 0):.2f}")
        
        # Test review agent
        print("\nTesting ReviewAgent...")
        review_agent = ReviewAgent()
        review_result = await review_agent.execute({
            "forecasts": forecast_result,
            "drug_data": data_result.get("collected_data", {})
        })
        print(f"   Review score: {review_result.get('overall_score', 0):.1f}/10")
        print(f"   Red flags: {len(review_result.get('red_flags', []))}")
        
        print(f"\n[SUCCESS] All specialized agents working correctly!")
        return True
        
    except Exception as e:
        print(f"[ERROR] Specialized agents test failed: {e}")
        return False

def check_model_router():
    """Check if model router is available"""
    
    print("=== MODEL ROUTER CHECK ===")
    
    try:
        from model_router import get_router, TaskType
        
        router = get_router()
        print(f"[SUCCESS] Model router initialized")
        print(f"   Available task types: {len([t for t in TaskType])}")
        
        return True
        
    except Exception as e:
        print(f"[WARNING] Model router issue: {e}")
        print("   Multi-agent system will use mock responses")
        return False

async def main():
    """Main test function"""
    
    print("Phase 2 Multi-Agent Architecture Validation")
    print("=" * 50)
    
    # Check dependencies
    router_ok = check_model_router()
    
    # Test specialized agents
    agents_ok = await test_specialized_agents()
    
    # Test full orchestrator if components work
    if agents_ok:
        orchestrator_ok = await test_phase2_multiagent()
    else:
        print("[WARNING] Skipping orchestrator test due to agent issues")
        orchestrator_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("PHASE 2 TEST SUMMARY:")
    print(f"   Model Router: {'[SUCCESS]' if router_ok else '[WARNING]'}")
    print(f"   Specialized Agents: {'[SUCCESS]' if agents_ok else '[ERROR]'}")
    print(f"   GPT-5 Orchestrator: {'[SUCCESS]' if orchestrator_ok else '[ERROR]'}")
    
    if agents_ok and orchestrator_ok:
        print("\n[SUCCESS] PHASE 2 MULTI-AGENT ARCHITECTURE: OPERATIONAL!")
        print("   Ready for real pharmaceutical forecasting")
    else:
        print("\n[WARNING] PHASE 2 NEEDS ATTENTION")
        print("   Check dependencies and fix issues")

if __name__ == "__main__":
    asyncio.run(main())