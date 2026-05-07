"""
Test API connectivity for all configured LLM providers
Quick validation that API keys and models are working
"""

import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
sys.path.insert(0, ai_scientist_path)

from model_router import get_router, TaskType

def test_api_connectivity():
    """Test each provider with a simple query"""
    
    router = get_router()
    
    print("=== API CONNECTIVITY TEST ===")
    print(f"Configured providers: {list(router.providers.keys())}")
    print(f"Daily budget: ${router.config['daily_budget_cents']/100:.2f}")
    print()
    
    # Test each task type routing
    test_cases = [
        (TaskType.HYPOTHESIS_GENERATION, "Generate one research hypothesis about pharmaceutical forecasting methods.", "openai"),
        (TaskType.BULK_PARSING, "Parse this: 'pediatric severe asthma biologic' → extract drug_type, indication, population", "deepseek"),
    ]
    
    for task_type, test_prompt, expected_provider in test_cases:
        print(f"Testing {task_type.value}...")
        
        try:
            response = router.generate(
                prompt=test_prompt,
                task_type=task_type,
                system_prompt="You are a pharmaceutical AI researcher. Respond concisely.",
                max_tokens=200,
                temperature=0.3
            )
            
            print(f"[SUCCESS] {response.provider} ({response.model_used})")
            print(f"   Tokens: {response.input_tokens} -> {response.output_tokens}")
            print(f"   Cost: ${response.cost_cents/100:.4f}")
            print(f"   Response preview: {response.content[:100]}...")
            
        except Exception as e:
            print(f"[FAILED] {expected_provider}: {str(e)}")
        
        print()
    
    # Show usage summary
    report = router.get_usage_report()
    print("=== USAGE SUMMARY ===")
    print(f"Total cost: ${report['total_cost_cents']/100:.4f}")
    print(f"Total calls: {report['total_calls']}")
    print(f"Budget remaining: ${report['budget_remaining_cents']/100:.4f}")
    
    if report['by_provider']:
        print("\nBy provider:")
        for provider, stats in report['by_provider'].items():
            print(f"  {provider}: {stats['calls']} calls, ${stats['cost_cents']/100:.4f}")
    
    return report['total_calls'] > 0

if __name__ == "__main__":
    success = test_api_connectivity()
    if success:
        print("\n[PASSED] API connectivity test PASSED!")
        print("Ready to proceed with AI Scientist implementation.")
    else:
        print("\n[ISSUES] API connectivity test had issues.")
        print("Check .env file and API keys.")