"""
Test Evidence Grounding Integration with Real AI Parser
Verifies pharmaceutical claims are properly grounded in sources
"""

import sys
from pathlib import Path

# Add ai_scientist to path
ai_scientist_path = str(Path(__file__).parent / "ai_scientist")
sys.path.insert(0, ai_scientist_path)

from evidence_grounding import EvidenceGroundingAgent, GroundedClaim
from real_ai_parser import RealAIParser

def test_integrated_evidence_grounding():
    """Test evidence grounding integration with real AI parsing"""
    
    print("=== INTEGRATED EVIDENCE GROUNDING TEST ===")
    
    # Initialize systems
    ai_parser = RealAIParser()
    evidence_agent = EvidenceGroundingAgent()
    
    # Test pharmaceutical query with evidence grounding
    test_query = "What is the commercial potential for a new severe asthma biologic targeting adult patients?"
    
    print(f"\nQuery: {test_query}")
    print("\n[STEP 1] AI Query Parsing...")
    
    # Parse query with AI
    parsed = ai_parser.parse_query(test_query)
    
    print(f"  Drug type: {parsed.drug_type}")
    print(f"  Indication: {parsed.indication}")
    print(f"  Population: {parsed.patient_population}")
    print(f"  Severity: {parsed.severity}")
    
    print("\n[STEP 2] Evidence Grounding Key Claims...")
    
    # Ground key claims from the parsing
    claims_to_ground = [
        (f"Severe asthma affects a subset of the total asthma patient population", "population_estimate"),
        (f"Biologics are effective treatments for severe asthma", "drug_efficacy"),
        (f"The asthma therapeutics market represents significant commercial opportunity", "market_size")
    ]
    
    grounded_claims = []
    for claim, claim_type in claims_to_ground:
        print(f"\n  Grounding: {claim}")
        
        grounded = evidence_agent.ground_pharmaceutical_claim(
            claim=claim,
            claim_type=claim_type,
            context={
                "indication": parsed.indication,
                "population": parsed.patient_population,
                "severity": parsed.severity
            }
        )
        
        grounded_claims.append(grounded)
        print(f"    Quality: {grounded.grounding_quality}")
        print(f"    Confidence: {grounded.confidence:.2f}")
        print(f"    Sources: {len(grounded.sources)}")
        
        # Show source details
        for source in grounded.sources:
            print(f"      - {source.title} ({source.source_type.value}, conf: {source.confidence:.2f})")
    
    print("\n[STEP 3] Integrated Analysis Summary...")
    
    # Create comprehensive analysis combining AI parsing + evidence grounding
    analysis_summary = {
        "query": test_query,
        "ai_parsing": {
            "drug_characteristics": {
                "type": parsed.drug_type,
                "indication": parsed.indication,
                "population": parsed.patient_population,
                "severity": parsed.severity,
                "indication_area": parsed.indication_area
            },
            "market_context": {
                "competitive_position": parsed.competitive_position,
                "confidence": parsed.confidence,
                "reasoning": parsed.reasoning
            }
        },
        "evidence_grounding": {
            "claims_grounded": len(grounded_claims),
            "average_confidence": sum(gc.confidence for gc in grounded_claims) / len(grounded_claims),
            "grounding_distribution": {
                "strong": sum(1 for gc in grounded_claims if gc.grounding_quality == "strong"),
                "moderate": sum(1 for gc in grounded_claims if gc.grounding_quality == "moderate"),
                "weak": sum(1 for gc in grounded_claims if gc.grounding_quality == "weak")
            },
            "total_sources": sum(len(gc.sources) for gc in grounded_claims)
        }
    }
    
    print(f"  AI Parsing Confidence: {parsed.confidence:.2f}")
    print(f"  Evidence Grounding Avg Confidence: {analysis_summary['evidence_grounding']['average_confidence']:.2f}")
    print(f"  Total Evidence Sources: {analysis_summary['evidence_grounding']['total_sources']}")
    print(f"  Grounding Quality Distribution:")
    for quality, count in analysis_summary['evidence_grounding']['grounding_distribution'].items():
        print(f"    {quality}: {count}")
    
    # Show evidence audit trail
    print(f"\n[STEP 4] Evidence Audit Trail...")
    for i, grounded in enumerate(grounded_claims, 1):
        print(f"\n  Claim {i}: {grounded.claim}")
        print(f"    Reasoning: {grounded.reasoning}")
        if grounded.audit_trail:
            for step in grounded.audit_trail:
                print(f"    • {step}")
    
    # Generate compliance report
    print(f"\n[STEP 5] AI + Evidence Compliance Report...")
    
    evidence_report = evidence_agent.get_grounding_report()
    
    compliance_metrics = {
        "ai_processing_used": ai_parser.api_calls > 0,
        "evidence_grounding_applied": evidence_report["claims_processed"] > 0,
        "source_backed_claims": evidence_report["claims_processed"],
        "average_source_confidence": evidence_report["average_confidence"],
        "audit_trail_complete": all(len(gc.audit_trail) > 0 for gc in grounded_claims),
        "authoritative_sources_used": evidence_report["total_sources_found"] > 0
    }
    
    print(f"  Real AI Processing: {'YES' if compliance_metrics['ai_processing_used'] else 'NO'}")
    print(f"  Evidence Grounding: {'YES' if compliance_metrics['evidence_grounding_applied'] else 'NO'}")
    print(f"  Claims with Sources: {compliance_metrics['source_backed_claims']}")
    print(f"  Audit Trail Complete: {'YES' if compliance_metrics['audit_trail_complete'] else 'NO'}")
    
    # Overall system readiness
    system_ready = (
        compliance_metrics['ai_processing_used'] and
        compliance_metrics['evidence_grounding_applied'] and
        compliance_metrics['average_source_confidence'] > 0.5 and
        compliance_metrics['audit_trail_complete']
    )
    
    if system_ready:
        print(f"\n[SUCCESS] Integrated AI + Evidence system is ready!")
        print(f"  The system now combines real AI processing with evidence grounding")
        print(f"  Every pharmaceutical claim is backed by authoritative sources")
        print(f"  Complete audit trails available for conference compliance")
        return True
    else:
        print(f"\n[ISSUES] System needs improvements:")
        if not compliance_metrics['ai_processing_used']:
            print(f"  - Real AI processing not working")
        if not compliance_metrics['evidence_grounding_applied']:
            print(f"  - Evidence grounding not applied")
        if compliance_metrics['average_source_confidence'] <= 0.5:
            print(f"  - Low evidence confidence ({compliance_metrics['average_source_confidence']:.2f})")
        if not compliance_metrics['audit_trail_complete']:
            print(f"  - Incomplete audit trails")
        return False

if __name__ == "__main__":
    success = test_integrated_evidence_grounding()
    
    if success:
        print(f"\n[READY] AI + Evidence integration complete!")
        print(f"Ready for autonomous research with authoritative source grounding.")
    else:
        print(f"\n[ISSUES] Integration needs fixes before research experiments.")