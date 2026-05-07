#!/usr/bin/env python3
"""
Comprehensive analysis of FDA API data to identify all useful fields
for pharmaceutical forecasting
"""

import json
from src.data.sources import fetch_fda_drug

def analyze_comprehensive_fda_data():
    """Analyze all potentially useful fields in FDA API response"""
    
    # Test with multiple drugs to see data variety
    drugs = ["Keytruda", "Humira", "Eliquis", "Repatha"]
    
    all_fields = {}
    
    for drug in drugs:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {drug}")
        print(f"{'='*60}")
        
        result = fetch_fda_drug(drug)
        
        if 'results' in result and result['results']:
            data = result['results'][0]
            
            # 1. Top-level application info
            print(f"\n1. APPLICATION INFO:")
            app_fields = ['application_number', 'sponsor_name']
            for field in app_fields:
                if field in data:
                    print(f"  {field}: {data[field]}")
                    all_fields[field] = all_fields.get(field, 0) + 1
            
            # 2. OpenFDA section (rich metadata)
            if 'openfda' in data:
                openfda = data['openfda']
                print(f"\n2. OPENFDA METADATA:")
                useful_openfda_fields = [
                    'brand_name', 'generic_name', 'manufacturer_name',
                    'route', 'substance_name', 'pharm_class_epc', 
                    'pharm_class_moa', 'product_type'
                ]
                for field in useful_openfda_fields:
                    if field in openfda:
                        print(f"  {field}: {openfda[field]}")
                        all_fields[f'openfda_{field}'] = all_fields.get(f'openfda_{field}', 0) + 1
            
            # 3. Products section (dosage, route, marketing status)
            if 'products' in data:
                products = data['products']
                print(f"\n3. PRODUCT INFO:")
                print(f"  Number of products: {len(products)}")
                if products:
                    product = products[0]
                    product_fields = [
                        'brand_name', 'active_ingredients', 'dosage_form', 
                        'route', 'marketing_status'
                    ]
                    for field in product_fields:
                        if field in product:
                            print(f"  {field}: {product[field]}")
                            all_fields[f'product_{field}'] = all_fields.get(f'product_{field}', 0) + 1
            
            # 4. Submissions analysis (approval timeline)
            if 'submissions' in data:
                submissions = data['submissions']
                print(f"\n4. SUBMISSIONS ANALYSIS:")
                print(f"  Total submissions: {len(submissions)}")
                
                # Find original approval
                orig_approval = None
                first_supplemental = None
                recent_activity = None
                
                for sub in submissions:
                    date = sub.get('submission_status_date', '')
                    if sub.get('submission_type') == 'ORIG' and sub.get('submission_status') == 'AP':
                        orig_approval = {
                            'date': date,
                            'priority': sub.get('review_priority', ''),
                            'class_code': sub.get('submission_class_code', '')
                        }
                    elif sub.get('submission_type') == 'SUPPL' and sub.get('submission_status') == 'AP' and not first_supplemental:
                        first_supplemental = {'date': date}
                
                # Most recent activity
                if submissions:
                    sorted_subs = sorted([s for s in submissions if s.get('submission_status_date')], 
                                       key=lambda x: x['submission_status_date'], reverse=True)
                    if sorted_subs:
                        recent_activity = {'date': sorted_subs[0]['submission_status_date']}
                
                if orig_approval:
                    print(f"  Original approval: {orig_approval['date']} (Priority: {orig_approval['priority']})")
                    all_fields['original_approval_date'] = all_fields.get('original_approval_date', 0) + 1
                    all_fields['review_priority'] = all_fields.get('review_priority', 0) + 1
                
                if first_supplemental:
                    print(f"  First supplemental: {first_supplemental['date']}")
                    all_fields['first_supplemental_date'] = all_fields.get('first_supplemental_date', 0) + 1
                
                if recent_activity:
                    print(f"  Most recent activity: {recent_activity['date']}")
                    all_fields['most_recent_activity'] = all_fields.get('most_recent_activity', 0) + 1
                
                # Count approved supplementals (indicates expansion success)
                approved_suppls = len([s for s in submissions 
                                     if s.get('submission_type') == 'SUPPL' and s.get('submission_status') == 'AP'])
                print(f"  Approved supplementals: {approved_suppls}")
                all_fields['approved_supplementals_count'] = all_fields.get('approved_supplementals_count', 0) + 1
        
        else:
            print(f"No results found for {drug}")
    
    # Summary of all useful fields found
    print(f"\n{'='*60}")
    print("SUMMARY: USEFUL FIELDS FOUND ACROSS ALL DRUGS")
    print(f"{'='*60}")
    
    for field, count in sorted(all_fields.items(), key=lambda x: x[1], reverse=True):
        print(f"  {field}: found in {count}/{len(drugs)} drugs")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR DATA EXTRACTION")
    print(f"{'='*60}")
    
    print("\n1. CORE APPROVAL DATA:")
    print("   - original_approval_date (ESSENTIAL for temporal splits)")
    print("   - review_priority (Standard/Priority/Fast Track - affects uptake speed)")
    print("   - application_number (unique identifier)")
    
    print("\n2. DRUG CHARACTERISTICS:")
    print("   - openfda_route (oral/injection - affects adoption)")
    print("   - openfda_pharm_class_moa (mechanism of action)")
    print("   - product_dosage_form (tablet/injection - convenience factor)")
    
    print("\n3. COMPETITIVE INTELLIGENCE:")
    print("   - approved_supplementals_count (indication expansion success)")
    print("   - first_supplemental_date (time to first expansion)")
    print("   - most_recent_activity (ongoing development)")
    
    print("\n4. MARKET ACCESS INDICATORS:")
    print("   - sponsor_name/manufacturer_name (company size affects access)")
    print("   - product_marketing_status (Prescription/OTC)")

if __name__ == "__main__":
    analyze_comprehensive_fda_data()