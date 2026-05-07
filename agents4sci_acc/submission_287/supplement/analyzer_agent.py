# Analyzer Agent: Processes drug extraction files and creates aggregated side effects analysis

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

def analyze_side_effects(aggregate_folder, output_file):
    aggregate_path = Path(aggregate_folder)
    
    side_effects_data = defaultdict(lambda: defaultdict(list))
    
    json_files = list(aggregate_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {aggregate_folder}")
        return
    
    print(f"Processing {len(json_files)} drug files...")
    
    total_extractions = 0
    
    for file_path in json_files:
        try:
            drug_canonical = file_path.stem
            
            with open(file_path, 'r', encoding='utf-8') as f:
                extractions = json.load(f)
            
            print(f"Processing {drug_canonical}: {len(extractions)} extractions")
            total_extractions += len(extractions)
            
            for extraction in extractions:
                side_effect = extraction.get('side_effect_medical', 'unknown')
                if(side_effect == 'unknown'):
                    side_effect = extraction.get('side_effect', 'unknown')
                
                metrics = {
                    'temporal_weight': extraction.get('temporal_weight'),
                    'confidence': extraction.get('confidence'),
                    'community_metric': extraction.get('community_metric'),
                    'confounders': extraction.get('confounders'),
                    'quote': extraction.get('quote')
                }
                
                side_effects_data[drug_canonical][side_effect].append(metrics)
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file {file_path}: {e}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    output_data = {
        drug: dict(side_effects) 
        for drug, side_effects in side_effects_data.items()
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nAnalysis complete!")
        print(f"Processed {total_extractions} total extractions")
        print(f"Found {len(output_data)} unique drugs")
        
        total_drug_side_effect_pairs = sum(len(side_effects) for side_effects in output_data.values())
        print(f"Created {total_drug_side_effect_pairs} drug-side_effect combinations")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error writing output file {output_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze side effects from aggregated drug extraction files'
    )
    
    parser.add_argument(
        '--aggregate_folder',
        help='Path to the folder containing aggregated drug JSON files'
    )
    
    parser.add_argument(
        '--output-file',
        '-o',
        default='side_effects_analysis.json',
        help='Path to the output JSON file (default: side_effects_analysis.json)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.aggregate_folder):
        print(f"Error: Aggregate folder '{args.aggregate_folder}' does not exist")
        return 1
    
    if not os.path.isdir(args.aggregate_folder):
        print(f"Error: '{args.aggregate_folder}' is not a directory")
        return 1
    
    analyze_side_effects(args.aggregate_folder, args.output_file)
    return 0

if __name__ == '__main__':
    exit(main())