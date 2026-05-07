#!/usr/bin/env python3
"""
Optimized UnitMath Evaluation System
Building on the enhanced version (52.5% F1) with improvements
"""

import json
import re
import argparse
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from difflib import SequenceMatcher
from collections import defaultdict
import math

class PredictionLabel(Enum):
    SUPPORTED = "Supported"
    REFUTED = "Refuted"
    NEI = "NEI"

@dataclass
class NumericValue:
    """Represents a numeric value with context"""
    value: float
    text: str
    context: str
    entity: Optional[str] = None
    unit: Optional[str] = None
    is_percentage: bool = False
    
@dataclass
class DetailedReasoning:
    """Enhanced reasoning with structured comparisons"""
    prediction: str
    confidence: float
    steps: List[str]
    evidence: List[str]
    comparisons: List[Any]
    numeric_matches: List[Any]
    reason: str

class OptimizedTableReasoner:
    """Optimized table reasoning with balanced predictions"""
    
    def __init__(self, binary_mode: bool = True):
        self.binary_mode = binary_mode
        
        # Comprehensive patterns
        self.patterns = {
            'superlative': {
                'positive': r'\b(best|highest|most|greatest|largest|top|maximum|peak|superior|optimal)\b',
                'negative': r'\b(worst|lowest|least|smallest|minimum|bottom|inferior|poorest)\b'
            },
            'comparison': {
                'better': r'\b(better|higher|greater|more|outperform[s]?|exceed[s]?|surpass|superior|improve)\b',
                'worse': r'\b(worse|lower|less|underperform[s]?|inferior|below|drop|decrease)\b',
                'equal': r'\b(same|equal|similar|comparable|match|consistent|close)\b'
            },
            'change': {
                'increase': r'\b(increase[sd]?|improv[e]?[sd]?|gain|boost|grow|enhance|rise)\b',
                'decrease': r'\b(decrease[sd]?|drop|fall|reduce|decline|lower|diminish)\b',
                'stable': r'\b(stable|maintain|constant|unchanged|steady)\b'
            },
            'significance': r'\b(significant[ly]?|substantial[ly]?|notable|considerably)\b',
            'negation': r'\b(not|no|n\'t|never|neither|none|without|fail[s]?|cannot)\b'
        }
        
    def extract_numeric_values(self, text: str) -> List[NumericValue]:
        """Enhanced numeric extraction"""
        values = []
        
        # Multiple patterns for different formats
        patterns = [
            (r'(\d+\.?\d*)\s*%', True),  # Percentages
            (r'(\d+\.\d+)', False),       # Decimals
            (r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', False),  # Numbers with commas
            (r'(\d+)', False),            # Integers
            (r'(\d+)\s*/\s*(\d+)', False), # Fractions
        ]
        
        seen_positions = set()
        
        for pattern, is_pct in patterns:
            for match in re.finditer(pattern, text):
                if match.start() in seen_positions:
                    continue
                    
                try:
                    if '/' in match.group():
                        parts = match.group().split('/')
                        value = float(parts[0]) / float(parts[1])
                    elif ',' in match.group():
                        value = float(match.group().replace(',', ''))
                    else:
                        value = float(match.group(1))
                    
                    # Get context
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    context = text[start:end]
                    
                    values.append(NumericValue(
                        value=value,
                        text=match.group(),
                        context=context,
                        is_percentage=is_pct
                    ))
                    
                    seen_positions.add(match.start())
                except:
                    continue
                    
        return values
    
    def extract_entities_from_table(self, table: List[List[str]]) -> Dict[str, List[float]]:
        """Extract entities and their values"""
        entities = {}
        
        if not table:
            return entities
            
        # Process each row
        for row in table:
            if row and isinstance(row[0], str):
                # Clean entity name
                entity = self._clean_entity_name(row[0])
                if entity and len(entity) > 1:
                    values = []
                    
                    for cell in row[1:]:
                        numeric_vals = self.extract_numeric_values(str(cell))
                        values.extend([v.value for v in numeric_vals])
                    
                    if values:
                        entities[entity] = values
                        
        # Also try column-wise extraction
        if table and len(table) > 1:
            headers = table[0]
            for col_idx, header in enumerate(headers[1:], 1):
                if isinstance(header, str):
                    entity = self._clean_entity_name(header)
                    if entity and len(entity) > 1 and entity not in entities:
                        values = []
                        for row in table[1:]:
                            if col_idx < len(row):
                                numeric_vals = self.extract_numeric_values(str(row[col_idx]))
                                values.extend([v.value for v in numeric_vals])
                        if values:
                            entities[entity] = values
                            
        return entities
    
    def _clean_entity_name(self, name: str) -> str:
        """Clean entity name"""
        # Remove markup
        cleaned = re.sub(r'\[.*?\]', '', name)
        cleaned = re.sub(r'<.*?>', '', cleaned)
        # Normalize
        cleaned = re.sub(r'[^a-zA-Z0-9\s\-_+]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip().lower()
        return cleaned
    
    def fuzzy_match_entity(self, entity: str, text: str, threshold: float = 0.7) -> bool:
        """Fuzzy matching for entities"""
        entity_clean = entity.lower()
        text_clean = text.lower()
        
        # Direct substring
        if entity_clean in text_clean:
            return True
            
        # Word-level matching
        entity_words = set(entity_clean.split())
        text_words = set(text_clean.split())
        
        if entity_words and text_words:
            overlap = len(entity_words.intersection(text_words))
            if overlap >= len(entity_words) * 0.6:
                return True
        
        # Sequence matching for longer entities
        if len(entity_clean) > 3:
            for word in text_clean.split():
                if len(word) > 3:
                    similarity = SequenceMatcher(None, entity_clean, word).ratio()
                    if similarity >= threshold:
                        return True
                        
        return False
    
    def verify_numeric_claim(self, claim_values: List[NumericValue], 
                            table_values: List[NumericValue]) -> Tuple[bool, float, List[str]]:
        """Verify numeric claims against table"""
        if not claim_values or not table_values:
            return False, 0.0, []
            
        matches = []
        matched_count = 0
        
        for claim_val in claim_values:
            best_match = None
            best_score = 0
            
            for table_val in table_values:
                # Exact match
                if abs(claim_val.value - table_val.value) < 0.01:
                    best_match = table_val
                    best_score = 1.0
                    break
                    
                # Percentage conversion
                if claim_val.is_percentage != table_val.is_percentage:
                    if claim_val.is_percentage and abs(claim_val.value - table_val.value * 100) < 0.1:
                        best_match = table_val
                        best_score = 0.95
                        break
                    elif not claim_val.is_percentage and abs(claim_val.value * 100 - table_val.value) < 0.1:
                        best_match = table_val
                        best_score = 0.95
                        break
                        
                # Approximate match (within 2%)
                if table_val.value != 0:
                    ratio = claim_val.value / table_val.value
                    if 0.98 <= ratio <= 1.02:
                        best_match = table_val
                        best_score = 0.9
                        
            if best_match and best_score >= 0.9:
                matched_count += 1
                matches.append(f"{claim_val.value} matched with {best_match.value}")
            else:
                matches.append(f"{claim_val.value} not found")
                
        confidence = matched_count / len(claim_values) if claim_values else 0
        verified = matched_count > 0
        
        return verified, confidence, matches
    
    def analyze_claim_type(self, claim: str) -> Dict[str, Any]:
        """Analyze claim characteristics"""
        claim_lower = claim.lower()
        analysis = {
            'has_negation': False,
            'is_superlative': False,
            'is_comparison': False,
            'is_change': False,
            'is_significance': False,
            'direction': None
        }
        
        # Check patterns
        if re.search(self.patterns['negation'], claim_lower):
            analysis['has_negation'] = True
            
        for direction, pattern in self.patterns['superlative'].items():
            if re.search(pattern, claim_lower):
                analysis['is_superlative'] = True
                analysis['direction'] = direction
                break
                
        for comp_type, pattern in self.patterns['comparison'].items():
            if re.search(pattern, claim_lower):
                analysis['is_comparison'] = True
                if not analysis['direction']:
                    analysis['direction'] = comp_type
                break
                
        for change_type, pattern in self.patterns['change'].items():
            if re.search(pattern, claim_lower):
                analysis['is_change'] = True
                if not analysis['direction']:
                    analysis['direction'] = change_type
                break
                
        if re.search(self.patterns['significance'], claim_lower):
            analysis['is_significance'] = True
            
        return analysis
    
    def predict(self, claim: str, table: List[List[str]]) -> Tuple[str, float, DetailedReasoning]:
        """Main prediction with balanced logic"""
        
        steps = []
        evidence = []
        comparisons = []
        numeric_matches = []
        
        # Extract information
        steps.append("Extracting numeric values and entities")
        claim_values = self.extract_numeric_values(claim)
        
        table_text = ' '.join([' '.join([str(cell) for cell in row]) for row in table])
        table_values = self.extract_numeric_values(table_text)
        
        table_entities = self.extract_entities_from_table(table)
        
        evidence.append(f"Found {len(claim_values)} numbers in claim")
        evidence.append(f"Found {len(table_values)} numbers in table")
        evidence.append(f"Found {len(table_entities)} entities in table")
        
        # Analyze claim
        claim_analysis = self.analyze_claim_type(claim)
        
        # Initialize prediction - start neutral
        prediction = PredictionLabel.NEI
        confidence = 0.5
        reason = "Analyzing..."
        
        # Priority 1: Numeric verification (most reliable)
        if claim_values and table_values:
            steps.append("Verifying numeric claims")
            verified, num_conf, matches = self.verify_numeric_claim(claim_values, table_values)
            numeric_matches = matches
            
            if verified and num_conf >= 0.5:
                if claim_analysis['has_negation']:
                    prediction = PredictionLabel.REFUTED
                    confidence = 0.65 + num_conf * 0.15
                    reason = "Negated claim contradicted by matching numbers"
                else:
                    prediction = PredictionLabel.SUPPORTED
                    confidence = 0.6 + num_conf * 0.2
                    reason = f"Numeric evidence found ({int(num_conf * len(claim_values))}/{len(claim_values)} matches)"
            elif num_conf == 0 and len(claim_values) > 0:
                # No matches at all
                if claim_analysis['has_negation']:
                    prediction = PredictionLabel.SUPPORTED
                    confidence = 0.6
                    reason = "Negated claim supported (no matching numbers)"
                else:
                    prediction = PredictionLabel.REFUTED
                    confidence = 0.55
                    reason = "No numeric matches found"
        
        # Priority 2: Superlative claims
        if prediction == PredictionLabel.NEI and claim_analysis['is_superlative'] and table_entities:
            steps.append("Analyzing superlative claim")
            
            # Find mentioned entities
            mentioned = []
            for entity in table_entities:
                if self.fuzzy_match_entity(entity, claim):
                    mentioned.append(entity)
            
            if mentioned:
                # Rank entities
                entity_ranks = [(e, np.mean(v)) for e, v in table_entities.items() if v]
                entity_ranks.sort(key=lambda x: x[1], reverse=True)
                
                for mentioned_entity in mentioned:
                    rank = next((i for i, (e, _) in enumerate(entity_ranks) if e == mentioned_entity), None)
                    
                    if rank is not None:
                        if claim_analysis['direction'] == 'positive':
                            if rank == 0:
                                prediction = PredictionLabel.SUPPORTED
                                confidence = 0.75
                                reason = f"{mentioned_entity} is highest ranked"
                            elif rank <= 2:
                                prediction = PredictionLabel.SUPPORTED
                                confidence = 0.6
                                reason = f"{mentioned_entity} ranks high ({rank+1}/{len(entity_ranks)})"
                            else:
                                prediction = PredictionLabel.REFUTED
                                confidence = 0.65
                                reason = f"{mentioned_entity} doesn't rank high ({rank+1}/{len(entity_ranks)})"
                        elif claim_analysis['direction'] == 'negative':
                            if rank == len(entity_ranks) - 1:
                                prediction = PredictionLabel.SUPPORTED
                                confidence = 0.75
                                reason = f"{mentioned_entity} is lowest ranked"
                            elif rank >= len(entity_ranks) - 3:
                                prediction = PredictionLabel.SUPPORTED
                                confidence = 0.6
                                reason = f"{mentioned_entity} ranks low ({rank+1}/{len(entity_ranks)})"
                            else:
                                prediction = PredictionLabel.REFUTED
                                confidence = 0.65
                                reason = f"{mentioned_entity} doesn't rank low ({rank+1}/{len(entity_ranks)})"
                        break
        
        # Priority 3: Comparison claims
        if prediction == PredictionLabel.NEI and claim_analysis['is_comparison'] and table_entities:
            steps.append("Analyzing comparison claim")
            
            # Find mentioned entities
            mentioned = []
            for entity in table_entities:
                if self.fuzzy_match_entity(entity, claim, 0.65):
                    mentioned.append(entity)
            
            if len(mentioned) >= 2:
                entity1, entity2 = mentioned[0], mentioned[1]
                values1 = table_entities.get(entity1, [])
                values2 = table_entities.get(entity2, [])
                
                if values1 and values2:
                    avg1, avg2 = np.mean(values1), np.mean(values2)
                    
                    if claim_analysis['direction'] == 'better':
                        if avg1 > avg2:
                            prediction = PredictionLabel.SUPPORTED
                            confidence = 0.7
                            reason = f"{entity1} ({avg1:.2f}) > {entity2} ({avg2:.2f})"
                        else:
                            prediction = PredictionLabel.REFUTED
                            confidence = 0.65
                            reason = f"{entity1} ({avg1:.2f}) ≤ {entity2} ({avg2:.2f})"
                    elif claim_analysis['direction'] == 'worse':
                        if avg1 < avg2:
                            prediction = PredictionLabel.SUPPORTED
                            confidence = 0.7
                            reason = f"{entity1} ({avg1:.2f}) < {entity2} ({avg2:.2f})"
                        else:
                            prediction = PredictionLabel.REFUTED
                            confidence = 0.65
                            reason = f"{entity1} ({avg1:.2f}) ≥ {entity2} ({avg2:.2f})"
                    elif claim_analysis['direction'] == 'equal':
                        if abs(avg1 - avg2) / max(avg1, avg2) < 0.05:
                            prediction = PredictionLabel.SUPPORTED
                            confidence = 0.65
                            reason = f"{entity1} ≈ {entity2}"
                        else:
                            prediction = PredictionLabel.REFUTED
                            confidence = 0.6
                            reason = f"{entity1} ≠ {entity2}"
        
        # Priority 4: Weak entity-based inference
        if prediction == PredictionLabel.NEI and table_entities:
            mentioned_count = 0
            for entity in table_entities:
                if self.fuzzy_match_entity(entity, claim, 0.7):
                    mentioned_count += 1
            
            if mentioned_count >= 2:
                # Multiple entities - slightly lean toward supported but stay conservative
                confidence = 0.52
                prediction = PredictionLabel.SUPPORTED
                reason = f"Multiple entities mentioned ({mentioned_count})"
            elif mentioned_count == 1:
                # Single entity - very weak
                confidence = 0.51
                # Don't commit strongly
                reason = "Single entity mentioned"
        
        # Binary mode conversion - be balanced
        if self.binary_mode and prediction == PredictionLabel.NEI:
            # Use claim characteristics to break ties
            if claim_analysis['has_negation']:
                # Negated claims slightly more likely to be refuted
                if confidence < 0.52:
                    prediction = PredictionLabel.REFUTED
                    reason = f"Binary: negated claim with weak evidence (conf={confidence:.2f})"
                else:
                    prediction = PredictionLabel.SUPPORTED
                    reason = f"Binary: negated claim with moderate evidence (conf={confidence:.2f})"
            else:
                # Non-negated claims - balanced decision
                if confidence < 0.5:
                    prediction = PredictionLabel.REFUTED
                    reason = f"Binary: weak evidence (conf={confidence:.2f})"
                else:
                    prediction = PredictionLabel.SUPPORTED
                    reason = f"Binary: moderate evidence (conf={confidence:.2f})"
        
        # Create detailed reasoning
        detailed_reasoning = DetailedReasoning(
            prediction=prediction.value,
            confidence=round(confidence, 3),
            steps=steps,
            evidence=evidence,
            comparisons=comparisons,
            numeric_matches=numeric_matches,
            reason=reason
        )
        
        return prediction.value, confidence, detailed_reasoning

def evaluate_on_dataset(data_path: str, binary_mode: bool = True, 
                        detailed: bool = False) -> Dict[str, Any]:
    """Evaluate on the entire dataset"""
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    reasoner = OptimizedTableReasoner(binary_mode=binary_mode)
    
    predictions = []
    true_labels = []
    detailed_results = []
    
    for idx, item in enumerate(data):
        claim = item.get('claim', '')
        table = item.get('table', [])
        true_label = item.get('label', 'NEI')
        
        # Get prediction
        pred_label, confidence, reasoning = reasoner.predict(claim, table)
        
        # Convert for binary mode
        if binary_mode:
            true_label_binary = true_label if true_label != "NEI" else "Refuted"
            predictions.append(pred_label)
            true_labels.append(true_label_binary)
        else:
            predictions.append(pred_label)
            true_labels.append(true_label)
        
        # Store detailed results
        if detailed:
            detailed_results.append({
                'index': idx,
                'claim': claim[:200] + '...' if len(claim) > 200 else claim,
                'true_label': true_label,
                'converted_label': true_label_binary if binary_mode else true_label,
                'prediction': pred_label,
                'confidence': round(confidence, 2),
                'reasoning': reasoning.__dict__
            })
    
    # Calculate metrics
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    # For macro F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='macro', zero_division=0
    )
    
    # Get prediction distribution
    pred_dist = {label: predictions.count(label) for label in set(predictions)}
    
    results = {
        'metrics': {
            'precision': f"{precision*100:.1f}",
            'recall': f"{recall*100:.1f}",
            'macro_f1': f"{f1*100:.1f}",
            'accuracy': f"{accuracy_score(true_labels, predictions)*100:.1f}"
        },
        'binary_mode': binary_mode,
        'prediction_distribution': pred_dist
    }
    
    if detailed:
        results['detailed_results'] = detailed_results
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Optimized UnitMath Evaluation')
    parser.add_argument('--data', type=str, default='sci_tab.json',
                       help='Path to the dataset')
    parser.add_argument('--binary', action='store_true',
                       help='Use binary classification')
    parser.add_argument('--detailed', action='store_true',
                       help='Save detailed results')
    parser.add_argument('--output', type=str, default='optimized_evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    print("Optimized UnitMath Evaluation System")
    print("=====================================")
    print(f"Dataset: {args.data}")
    print(f"Binary mode: {args.binary}")
    print(f"Detailed results: {args.detailed}")
    
    # Run evaluation
    results = evaluate_on_dataset(
        args.data,
        binary_mode=args.binary,
        detailed=args.detailed
    )
    
    # Print summary
    print("\nResults:")
    print(f"Precision: {results['metrics']['precision']}%")
    print(f"Recall: {results['metrics']['recall']}%")
    print(f"Macro F1: {results['metrics']['macro_f1']}%")
    print(f"Accuracy: {results['metrics']['accuracy']}%")
    
    print("\nPrediction Distribution:")
    for label, count in results['prediction_distribution'].items():
        print(f"  {label}: {count}")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()