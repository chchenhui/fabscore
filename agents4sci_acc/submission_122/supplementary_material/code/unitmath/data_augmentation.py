"""
Data Augmentation for Unit-Aware Table Reasoning
Generates counterfactual examples with unit rescaling and transformations
"""

import random
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from .unit_parser import Quantity, QuantityExtractor, UnitOntology
from .symbolic_calculator import SymbolicCalculator


@dataclass
class AugmentedSample:
    """Represents an augmented data sample"""
    original_table: Dict[str, Any]
    augmented_table: Dict[str, Any]
    original_claim: str
    augmented_claim: str
    transformation_type: str
    transformation_details: Dict[str, Any]
    label_preserved: bool


class UnitAugmentor:
    """Performs unit-aware data augmentation"""
    
    def __init__(self, seed: int = 42):
        self.quantity_extractor = QuantityExtractor()
        self.calculator = SymbolicCalculator()
        self.ontology = UnitOntology()
        random.seed(seed)
        np.random.seed(seed)
        
        # Define unit conversion mappings
        self.unit_conversions = {
            'mass': {
                'mg': {'g': 0.001, 'kg': 0.000001, 'μg': 1000},
                'g': {'mg': 1000, 'kg': 0.001, 'μg': 1000000},
                'kg': {'g': 1000, 'mg': 1000000, 'μg': 1000000000}
            },
            'length': {
                'mm': {'cm': 0.1, 'm': 0.001, 'μm': 1000},
                'cm': {'mm': 10, 'm': 0.01, 'μm': 10000},
                'm': {'cm': 100, 'mm': 1000, 'km': 0.001}
            },
            'concentration': {
                'mM': {'M': 0.001, 'μM': 1000, 'nM': 1000000},
                'μM': {'mM': 0.001, 'M': 0.000001, 'nM': 1000},
                'M': {'mM': 1000, 'μM': 1000000, 'nM': 1000000000}
            }
        }
    
    def augment_unit_rescaling(self, 
                               table: Dict[str, Any],
                               claim: str) -> AugmentedSample:
        """
        Augment by rescaling units (e.g., mg → g)
        
        Args:
            table: Original table data
            claim: Original claim text
        
        Returns:
            AugmentedSample with rescaled units
        """
        augmented_table = copy.deepcopy(table)
        augmented_claim = claim
        transformation_details = {'rescaled_units': []}
        
        # Extract quantities from table
        table_quantities = self.quantity_extractor.extract_from_table(table)
        
        # Choose units to rescale
        units_to_rescale = self._select_units_to_rescale(table_quantities)
        
        # Rescale table cells
        if 'data' in augmented_table:
            for i, row in enumerate(augmented_table['data']):
                new_row = []
                for j, cell in enumerate(row):
                    cell_str = str(cell)
                    quantities = self.quantity_extractor.parser.parse_text(cell_str)
                    
                    if quantities:
                        q = quantities[0]
                        new_cell = self._rescale_quantity(q, units_to_rescale)
                        if new_cell != cell_str:
                            transformation_details['rescaled_units'].append({
                                'position': (i, j),
                                'original': cell_str,
                                'augmented': new_cell
                            })
                        new_row.append(new_cell)
                    else:
                        new_row.append(cell)
                
                augmented_table['data'][i] = new_row
        
        # Rescale quantities in claim
        claim_quantities = self.quantity_extractor.extract_from_claim(claim)
        for q in claim_quantities:
            rescaled = self._rescale_quantity(q, units_to_rescale)
            if rescaled != q.original_text:
                augmented_claim = augmented_claim.replace(q.original_text, rescaled)
                transformation_details['rescaled_units'].append({
                    'position': 'claim',
                    'original': q.original_text,
                    'augmented': rescaled
                })
        
        return AugmentedSample(
            original_table=table,
            augmented_table=augmented_table,
            original_claim=claim,
            augmented_claim=augmented_claim,
            transformation_type='unit_rescaling',
            transformation_details=transformation_details,
            label_preserved=True  # Unit rescaling preserves truth value
        )
    
    def augment_percentage_swap(self,
                               table: Dict[str, Any],
                               claim: str) -> AugmentedSample:
        """
        Swap between percentage and percentage points
        
        Args:
            table: Original table data
            claim: Original claim text
        
        Returns:
            AugmentedSample with swapped percentage types
        """
        augmented_table = copy.deepcopy(table)
        augmented_claim = claim
        transformation_details = {'swapped_percentages': []}
        
        # Swap in claim
        if 'percentage points' in claim.lower():
            augmented_claim = claim.replace('percentage points', 'percent')
            augmented_claim = augmented_claim.replace('pp', '%')
            transformation_details['swapped_percentages'].append('pp_to_percent')
        elif 'percent' in claim.lower() and 'points' not in claim.lower():
            augmented_claim = claim.replace('percent', 'percentage points')
            augmented_claim = augmented_claim.replace('%', 'pp')
            transformation_details['swapped_percentages'].append('percent_to_pp')
        
        # Note: This changes the semantic meaning and likely the label
        label_preserved = False
        
        return AugmentedSample(
            original_table=table,
            augmented_table=augmented_table,
            original_claim=claim,
            augmented_claim=augmented_claim,
            transformation_type='percentage_swap',
            transformation_details=transformation_details,
            label_preserved=label_preserved
        )
    
    def augment_value_perturbation(self,
                                  table: Dict[str, Any],
                                  claim: str,
                                  perturbation_factor: float = 0.1) -> AugmentedSample:
        """
        Perturb numeric values slightly to test robustness
        
        Args:
            table: Original table data
            claim: Original claim text
            perturbation_factor: Maximum relative perturbation (e.g., 0.1 = ±10%)
        
        Returns:
            AugmentedSample with perturbed values
        """
        augmented_table = copy.deepcopy(table)
        transformation_details = {'perturbed_values': []}
        
        if 'data' in augmented_table:
            for i, row in enumerate(augmented_table['data']):
                new_row = []
                for j, cell in enumerate(row):
                    cell_str = str(cell)
                    quantities = self.quantity_extractor.parser.parse_text(cell_str)
                    
                    if quantities and random.random() < 0.3:  # Perturb 30% of values
                        q = quantities[0]
                        # Generate random perturbation
                        perturbation = random.uniform(-perturbation_factor, perturbation_factor)
                        new_value = q.value * (1 + perturbation)
                        
                        # Format with appropriate precision
                        if '.' in str(q.value):
                            decimals = len(str(q.value).split('.')[1])
                            new_cell = f"{new_value:.{decimals}f} {q.unit}"
                        else:
                            new_cell = f"{int(new_value)} {q.unit}"
                        
                        transformation_details['perturbed_values'].append({
                            'position': (i, j),
                            'original': cell_str,
                            'augmented': new_cell,
                            'perturbation': perturbation
                        })
                        new_row.append(new_cell)
                    else:
                        new_row.append(cell)
                
                augmented_table['data'][i] = new_row
        
        # Small perturbations might preserve label, large ones might not
        label_preserved = perturbation_factor < 0.05
        
        return AugmentedSample(
            original_table=table,
            augmented_table=augmented_table,
            original_claim=claim,
            augmented_claim=claim,  # Claim unchanged
            transformation_type='value_perturbation',
            transformation_details=transformation_details,
            label_preserved=label_preserved
        )
    
    def augment_row_shuffle(self,
                           table: Dict[str, Any],
                           claim: str) -> AugmentedSample:
        """
        Shuffle table rows to test position independence
        
        Args:
            table: Original table data
            claim: Original claim text
        
        Returns:
            AugmentedSample with shuffled rows
        """
        augmented_table = copy.deepcopy(table)
        transformation_details = {'shuffle_order': []}
        
        if 'data' in augmented_table:
            original_rows = augmented_table['data']
            indices = list(range(len(original_rows)))
            random.shuffle(indices)
            
            augmented_table['data'] = [original_rows[i] for i in indices]
            transformation_details['shuffle_order'] = indices
        
        return AugmentedSample(
            original_table=table,
            augmented_table=augmented_table,
            original_claim=claim,
            augmented_claim=claim,
            transformation_type='row_shuffle',
            transformation_details=transformation_details,
            label_preserved=True  # Row order shouldn't affect truth
        )
    
    def augment_synthetic_contradiction(self,
                                       table: Dict[str, Any],
                                       claim: str) -> AugmentedSample:
        """
        Create synthetic contradictions by negating comparisons
        
        Args:
            table: Original table data
            claim: Original claim text
        
        Returns:
            AugmentedSample with contradictory claim
        """
        augmented_claim = claim
        transformation_details = {'negations': []}
        
        # Negate comparison words
        negation_map = {
            'higher': 'lower',
            'lower': 'higher',
            'increased': 'decreased',
            'decreased': 'increased',
            'more': 'less',
            'less': 'more',
            'greater': 'smaller',
            'smaller': 'greater',
            'above': 'below',
            'below': 'above',
            'exceeds': 'falls below',
            'improved': 'worsened',
            'worsened': 'improved'
        }
        
        for original, negated in negation_map.items():
            if original in augmented_claim.lower():
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                augmented_claim = pattern.sub(negated, augmented_claim)
                transformation_details['negations'].append((original, negated))
        
        # If no negation was applied, add "not"
        if not transformation_details['negations']:
            # Simple heuristic: add "not" before the main verb
            words = augmented_claim.split()
            verb_keywords = ['is', 'are', 'was', 'were', 'has', 'have', 'shows', 'demonstrates']
            for i, word in enumerate(words):
                if word.lower() in verb_keywords:
                    words.insert(i, 'not')
                    transformation_details['negations'].append(('', 'not'))
                    break
            augmented_claim = ' '.join(words)
        
        return AugmentedSample(
            original_table=table,
            augmented_table=table,  # Table unchanged
            original_claim=claim,
            augmented_claim=augmented_claim,
            transformation_type='synthetic_contradiction',
            transformation_details=transformation_details,
            label_preserved=False  # Label is inverted
        )
    
    def augment_confidence_interval(self,
                                   table: Dict[str, Any],
                                   claim: str,
                                   ci_level: float = 0.95) -> AugmentedSample:
        """
        Add confidence intervals to point estimates
        
        Args:
            table: Original table data
            claim: Original claim text
            ci_level: Confidence level for intervals
        
        Returns:
            AugmentedSample with added confidence intervals
        """
        augmented_table = copy.deepcopy(table)
        transformation_details = {'added_cis': []}
        
        if 'data' in augmented_table:
            for i, row in enumerate(augmented_table['data']):
                new_row = []
                for j, cell in enumerate(row):
                    cell_str = str(cell)
                    quantities = self.quantity_extractor.parser.parse_text(cell_str)
                    
                    if quantities and random.random() < 0.2:  # Add CI to 20% of values
                        q = quantities[0]
                        # Generate synthetic CI
                        margin = abs(q.value * 0.1)  # 10% margin
                        lower = q.value - margin
                        upper = q.value + margin
                        
                        # Format with CI
                        if '.' in str(q.value):
                            decimals = len(str(q.value).split('.')[1])
                            new_cell = f"{q.value:.{decimals}f} ({lower:.{decimals}f}-{upper:.{decimals}f}) {q.unit}"
                        else:
                            new_cell = f"{int(q.value)} ({int(lower)}-{int(upper)}) {q.unit}"
                        
                        transformation_details['added_cis'].append({
                            'position': (i, j),
                            'original': cell_str,
                            'augmented': new_cell,
                            'ci_level': ci_level
                        })
                        new_row.append(new_cell)
                    else:
                        new_row.append(cell)
                
                augmented_table['data'][i] = new_row
        
        return AugmentedSample(
            original_table=table,
            augmented_table=augmented_table,
            original_claim=claim,
            augmented_claim=claim,
            transformation_type='confidence_interval',
            transformation_details=transformation_details,
            label_preserved=True  # Adding CIs doesn't change truth
        )
    
    def _select_units_to_rescale(self, 
                                 table_quantities: Dict[str, Any]) -> Dict[str, str]:
        """Select which units to rescale"""
        units_to_rescale = {}
        
        # Collect all unique units
        unique_units = set()
        for quantities in table_quantities.get('cells', {}).values():
            for q in quantities:
                unique_units.add(q.unit)
        
        # Select conversions for each unit
        for unit in unique_units:
            dimension = self.ontology.get_dimension(unit)
            if dimension in self.unit_conversions:
                if unit in self.unit_conversions[dimension]:
                    # Choose a random target unit
                    targets = list(self.unit_conversions[dimension][unit].keys())
                    if targets:
                        units_to_rescale[unit] = random.choice(targets)
        
        return units_to_rescale
    
    def _rescale_quantity(self, 
                         quantity: Quantity,
                         units_to_rescale: Dict[str, str]) -> str:
        """Rescale a single quantity"""
        if quantity.unit in units_to_rescale:
            target_unit = units_to_rescale[quantity.unit]
            dimension = self.ontology.get_dimension(quantity.unit)
            
            if dimension in self.unit_conversions:
                if quantity.unit in self.unit_conversions[dimension]:
                    if target_unit in self.unit_conversions[dimension][quantity.unit]:
                        factor = self.unit_conversions[dimension][quantity.unit][target_unit]
                        new_value = quantity.value * factor
                        
                        # Format appropriately
                        if '.' in str(quantity.value):
                            decimals = len(str(quantity.value).split('.')[1])
                            return f"{new_value:.{decimals}f} {target_unit}"
                        else:
                            return f"{int(new_value)} {target_unit}"
        
        return quantity.original_text
    
    def generate_augmented_dataset(self,
                                  tables: List[Dict[str, Any]],
                                  claims: List[str],
                                  labels: List[str],
                                  augmentation_types: Optional[List[str]] = None,
                                  samples_per_type: int = 1) -> List[Dict[str, Any]]:
        """
        Generate augmented dataset
        
        Args:
            tables: List of original tables
            claims: List of original claims
            labels: List of original labels
            augmentation_types: Types of augmentation to apply
            samples_per_type: Number of augmented samples per type
        
        Returns:
            List of augmented samples
        """
        if augmentation_types is None:
            augmentation_types = [
                'unit_rescaling',
                'percentage_swap',
                'value_perturbation',
                'row_shuffle',
                'synthetic_contradiction',
                'confidence_interval'
            ]
        
        augmented_samples = []
        
        for table, claim, label in zip(tables, claims, labels):
            for aug_type in augmentation_types:
                for _ in range(samples_per_type):
                    if aug_type == 'unit_rescaling':
                        aug_sample = self.augment_unit_rescaling(table, claim)
                    elif aug_type == 'percentage_swap':
                        aug_sample = self.augment_percentage_swap(table, claim)
                    elif aug_type == 'value_perturbation':
                        aug_sample = self.augment_value_perturbation(table, claim)
                    elif aug_type == 'row_shuffle':
                        aug_sample = self.augment_row_shuffle(table, claim)
                    elif aug_type == 'synthetic_contradiction':
                        aug_sample = self.augment_synthetic_contradiction(table, claim)
                    elif aug_type == 'confidence_interval':
                        aug_sample = self.augment_confidence_interval(table, claim)
                    else:
                        continue
                    
                    # Determine new label
                    if aug_sample.label_preserved:
                        new_label = label
                    else:
                        # Invert label for contradictions
                        if label == 'Supported':
                            new_label = 'Refuted'
                        elif label == 'Refuted':
                            new_label = 'Supported'
                        else:
                            new_label = label  # NEI stays NEI
                    
                    augmented_samples.append({
                        'table': aug_sample.augmented_table,
                        'claim': aug_sample.augmented_claim,
                        'label': new_label,
                        'augmentation': {
                            'type': aug_sample.transformation_type,
                            'details': aug_sample.transformation_details,
                            'original_label': label
                        }
                    })
        
        return augmented_samples