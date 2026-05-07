"""
Unit Parser and Quantity Extractor
Extracts quantities and units from table cells, headers, and claims
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pint
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import numpy as np


@dataclass
class Quantity:
    """Represents a quantity with value and unit"""
    value: float
    unit: str
    original_text: str
    confidence: float = 1.0
    
    def __repr__(self):
        return f"Quantity({self.value} {self.unit})"


class UnitOntology:
    """Canonical unit ontology with common scientific units"""
    
    def __init__(self):
        self.ureg = pint.UnitRegistry()
        
        # Define custom units and aliases
        self.ureg.define('percent = 0.01 = %')
        self.ureg.define('percentage_point = 1 = pp')
        self.ureg.define('fold = 1 = x')
        
        # Common unit patterns
        self.unit_patterns = {
            'mass': ['kg', 'g', 'mg', 'μg', 'ng', 'pg', 'lb', 'oz'],
            'length': ['m', 'cm', 'mm', 'μm', 'nm', 'km', 'ft', 'in'],
            'time': ['s', 'ms', 'μs', 'ns', 'min', 'h', 'hour', 'day', 'year'],
            'temperature': ['K', '°C', '°F', 'celsius', 'fahrenheit'],
            'concentration': ['M', 'mM', 'μM', 'nM', 'mol/L', 'g/L', 'mg/mL'],
            'percentage': ['%', 'percent', 'percentage'],
            'ratio': ['fold', 'x', 'times', 'ratio'],
            'currency': ['$', '€', '£', '¥', 'USD', 'EUR'],
            'count': ['cells', 'particles', 'molecules', 'atoms']
        }
        
    def normalize_unit(self, unit_str: str) -> str:
        """Normalize unit string to canonical form"""
        try:
            parsed = self.ureg.parse_expression(unit_str)
            return str(parsed.units)
        except:
            # Fallback to string normalization
            unit_str = unit_str.strip().lower()
            
            # Handle special cases
            if unit_str in ['percentage_points', 'pp', 'p.p.']:
                return 'percentage_point'
            if unit_str in ['x', 'fold', 'times']:
                return 'fold'
                
            return unit_str
    
    def get_dimension(self, unit_str: str) -> Optional[str]:
        """Get the dimensional category of a unit"""
        normalized = self.normalize_unit(unit_str)
        
        for dimension, units in self.unit_patterns.items():
            if any(u in normalized for u in units):
                return dimension
                
        try:
            parsed = self.ureg.parse_expression(normalized)
            return str(parsed.dimensionality)
        except:
            return None


class UnitParser:
    """Rule-enhanced neural tagger for unit extraction"""
    
    def __init__(self, model_name: str = "bert-base-cased"):
        self.ontology = UnitOntology()
        
        # Regex patterns for common quantity formats
        self.quantity_patterns = [
            # Number with unit (e.g., "5.2 mg", "3.14%")
            r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z%°μ]+(?:/[a-zA-Z]+)?)',
            # Percentage (e.g., "45%", "12.5 percent")
            r'([-+]?\d*\.?\d+)\s*(?:%|percent)',
            # Fold change (e.g., "2.5-fold", "3x")
            r'([-+]?\d*\.?\d+)[-\s]?(?:fold|x|times)',
            # Currency (e.g., "$100", "€50.5")
            r'([$€£¥])\s*([-+]?\d*\.?\d+)',
            # Range with units (e.g., "10-20 mg")
            r'([-+]?\d*\.?\d+)\s*-\s*([-+]?\d*\.?\d+)\s*([a-zA-Z%°μ]+)',
            # Confidence interval (e.g., "5.2 (4.1-6.3)")
            r'([-+]?\d*\.?\d+)\s*\(([-+]?\d*\.?\d+)\s*-\s*([-+]?\d*\.?\d+)\)'
        ]
        
        # Load neural model for advanced extraction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None  # Lazy load when needed
        
    def parse_text(self, text: str) -> List[Quantity]:
        """Extract quantities from text using rules and neural model"""
        quantities = []
        
        # Rule-based extraction
        for pattern in self.quantity_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    value_str = groups[0]
                    unit_str = groups[1] if len(groups) > 1 else ""
                    
                    try:
                        value = float(value_str)
                        unit = self.ontology.normalize_unit(unit_str)
                        quantities.append(Quantity(
                            value=value,
                            unit=unit,
                            original_text=match.group(0),
                            confidence=0.9  # Rule-based confidence
                        ))
                    except ValueError:
                        continue
        
        # Neural extraction (if model loaded)
        if self.model is not None:
            neural_quantities = self._neural_extract(text)
            quantities.extend(neural_quantities)
        
        return self._deduplicate_quantities(quantities)
    
    def _neural_extract(self, text: str) -> List[Quantity]:
        """Extract quantities using neural model"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Process predictions to extract quantities
        quantities = []
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        labels = predictions[0].tolist()
        
        # Group consecutive B-QTY and I-QTY tags
        current_quantity = []
        for token, label in zip(tokens, labels):
            if label == 1:  # B-QTY
                if current_quantity:
                    quantities.append(self._parse_quantity_tokens(current_quantity))
                current_quantity = [token]
            elif label == 2:  # I-QTY
                current_quantity.append(token)
            else:
                if current_quantity:
                    quantities.append(self._parse_quantity_tokens(current_quantity))
                    current_quantity = []
        
        return [q for q in quantities if q is not None]
    
    def _parse_quantity_tokens(self, tokens: List[str]) -> Optional[Quantity]:
        """Parse quantity from tokenized text"""
        text = " ".join(tokens).replace(" ##", "")
        
        # Try to extract value and unit
        match = re.match(r'([-+]?\d*\.?\d+)\s*(.+)?', text)
        if match:
            try:
                value = float(match.group(1))
                unit = match.group(2) if match.group(2) else ""
                return Quantity(
                    value=value,
                    unit=self.ontology.normalize_unit(unit),
                    original_text=text,
                    confidence=0.7  # Neural confidence
                )
            except ValueError:
                return None
        return None
    
    def _deduplicate_quantities(self, quantities: List[Quantity]) -> List[Quantity]:
        """Remove duplicate quantities, keeping highest confidence"""
        unique = {}
        for q in quantities:
            key = (q.value, q.unit)
            if key not in unique or unique[key].confidence < q.confidence:
                unique[key] = q
        return list(unique.values())


class QuantityExtractor:
    """Extract quantities from table structures"""
    
    def __init__(self):
        self.parser = UnitParser()
        
    def extract_from_table(self, table: Dict[str, Any]) -> Dict[str, List[Quantity]]:
        """Extract quantities from table headers and cells"""
        extracted = {
            'headers': {},
            'cells': {}
        }
        
        # Extract from headers
        if 'headers' in table:
            for i, header in enumerate(table['headers']):
                quantities = self.parser.parse_text(header)
                if quantities:
                    extracted['headers'][i] = quantities
        
        # Extract from cells
        if 'data' in table:
            for i, row in enumerate(table['data']):
                for j, cell in enumerate(row):
                    cell_text = str(cell)
                    quantities = self.parser.parse_text(cell_text)
                    if quantities:
                        extracted['cells'][(i, j)] = quantities
        
        return extracted
    
    def extract_from_claim(self, claim: str) -> List[Quantity]:
        """Extract quantities from a claim text"""
        return self.parser.parse_text(claim)
    
    def align_quantities(self, 
                        table_quantities: Dict[str, List[Quantity]], 
                        claim_quantities: List[Quantity]) -> List[Tuple[Quantity, Quantity]]:
        """Align quantities from table with those in claim"""
        alignments = []
        
        # Simple alignment based on dimension matching
        for claim_q in claim_quantities:
            claim_dim = self.parser.ontology.get_dimension(claim_q.unit)
            
            # Search in table cells
            for cell_quantities in table_quantities.get('cells', {}).values():
                for table_q in cell_quantities:
                    table_dim = self.parser.ontology.get_dimension(table_q.unit)
                    if claim_dim == table_dim:
                        alignments.append((table_q, claim_q))
        
        return alignments