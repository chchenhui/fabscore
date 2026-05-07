"""
Integration Templates for Table Encoders
Provides easy integration with existing table reasoning models
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod

from .unit_parser import QuantityExtractor
from .symbolic_calculator import SymbolicCalculator
from .operation_sketch import OperationSketch, SketchExecutor
from .neural_symbolic_model import NeuralSymbolicTableReasoner


class TableEncoderAdapter(ABC):
    """Abstract base class for adapting different table encoders"""
    
    @abstractmethod
    def encode_table(self, table: Dict[str, Any], claim: str) -> torch.Tensor:
        """Encode table and claim into embeddings"""
        pass
    
    @abstractmethod
    def get_hidden_size(self) -> int:
        """Get the hidden size of the encoder"""
        pass


class TAPASAdapter(TableEncoderAdapter):
    """Adapter for TAPAS table encoder"""
    
    def __init__(self, model_name: str = "google/tapas-base"):
        from transformers import TapasTokenizer, TapasModel
        self.tokenizer = TapasTokenizer.from_pretrained(model_name)
        self.model = TapasModel.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
    
    def encode_table(self, table: Dict[str, Any], claim: str) -> torch.Tensor:
        """Encode using TAPAS"""
        # Convert table to pandas DataFrame format expected by TAPAS
        import pandas as pd
        df = pd.DataFrame(table['data'], columns=table['headers'])
        
        # Tokenize
        inputs = self.tokenizer(
            table=df,
            queries=[claim],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.pooler_output
    
    def get_hidden_size(self) -> int:
        return self.hidden_size


class TaBERTAdapter(TableEncoderAdapter):
    """Adapter for TaBERT table encoder"""
    
    def __init__(self, model_name: str = "tabert_base_k3"):
        # Note: TaBERT requires special installation
        # This is a placeholder implementation
        self.hidden_size = 768
    
    def encode_table(self, table: Dict[str, Any], claim: str) -> torch.Tensor:
        """Encode using TaBERT"""
        # Placeholder - actual implementation would use TaBERT
        batch_size = 1
        return torch.randn(batch_size, self.hidden_size)
    
    def get_hidden_size(self) -> int:
        return self.hidden_size


class UnitMathIntegration(nn.Module):
    """
    Main integration class for adding UnitMath to existing models
    """
    
    def __init__(self,
                 base_encoder: TableEncoderAdapter,
                 use_symbolic: bool = True,
                 use_operation_sketch: bool = True,
                 num_classes: int = 2):
        super().__init__()
        
        self.base_encoder = base_encoder
        self.use_symbolic = use_symbolic
        self.use_operation_sketch = use_operation_sketch
        
        # UnitMath components
        self.quantity_extractor = QuantityExtractor()
        self.calculator = SymbolicCalculator()
        self.sketch_executor = SketchExecutor()
        
        # Feature dimensions
        base_hidden = base_encoder.get_hidden_size()
        calc_features = 64 if use_symbolic else 0
        sketch_features = 128 if use_operation_sketch else 0
        
        # Feature projection layers
        if use_symbolic:
            self.calc_feature_encoder = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, calc_features)
            )
        
        if use_operation_sketch:
            self.sketch_feature_encoder = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, sketch_features)
            )
        
        # Fusion layer
        total_features = base_hidden + calc_features + sketch_features
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, table: Dict[str, Any], claim: str) -> Dict[str, Any]:
        """
        Forward pass with UnitMath integration
        
        Returns:
            Dictionary containing predictions and intermediate results
        """
        # Base encoding
        base_features = self.base_encoder.encode_table(table, claim)
        all_features = [base_features]
        
        results = {
            'base_features': base_features
        }
        
        # Symbolic calculator features
        if self.use_symbolic:
            calc_features = self._extract_calculator_features(table, claim)
            calc_features_encoded = self.calc_feature_encoder(calc_features)
            all_features.append(calc_features_encoded)
            results['calc_features'] = calc_features_encoded
        
        # Operation sketch features
        if self.use_operation_sketch:
            sketch_features = self._extract_sketch_features(table, claim)
            sketch_features_encoded = self.sketch_feature_encoder(sketch_features)
            all_features.append(sketch_features_encoded)
            results['sketch_features'] = sketch_features_encoded
        
        # Fusion
        combined = torch.cat(all_features, dim=-1)
        fused = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused)
        
        results['logits'] = logits
        results['prediction'] = torch.argmax(logits, dim=-1)
        results['probabilities'] = torch.softmax(logits, dim=-1)
        
        return results
    
    def _extract_calculator_features(self, 
                                    table: Dict[str, Any],
                                    claim: str) -> torch.Tensor:
        """Extract features using symbolic calculator"""
        features = []
        
        # Extract quantities
        table_quantities = self.quantity_extractor.extract_from_table(table)
        claim_quantities = self.quantity_extractor.extract_from_claim(claim)
        
        # Basic quantity statistics
        num_table_quantities = sum(
            len(qs) for qs in table_quantities.get('cells', {}).values()
        )
        num_claim_quantities = len(claim_quantities)
        features.extend([num_table_quantities, num_claim_quantities])
        
        # Try simple calculations between aligned quantities
        alignments = self.quantity_extractor.align_quantities(
            table_quantities, claim_quantities
        )
        
        if alignments:
            # Calculate differences and ratios
            differences = []
            ratios = []
            
            for table_q, claim_q in alignments[:3]:  # Limit to first 3
                # Difference
                diff_result = self.calculator.subtract(table_q, claim_q)
                if not diff_result.error_message:
                    differences.append(diff_result.value)
                
                # Ratio
                ratio_result = self.calculator.divide(table_q, claim_q)
                if not ratio_result.error_message:
                    ratios.append(ratio_result.value)
            
            # Add statistics
            if differences:
                features.extend([
                    min(differences),
                    max(differences),
                    sum(differences) / len(differences)
                ])
            else:
                features.extend([0, 0, 0])
            
            if ratios:
                features.extend([
                    min(ratios),
                    max(ratios),
                    sum(ratios) / len(ratios)
                ])
            else:
                features.extend([0, 0, 0])
        else:
            features.extend([0, 0, 0, 0, 0, 0])
        
        # Check for percentage/percentage point keywords
        has_percentage = 'percent' in claim.lower()
        has_pp = 'percentage point' in claim.lower() or 'pp' in claim.lower()
        features.extend([float(has_percentage), float(has_pp)])
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return torch.tensor(features[:10], dtype=torch.float32).unsqueeze(0)
    
    def _extract_sketch_features(self,
                                table: Dict[str, Any],
                                claim: str) -> torch.Tensor:
        """Extract features from operation sketches"""
        # Generate simple sketch based on claim keywords
        sketch_str = self._generate_simple_sketch(claim, table)
        
        # Try to execute sketch
        try:
            sketch = OperationSketch(sketch_str)
            result = self.sketch_executor.execute(sketch, table)
            
            # Extract features from result
            features = [
                result.value if not result.error_message else 0,
                result.confidence,
                1.0 if result.error_message else 0.0
            ]
        except:
            features = [0, 0, 1.0]
        
        # Add sketch complexity features
        features.extend([
            len(sketch_str),
            sketch_str.count('('),
            sketch_str.count(','),
            float('compare' in sketch_str),
            float('percentage' in sketch_str),
            float('fold' in sketch_str)
        ])
        
        # One-hot encoding of detected operations
        operations = [
            'add', 'subtract', 'multiply', 'divide',
            'compare', 'percentage_change', 'fold_change',
            'mean', 'max', 'min'
        ]
        
        for op in operations:
            features.append(float(op in sketch_str))
        
        # Pad to fixed size
        while len(features) < 256:
            features.append(0.0)
        
        return torch.tensor(features[:256], dtype=torch.float32).unsqueeze(0)
    
    def _generate_simple_sketch(self, claim: str, table: Dict[str, Any]) -> str:
        """Generate a simple operation sketch from claim"""
        claim_lower = claim.lower()
        
        # Simple heuristics for sketch generation
        if 'increase' in claim_lower or 'decrease' in claim_lower:
            return "percentage_change(cell(0,1), cell(1,1))"
        elif 'higher' in claim_lower or 'lower' in claim_lower:
            return "compare(cell(0,1), cell(1,1), gt)"
        elif 'average' in claim_lower or 'mean' in claim_lower:
            return "mean(col(1))"
        elif 'maximum' in claim_lower or 'highest' in claim_lower:
            return "max(col(1))"
        elif 'minimum' in claim_lower or 'lowest' in claim_lower:
            return "min(col(1))"
        elif 'fold' in claim_lower:
            return "fold_change(cell(0,1), cell(1,1))"
        else:
            return "compare(cell(0,1), cell(1,1), eq)"


class LightweightUnitMathPlugin:
    """
    Lightweight plugin for adding UnitMath features to any model
    """
    
    def __init__(self):
        self.quantity_extractor = QuantityExtractor()
        self.calculator = SymbolicCalculator()
    
    def extract_features(self, 
                        table: Dict[str, Any],
                        claim: str) -> np.ndarray:
        """
        Extract UnitMath features as numpy array
        
        Args:
            table: Table data
            claim: Claim text
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # Extract quantities
        table_quantities = self.quantity_extractor.extract_from_table(table)
        claim_quantities = self.quantity_extractor.extract_from_claim(claim)
        
        # Quantity counts
        num_table_quantities = sum(
            len(qs) for qs in table_quantities.get('cells', {}).values()
        )
        num_claim_quantities = len(claim_quantities)
        
        features.extend([
            num_table_quantities,
            num_claim_quantities,
            num_table_quantities / max(1, len(table.get('data', [])) * len(table.get('headers', [])))
        ])
        
        # Unit diversity
        table_units = set()
        for quantities in table_quantities.get('cells', {}).values():
            for q in quantities:
                table_units.add(q.unit)
        
        claim_units = set(q.unit for q in claim_quantities)
        
        features.extend([
            len(table_units),
            len(claim_units),
            len(table_units.intersection(claim_units))
        ])
        
        # Dimensional consistency
        alignments = self.quantity_extractor.align_quantities(
            table_quantities, claim_quantities
        )
        
        features.extend([
            len(alignments),
            len(alignments) / max(1, num_claim_quantities)
        ])
        
        # Keyword features
        keywords = {
            'increase': ['increase', 'rise', 'grow', 'improve'],
            'decrease': ['decrease', 'fall', 'drop', 'decline'],
            'comparison': ['higher', 'lower', 'more', 'less', 'greater', 'smaller'],
            'percentage': ['percent', '%', 'percentage'],
            'significance': ['significant', 'p-value', 'confidence', 'correlation']
        }
        
        claim_lower = claim.lower()
        for category, words in keywords.items():
            features.append(float(any(word in claim_lower for word in words)))
        
        return np.array(features)
    
    def augment_dataset(self,
                       X: np.ndarray,
                       tables: List[Dict[str, Any]],
                       claims: List[str]) -> np.ndarray:
        """
        Augment existing feature matrix with UnitMath features
        
        Args:
            X: Original feature matrix [n_samples, n_features]
            tables: List of table data
            claims: List of claims
        
        Returns:
            Augmented feature matrix
        """
        unitmath_features = []
        
        for table, claim in zip(tables, claims):
            features = self.extract_features(table, claim)
            unitmath_features.append(features)
        
        unitmath_features = np.array(unitmath_features)
        
        # Concatenate with original features
        return np.hstack([X, unitmath_features])


def create_unitmath_model(base_encoder_type: str = "tapas",
                         use_symbolic: bool = True,
                         use_operation_sketch: bool = True,
                         num_classes: int = 2) -> UnitMathIntegration:
    """
    Factory function to create UnitMath integrated model
    
    Args:
        base_encoder_type: Type of base encoder ("tapas", "tabert")
        use_symbolic: Whether to use symbolic calculator
        use_operation_sketch: Whether to use operation sketches
        num_classes: Number of classification classes
    
    Returns:
        UnitMathIntegration model
    """
    if base_encoder_type.lower() == "tapas":
        base_encoder = TAPASAdapter()
    elif base_encoder_type.lower() == "tabert":
        base_encoder = TaBERTAdapter()
    else:
        raise ValueError(f"Unknown encoder type: {base_encoder_type}")
    
    return UnitMathIntegration(
        base_encoder=base_encoder,
        use_symbolic=use_symbolic,
        use_operation_sketch=use_operation_sketch,
        num_classes=num_classes
    )


# Example usage
def integrate_with_existing_model():
    """Example of integrating UnitMath with an existing model"""
    
    # Create integrated model
    model = create_unitmath_model(
        base_encoder_type="tapas",
        use_symbolic=True,
        use_operation_sketch=True,
        num_classes=3  # Supported, Refuted, NEI
    )
    
    # Example table and claim
    table = {
        'headers': ['Model', 'Accuracy', 'F1 Score'],
        'data': [
            ['Baseline', '85.2%', '0.83'],
            ['Our Model', '89.7%', '0.88']
        ]
    }
    
    claim = "Our model shows a 4.5 percentage point improvement in accuracy"
    
    # Forward pass
    results = model(table, claim)
    
    print(f"Prediction: {results['prediction']}")
    print(f"Probabilities: {results['probabilities']}")
    
    return model