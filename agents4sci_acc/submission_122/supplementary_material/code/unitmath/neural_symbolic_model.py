"""
Neural-Symbolic Fusion Model
Integrates neural table encoders with symbolic calculator for unit-aware reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from transformers import AutoModel, AutoTokenizer

from .unit_parser import QuantityExtractor, Quantity
from .symbolic_calculator import SymbolicCalculator, CalculationResult
from .operation_sketch import OperationSketch, SketchExecutor


@dataclass
class TableEncoding:
    """Encoded representation of a table"""
    cell_embeddings: torch.Tensor  # [batch, rows, cols, hidden]
    row_embeddings: torch.Tensor   # [batch, rows, hidden]
    col_embeddings: torch.Tensor   # [batch, cols, hidden]
    global_embedding: torch.Tensor # [batch, hidden]
    attention_mask: torch.Tensor   # [batch, rows, cols]


@dataclass
class ReasoningOutput:
    """Output of neural-symbolic reasoning"""
    prediction: torch.Tensor       # Classification logits
    operation_sketch: str          # Generated operation sketch
    calculation_result: CalculationResult  # Result from symbolic calculator
    features: torch.Tensor         # Combined neural-symbolic features
    confidence: float              # Overall confidence score


class TableEncoder(nn.Module):
    """Neural encoder for table data"""
    
    def __init__(self, model_name: str = "bert-base-uncased", hidden_size: int = 768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = hidden_size
        
        # Projection layers
        self.cell_projection = nn.Linear(hidden_size, hidden_size)
        self.row_projection = nn.Linear(hidden_size, hidden_size)
        self.col_projection = nn.Linear(hidden_size, hidden_size)
        
        # Attention layers
        self.row_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.col_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        
    def forward(self, table_data: Dict[str, Any], claim: str) -> TableEncoding:
        """Encode table and claim"""
        # Prepare table text
        table_text = self._table_to_text(table_data)
        
        # Encode with BERT
        inputs = self.tokenizer(
            claim,
            table_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
        
        outputs = self.bert(**inputs)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        
        # Extract cell representations
        cell_embeddings = self._extract_cell_embeddings(sequence_output, table_data)
        
        # Compute row and column embeddings
        row_embeddings = self._compute_row_embeddings(cell_embeddings)
        col_embeddings = self._compute_col_embeddings(cell_embeddings)
        
        # Create attention mask
        batch_size, num_rows, num_cols, _ = cell_embeddings.shape
        attention_mask = torch.ones(batch_size, num_rows, num_cols)
        
        return TableEncoding(
            cell_embeddings=cell_embeddings,
            row_embeddings=row_embeddings,
            col_embeddings=col_embeddings,
            global_embedding=pooled_output,
            attention_mask=attention_mask
        )
    
    def _table_to_text(self, table_data: Dict[str, Any]) -> str:
        """Convert table to text representation"""
        lines = []
        
        # Headers
        if 'headers' in table_data:
            lines.append(" | ".join(str(h) for h in table_data['headers']))
        
        # Data rows
        if 'data' in table_data:
            for row in table_data['data']:
                lines.append(" | ".join(str(cell) for cell in row))
        
        return " [SEP] ".join(lines)
    
    def _extract_cell_embeddings(self, sequence_output: torch.Tensor, 
                                 table_data: Dict[str, Any]) -> torch.Tensor:
        """Extract embeddings for each cell"""
        batch_size = sequence_output.shape[0]
        num_rows = len(table_data.get('data', []))
        num_cols = len(table_data.get('headers', []))
        hidden_size = sequence_output.shape[-1]
        
        # Initialize cell embeddings
        cell_embeddings = torch.zeros(batch_size, num_rows, num_cols, hidden_size)
        
        # Simple approach: use mean pooling over sequence
        # In practice, you'd want more sophisticated cell extraction
        cell_embeddings[:, :, :, :] = sequence_output.mean(dim=1).unsqueeze(1).unsqueeze(1)
        
        return self.cell_projection(cell_embeddings)
    
    def _compute_row_embeddings(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute row embeddings from cell embeddings"""
        batch_size, num_rows, num_cols, hidden_size = cell_embeddings.shape
        
        # Reshape for attention
        cells_flat = cell_embeddings.reshape(batch_size * num_rows, num_cols, hidden_size)
        
        # Apply row attention
        row_emb, _ = self.row_attention(cells_flat, cells_flat, cells_flat)
        row_emb = row_emb.mean(dim=1)  # Pool over columns
        
        # Reshape back
        row_emb = row_emb.reshape(batch_size, num_rows, hidden_size)
        
        return self.row_projection(row_emb)
    
    def _compute_col_embeddings(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute column embeddings from cell embeddings"""
        batch_size, num_rows, num_cols, hidden_size = cell_embeddings.shape
        
        # Transpose to work with columns
        cells_transposed = cell_embeddings.permute(0, 2, 1, 3)
        cells_flat = cells_transposed.reshape(batch_size * num_cols, num_rows, hidden_size)
        
        # Apply column attention
        col_emb, _ = self.col_attention(cells_flat, cells_flat, cells_flat)
        col_emb = col_emb.mean(dim=1)  # Pool over rows
        
        # Reshape back
        col_emb = col_emb.reshape(batch_size, num_cols, hidden_size)
        
        return self.col_projection(col_emb)


class OperationGenerator(nn.Module):
    """Generates operation sketches from table encodings"""
    
    def __init__(self, hidden_size: int = 768, vocab_size: int = 1000):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Operation vocabulary
        self.operation_vocab = [
            'add', 'subtract', 'multiply', 'divide',
            'compare', 'percentage_change', 'fold_change',
            'mean', 'max', 'min', 'ci_overlap',
            'cell', 'col', 'row', '(', ')', ','
        ]
        
        # LSTM decoder
        self.decoder_lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, len(self.operation_vocab))
        
        # Attention over table
        self.table_attention = nn.MultiheadAttention(
            hidden_size,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, table_encoding: TableEncoding, 
                max_length: int = 50) -> Tuple[str, torch.Tensor]:
        """Generate operation sketch"""
        batch_size = table_encoding.global_embedding.shape[0]
        device = table_encoding.global_embedding.device
        
        # Initialize decoder hidden state with global embedding
        h0 = table_encoding.global_embedding.unsqueeze(0).repeat(2, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Start token
        current_input = table_encoding.global_embedding.unsqueeze(1)
        
        generated_tokens = []
        logits_sequence = []
        
        for step in range(max_length):
            # LSTM step
            lstm_out, (h0, c0) = self.decoder_lstm(current_input, (h0, c0))
            
            # Attention over table cells
            cell_flat = table_encoding.cell_embeddings.reshape(
                batch_size, -1, self.hidden_size
            )
            attended, _ = self.table_attention(lstm_out, cell_flat, cell_flat)
            
            # Combine LSTM output and attention
            combined = lstm_out + attended
            
            # Generate token
            logits = self.output_projection(combined.squeeze(1))
            token_idx = torch.argmax(logits, dim=-1)
            
            generated_tokens.append(token_idx.item())
            logits_sequence.append(logits)
            
            # Stop if we generate end token (closing parenthesis after complete expression)
            if token_idx.item() < len(self.operation_vocab):
                token_str = self.operation_vocab[token_idx.item()]
                if token_str == ')' and self._is_complete_expression(generated_tokens):
                    break
            
            # Prepare next input
            current_input = table_encoding.global_embedding.unsqueeze(1)
        
        # Convert tokens to sketch string
        sketch_str = self._tokens_to_sketch(generated_tokens)
        logits_tensor = torch.stack(logits_sequence, dim=1)
        
        return sketch_str, logits_tensor
    
    def _tokens_to_sketch(self, tokens: List[int]) -> str:
        """Convert token indices to sketch string"""
        sketch_parts = []
        for token_idx in tokens:
            if token_idx < len(self.operation_vocab):
                sketch_parts.append(self.operation_vocab[token_idx])
        
        return "".join(sketch_parts)
    
    def _is_complete_expression(self, tokens: List[int]) -> bool:
        """Check if generated tokens form a complete expression"""
        # Simple check: count parentheses
        open_count = 0
        for token_idx in tokens:
            if token_idx < len(self.operation_vocab):
                token = self.operation_vocab[token_idx]
                if token == '(':
                    open_count += 1
                elif token == ')':
                    open_count -= 1
        
        return open_count == 0 and len(tokens) > 3


class NeuralSymbolicTableReasoner(nn.Module):
    """Main neural-symbolic model for table reasoning"""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 hidden_size: int = 768,
                 num_classes: int = 2):
        super().__init__()
        
        # Components
        self.table_encoder = TableEncoder(model_name, hidden_size)
        self.operation_generator = OperationGenerator(hidden_size)
        self.quantity_extractor = QuantityExtractor()
        self.sketch_executor = SketchExecutor()
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),  # +64 for calculator features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
        # Calculator feature encoder
        self.calc_feature_encoder = nn.Linear(10, 64)  # 10 calc features -> 64 dims
        
    def forward(self, table_data: Dict[str, Any], claim: str) -> ReasoningOutput:
        """Forward pass with neural-symbolic reasoning"""
        
        # Encode table and claim
        table_encoding = self.table_encoder(table_data, claim)
        
        # Generate operation sketch
        sketch_str, sketch_logits = self.operation_generator(table_encoding)
        
        # Execute sketch with symbolic calculator
        try:
            sketch = OperationSketch(sketch_str)
            calc_result = self.sketch_executor.execute(sketch, table_data)
        except Exception as e:
            # Fallback if sketch execution fails
            calc_result = CalculationResult(
                value=0,
                unit="",
                operation=None,
                inputs=[],
                confidence=0,
                error_message=str(e)
            )
        
        # Extract calculator features
        calc_features = self._extract_calc_features(calc_result, table_data, claim)
        calc_features_encoded = self.calc_feature_encoder(calc_features)
        
        # Combine neural and symbolic features
        combined_features = torch.cat([
            table_encoding.global_embedding,
            calc_features_encoded
        ], dim=-1)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return ReasoningOutput(
            prediction=logits,
            operation_sketch=sketch_str,
            calculation_result=calc_result,
            features=fused_features,
            confidence=torch.sigmoid(logits).max().item()
        )
    
    def _extract_calc_features(self, calc_result: CalculationResult,
                               table_data: Dict[str, Any],
                               claim: str) -> torch.Tensor:
        """Extract features from calculator result"""
        features = []
        
        # Basic features
        features.append(calc_result.value if not calc_result.error_message else 0)
        features.append(calc_result.confidence)
        features.append(1.0 if calc_result.error_message else 0.0)
        
        # Unit compatibility features
        claim_quantities = self.quantity_extractor.extract_from_claim(claim)
        table_quantities = self.quantity_extractor.extract_from_table(table_data)
        
        # Check if result unit matches claim quantities
        unit_match = 0.0
        if calc_result.unit and claim_quantities:
            for q in claim_quantities:
                if calc_result.unit == q.unit:
                    unit_match = 1.0
                    break
        features.append(unit_match)
        
        # Number of quantities in claim
        features.append(len(claim_quantities))
        
        # Number of quantities in table
        total_table_quantities = sum(
            len(qs) for qs in table_quantities.get('cells', {}).values()
        )
        features.append(total_table_quantities)
        
        # Dimensional consistency score
        alignments = self.quantity_extractor.align_quantities(
            table_quantities,
            claim_quantities
        )
        features.append(len(alignments))
        
        # Operation type encoding (one-hot subset)
        op_features = [0.0, 0.0, 0.0]  # add/sub, mul/div, comparison
        if calc_result.operation:
            if calc_result.operation.value in ['add', 'subtract']:
                op_features[0] = 1.0
            elif calc_result.operation.value in ['multiply', 'divide']:
                op_features[1] = 1.0
            elif calc_result.operation.value in ['compare', 'percentage_change']:
                op_features[2] = 1.0
        features.extend(op_features)
        
        # Pad to fixed size
        while len(features) < 10:
            features.append(0.0)
        
        return torch.tensor(features[:10], dtype=torch.float32).unsqueeze(0)
    
    def train_step(self, batch: Dict[str, Any], 
                   labels: torch.Tensor,
                   operation_labels: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """Training step with optional operation supervision"""
        total_loss = 0
        losses = {}
        
        # Forward pass
        outputs = []
        for i in range(len(batch['tables'])):
            output = self.forward(batch['tables'][i], batch['claims'][i])
            outputs.append(output)
        
        # Classification loss
        predictions = torch.stack([o.prediction for o in outputs])
        classification_loss = F.cross_entropy(predictions, labels)
        losses['classification'] = classification_loss
        total_loss += classification_loss
        
        # Operation generation loss (if supervised)
        if operation_labels is not None:
            operation_loss = self._compute_operation_loss(outputs, operation_labels)
            losses['operation'] = operation_loss
            total_loss += 0.5 * operation_loss  # Weight the auxiliary loss
        
        # Consistency loss (ensure calculator results are used)
        consistency_loss = self._compute_consistency_loss(outputs, labels)
        losses['consistency'] = consistency_loss
        total_loss += 0.1 * consistency_loss
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_operation_loss(self, outputs: List[ReasoningOutput],
                                operation_labels: List[str]) -> torch.Tensor:
        """Compute loss for operation generation"""
        # This would require tokenizing the operation labels
        # and computing cross-entropy with generated sketches
        # Simplified version:
        return torch.tensor(0.0)
    
    def _compute_consistency_loss(self, outputs: List[ReasoningOutput],
                                  labels: torch.Tensor) -> torch.Tensor:
        """Ensure calculator results influence predictions"""
        consistency_losses = []
        
        for i, output in enumerate(outputs):
            # If calculation succeeded, prediction confidence should be high
            if not output.calculation_result.error_message:
                calc_confidence = output.calculation_result.confidence
                pred_confidence = torch.sigmoid(output.prediction).max()
                
                # Encourage alignment between calc and pred confidence
                consistency = F.mse_loss(pred_confidence, torch.tensor(calc_confidence))
                consistency_losses.append(consistency)
        
        if consistency_losses:
            return torch.stack(consistency_losses).mean()
        return torch.tensor(0.0)