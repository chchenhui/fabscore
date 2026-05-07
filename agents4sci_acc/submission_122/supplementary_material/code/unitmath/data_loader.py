"""
Data Loader for SciTab Dataset
Handles loading and preprocessing of table-claim pairs from JSON
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class SciTabSample:
    """Represents a single SciTab sample"""
    paper: str
    paper_id: str
    table_caption: str
    table_column_names: List[str]
    table_content_values: List[List[str]]
    sample_id: str
    claim: str
    label: str
    table_id: str
    
    def to_table_dict(self) -> Dict[str, Any]:
        """Convert to table dictionary format used by UnitMath"""
        return {
            'headers': self.table_column_names,
            'data': self.table_content_values,
            'caption': self.table_caption,
            'metadata': {
                'paper': self.paper,
                'paper_id': self.paper_id,
                'table_id': self.table_id,
                'sample_id': self.sample_id
            }
        }
    
    def clean_cell_values(self):
        """Clean cell values by removing markup like [BOLD]"""
        cleaned_values = []
        for row in self.table_content_values:
            cleaned_row = []
            for cell in row:
                # Remove [BOLD] and other markup
                cleaned_cell = cell.replace('[BOLD] ', '').replace('[BOLD]', '')
                cleaned_cell = cleaned_cell.strip()
                cleaned_row.append(cleaned_cell)
            cleaned_values.append(cleaned_row)
        self.table_content_values = cleaned_values


class SciTabDataset(Dataset):
    """PyTorch Dataset for SciTab data"""
    
    def __init__(self, json_path: str, 
                 label_mapping: Optional[Dict[str, int]] = None,
                 clean_cells: bool = True):
        """
        Initialize SciTab dataset
        
        Args:
            json_path: Path to JSON file containing SciTab data
            label_mapping: Mapping from label strings to integers
            clean_cells: Whether to clean cell values (remove markup)
        """
        self.json_path = json_path
        self.clean_cells = clean_cells
        
        # Default label mapping for 3-way classification
        if label_mapping is None:
            self.label_mapping = {
                'Supported': 0,
                'Refuted': 1,
                'NEI': 2  # Not Enough Information
            }
        else:
            self.label_mapping = label_mapping
        
        # Load data
        self.samples = self._load_json()
        
        # Clean cell values if requested
        if self.clean_cells:
            for sample in self.samples:
                sample.clean_cell_values()
    
    def _load_json(self) -> List[SciTabSample]:
        """Load and parse JSON file"""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            sample = SciTabSample(
                paper=item.get('paper', ''),
                paper_id=item.get('paper_id', ''),
                table_caption=item.get('table_caption', ''),
                table_column_names=item.get('table_column_names', []),
                table_content_values=item.get('table_content_values', []),
                sample_id=item.get('id', ''),
                claim=item.get('claim', ''),
                label=item.get('label', 'NEI'),
                table_id=item.get('table_id', '')
            )
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        return {
            'table': sample.to_table_dict(),
            'claim': sample.claim,
            'label': self.label_mapping.get(sample.label, 2),  # Default to NEI
            'label_str': sample.label,
            'metadata': {
                'paper': sample.paper,
                'paper_id': sample.paper_id,
                'sample_id': sample.sample_id,
                'table_id': sample.table_id
            }
        }
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in dataset"""
        distribution = {}
        for sample in self.samples:
            label = sample.label
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def filter_by_label(self, labels: List[str]) -> 'SciTabDataset':
        """Create a filtered dataset with only specified labels"""
        filtered_samples = [s for s in self.samples if s.label in labels]
        
        # Create new dataset instance
        new_dataset = SciTabDataset.__new__(SciTabDataset)
        new_dataset.json_path = self.json_path
        new_dataset.clean_cells = self.clean_cells
        new_dataset.label_mapping = self.label_mapping
        new_dataset.samples = filtered_samples
        
        return new_dataset


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching"""
    tables = [item['table'] for item in batch]
    claims = [item['claim'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    label_strs = [item['label_str'] for item in batch]
    metadata = [item['metadata'] for item in batch]
    
    return {
        'tables': tables,
        'claims': claims,
        'labels': labels,
        'label_strs': label_strs,
        'metadata': metadata
    }


def load_scitab_data(json_path: str,
                     batch_size: int = 32,
                     shuffle: bool = True,
                     num_workers: int = 0,
                     clean_cells: bool = True,
                     filter_labels: Optional[List[str]] = None) -> DataLoader:
    """
    Load SciTab data and create DataLoader
    
    Args:
        json_path: Path to JSON file
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        num_workers: Number of workers for DataLoader
        clean_cells: Whether to clean cell markup
        filter_labels: If provided, only include samples with these labels
    
    Returns:
        DataLoader for SciTab dataset
    """
    dataset = SciTabDataset(json_path, clean_cells=clean_cells)
    
    # Filter by labels if specified
    if filter_labels:
        dataset = dataset.filter_by_label(filter_labels)
    
    # Print dataset statistics
    print(f"Loaded {len(dataset)} samples from {json_path}")
    print(f"Label distribution: {dataset.get_label_distribution()}")
    
    # Create DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return loader


class SciTabProcessor:
    """Processes SciTab data for UnitMath system"""
    
    def __init__(self):
        from .unit_parser import QuantityExtractor
        self.quantity_extractor = QuantityExtractor()
    
    def preprocess_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single sample for UnitMath
        
        Args:
            sample: Dictionary containing table, claim, etc.
        
        Returns:
            Preprocessed sample with extracted quantities
        """
        table = sample['table']
        claim = sample['claim']
        
        # Extract quantities from table
        table_quantities = self.quantity_extractor.extract_from_table(table)
        
        # Extract quantities from claim
        claim_quantities = self.quantity_extractor.extract_from_claim(claim)
        
        # Find alignments
        alignments = self.quantity_extractor.align_quantities(
            table_quantities,
            claim_quantities
        )
        
        # Add to sample
        sample['table_quantities'] = table_quantities
        sample['claim_quantities'] = claim_quantities
        sample['quantity_alignments'] = alignments
        
        # Add statistics
        sample['stats'] = {
            'num_table_quantities': sum(
                len(qs) for qs in table_quantities.get('cells', {}).values()
            ),
            'num_claim_quantities': len(claim_quantities),
            'num_alignments': len(alignments),
            'has_quantities': len(claim_quantities) > 0
        }
        
        return sample
    
    def identify_numeric_claims(self, dataset: SciTabDataset) -> List[int]:
        """
        Identify samples that involve numeric reasoning
        
        Args:
            dataset: SciTabDataset instance
        
        Returns:
            List of indices for samples with numeric claims
        """
        numeric_indices = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            claim = sample['claim']
            
            # Check if claim contains quantities
            claim_quantities = self.quantity_extractor.extract_from_claim(claim)
            
            # Check for numeric keywords
            numeric_keywords = [
                'increase', 'decrease', 'higher', 'lower', 'more', 'less',
                'percentage', 'percent', '%', 'fold', 'times', 'ratio',
                'average', 'mean', 'median', 'sum', 'total',
                'significant', 'correlation', 'p-value', 'confidence'
            ]
            
            has_quantities = len(claim_quantities) > 0
            has_keywords = any(keyword in claim.lower() for keyword in numeric_keywords)
            
            if has_quantities or has_keywords:
                numeric_indices.append(i)
        
        return numeric_indices
    
    def create_train_test_split(self, 
                               dataset: SciTabDataset,
                               test_ratio: float = 0.2,
                               seed: int = 42) -> Tuple[SciTabDataset, SciTabDataset]:
        """
        Create train/test split of dataset
        
        Args:
            dataset: SciTabDataset instance
            test_ratio: Ratio of samples for test set
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        np.random.seed(seed)
        
        # Get indices
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        # Split indices
        n_test = int(n_samples * test_ratio)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        # Create subset datasets
        train_samples = [dataset.samples[i] for i in train_indices]
        test_samples = [dataset.samples[i] for i in test_indices]
        
        # Create new dataset instances
        train_dataset = SciTabDataset.__new__(SciTabDataset)
        train_dataset.json_path = dataset.json_path
        train_dataset.clean_cells = dataset.clean_cells
        train_dataset.label_mapping = dataset.label_mapping
        train_dataset.samples = train_samples
        
        test_dataset = SciTabDataset.__new__(SciTabDataset)
        test_dataset.json_path = dataset.json_path
        test_dataset.clean_cells = dataset.clean_cells
        test_dataset.label_mapping = dataset.label_mapping
        test_dataset.samples = test_samples
        
        return train_dataset, test_dataset


# Example usage functions
def load_and_analyze_scitab(json_path: str = "sci_tab.json"):
    """
    Load and analyze SciTab dataset
    
    Args:
        json_path: Path to SciTab JSON file
    """
    # Load dataset
    dataset = SciTabDataset(json_path)
    processor = SciTabProcessor()
    
    print(f"\n=== SciTab Dataset Analysis ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Label distribution: {dataset.get_label_distribution()}")
    
    # Identify numeric claims
    numeric_indices = processor.identify_numeric_claims(dataset)
    print(f"\nNumeric claims: {len(numeric_indices)} ({len(numeric_indices)/len(dataset)*100:.1f}%)")
    
    # Analyze a sample
    if len(dataset) > 0:
        sample = dataset[0]
        processed = processor.preprocess_sample(sample)
        
        print(f"\n=== Sample Analysis ===")
        print(f"Claim: {sample['claim'][:100]}...")
        print(f"Label: {sample['label_str']}")
        print(f"Table size: {len(sample['table']['headers'])} columns x {len(sample['table']['data'])} rows")
        print(f"Quantities in table: {processed['stats']['num_table_quantities']}")
        print(f"Quantities in claim: {processed['stats']['num_claim_quantities']}")
        print(f"Alignments found: {processed['stats']['num_alignments']}")
    
    return dataset


if __name__ == "__main__":
    # Example: Load and analyze the dataset
    dataset = load_and_analyze_scitab("sci_tab.json")