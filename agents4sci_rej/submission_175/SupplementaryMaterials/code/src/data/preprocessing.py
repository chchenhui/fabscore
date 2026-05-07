"""
Data Preprocessing Pipeline for TCGA Pathway Signatures
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pyreadr
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import os
import logging
from pathlib import Path


class TCGAPathwayDataset:
    """
    Main dataset class for loading and preprocessing TCGA pathway signature data.
    """
    
    def __init__(self, 
                 data_dir: str,
                 normalization: str = 'quantile',
                 min_samples_per_cancer: int = 50,
                 pathway_filter_threshold: float = 0.1,
                 test_size: float = 0.2,
                 val_size: float = 0.1,
                 random_state: int = 42):
        
        self.data_dir = Path(data_dir)
        self.normalization = normalization
        self.min_samples_per_cancer = min_samples_per_cancer
        self.pathway_filter_threshold = pathway_filter_threshold
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Initialize storage
        self.pathway_data = None
        self.clinical_data = None
        self.hierarchy_mapping = None
        self.cancer_type_counts = None
        
        # Scalers for normalization
        self.pathway_scaler = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_and_preprocess(self) -> Dict:
        """
        Main method to load and preprocess all TCGA data.
        
        Returns:
            processed_data: Dictionary containing processed datasets
        """
        self.logger.info("Starting TCGA data loading and preprocessing...")
        
        # Load raw data
        raw_data = self._load_raw_data()
        
        # Quality control and filtering
        filtered_data = self._quality_control(raw_data)
        
        # Normalize pathway signatures
        normalized_data = self._normalize_pathways(filtered_data)
        
        # Create hierarchical labels
        hierarchical_data = self._create_hierarchical_labels(normalized_data)
        
        # Split into train/val/test
        splits = self._create_splits(hierarchical_data)
        
        self.logger.info("Data preprocessing completed successfully!")
        
        return splits
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load raw RDS files and combine into single dataframe."""
        self.logger.info("Loading raw TCGA pathway data...")
        
        combined_data = []
        cancer_types = []
        
        # Get all RDS files
        rds_files = list(self.data_dir.glob("*_data.rds"))
        
        if not rds_files:
            raise FileNotFoundError(f"No RDS files found in {self.data_dir}")
        
        for rds_file in rds_files:
            try:
                # Extract cancer type from filename
                cancer_type = rds_file.stem.replace('_data', '')
                
                # Load RDS file
                result = pyreadr.read_r(str(rds_file))
                
                # Get the data (usually first key)
                data_key = list(result.keys())[0]
                data = result[data_key]
                
                # Add cancer type column
                data['cancer_type'] = cancer_type
                
                combined_data.append(data)
                cancer_types.append(cancer_type)
                
                self.logger.info(f"Loaded {cancer_type}: {len(data)} samples")
                
            except Exception as e:
                self.logger.warning(f"Failed to load {rds_file}: {e}")
                continue
        
        if not combined_data:
            raise ValueError("No data was successfully loaded")
        
        # Combine all datasets
        full_data = pd.concat(combined_data, ignore_index=True)
        
        self.logger.info(f"Total loaded: {len(full_data)} samples from {len(cancer_types)} cancer types")
        
        return full_data
    
    def _quality_control(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply quality control filters to the data."""
        self.logger.info("Applying quality control filters...")
        
        initial_samples = len(data)
        
        # Remove samples with too many missing pathway values
        pathway_cols = [col for col in data.columns if col != 'cancer_type']
        missing_threshold = 0.3  # Allow up to 30% missing
        
        missing_fractions = data[pathway_cols].isnull().sum(axis=1) / len(pathway_cols)
        valid_samples = missing_fractions < missing_threshold
        
        data = data[valid_samples].copy()
        self.logger.info(f"Removed {initial_samples - len(data)} samples with >30% missing pathways")
        
        # Filter cancer types with insufficient samples
        cancer_counts = data['cancer_type'].value_counts()
        valid_cancers = cancer_counts[cancer_counts >= self.min_samples_per_cancer].index
        
        data = data[data['cancer_type'].isin(valid_cancers)].copy()
        removed_cancers = set(cancer_counts.index) - set(valid_cancers)
        
        if removed_cancers:
            self.logger.info(f"Removed cancer types with <{self.min_samples_per_cancer} samples: {removed_cancers}")
        
        # Remove pathways with too many missing values across samples
        pathway_missing_fractions = data[pathway_cols].isnull().sum() / len(data)
        valid_pathways = pathway_missing_fractions[pathway_missing_fractions < self.pathway_filter_threshold].index
        
        invalid_pathways = set(pathway_cols) - set(valid_pathways)
        if invalid_pathways:
            self.logger.info(f"Removed {len(invalid_pathways)} pathways with >{self.pathway_filter_threshold*100}% missing values")
        
        # Keep only valid pathways plus cancer_type
        final_cols = list(valid_pathways) + ['cancer_type']
        data = data[final_cols].copy()
        
        # Fill remaining missing values with median
        for col in valid_pathways:
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
        
        self.cancer_type_counts = data['cancer_type'].value_counts()
        self.logger.info(f"Final dataset: {len(data)} samples, {len(valid_pathways)} pathways")
        self.logger.info(f"Cancer type distribution:\\n{self.cancer_type_counts}")
        
        return data
    
    def _normalize_pathways(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize pathway signature values."""
        self.logger.info(f"Normalizing pathways using {self.normalization} method...")
        
        pathway_cols = [col for col in data.columns if col != 'cancer_type']
        pathway_data = data[pathway_cols].values
        
        if self.normalization == 'quantile':
            self.pathway_scaler = QuantileTransformer(n_quantiles=1000, random_state=self.random_state)
        elif self.normalization == 'standard':
            self.pathway_scaler = StandardScaler()
        else:
            self.logger.warning(f"Unknown normalization method: {self.normalization}")
            return data
        
        # Fit and transform
        normalized_pathways = self.pathway_scaler.fit_transform(pathway_data)
        
        # Create normalized dataframe
        normalized_data = data.copy()
        normalized_data[pathway_cols] = normalized_pathways
        
        self.logger.info("Pathway normalization completed")
        
        return normalized_data
    
    def _create_hierarchical_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create hierarchical labels for multi-level classification."""
        try:
            from ..models.hierarchical_maml import create_hierarchy_mapping
        except ImportError:
            from models.hierarchical_maml import create_hierarchy_mapping
        
        self.logger.info("Creating hierarchical labels...")
        
        self.hierarchy_mapping = create_hierarchy_mapping()
        
        # Create hierarchical label columns
        data['organ_label'] = data['cancer_type'].map(self.hierarchy_mapping['organ'])
        data['histology_label'] = data['cancer_type'].map(self.hierarchy_mapping['histology'])
        data['molecular_label'] = data['cancer_type'].map(self.hierarchy_mapping['molecular'])
        
        # Check for unmapped cancer types
        unmapped = data[data['organ_label'].isnull()]['cancer_type'].unique()
        if len(unmapped) > 0:
            self.logger.warning(f"Unmapped cancer types: {unmapped}")
            # Remove unmapped samples
            data = data.dropna(subset=['organ_label', 'histology_label', 'molecular_label'])
        
        self.logger.info("Hierarchical labels created successfully")
        
        return data
    
    def _create_splits(self, data: pd.DataFrame) -> Dict:
        """Create train/validation/test splits for meta-learning."""
        self.logger.info("Creating train/validation/test splits...")
        
        pathway_cols = [col for col in data.columns 
                       if col not in ['cancer_type', 'organ_label', 'histology_label', 'molecular_label']]
        
        # Split by cancer types for meta-learning
        unique_cancers = data['cancer_type'].unique()
        
        # First split: train+val vs test
        train_val_cancers, test_cancers = train_test_split(
            unique_cancers, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Second split: train vs val
        train_cancers, val_cancers = train_test_split(
            train_val_cancers,
            test_size=self.val_size / (1 - self.test_size),
            random_state=self.random_state
        )
        
        # Create splits
        train_data = data[data['cancer_type'].isin(train_cancers)].copy()
        val_data = data[data['cancer_type'].isin(val_cancers)].copy()
        test_data = data[data['cancer_type'].isin(test_cancers)].copy()
        
        self.logger.info(f"Train cancers ({len(train_cancers)}): {sorted(train_cancers)}")
        self.logger.info(f"Val cancers ({len(val_cancers)}): {sorted(val_cancers)}")
        self.logger.info(f"Test cancers ({len(test_cancers)}): {sorted(test_cancers)}")
        
        # Extract features and labels
        splits = {}
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            splits[split_name] = {
                'pathway_data': split_data[pathway_cols].values.astype(np.float32),
                'cancer_types': split_data['cancer_type'].values,
                'organ_labels': split_data['organ_label'].values.astype(np.int64),
                'histology_labels': split_data['histology_label'].values.astype(np.int64),
                'molecular_labels': split_data['molecular_label'].values.astype(np.int64),
                'sample_ids': split_data.index.values
            }
            
            self.logger.info(f"{split_name.title()} set: {len(split_data)} samples")
        
        return splits


class MetaLearningDataLoader:
    """
    Data loader for meta-learning episodes with hierarchical classification.
    """
    
    def __init__(self,
                 pathway_data: np.ndarray,
                 cancer_types: np.ndarray,
                 hierarchical_labels: Dict[str, np.ndarray],
                 n_way: int = 5,
                 k_shot: int = 5,
                 n_query: int = 15,
                 n_tasks_per_batch: int = 8):
        
        self.pathway_data = pathway_data
        self.cancer_types = cancer_types
        self.hierarchical_labels = hierarchical_labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_tasks_per_batch = n_tasks_per_batch
        
        # Group samples by cancer type
        self.cancer_to_indices = {}
        for i, cancer_type in enumerate(cancer_types):
            if cancer_type not in self.cancer_to_indices:
                self.cancer_to_indices[cancer_type] = []
            self.cancer_to_indices[cancer_type].append(i)
        
        self.available_cancers = list(self.cancer_to_indices.keys())
        
    def sample_task(self) -> Dict:
        """Sample a single meta-learning task."""
        # Sample cancer types for this task
        if len(self.available_cancers) < self.n_way:
            # If not enough cancer types, sample with replacement
            task_cancers = np.random.choice(
                self.available_cancers, size=self.n_way, replace=True
            )
        else:
            task_cancers = np.random.choice(
                self.available_cancers, size=self.n_way, replace=False
            )
        
        support_data, support_labels = [], {}
        query_data, query_labels = [], {}
        
        # Initialize label containers
        for level in ['organ', 'histology', 'molecular']:
            support_labels[level] = []
            query_labels[level] = []
        
        for i, cancer_type in enumerate(task_cancers):
            # Get indices for this cancer type
            cancer_indices = self.cancer_to_indices[cancer_type]
            
            # Sample support and query sets
            total_needed = self.k_shot + self.n_query
            if len(cancer_indices) < total_needed:
                # Oversample if not enough samples
                sampled_indices = np.random.choice(
                    cancer_indices, size=total_needed, replace=True
                )
            else:
                sampled_indices = np.random.choice(
                    cancer_indices, size=total_needed, replace=False
                )
            
            # Split into support and query
            support_indices = sampled_indices[:self.k_shot]
            query_indices = sampled_indices[self.k_shot:]
            
            # Get data
            support_data.append(self.pathway_data[support_indices])
            query_data.append(self.pathway_data[query_indices])
            
            # Get hierarchical labels for task-specific labels (0 to n_way-1)
            support_labels['organ'].extend([i] * self.k_shot)
            support_labels['histology'].extend([i] * self.k_shot)
            support_labels['molecular'].extend([i] * self.k_shot)
            
            query_labels['organ'].extend([i] * self.n_query)
            query_labels['histology'].extend([i] * self.n_query)
            query_labels['molecular'].extend([i] * self.n_query)
        
        # Convert to tensors
        support_x = torch.FloatTensor(np.concatenate(support_data, axis=0))
        query_x = torch.FloatTensor(np.concatenate(query_data, axis=0))
        
        support_y = {
            level: torch.LongTensor(labels) 
            for level, labels in support_labels.items()
        }
        query_y = {
            level: torch.LongTensor(labels) 
            for level, labels in query_labels.items()
        }
        
        return {
            'support': (support_x, support_y),
            'query': (query_x, query_y),
            'cancer_types': task_cancers
        }
    
    def get_batch(self) -> List[Dict]:
        """Generate a batch of meta-learning tasks."""
        batch = []
        for _ in range(self.n_tasks_per_batch):
            task = self.sample_task()
            batch.append(task)
        return batch
    
    def __iter__(self):
        """Make the dataloader iterable."""
        while True:
            yield self.get_batch()