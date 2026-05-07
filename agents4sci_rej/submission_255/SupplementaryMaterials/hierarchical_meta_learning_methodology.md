# Hierarchical Meta-Learning for Cancer Pathway Signatures: Comprehensive Methodology

## 1. Experimental Design

### 1.1 Meta-Learning Task Formulation

**Task Definition**: Each "task" τᵢ represents learning to classify a specific cancer type using pathway signatures from a limited number of samples.

- **Support Set S**: K samples per cancer type (K-shot learning, typically K ∈ {1, 5, 10})
- **Query Set Q**: N samples for evaluation within the same cancer type
- **Task Distribution**: p(τ) samples tasks from 36 cancer types with varying frequencies

**Mathematical Formulation**:
```
τᵢ = {(Sᵢ, Qᵢ, yᵢ)} where:
- Sᵢ = {(x₁, y₁), ..., (xₖ, yₖ)} ∈ ℝᵏˣ³²
- Qᵢ = {(x₁, y₁), ..., (xₙ, yₙ)} ∈ ℝⁿˣ³²
- xⱼ ∈ ℝ³² (32 pathway signatures)
- yᵢ ∈ {1, ..., C} (cancer type labels)
```

### 1.2 Training/Validation/Testing Splits

**Hierarchical Splitting Strategy**:

1. **Cancer Type Level** (36 types):
   - Meta-Training: 24 cancer types (67%)
   - Meta-Validation: 6 cancer types (17%)
   - Meta-Testing: 6 cancer types (16%)

2. **Sample Level** (within each cancer type):
   - Training Pool: 70% of samples
   - Validation Pool: 15% of samples  
   - Testing Pool: 15% of samples

3. **Episode Sampling**:
   - Support/Query split: Dynamic sampling from training pool
   - Ensure balanced representation across cancer subtypes

### 1.3 Few-Shot Evaluation Scenarios

**Multi-Scale Evaluation**:
- **1-shot**: Single sample per cancer type in support set
- **5-shot**: Five samples per cancer type in support set
- **10-shot**: Ten samples per cancer type in support set
- **Cross-domain**: Train on primary tumors, test on metastases

**Query Set Construction**:
- Fixed query size: 20 samples per task
- Stratified sampling to maintain cancer subtype distribution
- Temporal holdout for longitudinal validation

### 1.4 Cancer Type Hierarchy Construction

**Multi-Level Hierarchy**:

Level 1 (Organ System):
- Gastrointestinal (COAD, READ, STAD, ESCA, LIHC, PAAD, CHOL)
- Genitourinary (KIRC, KIRP, KICH, BLCA, PRAD, TGCT, CESC, UCEC, OV)
- Thoracic (LUAD, LUSC, MESO, THYM)
- Hematologic (LAML, DLBC, THCA)
- Nervous System (GBM, LGG)
- Skin/Soft Tissue (SKCM, SARC, UCS)
- Head/Neck (HNSC)
- Breast (BRCA)
- Other (ACC, PCPG, UVM)

Level 2 (Histology):
- Adenocarcinoma
- Squamous Cell Carcinoma
- Sarcoma
- Hematologic Malignancies

Level 3 (Molecular Subtypes):
- Based on pathway signature clustering

## 2. Technical Architecture

### 2.1 Base Pathway Signature Encoder

**Neural Architecture**:
```python
class PathwayEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[64, 128, 64], output_dim=32):
        super().__init__()
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims)-2:  # No activation after last layer
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(0.1))
        
        # Pathway-specific attention
        self.pathway_attention = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=4, batch_first=True
        )
        
    def forward(self, x):
        # x: [batch_size, 32] pathway signatures
        for layer in self.layers:
            x = layer(x)
        
        # Self-attention across pathway dimensions
        x_attended, _ = self.pathway_attention(x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1))
        return x_attended.squeeze(1)
```

### 2.2 Hierarchical Meta-Learner

**MAML-inspired Hierarchical Architecture**:
```python
class HierarchicalMAML(nn.Module):
    def __init__(self, encoder, hierarchy_levels=3):
        super().__init__()
        self.encoder = encoder
        self.hierarchy_levels = hierarchy_levels
        
        # Hierarchical classifiers
        self.level_classifiers = nn.ModuleDict({
            'organ': nn.Linear(32, 9),      # 9 organ systems
            'histology': nn.Linear(32, 4),  # 4 histology types
            'molecular': nn.Linear(32, 36)  # 36 cancer types
        })
        
        # Hierarchical attention weights
        self.level_attention = nn.Parameter(torch.ones(hierarchy_levels))
        
    def forward(self, x, adaptation_params=None):
        # Encode pathway signatures
        features = self.encoder(x)
        
        # Hierarchical predictions
        predictions = {}
        for level, classifier in self.level_classifiers.items():
            if adaptation_params and level in adaptation_params:
                # Use adapted parameters
                pred = F.linear(features, adaptation_params[level]['weight'], 
                              adaptation_params[level]['bias'])
            else:
                pred = classifier(features)
            predictions[level] = pred
            
        return predictions, features
```

### 2.3 Hierarchical Attention Mechanism

**Cross-Level Attention**:
```python
class HierarchicalAttention(nn.Module):
    def __init__(self, feature_dim=32, num_levels=3):
        super().__init__()
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4, batch_first=True
        )
        self.level_embeddings = nn.Embedding(num_levels, feature_dim)
        
    def forward(self, features, hierarchy_level):
        # Add positional encoding for hierarchy level
        level_emb = self.level_embeddings(hierarchy_level)
        features_with_level = features + level_emb
        
        # Cross-level attention
        attended_features, attention_weights = self.cross_level_attention(
            features_with_level, features_with_level, features_with_level
        )
        
        return attended_features, attention_weights
```

### 2.4 Loss Functions and Optimization

**Hierarchical Loss Function**:
```python
def hierarchical_loss(predictions, targets, hierarchy_weights=[1.0, 0.7, 0.5]):
    """
    Multi-level loss combining organ, histology, and molecular predictions
    """
    total_loss = 0
    
    # Organ-level loss
    organ_targets = map_to_organ_level(targets)
    organ_loss = F.cross_entropy(predictions['organ'], organ_targets)
    
    # Histology-level loss
    histology_targets = map_to_histology_level(targets)
    histology_loss = F.cross_entropy(predictions['histology'], histology_targets)
    
    # Molecular-level loss (final cancer type)
    molecular_loss = F.cross_entropy(predictions['molecular'], targets)
    
    # Weighted combination
    total_loss = (hierarchy_weights[0] * molecular_loss + 
                 hierarchy_weights[1] * histology_loss + 
                 hierarchy_weights[2] * organ_loss)
    
    return total_loss

def meta_loss(model, tasks, inner_lr=0.01, inner_steps=5):
    """
    MAML-style meta-learning loss with hierarchical components
    """
    meta_loss = 0
    
    for task in tasks:
        support_x, support_y = task['support']
        query_x, query_y = task['query']
        
        # Inner loop adaptation
        adapted_params = {}
        current_params = {name: param.clone() for name, param in model.named_parameters()}
        
        for step in range(inner_steps):
            predictions, _ = model(support_x, adapted_params)
            support_loss = hierarchical_loss(predictions, support_y)
            
            # Compute gradients and update adapted parameters
            grads = torch.autograd.grad(support_loss, current_params.values(), 
                                      create_graph=True, retain_graph=True)
            
            adapted_params = {
                name: param - inner_lr * grad 
                for (name, param), grad in zip(current_params.items(), grads)
            }
            current_params = adapted_params
        
        # Outer loop evaluation
        query_predictions, _ = model(query_x, adapted_params)
        query_loss = hierarchical_loss(query_predictions, query_y)
        meta_loss += query_loss
    
    return meta_loss / len(tasks)
```

## 3. Implementation Pipeline

### 3.1 Data Preprocessing Strategy

```python
class PathwayDataPreprocessor:
    def __init__(self, normalization='quantile', batch_correction=True):
        self.normalization = normalization
        self.batch_correction = batch_correction
        self.scalers = {}
        self.batch_corrector = None
        
    def preprocess(self, pathway_data, batch_info=None, clinical_data=None):
        """
        Comprehensive preprocessing pipeline for pathway signatures
        """
        # 1. Quality control
        pathway_data = self.quality_control(pathway_data)
        
        # 2. Batch effect correction
        if self.batch_correction and batch_info is not None:
            pathway_data = self.correct_batch_effects(pathway_data, batch_info)
        
        # 3. Normalization
        pathway_data = self.normalize_pathways(pathway_data)
        
        # 4. Feature selection/filtering
        pathway_data = self.filter_pathways(pathway_data)
        
        return pathway_data
    
    def quality_control(self, data):
        # Remove samples with >20% missing pathway values
        missing_threshold = 0.2
        valid_samples = (data.isnull().sum(axis=1) / data.shape[1]) < missing_threshold
        return data[valid_samples]
    
    def correct_batch_effects(self, data, batch_info):
        # Use Combat or similar batch correction
        from pycombat import pycombat
        corrected_data = pycombat(data.T, batch_info).T
        return corrected_data
    
    def normalize_pathways(self, data):
        if self.normalization == 'quantile':
            # Quantile normalization across samples
            from sklearn.preprocessing import quantile_transform
            normalized = quantile_transform(data.T, n_quantiles=1000).T
        elif self.normalization == 'zscore':
            # Z-score normalization per pathway
            normalized = (data - data.mean()) / data.std()
        else:
            normalized = data
            
        return pd.DataFrame(normalized, index=data.index, columns=data.columns)
```

### 3.2 Meta-Learning Batch Construction

```python
class MetaLearningDataLoader:
    def __init__(self, pathway_data, cancer_labels, hierarchy_map, 
                 n_way=5, k_shot=5, n_query=15, n_tasks_per_batch=8):
        self.pathway_data = pathway_data
        self.cancer_labels = cancer_labels
        self.hierarchy_map = hierarchy_map
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.n_tasks_per_batch = n_tasks_per_batch
        
    def sample_task(self, available_cancer_types):
        """Sample a single meta-learning task"""
        # Sample cancer types for this task
        task_cancer_types = np.random.choice(
            available_cancer_types, size=self.n_way, replace=False
        )
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for i, cancer_type in enumerate(task_cancer_types):
            # Get samples for this cancer type
            cancer_samples = self.pathway_data[self.cancer_labels == cancer_type]
            
            # Sample support and query sets
            total_needed = self.k_shot + self.n_query
            if len(cancer_samples) < total_needed:
                # Oversample if not enough samples
                sampled_indices = np.random.choice(
                    len(cancer_samples), size=total_needed, replace=True
                )
            else:
                sampled_indices = np.random.choice(
                    len(cancer_samples), size=total_needed, replace=False
                )
            
            sampled_data = cancer_samples.iloc[sampled_indices]
            
            # Split into support and query
            support_data.append(sampled_data.iloc[:self.k_shot])
            query_data.append(sampled_data.iloc[self.k_shot:])
            
            # Create labels (0 to n_way-1 for this task)
            support_labels.extend([i] * self.k_shot)
            query_labels.extend([i] * self.n_query)
        
        # Combine and convert to tensors
        support_x = torch.FloatTensor(pd.concat(support_data).values)
        support_y = torch.LongTensor(support_labels)
        query_x = torch.FloatTensor(pd.concat(query_data).values)
        query_y = torch.LongTensor(query_labels)
        
        return {
            'support': (support_x, support_y),
            'query': (query_x, query_y),
            'cancer_types': task_cancer_types
        }
    
    def get_batch(self, available_cancer_types):
        """Generate a batch of meta-learning tasks"""
        batch = []
        for _ in range(self.n_tasks_per_batch):
            task = self.sample_task(available_cancer_types)
            batch.append(task)
        return batch
```

### 3.3 Training Procedures

```python
class HierarchicalMetaTrainer:
    def __init__(self, model, meta_lr=0.001, inner_lr=0.01, inner_steps=5):
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
    def train_epoch(self, dataloader, device):
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, task_batch in enumerate(dataloader):
            # Compute meta-loss for this batch
            batch_loss = meta_loss(self.model, task_batch, 
                                 self.inner_lr, self.inner_steps)
            
            # Meta-gradient update
            self.meta_optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.meta_optimizer.step()
            
            epoch_loss += batch_loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {batch_loss.item():.4f}')
        
        return epoch_loss / len(dataloader)
    
    def validate(self, val_dataloader, device):
        self.model.eval()
        total_accuracy = 0
        total_tasks = 0
        
        with torch.no_grad():
            for task_batch in val_dataloader:
                for task in task_batch:
                    support_x, support_y = task['support']
                    query_x, query_y = task['query']
                    
                    # Fast adaptation
                    adapted_params = self.fast_adapt(support_x, support_y)
                    
                    # Evaluate on query set
                    predictions, _ = self.model(query_x, adapted_params)
                    accuracy = (predictions['molecular'].argmax(dim=1) == query_y).float().mean()
                    
                    total_accuracy += accuracy.item()
                    total_tasks += 1
        
        return total_accuracy / total_tasks
```

## 4. Baseline Comparisons

### 4.1 Standard Machine Learning Baselines

1. **Random Forest Classifier**
   - Standard RF with 100 trees
   - Feature importance analysis
   - Cross-validation with stratification

2. **Support Vector Machine**
   - RBF and linear kernels
   - Hyperparameter tuning via grid search
   - One-vs-rest multi-class strategy

3. **Gradient Boosting (XGBoost)**
   - Tree-based ensemble method
   - Early stopping and regularization
   - SHAP values for interpretability

4. **Neural Network (Standard)**
   - Multi-layer perceptron
   - Dropout and batch normalization
   - Standard cross-entropy loss

### 4.2 Recent Cancer Classification Methods

1. **DeepCC** (Deep Cancer Classification)
   - Deep learning approach for cancer subtype classification
   - Comparison on pathway-level features

2. **MOFA+** (Multi-Omics Factor Analysis)
   - Dimensionality reduction approach
   - Followed by standard classification

3. **Pathway Activity Score Methods**
   - GSVA (Gene Set Variation Analysis)
   - ssGSEA (single-sample GSEA)
   - Combined with ensemble methods

### 4.3 Meta-Learning Baselines

1. **Prototypical Networks**
   - Distance-based classification in embedding space
   - Euclidean and cosine distance metrics

2. **Relation Networks**
   - Learning similarity metrics between samples
   - Adapted for pathway signature data

3. **Model-Agnostic Meta-Learning (MAML)**
   - Standard MAML without hierarchical components
   - Direct comparison to hierarchical version

### 4.4 Ablation Study Design

**Component Ablation**:
1. Remove hierarchical structure (flat classification)
2. Remove attention mechanisms
3. Remove pathway-specific encoding
4. Remove meta-learning (standard fine-tuning)
5. Different hierarchy levels (2-level vs 3-level)

**Architecture Ablation**:
1. Different encoder architectures
2. Various meta-learning algorithms
3. Alternative attention mechanisms
4. Different loss function weights

## 5. Biological Validation

### 5.1 Pathway Importance Interpretation

```python
class PathwayImportanceAnalyzer:
    def __init__(self, model, pathway_names):
        self.model = model
        self.pathway_names = pathway_names
        
    def compute_pathway_importance(self, test_data, method='integrated_gradients'):
        """Compute pathway importance scores"""
        if method == 'integrated_gradients':
            return self.integrated_gradients(test_data)
        elif method == 'attention_weights':
            return self.extract_attention_weights(test_data)
        elif method == 'permutation':
            return self.permutation_importance(test_data)
    
    def integrated_gradients(self, data, n_steps=50):
        """Compute integrated gradients for pathway importance"""
        baseline = torch.zeros_like(data)
        importance_scores = []
        
        for step in range(n_steps):
            alpha = step / n_steps
            interpolated = baseline + alpha * (data - baseline)
            interpolated.requires_grad_(True)
            
            predictions, _ = self.model(interpolated)
            target_class = predictions['molecular'].argmax(dim=1)
            
            gradients = torch.autograd.grad(
                predictions['molecular'][range(len(target_class)), target_class].sum(),
                interpolated, retain_graph=True
            )[0]
            
            importance_scores.append(gradients)
        
        # Integrate gradients
        integrated_grads = torch.stack(importance_scores).mean(dim=0)
        pathway_importance = (data - baseline) * integrated_grads
        
        return pathway_importance.abs().mean(dim=0)
```

### 5.2 Cross-Cancer Transferability Analysis

```python
def analyze_transferability(model, source_cancer, target_cancer, pathway_data):
    """Analyze how well patterns transfer between cancer types"""
    # Train on source cancer type
    source_data = pathway_data[pathway_data['cancer_type'] == source_cancer]
    
    # Adapt to target cancer type with few samples
    target_data = pathway_data[pathway_data['cancer_type'] == target_cancer]
    few_shot_target = target_data.sample(n=10)  # 10-shot adaptation
    
    # Measure adaptation efficiency
    adaptation_curves = []
    for n_shots in [1, 2, 5, 10]:
        subset = few_shot_target.iloc[:n_shots]
        adapted_model = fast_adapt(model, subset)
        accuracy = evaluate_model(adapted_model, target_data)
        adaptation_curves.append((n_shots, accuracy))
    
    return adaptation_curves

def pathway_transfer_heatmap(model, all_cancer_types, pathway_data):
    """Create transferability heatmap between cancer types"""
    transfer_matrix = np.zeros((len(all_cancer_types), len(all_cancer_types)))
    
    for i, source in enumerate(all_cancer_types):
        for j, target in enumerate(all_cancer_types):
            if i != j:  # Don't compute self-transfer
                curves = analyze_transferability(model, source, target, pathway_data)
                # Use 5-shot performance as transfer score
                transfer_score = curves[2][1]  # 5-shot accuracy
                transfer_matrix[i, j] = transfer_score
    
    return transfer_matrix
```

### 5.3 Clinical Outcome Correlation

```python
def correlate_with_survival(pathway_importance, clinical_data, cancer_type):
    """Correlate pathway importance with survival outcomes"""
    from lifelines import CoxPHFitter
    
    # Prepare survival data
    survival_data = clinical_data[clinical_data['cancer_type'] == cancer_type].copy()
    survival_data = survival_data.merge(pathway_importance, on='sample_id')
    
    # Cox proportional hazards model
    cph = CoxPHFitter()
    
    # Test each pathway individually
    pathway_hazard_ratios = {}
    for pathway in pathway_importance.columns:
        if pathway in ['sample_id', 'cancer_type']:
            continue
            
        temp_data = survival_data[['duration', 'event', pathway]].dropna()
        if len(temp_data) > 50:  # Minimum sample size
            cph.fit(temp_data, duration_col='duration', event_col='event')
            hr = np.exp(cph.params_[pathway])
            p_value = cph.summary.loc[pathway, 'p']
            pathway_hazard_ratios[pathway] = {'HR': hr, 'p_value': p_value}
    
    return pathway_hazard_ratios

def drug_response_correlation(pathway_signatures, drug_response_data):
    """Correlate pathway signatures with drug response"""
    from scipy.stats import spearmanr
    
    correlations = {}
    for drug in drug_response_data.columns:
        drug_correlations = {}
        for pathway in pathway_signatures.columns:
            corr, p_val = spearmanr(
                pathway_signatures[pathway], 
                drug_response_data[drug]
            )
            drug_correlations[pathway] = {'correlation': corr, 'p_value': p_val}
        correlations[drug] = drug_correlations
    
    return correlations
```

### 5.4 Literature Validation

```python
def validate_with_literature(important_pathways, cancer_type):
    """Validate important pathways against known literature"""
    
    # Known cancer-pathway associations from literature
    literature_associations = {
        'BRCA': ['PI3K-AKT', 'p53', 'MAPK', 'Cell Cycle', 'DNA Repair'],
        'LUAD': ['EGFR', 'KRAS', 'p53', 'Apoptosis', 'Angiogenesis'],
        'COAD': ['WNT', 'p53', 'KRAS', 'TGF-beta', 'Cell Adhesion'],
        # Add more associations...
    }
    
    if cancer_type in literature_associations:
        known_pathways = set(literature_associations[cancer_type])
        discovered_pathways = set(important_pathways.keys())
        
        # Calculate validation metrics
        overlap = known_pathways.intersection(discovered_pathways)
        precision = len(overlap) / len(discovered_pathways) if discovered_pathways else 0
        recall = len(overlap) / len(known_pathways) if known_pathways else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
            'overlapping_pathways': list(overlap),
            'novel_pathways': list(discovered_pathways - known_pathways)
        }
    
    return None
```

## 6. Reproducibility Plan

### 6.1 Code Organization

```
hierarchical_meta_learning/
├── src/
│   ├── models/
│   │   ├── hierarchical_maml.py
│   │   ├── pathway_encoder.py
│   │   └── attention_mechanisms.py
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── meta_dataloader.py
│   │   └── hierarchy_construction.py
│   ├── training/
│   │   ├── meta_trainer.py
│   │   ├── evaluation.py
│   │   └── baselines.py
│   ├── analysis/
│   │   ├── pathway_importance.py
│   │   ├── transferability.py
│   │   └── clinical_validation.py
│   └── utils/
│       ├── logging.py
│       ├── visualization.py
│       └── statistical_tests.py
├── configs/
│   ├── model_configs.yaml
│   ├── training_configs.yaml
│   └── evaluation_configs.yaml
├── experiments/
│   ├── baseline_comparison.py
│   ├── ablation_studies.py
│   └── biological_validation.py
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── results_analysis.ipynb
│   └── figure_generation.ipynb
├── tests/
│   ├── test_models.py
│   ├── test_data_processing.py
│   └── test_evaluation.py
├── requirements.txt
├── environment.yml
└── README.md
```

### 6.2 Experiment Tracking

```python
import mlflow
import wandb
from typing import Dict, Any

class ExperimentLogger:
    def __init__(self, experiment_name: str, use_mlflow: bool = True, use_wandb: bool = True):
        self.experiment_name = experiment_name
        
        if use_mlflow:
            mlflow.set_experiment(experiment_name)
        
        if use_wandb:
            wandb.init(project="hierarchical-meta-learning", name=experiment_name)
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to tracking systems"""
        if hasattr(self, 'mlflow'):
            mlflow.log_params(params)
        
        if hasattr(self, 'wandb'):
            wandb.config.update(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics during training/evaluation"""
        if hasattr(self, 'mlflow'):
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        
        if hasattr(self, 'wandb'):
            wandb.log(metrics, step=step)
    
    def log_model(self, model, model_name: str):
        """Save model artifacts"""
        model_path = f"models/{model_name}"
        torch.save(model.state_dict(), f"{model_path}.pth")
        
        if hasattr(self, 'mlflow'):
            mlflow.pytorch.log_model(model, model_name)
        
        if hasattr(self, 'wandb'):
            wandb.save(f"{model_path}.pth")
```

### 6.3 Statistical Significance Testing

```python
def statistical_significance_testing(results_dict, baseline_results, alpha=0.05):
    """
    Perform statistical significance testing for model comparisons
    """
    from scipy import stats
    import numpy as np
    
    significance_results = {}
    
    for method_name, method_results in results_dict.items():
        if method_name == 'baseline':
            continue
            
        # Paired t-test for cross-validation scores
        t_stat, p_value = stats.ttest_rel(method_results, baseline_results)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((np.var(method_results) + np.var(baseline_results)) / 2))
        cohens_d = (np.mean(method_results) - np.mean(baseline_results)) / pooled_std
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(method_results, baseline_results)
        
        significance_results[method_name] = {
            'paired_t_test': {'t_statistic': t_stat, 'p_value': p_value},
            'cohens_d': cohens_d,
            'wilcoxon_test': {'statistic': wilcoxon_stat, 'p_value': wilcoxon_p},
            'significant': p_value < alpha,
            'effect_size_interpretation': interpret_cohens_d(cohens_d)
        }
    
    return significance_results

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def multiple_comparison_correction(p_values, method='bonferroni'):
    """Apply multiple comparison correction"""
    from statsmodels.stats.multitest import multipletests
    
    _, corrected_p_values, _, _ = multipletests(p_values, method=method)
    return corrected_p_values
```

### 6.4 Computational Requirements Estimation

```python
def estimate_computational_requirements():
    """
    Estimate computational requirements for the full pipeline
    """
    
    # Model parameters
    pathway_encoder_params = 32 * 64 + 64 * 128 + 128 * 64 + 64 * 32  # ~20K params
    hierarchical_classifiers_params = 32 * (9 + 4 + 36)  # ~1.6K params
    attention_params = 32 * 32 * 4 * 3  # ~12K params
    total_params = pathway_encoder_params + hierarchical_classifiers_params + attention_params
    
    # Training estimates
    samples_per_cancer_type = 500  # Conservative estimate
    total_samples = 36 * samples_per_cancer_type  # 18K samples
    
    # Memory requirements
    batch_size = 8  # Meta-batch size
    k_shot = 5
    n_query = 15
    n_way = 5
    
    memory_per_task = (k_shot + n_query) * n_way * 32 * 4  # bytes (float32)
    memory_per_batch = memory_per_task * batch_size
    model_memory = total_params * 4 * 2  # parameters + gradients
    
    estimated_memory_gb = (memory_per_batch + model_memory) / (1024**3)
    
    # Training time estimates
    inner_steps = 5
    meta_epochs = 100
    tasks_per_epoch = 1000
    
    estimated_forward_passes = meta_epochs * tasks_per_epoch * (inner_steps + 1) * batch_size
    
    return {
        'total_parameters': total_params,
        'estimated_memory_gb': estimated_memory_gb,
        'estimated_training_hours': estimated_forward_passes / 10000,  # Rough estimate
        'recommended_hardware': {
            'gpu_memory': '16GB minimum, 32GB recommended',
            'cpu_cores': '8+ cores',
            'ram': '32GB minimum'
        }
    }
```

This comprehensive methodology provides a rigorous foundation for your hierarchical meta-learning research with clear technical specifications, implementation details, and validation strategies. The framework is designed to meet NeurIPS standards while ensuring biological validity and reproducibility.