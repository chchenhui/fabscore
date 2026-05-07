import argparse
import json
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_to_jsonl(data, file_path):
    """Saves a list of dictionaries to a JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logging.info(f"Successfully saved {len(data)} records to {file_path}")

def process_jbb(output_dir: Path, seed: int):
    """
    Loads and preprocesses the JailbreakBench dataset.
    - Splits into calibration (20%) and test (80%) sets.
    - Creates a 'benign-but-hard' subset from the test set.
    """
    logging.info("Processing JailbreakBench dataset...")
    try:
        dataset = load_dataset("JailbreakBench/JBB-Behaviors", name="behaviors")
    except Exception as e:
        logging.error(f"Failed to load JailbreakBench dataset: {e}")
        return

    # Process harmful and benign subsets separately
    harmful_df = dataset['harmful'].to_pandas()
    harmful_df['label'] = 1  # harmful
    harmful_df['prompt'] = harmful_df['Goal']  # Use 'Goal' as prompt
    
    benign_df = dataset['benign'].to_pandas()  
    benign_df['label'] = 0  # benign
    benign_df['prompt'] = benign_df['Goal']  # Use 'Goal' as prompt
    
    # Combine datasets
    df = pd.concat([harmful_df, benign_df], ignore_index=True)
    
    logging.info(f"JBB DataFrame columns: {df.columns.tolist()}")
    logging.info(f"JBB DataFrame shape: {df.shape}, harmful: {sum(df['label']==1)}, benign: {sum(df['label']==0)}")
    
    # Create prompt IDs and keep only required columns
    df['prompt_id'] = [f"jbb_{i}" for i in range(len(df))]
    
    # Ensure required columns are present
    df = df[['prompt_id', 'prompt', 'label']]

    # Stratified split for calibration and test sets
    train_df, test_df = train_test_split(
        df, test_size=0.8, random_state=seed, stratify=df['label']
    )

    save_to_jsonl(train_df.to_dict('records'), output_dir / "jbb_calibration.jsonl")
    save_to_jsonl(test_df.to_dict('records'), output_dir / "jbb_test.jsonl")

    # Create 'benign-but-hard' subset from the test set
    # Heuristic: Benign prompts in the top 25th percentile of prompt length
    benign_test_df = test_df[test_df['label'] == 0].copy()
    benign_test_df['prompt_length'] = benign_test_df['prompt'].str.len()
    hardness_threshold = benign_test_df['prompt_length'].quantile(0.75)
    
    benign_hard_df = benign_test_df[benign_test_df['prompt_length'] >= hardness_threshold]
    benign_hard_df = benign_hard_df.drop(columns=['prompt_length'])
    
    save_to_jsonl(benign_hard_df.to_dict('records'), output_dir / "jbb_benign_hard_test.jsonl")
    logging.info(f"Created 'benign-but-hard' set with {len(benign_hard_df)} samples.")

def process_harmbench(output_dir: Path):
    """
    Loads and preprocesses the HarmBench dataset for 'contextual' behaviors.
    """
    logging.info("Processing HarmBench dataset...")
    try:
        # Load contextual config directly as specified in the plan
        dataset = load_dataset("walledai/HarmBench", name="contextual")
    except Exception as e:
        logging.error(f"Failed to load HarmBench dataset: {e}")
        return
        
    if isinstance(dataset, dict):
        df = pd.concat([ds.to_pandas() for ds in dataset.values()]).reset_index(drop=True)
    else:
        df = dataset.to_pandas()
    
    logging.info(f"HarmBench DataFrame columns: {df.columns.tolist()}")
    logging.info(f"HarmBench DataFrame shape: {df.shape}")
    
    # Map labels: harmful -> 1, benign -> 0 (assuming harmful prompts are labeled as 1)
    # We'll need to inspect the actual labels to be sure
    if 'label' in df.columns:
        logging.info(f"Unique labels in HarmBench: {df['label'].unique()}")
        df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)  # Keep as is if already 0/1
    else:
        # If no label column, assume all are harmful since it's HarmBench
        df['label'] = 1
        logging.info("No label column found, setting all samples as harmful (label=1)")
    
    # For contextual HarmBench, we need to combine prompt + context for complete prompts
    if 'context' in df.columns and 'prompt' in df.columns:
        # Combine prompt and context to create complete, actionable prompts (prompt first, then context)
        df['full_prompt'] = df.apply(lambda row: 
            f"{row['prompt']}\n\n{row['context']}" if pd.notna(row['context']) and row['context'].strip() 
            else row['prompt'], axis=1)
        df = df.rename(columns={'full_prompt': 'prompt'})
        logging.info("Combined prompt and context for contextual behaviors (prompt first, then context)")
    else:
        prompt_col = 'prompt' if 'prompt' in df.columns else 'goal' if 'goal' in df.columns else df.columns[1]
        df = df.rename(columns={prompt_col: 'prompt'})
    
    df['prompt_id'] = [f"harmbench_{i}" for i in range(len(df))]
    
    df = df[['prompt_id', 'prompt', 'label']]
    
    save_to_jsonl(df.to_dict('records'), output_dir / "harmbench_contextual_test.jsonl")

def process_harmbench_separated(output_dir: Path):
    """
    Loads HarmBench dataset preserving original structure (prompt, context, category).
    Adds only our required columns: prompt_id and label.
    For refined H2 twin generation that rephrases prompts while keeping context intact.
    """
    logging.info("Processing HarmBench dataset with original structure preserved...")
    try:
        dataset = load_dataset("walledai/HarmBench", name="contextual")
    except Exception as e:
        logging.error(f"Failed to load HarmBench dataset: {e}")
        return
        
    if isinstance(dataset, dict):
        df = pd.concat([ds.to_pandas() for ds in dataset.values()]).reset_index(drop=True)
    else:
        df = dataset.to_pandas()
    
    logging.info(f"HarmBench DataFrame columns: {df.columns.tolist()}")
    logging.info(f"HarmBench DataFrame shape: {df.shape}")
    
    # Add our required columns while preserving original structure
    df['prompt_id'] = [f"harmbench_{i}" for i in range(len(df))]
    df['label'] = 1  # All HarmBench samples are harmful
    
    # Keep original columns + our additions
    df = df[['prompt_id', 'prompt', 'context', 'category', 'label']]
    
    save_to_jsonl(df.to_dict('records'), output_dir / "harmbench_contextual_separated.jsonl")
    logging.info(f"Saved HarmBench with original structure preserved: {len(df)} samples")

def process_wildguard(output_dir: Path):
    """
    Loads and preprocesses the WildGuard test dataset using alternative source.
    """
    logging.info("Processing WildGuard test dataset...")
    try:
        # Use alternative dataset that's accessible
        dataset = load_dataset("walledai/WildGuardTest")
    except Exception as e:
        logging.error(f"Failed to load WildGuard dataset: {e}")
        return

    if isinstance(dataset, dict):
        df = pd.concat([ds.to_pandas() for ds in dataset.values()]).reset_index(drop=True)
    else:
        df = dataset.to_pandas()
    
    logging.info(f"WildGuard DataFrame columns: {df.columns.tolist()}")
    logging.info(f"WildGuard DataFrame shape: {df.shape}")
    
    # Inspect the label format
    if 'label' in df.columns:
        logging.info(f"Unique labels in WildGuard: {df['label'].unique()}")
        # Convert string labels to binary: harmful=1, unharmful=0
        df['label'] = df['label'].apply(lambda x: 1 if x == 'harmful' else 0)
        logging.info(f"Converted to binary labels: {df['label'].value_counts().to_dict()}")
    
    # Use prompt column (already exists)
    df['prompt_id'] = [f"wildguard_{i}" for i in range(len(df))]
    
    # Keep only required columns
    df = df[['prompt_id', 'prompt', 'label']]
    
    save_to_jsonl(df.to_dict('records'), output_dir / "wildguard_test.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Load and preprocess datasets for the research project.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        required=True, 
        choices=["jbb", "harmbench", "harmbench_separated", "wildguard", "all"],
        help="The dataset to process."
    )
    parser.add_argument(
        "--output_dir", 
        type=Path, 
        default=Path("idea_14_workspace/data/processed"),
        help="The directory to save the processed data files."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for data splitting."
    )
    args = parser.parse_args()

    if args.dataset == "jbb" or args.dataset == "all":
        process_jbb(args.output_dir, args.seed)
    
    if args.dataset == "harmbench" or args.dataset == "all":
        process_harmbench(args.output_dir)
    
    if args.dataset == "harmbench_separated":
        process_harmbench_separated(args.output_dir)
        
    if args.dataset == "wildguard" or args.dataset == "all":
        process_wildguard(args.output_dir)

if __name__ == "__main__":
    main()
