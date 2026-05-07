
import argparse
import json
import logging
from pathlib import Path
import modal
import yaml
import os
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "openai", "requests", "pyyaml", "numpy", "scikit-learn", 
    "sentence-transformers", "torch", "bert-score", "python-Levenshtein"
]).add_local_python_source("src").add_local_dir("configs", "/configs").add_local_dir("data", "/data")

# Persistent storage volume for research outputs
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h1-response-generation")

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("openrouter-secret")],
    volumes={"/research_storage": volume}
)
def generate_h1_responses(test_mode=False):
    """Generate H1 responses using consistent config settings
    
    Args:
        test_mode: If True, only process first 10 samples for testing
    """
    from src.core.response_generator_openrouter import OpenRouterResponseGenerator
    import yaml
    import json
    import logging
    import os
    from tqdm import tqdm
    import time
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration for consistent parameters
    with open('/configs/project_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get H1 configuration (model + decoding)
    h1_config = config['hypotheses']['h1']
    model_name = h1_config['model_test']
    decoding_config = h1_config['decoding']
    
    logging.info("="*80)
    logging.info("H1 RESPONSE GENERATION CONFIGURATION VERIFICATION")
    logging.info("="*80)
    logging.info(f"🤖 Model: {model_name}")
    logging.info(f"   - Expected: meta-llama/Llama-4-Scout-17B-16E-Instruct")
    logging.info(f"   - Match: {'✅' if model_name == 'meta-llama/Llama-4-Scout-17B-16E-Instruct' else '❌'}")
    logging.info(f"📊 Decoding parameters (matching hyperparameter tuning):")
    logging.info(f"   - N (responses): {decoding_config['N']} (expected: 5)")
    logging.info(f"   - Temperature: {decoding_config['temperature']} (expected: 0.7)")
    logging.info(f"   - Top-p: {decoding_config['top_p']} (expected: 0.95)")
    logging.info(f"   - Max tokens: {decoding_config['max_new_tokens']} (expected: 1024)")
    logging.info(f"   - Seed: {config['reproducibility']['global_seed']} (expected: 42)")
    logging.info("="*80)
    
    # Set up paths - data is mounted directly, outputs go to persistent storage
    # Data files are mounted at /data via .add_local_dir("data", "/data") in image setup
    train_file = "/data/processed/jbb_train.jsonl"
    val_file = "/data/processed/jbb_validation.jsonl"
    
    # Clear filename indicating model, dataset size, and parameters matching hyperparameter tuning
    if test_mode:
        output_file = "/research_storage/outputs/h1/TEST_llama4scout_10samples_N5_temp0.7_top0.95_tokens1024_responses.jsonl"
    else:
        output_file = "/research_storage/outputs/h1/llama4scout_120val_N5_temp0.7_top0.95_tokens1024_responses.jsonl"
    
    # Create output directory
    os.makedirs("/research_storage/outputs/h1", exist_ok=True)
    
    # Get API key from Modal secret
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OpenRouter API key not found in Modal secret")
    
    generator = OpenRouterResponseGenerator(api_key)
    
    # Load same expanded validation set as hyperparameter tuning (train + validation)
    train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    validation_data = []
    with open(val_file, 'r') as f:
        for line in f:
            validation_data.append(json.loads(line))
    
    # Combine train + validation for direct comparison with hyperparameter tuning
    expanded_validation = train_data + validation_data
    
    # Apply test mode if requested
    if test_mode:
        expanded_validation = expanded_validation[:10]  # Take first 10 samples only
        logging.info("🧪 TEST MODE ACTIVATED: Processing first 10 samples only")
    
    logging.info("="*80)
    logging.info("DATASET VERIFICATION (Matching Hyperparameter Tuning)")
    logging.info("="*80)
    if test_mode:
        logging.info(f"🧪 TEST MODE: Using first 10 samples for verification")
        logging.info(f"  - Test subset: {len(expanded_validation)} samples")
    else:
        logging.info(f"📊 Using SAME 120 samples as hyperparameter tuning:")
        logging.info(f"  - JBB Train samples: {len(train_data)} (expected: 80)")
        logging.info(f"  - JBB Validation samples: {len(validation_data)} (expected: 40)")  
        logging.info(f"  - Total samples: {len(expanded_validation)} (expected: 120)")
    
    # Verify label distribution
    harmful_count = sum(1 for d in expanded_validation if d['label'] == 1)
    benign_count = sum(1 for d in expanded_validation if d['label'] == 0)
    logging.info(f"  - Harmful prompts: {harmful_count} (expected: 60)")
    logging.info(f"  - Benign prompts: {benign_count} (expected: 60)")
    logging.info(f"  - Label balance: {'✅ Balanced' if harmful_count == benign_count == 60 else '❌ Imbalanced'}")
    logging.info("="*80)
    
    # Track timing for test mode
    start_time = time.time() if test_mode else None
    response_times = [] if test_mode else None
    
    with open(output_file, 'w') as f_out:
        for i, data in enumerate(tqdm(expanded_validation, desc="Generating H1 Responses")):
            prompt = data['prompt']
            prompt_id = data['prompt_id']
            
            # Time tracking for test mode
            prompt_start = time.time() if test_mode else None
            
            responses = generator.generate_responses(
                prompt=prompt,
                model_name=model_name,
                n=decoding_config['N'],
                temperature=decoding_config['temperature'],
                top_p=decoding_config['top_p'],
                max_new_tokens=decoding_config['max_new_tokens']
            )
            
            # Track timing and log samples in test mode
            if test_mode:
                prompt_time = time.time() - prompt_start
                response_times.append(prompt_time)
                
                if i < 3:  # Show first 3 samples for quality check
                    logging.info(f"\n📝 Sample {i+1} - Response preview:")
                    logging.info(f"   First 200 chars: {responses[0][:200]}...")
                    logging.info(f"   Response lengths: {[len(r) for r in responses]}")
                    logging.info(f"   Generation time: {prompt_time:.2f}s")
            
            output_record = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "label": data['label'],  # Preserve label like hyperparameter tuning
                "responses": responses
            }
            f_out.write(json.dumps(output_record) + '\n')
    
    # Commit volume changes
    volume.commit()
    
    logging.info("="*80)
    if test_mode:
        logging.info("H1 TEST RUN COMPLETE - TIMING ANALYSIS")
    else:
        logging.info("H1 RESPONSE GENERATION COMPLETE")
    logging.info("="*80)
    logging.info(f"✅ Generated {decoding_config['N']} responses for each of {len(expanded_validation)} prompts")
    logging.info(f"📁 Output file: {output_file}")
    
    # Show timing analysis for test mode
    if test_mode and response_times:
        total_time = time.time() - start_time
        avg_time = sum(response_times) / len(response_times)
        logging.info(f"⏱️ Timing Statistics:")
        logging.info(f"   - Total time: {total_time:.2f}s")
        logging.info(f"   - Avg per prompt: {avg_time:.2f}s")
        logging.info(f"   - Estimated for 120 samples: {(avg_time * 120)/60:.1f} minutes")
        logging.info(f"   - Estimated for 120 samples: {(avg_time * 120)/3600:.2f} hours")
        logging.info("="*80)
        logging.info("🎯 RECOMMENDATION:")
        if avg_time < 60:
            logging.info("   ✅ Performance looks good! Safe to run full 120 samples")
        else:
            logging.info(f"   ⚠️ Slow ({avg_time:.1f}s/prompt). Consider optimizations.")
    else:
        logging.info(f"🔧 Parameters used:")
        logging.info(f"   - Model: Llama-4-Scout-17B")
        logging.info(f"   - N=5, temp=0.7, top_p=0.95, max_tokens=1024")
        logging.info(f"   - Same 120 samples as hyperparameter tuning")
    logging.info("="*80)
    
    return len(expanded_validation)

@app.local_entrypoint()
def main():
    # Check for test mode via environment variable or create separate entry point
    test_mode = os.environ.get('H1_TEST_MODE', 'false').lower() == 'true'
    
    if test_mode:
        logging.info(f"🧪 Starting H1 TEST RUN with 10 samples")
        num_generated = generate_h1_responses.remote(test_mode=True)
        logging.info(f"✅ Test complete - generated responses for {num_generated} samples")
    else:
        logging.info(f"🚀 Starting FULL H1 response generation (120 samples)")
        num_generated = generate_h1_responses.remote(test_mode=False)
        logging.info(f"✅ Generated responses for {num_generated} samples")

# Entry point handled by Modal
