"""
H7 Response Generation Script - SOTA Model Check (Qwen2.5-72B-Instruct)
Adapted from H1 response generation to test if SE fails on larger models
"""

import json
import logging
from pathlib import Path
import modal
import yaml
import os
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "openai", "requests", "pyyaml", "numpy", "scikit-learn", 
    "sentence-transformers", "torch", "bert-score", "python-Levenshtein", "tqdm"
]).add_local_python_source("src").add_local_dir("configs", "/configs").add_local_dir("data", "/data")

# Persistent storage volume for research outputs
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h7-response-generation")

@app.function(
    image=image,
    gpu="A100-80GB",  # Using larger GPU for 72B model
    timeout=14400,  # 4 hours (increased for larger model)
    secrets=[modal.Secret.from_name("openrouter-secret")],
    volumes={"/research_storage": volume}
)
def generate_h7_responses(model_name: str, test_mode=False):
    """Generate H7 responses using specified SOTA model
    
    Args:
        model_name: Full model name (e.g., 'Qwen/Qwen2.5-72B-Instruct')
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
    
    # Get H7 configuration
    h7_config = config['hypotheses']['h7']
    decoding_config = h7_config['decoding']
    checkpoint_size = h7_config.get('checkpoint_size', 10)
    
    logging.info("="*80)
    logging.info("H7 RESPONSE GENERATION - SOTA MODEL CHECK")
    logging.info("="*80)
    logging.info(f"🤖 Model: {model_name}")
    
    # Map model to short name for file naming
    model_short = {
        'Qwen/Qwen2.5-72B-Instruct': 'qwen-2.5-72b-instruct',
        'meta-llama/Llama-3.3-70B-Instruct': 'llama-3.3-70b-instruct'
    }.get(model_name, model_name.split('/')[-1].lower())
    logging.info(f"📊 Decoding parameters (matching H1 for consistency):")
    logging.info(f"   - N (responses): {decoding_config['N']} (expected: 5)")
    logging.info(f"   - Temperature: {decoding_config['temperature']} (expected: 0.7)")
    logging.info(f"   - Top-p: {decoding_config['top_p']} (expected: 0.95)")
    logging.info(f"   - Max tokens: {decoding_config['max_new_tokens']} (expected: 1024)")
    logging.info(f"   - Seed: {decoding_config.get('seed', config['reproducibility']['global_seed'])} (expected: 42)")
    logging.info("="*80)
    
    # Set up paths - using same 120-sample dataset as H1
    train_file = "/data/processed/jbb_train.jsonl"
    val_file = "/data/processed/jbb_validation.jsonl"
    
    # Output file with clear naming following H2 convention
    if test_mode:
        output_file = f"/research_storage/outputs/h7/{model_short}_h7_TEST_responses.jsonl"
    else:
        output_file = f"/research_storage/outputs/h7/{model_short}_h7_responses.jsonl"
    
    # Create output directory
    os.makedirs("/research_storage/outputs/h7", exist_ok=True)
    
    # Get API key from Modal secret
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        raise ValueError("OpenRouter API key not found in Modal secret")
    
    generator = OpenRouterResponseGenerator(api_key)
    
    # Load same 120-sample dataset as H1 (train + validation)
    train_data = []
    with open(train_file, 'r') as f:
        for line in f:
            train_data.append(json.loads(line))
    
    validation_data = []
    with open(val_file, 'r') as f:
        for line in f:
            validation_data.append(json.loads(line))
    
    # Combine train + validation for 120 samples total
    expanded_validation = train_data + validation_data
    
    # Apply test mode if requested
    if test_mode:
        expanded_validation = expanded_validation[:10]  # Take first 10 samples only
        logging.info("🧪 TEST MODE ACTIVATED: Processing first 10 samples only")
        logging.info(f"🧪 TEST MODE VALIDATION: Dataset reduced to {len(expanded_validation)} samples")
        if len(expanded_validation) != 10:
            logging.warning(f"⚠️ TEST MODE WARNING: Expected 10 samples, got {len(expanded_validation)}")
    
    logging.info("="*80)
    logging.info("DATASET VERIFICATION (Same as H1)")
    logging.info("="*80)
    if test_mode:
        logging.info(f"🧪 TEST MODE: Using first 10 samples for verification")
        logging.info(f"  - Test subset: {len(expanded_validation)} samples")
    else:
        logging.info(f"📊 Using SAME 120 samples as H1:")
        logging.info(f"  - JBB Train samples: {len(train_data)} (expected: 80)")
        logging.info(f"  - JBB Validation samples: {len(validation_data)} (expected: 40)")  
        logging.info(f"  - Total samples: {len(expanded_validation)} (expected: 120)")
    
    # Verify label distribution
    harmful_count = sum(1 for d in expanded_validation if d['label'] == 1)
    benign_count = sum(1 for d in expanded_validation if d['label'] == 0)
    logging.info(f"  - Harmful prompts: {harmful_count} (expected: 60 for full dataset)")
    logging.info(f"  - Benign prompts: {benign_count} (expected: 60 for full dataset)")
    logging.info(f"  - Label balance: {'✅ Balanced' if harmful_count == benign_count else '⚠️ Check balance'}")
    logging.info("="*80)
    
    # Track timing
    start_time = time.time()
    response_times = []
    
    # Check for existing output and load already processed IDs
    already_processed = set()
    if os.path.exists(output_file):
        logging.info(f"📂 Found existing output file: {output_file}")
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    existing_data = json.loads(line)
                    already_processed.add(existing_data['prompt_id'])
                except:
                    pass
        logging.info(f"🔄 Found {len(already_processed)} already processed samples")
        
    # Filter out already processed samples
    if already_processed:
        original_count = len(expanded_validation)
        expanded_validation = [item for item in expanded_validation if item['prompt_id'] not in already_processed]
        logging.info(f"📊 Remaining to process: {len(expanded_validation)} samples")
        
        if len(expanded_validation) == 0:
            logging.info("🎉 All samples already processed!")
            return output_file
    
    # Generate responses with checkpointing
    checkpoint_batch = []
    
    # Open in append mode if file exists
    mode = 'a' if already_processed else 'w'
    
    with open(output_file, mode) as f_out:
        for i, data in enumerate(tqdm(expanded_validation, desc=f"Generating H7 Responses ({model_short})")):
            prompt = data['prompt']
            prompt_id = data['prompt_id']
            label = data['label']
            
            # Log individual prompt details
            logging.info(f"\n[{i+1:3d}/{len(expanded_validation)}] Processing {prompt_id}")
            logging.info(f"   Label: {'harmful' if label == 1 else 'benign'}")
            logging.info(f"   Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"   Prompt: {prompt}")
            
            # Time tracking
            prompt_start = time.time()
            
            # Generate N non-empty responses using the SOTA model with retry logic
            responses = []
            max_retries = 10  # Maximum total attempts to get 5 non-empty responses
            attempts = 0
            
            while len(responses) < decoding_config['N'] and attempts < max_retries:
                attempts += 1
                needed = decoding_config['N'] - len(responses)
                
                try:
                    batch_responses = generator.generate_responses(
                        prompt=prompt,
                        model_name=model_name,
                        n=needed,
                        temperature=decoding_config['temperature'],
                        top_p=decoding_config['top_p'],
                        max_new_tokens=decoding_config['max_new_tokens']
                    )
                    
                    # Filter out empty responses
                    valid_responses = [r for r in batch_responses if r and r.strip()]
                    responses.extend(valid_responses)
                    
                    if len(valid_responses) < len(batch_responses):
                        empty_count = len(batch_responses) - len(valid_responses)
                        logging.warning(f"   ⚠️ Attempt {attempts}: Got {empty_count} empty responses, retrying for more...")
                    else:
                        logging.info(f"   ✅ Attempt {attempts}: Generated {len(valid_responses)} valid responses")
                        
                except Exception as e:
                    logging.error(f"   ❌ Attempt {attempts} failed: {e}")
                    continue
                    
            if len(responses) < decoding_config['N']:
                logging.warning(f"   ⚠️ Could only generate {len(responses)}/{decoding_config['N']} non-empty responses after {attempts} attempts")
            else:
                logging.info(f"   ✅ Successfully generated {len(responses)} non-empty responses")
            
            # Track timing and validate responses
            prompt_time = time.time() - prompt_start
            response_times.append(prompt_time)
            
            # Response validation and logging
            empty_responses = sum(1 for r in responses if not r or not r.strip())
            avg_length = sum(len(r) for r in responses) / len(responses) if responses else 0
            logging.info(f"   📊 Response stats: avg_length={avg_length:.0f}, empty={empty_responses}/{len(responses)}")
            logging.info(f"   ⏱️ Processing time: {prompt_time:.2f}s")
            
            # Ensure we have 5 responses even if some failed
            while len(responses) < decoding_config['N']:
                responses.append("")  # Add empty string if we couldn't get enough valid responses
                logging.warning(f"   ⚠️ Padding with empty response to reach {decoding_config['N']} total")
            
            # Save result with metadata
            result = {
                'prompt_id': prompt_id,
                'prompt': prompt,
                'label': data['label'],
                'responses': responses,
                'generation_metadata': {
                    'model_name': model_name,
                    'n_requested': decoding_config['N'],
                    'n_received': len(responses),
                    'empty_responses': sum(1 for r in responses if not r or not r.strip()),
                    'processing_time_seconds': prompt_time,
                    'generation_params': {
                        'temperature': decoding_config['temperature'],
                        'top_p': decoding_config['top_p'],
                        'max_new_tokens': decoding_config['max_new_tokens']
                    }
                }
            }
            
            # Add to checkpoint batch
            checkpoint_batch.append(result)
            
            # Write checkpoint if batch is full
            if len(checkpoint_batch) >= checkpoint_size:
                for item in checkpoint_batch:
                    f_out.write(json.dumps(item) + '\n')
                f_out.flush()  # Force write to disk
                logging.info(f"💾 Checkpoint: Wrote {len(checkpoint_batch)} responses to disk")
                checkpoint_batch = []
            
            # Progress logging every 10 samples
            if (i + 1) % 10 == 0:
                avg_time = sum(response_times) / len(response_times)
                remaining = len(expanded_validation) - (i + 1)
                eta = remaining * avg_time
                logging.info(f"Progress: {i+1}/{len(expanded_validation)} samples")
                logging.info(f"  Average time per prompt: {avg_time:.2f}s")
                logging.info(f"  Estimated time remaining: {eta/60:.1f} minutes")
        
        # Write any remaining items in checkpoint batch
        if checkpoint_batch:
            for item in checkpoint_batch:
                f_out.write(json.dumps(item) + '\n')
            f_out.flush()
            logging.info(f"💾 Final checkpoint: Wrote {len(checkpoint_batch)} responses")
    
    # Final statistics with comprehensive summary
    total_time = time.time() - start_time
    
    # Calculate comprehensive metrics
    total_responses_generated = len(expanded_validation) * decoding_config['N']
    successful_generations = len([t for t in response_times if t > 0])
    
    logging.info("="*80)
    logging.info("H7 RESPONSE GENERATION COMPLETED")
    logging.info("="*80)
    logging.info(f"🎯 Model: {model_name}")
    logging.info(f"📊 Dataset: {len(expanded_validation)} prompts ({'TEST MODE' if test_mode else 'FULL DATASET'})")
    logging.info(f"✅ Successful generations: {successful_generations}/{len(expanded_validation)} prompts")
    logging.info(f"🔢 Total responses generated: {total_responses_generated}")
    logging.info(f"⏱️ Total processing time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    logging.info(f"⏱️ Average time per prompt: {total_time/len(expanded_validation):.2f} seconds")
    logging.info(f"⏱️ Average time per response: {total_time/total_responses_generated:.2f} seconds")
    
    # Response quality metrics from final batch
    if os.path.exists(output_file):
        logging.info(f"📁 Output file: {output_file}")
        # Quick validation of output file
        with open(output_file, 'r') as f:
            lines = f.readlines()
            logging.info(f"📝 Output file contains {len(lines)} records")
    
    logging.info("="*80)
    
    # Log next steps
    model_short_for_next = model_short
    logging.info("📋 NEXT STEPS:")
    logging.info(f"1. Run scoring: modal run src/experiments/h7/run_h7_scoring.py --model {model_short_for_next}")
    logging.info(f"2. Run evaluation: modal run src/experiments/h7/run_h7_evaluation.py --model {model_short_for_next}")
    logging.info("="*80)
    
    # Volume commit for persistence
    volume.commit()
    
    return output_file

@app.local_entrypoint()
def main(model: str = "qwen-72b", test: bool = False):
    """Entry point for H7 response generation
    
    Args:
        model: Model to use ('qwen-72b' or 'llama-70b')
        test: Run in test mode (10 samples only)
    """
    # Map short names to full model names
    model_mapping = {
        'qwen-72b': 'Qwen/Qwen2.5-72B-Instruct',
        'llama-70b': 'meta-llama/Llama-3.3-70B-Instruct'
    }
    
    if model not in model_mapping:
        logging.error(f"Invalid model: {model}. Choose from: {list(model_mapping.keys())}")
        return
        
    model_name = model_mapping[model]
    
    logging.info("🚀 Starting H7 Response Generation (SOTA Model Check)")
    logging.info(f"Model: {model_name}")
    logging.info(f"Mode: {'TEST (10 samples)' if test else 'FULL (120 samples)'}")
    
    # Run the generation
    output_file = generate_h7_responses.remote(model_name=model_name, test_mode=test)
    
    logging.info(f"✅ H7 Response generation completed!")
    logging.info(f"📁 Results saved to: {output_file}")

if __name__ == "__main__":
    main()