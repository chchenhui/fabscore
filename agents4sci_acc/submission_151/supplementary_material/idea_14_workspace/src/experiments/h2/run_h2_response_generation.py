
#!/usr/bin/env python3
"""
H2 Response Generation - Generate responses for HarmBench twins dataset on Modal.
Updated to work with the new twins approach and Modal infrastructure.

DATASET INPUT FORMAT:
- prompt_id: Unique identifier (e.g., h2_harmful_061, h2_benign_033)
- prompt: The actual harmful/benign request
- context: Detailed contextual information (technical details, examples, etc.)
- label: 0=benign, 1=harmful
- original_harmbench_id: Original HarmBench ID for traceability
- Other metadata: category, structure_features, generation_metadata, etc.

OUTPUT FILE FORMAT (each line is a JSON record with):
{
    # Core identification
    "prompt_id": "h2_harmful_061",
    "prompt": "Modify this blind SQL injection code...",  # Request only
    "context": "import requests\nimport time...",  # Context only  
    "full_prompt": "Modify this...\n\nimport requests...",  # Combined for generation
    "label": 1,  # 0=benign, 1=harmful
    
    # Generated responses (list of N=5 responses)
    "responses": ["Response 1", "Response 2", ...],
    
    # Generation metadata for analysis
    "generation_metadata": {
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "api_model_name": "qwen/qwen-2.5-7b-instruct", 
        "n_requested": 5,
        "n_received": 5,
        "empty_responses": 0,
        "processing_time_seconds": 12.34,
        "generation_params": {...},
        "prompt_lengths": {"request_chars": 78, "context_chars": 1856, "full_prompt_chars": 1936}
    },
    
    # Original dataset metadata for analysis
    "original_harmbench_id": "harmbench_61",
    "harmbench_parent": "...",
    "category": "...", 
    "dataset_source": "harmbench_contextual",
    "structure_features": {...},
    "twin_generation_metadata": {...}
}

This format preserves all information needed for:
1. Semantic entropy computation (responses)
2. Baseline metric computation (responses)
3. Analysis by label (harmful vs benign)
4. Traceability to original HarmBench
5. Twin relationship analysis
6. Performance metrics and debugging
"""

import modal
import json
import logging
import os
from pathlib import Path
from typing import Dict, List
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Modal setup - consistent with existing infrastructure
image = modal.Image.debian_slim(python_version="3.11").pip_install([
    "openai", 
    "requests", 
    "pyyaml", 
    "numpy",
    "tqdm"
]).add_local_python_source("src").add_local_dir("configs", "/configs").add_local_dir("data", "/data")

# Use same persistent storage volume
volume = modal.Volume.from_name("alignment-research-storage", create_if_missing=True)

app = modal.App("h2-response-generation")

@app.function(
    image=image,
    timeout=7200,  # 2 hours for both models
    secrets=[modal.Secret.from_name("openrouter-secret")],
    volumes={"/research_storage": volume}
)
def generate_h2_responses(model_name: str):
    """Generate responses for H2 twins dataset using specified model."""
    from src.core.response_generator_openrouter import OpenRouterResponseGenerator
    import json
    import logging
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 100)
    logger.info(f"H2 RESPONSE GENERATION - {model_name} - FULL DATASET")
    logger.info("=" * 100)
    
    # Load config for consistency
    try:
        with open('/configs/project_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✅ Loaded project configuration")
    except Exception as e:
        logger.warning(f"⚠️ Could not load config: {e}, using defaults")
        config = {}
    
    # Define paths using config structure
    data_processed_path = config.get('paths', {}).get('data_processed', 'data/processed/')
    outputs_path = config.get('paths', {}).get('outputs', 'outputs/')
    
    # Use Modal storage paths
    input_path = Path('/research_storage') / data_processed_path / 'h2_harmbench_twins_test.jsonl'
    output_dir = Path('/research_storage') / outputs_path / 'h2'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model-specific output filename - PRODUCTION VERSION
    model_short = model_name.split('/')[-1].lower()
    output_path = output_dir / f"{model_short}_h2_responses.jsonl"
    
    logger.info(f"📁 Input: {input_path}")
    logger.info(f"📁 Output: {output_path}")
    
    # Check if twins dataset exists
    if not input_path.exists():
        raise FileNotFoundError(f"H2 twins dataset not found at {input_path}. "
                               "Run generate_h2_twins_modal.py first.")
    
    # CHECKPOINTING: Check for already processed samples
    already_processed = set()
    if output_path.exists():
        logger.info("🔄 Found existing output file - checking for completed samples")
        try:
            with open(output_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)
                        already_processed.add(data['prompt_id'])
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
            
            logger.info(f"✅ Found {len(already_processed)} already processed samples")
            if already_processed:
                logger.info(f"   Will resume from where we left off")
        except Exception as e:
            logger.warning(f"⚠️ Could not read existing output file: {e}")
            already_processed = set()
    else:
        logger.info("📝 Starting fresh generation - no existing output file")
    
    # Load twins dataset - FULL DATASET FOR PRODUCTION
    twins_data = []
    try:
        with open(input_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    twins_data.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON error on line {line_num}: {e}")
                    
        logger.info(f"✅ Loaded {len(twins_data)} samples from twins dataset")
    except Exception as e:
        logger.error(f"❌ Failed to load twins dataset: {e}")
        raise
    
    # Filter out already processed samples
    if already_processed:
        original_count = len(twins_data)
        twins_data = [item for item in twins_data if item['prompt_id'] not in already_processed]
        filtered_count = original_count - len(twins_data)
        logger.info(f"🔄 Filtered out {filtered_count} already processed samples")
        logger.info(f"📊 Remaining to process: {len(twins_data)} samples")
        
        if len(twins_data) == 0:
            logger.info("🎉 All samples already processed! Nothing to do.")
            return {
                'success': True,
                'message': 'All samples already processed',
                'total_prompts': original_count,
                'already_processed': len(already_processed),
                'newly_processed': 0
            }
    
    # Analyze remaining dataset
    harmful_count = sum(1 for item in twins_data if item['label'] == 1)
    benign_count = sum(1 for item in twins_data if item['label'] == 0)
    
    logger.info(f"📊 Dataset composition: {harmful_count} harmful + {benign_count} benign = {len(twins_data)} total")
    
    # Validate model is in H2 config
    h2_models = config.get('hypotheses', {}).get('h2', {}).get('models', [])
    if model_name not in h2_models:
        logger.warning(f"⚠️ Model {model_name} not in H2 config. Available models: {h2_models}")
    
    # Get OpenRouter mapping
    openrouter_mapping = config.get('openrouter', {}).get('model_mappings', {})
    api_model_name = openrouter_mapping.get(model_name, model_name)
    
    logger.info(f"🔄 Model mapping: {model_name} → {api_model_name}")
    
    # Setup generation parameters (from config or defaults)
    generation_params = {
        'n_responses': config.get('hypotheses', {}).get('h2', {}).get('decoding', {}).get('N', 5),
        'temperature': config.get('hypotheses', {}).get('h2', {}).get('decoding', {}).get('temperature', 0.7),
        'top_p': config.get('hypotheses', {}).get('h2', {}).get('decoding', {}).get('top_p', 0.95),
        'max_new_tokens': config.get('hypotheses', {}).get('h2', {}).get('decoding', {}).get('max_new_tokens', 1024)
    }
    
    logger.info(f"⚙️ Generation parameters:")
    for param, value in generation_params.items():
        logger.info(f"   {param}: {value}")
    
    # Get OpenRouter API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")
    
    # Initialize response generator
    generator = OpenRouterResponseGenerator(api_key)
    logger.info(f"✅ Initialized OpenRouter generator for {model_name}")
    
    # Generate responses with STATIC CHECKPOINTING (every 20 responses)
    logger.info("🚀 Starting response generation with static checkpointing...")
    
    successful_generations = 0
    failed_generations = []
    checkpoint_batch = []  # Collect responses in batches
    checkpoint_size = 5    # Write every 5 responses (testing)
    
    # Detailed metrics tracking
    metrics = {
        'harmful_successful': 0,
        'benign_successful': 0,
        'harmful_failed': 0,
        'benign_failed': 0,
        'total_responses_generated': 0,
        'response_lengths': [],
        'empty_responses': 0,
        'processing_times': []
    }
    
    def write_checkpoint(batch, append_mode=True):
        """Write a batch of responses to the output file."""
        if not batch:
            return
        
        mode = 'a' if append_mode else 'w'
        try:
            with open(output_path, mode, encoding='utf-8') as f:
                for record in batch:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            logger.info(f"✅ Checkpoint saved: {len(batch)} responses written to file")
        except Exception as e:
            logger.error(f"❌ Failed to write checkpoint: {e}")
            raise
    
    try:
        for idx, item in enumerate(twins_data):
            prompt_id = item['prompt_id']
            prompt = item['prompt']
            context = item.get('context', '')
            label = item['label']
            
            # Combine prompt and context properly
            if context:
                full_prompt = f"{prompt}\n\n{context}"
            else:
                full_prompt = prompt
                logger.warning(f"⚠️ No context found for {prompt_id}")
            
            logger.info(f"\n[{idx+1:3d}/{len(twins_data)}] Generating responses for {prompt_id}")
            logger.info(f"   Label: {'harmful' if label == 1 else 'benign'}")
            logger.info(f"   Request length: {len(prompt)} chars")
            logger.info(f"   Context length: {len(context)} chars")
            logger.info(f"   Full prompt length: {len(full_prompt)} chars")
            logger.info(f"   Request preview: {prompt[:80]}...")
            if context:
                logger.info(f"   Context preview: {context[:80]}...")
            
            import time
            start_time = time.time()
            
            # RATE LIMITING: Add delay between samples to avoid API timeouts
            if idx > 0:  # Don't delay before first sample
                logger.info(f"   ⏱️ Rate limiting: waiting 2 seconds...")
                time.sleep(2)
            
            try:
                responses = generator.generate_responses(
                    prompt=full_prompt,  # Use combined prompt + context
                    model_name=api_model_name,  # Use mapped API model name
                    n=generation_params['n_responses'],
                    temperature=generation_params['temperature'],
                    top_p=generation_params['top_p'],
                    max_new_tokens=generation_params['max_new_tokens']
                )
                
                # Verify responses
                valid_responses = [r for r in responses if r and r.strip()]
                empty_count = len([r for r in responses if not r or not r.strip()])
                
                processing_time = time.time() - start_time
                metrics['processing_times'].append(processing_time)
                
                if len(valid_responses) != generation_params['n_responses']:
                    logger.warning(f"   ⚠️ Got {len(valid_responses)}/{generation_params['n_responses']} valid responses")
                
                # Update detailed metrics
                metrics['total_responses_generated'] += len(valid_responses)
                metrics['empty_responses'] += empty_count
                if label == 1:  # harmful
                    metrics['harmful_successful'] += 1
                else:  # benign
                    metrics['benign_successful'] += 1
                
                # Create comprehensive output record with all metadata needed for analysis
                output_record = {
                    # Core identification fields
                    'prompt_id': prompt_id,
                    'prompt': prompt,  # Original request part only
                    'context': context,  # Context part separately
                    'full_prompt': full_prompt,  # Combined prompt used for generation
                    'label': label,  # 0=benign, 1=harmful
                    
                    # Generated responses
                    'responses': valid_responses,
                    
                    # Response generation metadata for analysis
                    'generation_metadata': {
                        'model_name': model_name,  # Original model name from config
                        'api_model_name': api_model_name,  # Actual API model used
                        'n_requested': generation_params['n_responses'],
                        'n_received': len(valid_responses),
                        'empty_responses': empty_count,
                        'processing_time_seconds': processing_time,
                        'generation_params': generation_params,
                        'prompt_lengths': {
                            'request_chars': len(prompt),
                            'context_chars': len(context),
                            'full_prompt_chars': len(full_prompt)
                        }
                    },
                    
                    # Original dataset metadata preserved for analysis
                    'original_harmbench_id': item.get('original_harmbench_id'),
                    'harmbench_parent': item.get('harmbench_parent'),
                    'category': item.get('category'),
                    'dataset_source': item.get('dataset_source'),
                    'structure_features': item.get('structure_features'),
                    'twin_generation_metadata': item.get('generation_metadata')
                }
                
                # STATIC CHECKPOINTING: Add to batch for periodic writing
                checkpoint_batch.append(output_record)
                successful_generations += 1
                logger.info(f"   ✅ Successfully generated {len(valid_responses)} responses in {processing_time:.2f}s")
                logger.info(f"   📋 Added to checkpoint batch ({len(checkpoint_batch)}/{checkpoint_size})")
                
                # Write checkpoint when batch is full
                if len(checkpoint_batch) >= checkpoint_size:
                    logger.info(f"💾 Writing checkpoint: {len(checkpoint_batch)} responses...")
                    write_checkpoint(checkpoint_batch)
                    logger.info(f"✅ Checkpoint {len(checkpoint_batch)} responses saved to file")
                    checkpoint_batch.clear()  # Clear batch after writing
                
                # Log response lengths for monitoring
                response_lengths = [len(r) for r in valid_responses]
                if response_lengths:
                    avg_length = sum(response_lengths) / len(response_lengths)
                    metrics['response_lengths'].extend(response_lengths)
                    logger.info(f"   📏 Response lengths: avg={avg_length:.0f}, min={min(response_lengths)}, max={max(response_lengths)}")
                    logger.info(f"   ⚡ Processing time: {processing_time:.2f}s")
                
            except Exception as e:
                processing_time = time.time() - start_time
                metrics['processing_times'].append(processing_time)
                
                error_msg = f"Generation failed: {str(e)}"
                logger.error(f"   ❌ {error_msg} (after {processing_time:.2f}s)")
                
                # Update failure metrics
                if label == 1:  # harmful
                    metrics['harmful_failed'] += 1
                else:  # benign
                    metrics['benign_failed'] += 1
                
                failed_generations.append({
                    'prompt_id': prompt_id,
                    'label': label,
                    'error': error_msg,
                    'processing_time_seconds': processing_time,
                    'prompt_preview': prompt[:100] + "..." if len(prompt) > 100 else prompt
                })
            
            # Progress logging every 20 prompts
            if (idx + 1) % 20 == 0:
                success_rate = successful_generations / (idx + 1) * 100
                harmful_success_rate = metrics['harmful_successful'] / max(1, metrics['harmful_successful'] + metrics['harmful_failed']) * 100
                benign_success_rate = metrics['benign_successful'] / max(1, metrics['benign_successful'] + metrics['benign_failed']) * 100
                avg_processing_time = sum(metrics['processing_times']) / len(metrics['processing_times']) if metrics['processing_times'] else 0
                
                logger.info(f"\n📊 PROGRESS UPDATE: {idx+1}/{len(twins_data)} processed")
                logger.info(f"   Overall success rate: {success_rate:.1f}% ({successful_generations} successful)")
                logger.info(f"   Harmful success rate: {harmful_success_rate:.1f}% ({metrics['harmful_successful']}/{metrics['harmful_successful'] + metrics['harmful_failed']})")
                logger.info(f"   Benign success rate: {benign_success_rate:.1f}% ({metrics['benign_successful']}/{metrics['benign_successful'] + metrics['benign_failed']})")
                logger.info(f"   Total responses generated: {metrics['total_responses_generated']}")
                logger.info(f"   Empty responses: {metrics['empty_responses']}")
                logger.info(f"   Avg processing time: {avg_processing_time:.2f}s")
                logger.info(f"   Failed generations: {len(failed_generations)}")
    
    finally:
        # Write any remaining responses in the checkpoint batch
        if checkpoint_batch:
            logger.info(f"💾 Writing final checkpoint: {len(checkpoint_batch)} remaining responses...")
            try:
                write_checkpoint(checkpoint_batch)
                logger.info(f"✅ Final checkpoint saved: {len(checkpoint_batch)} responses")
            except Exception as e:
                logger.error(f"❌ Failed to write final checkpoint: {e}")
    
    # Calculate final comprehensive metrics
    total_success_rate = successful_generations / len(twins_data) * 100
    harmful_success_rate = metrics['harmful_successful'] / max(1, metrics['harmful_successful'] + metrics['harmful_failed']) * 100
    benign_success_rate = metrics['benign_successful'] / max(1, metrics['benign_successful'] + metrics['benign_failed']) * 100
    avg_processing_time = sum(metrics['processing_times']) / len(metrics['processing_times']) if metrics['processing_times'] else 0
    total_processing_time = sum(metrics['processing_times'])
    
    # Response length statistics
    if metrics['response_lengths']:
        avg_response_length = sum(metrics['response_lengths']) / len(metrics['response_lengths'])
        min_response_length = min(metrics['response_lengths'])
        max_response_length = max(metrics['response_lengths'])
    else:
        avg_response_length = min_response_length = max_response_length = 0
    
    # Final results
    logger.info("\n" + "=" * 100)
    logger.info("H2 RESPONSE GENERATION COMPLETE")
    logger.info("=" * 100)
    
    logger.info(f"📊 COMPREHENSIVE FINAL STATISTICS:")
    logger.info(f"   Total prompts processed: {len(twins_data)}")
    logger.info(f"   Overall success rate: {total_success_rate:.1f}% ({successful_generations} successful)")
    logger.info(f"   Harmful prompts: {harmful_count} | Success: {metrics['harmful_successful']} | Failed: {metrics['harmful_failed']} | Rate: {harmful_success_rate:.1f}%")
    logger.info(f"   Benign prompts: {benign_count} | Success: {metrics['benign_successful']} | Failed: {metrics['benign_failed']} | Rate: {benign_success_rate:.1f}%")
    logger.info(f"   Total responses generated: {metrics['total_responses_generated']}")
    logger.info(f"   Empty responses encountered: {metrics['empty_responses']}")
    logger.info(f"   Response length stats: avg={avg_response_length:.0f}, min={min_response_length}, max={max_response_length}")
    logger.info(f"   Processing time: total={total_processing_time:.1f}s, avg={avg_processing_time:.2f}s per prompt")
    logger.info(f"   Output file samples: {successful_generations}")
    
    # Responses saved via static checkpointing during generation
    logger.info(f"✅ All responses saved via checkpointing to {output_path}")
    logger.info(f"📊 Final output contains {successful_generations} completed samples")
    
    # Generate detailed log
    log_path = output_dir / f"{model_short}_h2_generation_log.md"
    try:
        with open(log_path, 'w') as f:
            import datetime
            f.write(f"# H2 Response Generation Log - {model_name}\n\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Model:** {model_name} → {api_model_name}\n")
            f.write(f"- **Input dataset:** {input_path}\n")
            f.write(f"- **Total prompts:** {len(twins_data)}\n")
            f.write(f"- **Overall success rate:** {total_success_rate:.1f}% ({successful_generations} successful)\n")
            f.write(f"- **Output file:** {output_path}\n\n")
            
            f.write("## Comprehensive Metrics\n\n")
            f.write(f"### Success Rates by Label\n")
            f.write(f"- **Harmful prompts:** {harmful_count} total | {metrics['harmful_successful']} success | {metrics['harmful_failed']} failed | **{harmful_success_rate:.1f}% success rate**\n")
            f.write(f"- **Benign prompts:** {benign_count} total | {metrics['benign_successful']} success | {metrics['benign_failed']} failed | **{benign_success_rate:.1f}% success rate**\n\n")
            
            f.write(f"### Response Generation Metrics\n")
            f.write(f"- **Total responses generated:** {metrics['total_responses_generated']}\n")
            f.write(f"- **Empty responses encountered:** {metrics['empty_responses']}\n")
            f.write(f"- **Average response length:** {avg_response_length:.0f} characters\n")
            f.write(f"- **Response length range:** {min_response_length} - {max_response_length} characters\n\n")
            
            f.write(f"### Processing Performance\n")
            f.write(f"- **Total processing time:** {total_processing_time:.1f} seconds\n")
            f.write(f"- **Average time per prompt:** {avg_processing_time:.2f} seconds\n")
            f.write(f"- **Processing rate:** {len(twins_data) / total_processing_time * 60:.1f} prompts/minute\n\n")
            
            f.write("## Generation Parameters\n\n")
            for param, value in generation_params.items():
                f.write(f"- **{param}:** {value}\n")
            f.write(f"\n")
            
            if failed_generations:
                f.write(f"## Failed Generations ({len(failed_generations)})\n\n")
                f.write("| Prompt ID | Label | Error | Processing Time | Preview |\n")
                f.write("|-----------|-------|-------|-----------------|----------|\n")
                for fail in failed_generations[:20]:  # First 20
                    label_str = "harmful" if fail['label'] == 1 else "benign"
                    f.write(f"| {fail['prompt_id']} | {label_str} | {fail['error'][:40]}... | {fail['processing_time_seconds']:.2f}s | {fail['prompt_preview'][:40]}... |\n")
            
        logger.info(f"✅ Generation log saved to {log_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save generation log: {e}")
    
    # Commit volume changes
    try:
        volume.commit()
        logger.info("✅ Volume changes committed")
    except Exception as e:
        logger.error(f"❌ Volume commit failed: {e}")
    
    return {
        'success': True,
        'model_name': model_name,
        'api_model_name': api_model_name,
        'total_prompts': len(twins_data),
        'successful_generations': successful_generations,
        'success_rate': total_success_rate,
        'harmful_success_rate': harmful_success_rate,
        'benign_success_rate': benign_success_rate,
        'total_responses_generated': metrics['total_responses_generated'],
        'empty_responses': metrics['empty_responses'],
        'avg_response_length': avg_response_length,
        'total_processing_time': total_processing_time,
        'avg_processing_time': avg_processing_time,
        'output_path': str(output_path),
        'output_samples': successful_generations,
        'log_path': str(log_path)
    }

@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Main entrypoint for H2 response generation."""
    
    print("=" * 100)
    print("H2 RESPONSE GENERATION - FULL DATASET WITH STATIC CHECKPOINTING")
    print("=" * 100)
    print(f"Model: {model_name}")
    print("This will:")
    print("1. Load full H2 twins dataset (162 samples)")
    print("2. Resume from any existing progress (checkpointing)")
    print("3. Generate N=5 responses per prompt using OpenRouter")
    print("4. Save responses in batches of 5 with rate limiting (testing)")
    print("5. Create comprehensive logs and metadata")
    print("=" * 100)
    
    try:
        result = generate_h2_responses.remote(model_name)
        
        print("\n" + "=" * 100)
        print("✅ H2 RESPONSE GENERATION COMPLETE!")
        print("=" * 100)
        print(f"Model: {result['model_name']}")
        print(f"Success rate: {result['success_rate']:.1f}%")
        print(f"Generated responses for: {result['successful_generations']}/{result['total_prompts']} prompts")
        print(f"Output: {result['output_path']}")
        print("=" * 100)
        
        return result
        
    except Exception as e:
        print(f"\n❌ H2 RESPONSE GENERATION FAILED: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen-2.5-7b-instruct"
    main(model)
