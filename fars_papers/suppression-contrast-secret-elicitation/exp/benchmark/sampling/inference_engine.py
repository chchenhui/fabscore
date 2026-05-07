#!/usr/bin/env python3

import warnings
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

# Disable warnings from transformers
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class InferenceEngine:
    """Unified inference engine that handles all generation strategies."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_batch(
        self,
        formatted_prompts: List[str],
        num_responses_per_prompt: int = 1,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        noise_hook=None,
        quiet: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Generate responses for multiple prompts using a unified approach.

        Args:
            formatted_prompts: List of prompts already formatted with chat template
            num_responses_per_prompt: Number of responses to generate per prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (0.0 = greedy, >0 = sampling)
            batch_size: Total batch size (must be multiple of num_responses_per_prompt)
            eos_token_id: EOS token ID for generation
            noise_hook: Optional noise injection hook
            quiet: If True, suppress progress messages

        Returns:
            Dictionary mapping each formatted prompt to list of generated responses
        """
        # Automatically determine sampling based on temperature
        do_sample = temperature > 0.0

        if not quiet:
            print(f"🚀 Starting inference for {len(formatted_prompts)} prompts...")
            print(f"   • Responses per prompt: {num_responses_per_prompt}")
            print(f"   • Max new tokens: {max_new_tokens}")
            print(f"   • Temperature: {temperature} ({'sampling' if do_sample else 'greedy'})")
            if batch_size:
                print(f"   • Batch size: {batch_size}")
            if noise_hook:
                print(f"   • Fuzzing enabled (layer {noise_hook.layer_idx}, magnitude {noise_hook.noise_magnitude})")

        # Register intervention hooks if provided
        if noise_hook:
            noise_hook.register_hook(self.model)

        try:
            # Calculate batch processing parameters
            prompts_per_batch = None
            if batch_size is not None:
                if batch_size <= 0:
                    raise ValueError("batch_size must be positive")
                if batch_size % num_responses_per_prompt != 0:
                    raise ValueError(f"batch_size ({batch_size}) must be multiple of num_responses_per_prompt ({num_responses_per_prompt})")
                prompts_per_batch = batch_size // num_responses_per_prompt

            # Generate responses
            if prompts_per_batch is None or prompts_per_batch >= len(formatted_prompts):
                # Process all prompts in one batch
                results = self._generate_batch_internal(
                    formatted_prompts=formatted_prompts,
                    num_responses_per_prompt=num_responses_per_prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    eos_token_id=eos_token_id,
                )
            else:
                # Process in smaller batches
                results = {}
                total_batches = (len(formatted_prompts) + prompts_per_batch - 1) // prompts_per_batch

                iterator = (
                    tqdm(range(total_batches), desc="Processing batches")
                )

                for batch_idx in iterator:
                    start_idx = batch_idx * prompts_per_batch
                    end_idx = min(start_idx + prompts_per_batch, len(formatted_prompts))
                    batch_prompts = formatted_prompts[start_idx:end_idx]

                    batch_results = self._generate_batch_internal(
                        formatted_prompts=batch_prompts,
                        num_responses_per_prompt=num_responses_per_prompt,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature,
                        eos_token_id=eos_token_id,
                    )

                    results.update(batch_results)

            total_responses = sum(len(responses) for responses in results.values())
            if not quiet:
                print(f"✅ Generated {total_responses} total responses")

            return results

        finally:
            # Always cleanup hooks
            if noise_hook:
                noise_hook.remove_hook()

    def _generate_batch_internal(
        self,
        formatted_prompts: List[str],
        num_responses_per_prompt: int,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        eos_token_id: Optional[int],
    ) -> Dict[str, List[str]]:
        """Internal method to generate responses for a batch of prompts."""

        # Tokenize all prompts
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=False,
            padding="longest",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate responses
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_responses_per_prompt,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id,
            )

        # Extract responses for each prompt
        results = {}
        input_lengths = [len(inputs['input_ids'][i]) for i in range(len(formatted_prompts))]

        for prompt_idx, formatted_prompt in enumerate(formatted_prompts):
            prompt_responses = []
            input_length = input_lengths[prompt_idx]

            # Get responses for this prompt
            start_seq = prompt_idx * num_responses_per_prompt
            end_seq = start_seq + num_responses_per_prompt

            for seq_idx in range(start_seq, end_seq):
                if seq_idx < len(outputs.sequences):
                    response_ids = outputs.sequences[seq_idx][input_length:]
                    response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    prompt_responses.append(response.strip())

            results[formatted_prompt] = prompt_responses

        return results


class PrefillInferenceEngine(InferenceEngine):
    """Specialized inference engine for prefill strategies."""

    def generate_prefill_batch(
        self,
        formatted_prompts: List[str],
        prefills: List[str],
        num_responses_per_prompt: int = 1,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        noise_hook=None,
        quiet: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Generate responses for prefill prompts where the model continues from prefills.

        This method handles the special case where responses should include the original
        prefill text plus the generated continuation.
        """
        if not quiet:
            print(f"🚀 Starting prefill inference for {len(formatted_prompts)} prompts...")

        # Generate using standard method
        raw_results = self.generate_batch(
            formatted_prompts=formatted_prompts,
            num_responses_per_prompt=num_responses_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            eos_token_id=eos_token_id,
            noise_hook=noise_hook,
            quiet=quiet,
        )

        # Reconstruct full responses (prefill + continuation)
        results = {}
        for i, (formatted_prompt, responses) in enumerate(raw_results.items()):
            prefill = prefills[i % len(prefills)]  # Handle case where there are more prompts than prefills
            full_responses = []

            for response in responses:
                # The response from generation is just the continuation
                # We need to combine it with the original prefill
                full_response = prefill + response
                full_responses.append(full_response)

            results[formatted_prompt] = full_responses

        return results


class UserPersonaInferenceEngine(InferenceEngine):
    """Specialized inference engine for user persona strategies."""

    def generate_user_persona_batch(
        self,
        formatted_prompts: List[str],
        persona_user_prefills: List[str],
        num_responses_per_prompt: int = 1,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        batch_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        noise_hook=None,
        quiet: bool = False,
    ) -> Dict[str, List[str]]:
        """
        Generate responses for user persona prompts where model continues from user prefills.

        Similar to prefill but specifically for the 3-turn user persona case.
        """
        if not quiet:
            print(f"🚀 Starting user persona inference for {len(formatted_prompts)} prompts...")

        # Generate using standard method
        raw_results = self.generate_batch(
            formatted_prompts=formatted_prompts,
            num_responses_per_prompt=num_responses_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            eos_token_id=eos_token_id,
            noise_hook=noise_hook,
            quiet=quiet,
        )

        # Reconstruct full responses (user prefill + continuation)
        results = {}
        for i, (formatted_prompt, responses) in enumerate(raw_results.items()):
            user_prefill = persona_user_prefills[i % len(persona_user_prefills)]
            full_responses = []

            for response in responses:
                # The response from generation is just the continuation
                # We need to combine it with the original user prefill
                full_response = user_prefill + response
                full_responses.append(full_response)

            results[formatted_prompt] = full_responses

        return results


def get_inference_engine(strategy: str, model, tokenizer):
    """Get the appropriate inference engine for a given strategy."""
    if strategy == "prefill":
        return PrefillInferenceEngine(model, tokenizer)
    elif strategy == "user_persona":
        return UserPersonaInferenceEngine(model, tokenizer)
    else:
        return InferenceEngine(model, tokenizer)
