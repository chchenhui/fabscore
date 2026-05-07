"""
Utilities for SAE feature exploration.
"""
import json
import os
from datetime import datetime
from typing import List, Optional, Tuple

import nnsight
import requests
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer


class SparseAutoEncoder(torch.nn.Module):
    """Sparse AutoEncoder implementation."""

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.device = device
        self.encoder_linear = torch.nn.Linear(d_in, d_hidden)
        self.decoder_linear = torch.nn.Linear(d_hidden, d_in)
        self.dtype = dtype
        self.to(self.device, self.dtype)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of data using a linear, followed by a ReLU."""
        return torch.nn.functional.relu(self.encoder_linear(x))

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a batch of data using a linear."""
        return self.decoder_linear(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """SAE forward pass. Returns the reconstruction and the encoded features."""
        f = self.encode(x)
        return self.decode(f), f


class ObservableLanguageModelGemma2:
    """Wrapper for language model with observation capabilities."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype
        self.device = device
        self._original_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )
        self._original_tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-2-9b-it",
            trust_remote_code=True,
        )

        self._model = nnsight.LanguageModel(
            self._original_model,
            tokenizer=self._original_tokenizer,
            device=device,
            torch_dtype=dtype,
        )

        # Force model download
        input_tokens = self._original_tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}]
        )
        with self._model.trace(input_tokens):
            pass

        self.tokenizer = self._original_tokenizer
        self.d_model = self._attempt_to_infer_hidden_layer_dimensions()
        self.safe_mode = False

    def _attempt_to_infer_hidden_layer_dimensions(self):
        """Infer hidden layer dimensions from model config."""
        config = self._original_model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        raise Exception("Could not infer hidden layer dimensions from model config")

    def _find_module(self, hook_point: str):
        """Find module by hook point string."""
        submodules = hook_point.split(".")
        module = self._model
        while submodules:
            module = getattr(module, submodules.pop(0))
        return module

    def forward(
        self,
        inputs: torch.Tensor,
        cache_activations_at: Optional[List[str]] = None,
        interventions=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass with optional activation caching and interventions."""
        cache = {}

        with self._model.trace(inputs, scan=self.safe_mode, validate=self.safe_mode):
            # Apply interventions
            if interventions:
                for hook_site, intervention in interventions.items():
                    if intervention is None:
                        continue
                    module = self._find_module(hook_site)
                    intervened_acts = intervention(module.output[0])
                    module.output = (intervened_acts,)

            # Cache activations
            if cache_activations_at is not None:
                for hook_point in cache_activations_at:
                    module = self._find_module(hook_point)
                    cache[hook_point] = module.output.save()

            logits = self._model.output[0].squeeze(1).save()
            kv_cache = self._model.output.past_key_values.save()

        return (
            logits.detach(),
            kv_cache,
            {k: v[0].detach() for k, v in cache.items()},
        )

class ObservableLanguageModel:
    """Wrapper for language model with observation capabilities."""

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.dtype = dtype
        self.device = device
        self._original_model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=dtype,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self._original_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.3-70B-Instruct",
            trust_remote_code=True,
        )

        self._model = nnsight.LanguageModel(
            self._original_model,
            tokenizer=self._original_tokenizer,
            device=device,
            torch_dtype=dtype,
        )

        # Force model download
        input_tokens = self._original_tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}]
        )
        with self._model.trace(input_tokens):
            pass

        self.tokenizer = self._original_tokenizer
        self.d_model = self._attempt_to_infer_hidden_layer_dimensions()
        self.safe_mode = False

    def _attempt_to_infer_hidden_layer_dimensions(self):
        """Infer hidden layer dimensions from model config."""
        config = self._original_model.config
        if hasattr(config, "hidden_size"):
            return int(config.hidden_size)
        raise Exception("Could not infer hidden layer dimensions from model config")

    def _find_module(self, hook_point: str):
        """Find module by hook point string."""
        submodules = hook_point.split(".")
        module = self._model
        while submodules:
            module = getattr(module, submodules.pop(0))
        return module

    def forward(
        self,
        inputs: torch.Tensor,
        cache_activations_at: Optional[List[str]] = None,
        interventions=None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor], dict[str, torch.Tensor]]:
        """Forward pass with optional activation caching and interventions."""
        cache = {}

        with self._model.trace(inputs, scan=self.safe_mode, validate=self.safe_mode):
            # Apply interventions
            if interventions:
                for hook_site, intervention in interventions.items():
                    if intervention is None:
                        continue
                    module = self._find_module(hook_site)
                    intervened_acts = intervention(module.output[0])
                    module.output = (intervened_acts,)

            # Cache activations
            if cache_activations_at is not None:
                for hook_point in cache_activations_at:
                    module = self._find_module(hook_point)
                    cache[hook_point] = module.output.save()

            logits = self._model.output[0].squeeze(1).save()
            kv_cache = self._model.output.past_key_values.save()

        return (
            logits.detach(),
            kv_cache,
            {k: v[0].detach() for k, v in cache.items()},
        )


def load_sae(
    sae_name: str,
    d_model: int,
    expansion_factor: int,
    device: str = "cuda",
) -> SparseAutoEncoder:
    """Load SAE from HuggingFace Hub."""
    device_obj = torch.device(device)

    # Download SAE weights
    file_path = hf_hub_download(
        repo_id=f"Goodfire/{sae_name}", filename=f"{sae_name}.pt", repo_type="model"
    )

    # Initialize and load SAE
    sae = SparseAutoEncoder(
        d_model,
        d_model * expansion_factor,
        device_obj,
    )
    sae_dict = torch.load(file_path, weights_only=True, map_location=device_obj)
    sae.load_state_dict(sae_dict)

    return sae


def get_top_features(features: torch.Tensor, top_k: int = 10) -> List[Tuple[int, float]]:
    """Get top-k activating features with their values."""
    top_values, top_indices = torch.topk(features, top_k)
    return [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)]


def string_to_base64(text: str) -> str:
    """Convert a string to base64 encoding."""
    import base64

    # Encode string to bytes, then to base64
    encoded_bytes = base64.b64encode(text.encode("utf-8"))
    # Convert bytes back to string
    return encoded_bytes.decode("utf-8")


def base64_to_string(encoded_text: str) -> str:
    """Convert base64 encoded string back to original string."""
    import base64

    # Decode base64 to bytes, then to string
    decoded_bytes = base64.b64decode(encoded_text.encode("utf-8"))
    return decoded_bytes.decode("utf-8")

def get_cache_key(feature_index: int, layer: int = 32, width_k: int = 131) -> str:
    """Generate a cache key for a feature."""
    return f"{layer}-{width_k}k-{feature_index}"


def fetch_sae_feature_description(
    feature_index: int, layer: int = 32, width_k: int = 131
) -> str:
    """Fetch SAE feature description from Neuronpedia API with caching.

    Args:
        feature_index: The SAE feature index
        layer: The layer number (default: 32)
        width_k: The width in thousands (default: 131)

    Returns:
        Feature description string, or "No description available" if API call fails
    """
    cache = load_sae_feature_cache()
    cache_key = get_cache_key(feature_index, layer, width_k)

    # Check if description is cached
    if cache_key in cache["features"] and "description" in cache["features"][cache_key]:
        return cache["features"][cache_key]["description"]

    try:
        url = f"https://www.neuronpedia.org/api/feature/gemma-2-9b/{layer}-gemmascope-res-{width_k}k/{feature_index}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Try to extract description from the response
        description = "No description available"
        explanations = data.get("explanations", [])
        if explanations and len(explanations) > 0:
            # Get the first explanation's description
            first_explanation = explanations[0]
            if "description" in first_explanation:
                description = first_explanation["description"].strip()

        # Fallback: check if there's a direct description field
        if description == "No description available" and "description" in data:
            description = data["description"].strip()

        # Cache the result and immediately save to disk
        if cache_key not in cache["features"]:
            cache["features"][cache_key] = {}
        cache["features"][cache_key]["description"] = description
        cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()

        # Immediately save cache to disk
        save_sae_feature_cache()

        return description

    except requests.RequestException as e:
        print(f"Warning: Failed to fetch description for feature {feature_index}: {e}")
        # Cache the failure to avoid repeated API calls and immediately save
        if cache_key not in cache["features"]:
            cache["features"][cache_key] = {}
        cache["features"][cache_key]["description"] = "No description available"
        cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()
        save_sae_feature_cache()
        return "No description available"
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Unexpected response format for feature {feature_index}: {e}")
        # Cache this failure too and save immediately
        if cache_key not in cache["features"]:
            cache["features"][cache_key] = {}
        cache["features"][cache_key]["description"] = "No description available"
        cache["features"][cache_key]["last_fetched"] = datetime.now().isoformat()
        save_sae_feature_cache()
        return "No description available"


_sae_feature_cache = None
_cache_file_path = "sae_feature_cache.json"


def load_sae_feature_cache():
    """Load the SAE feature cache from disk."""
    global _sae_feature_cache
    if _sae_feature_cache is not None:
        return _sae_feature_cache

    try:
        if os.path.exists(_cache_file_path):
            with open(_cache_file_path, "r", encoding="utf-8") as f:
                _sae_feature_cache = json.load(f)
            print(
                f"Loaded SAE feature cache with {len(_sae_feature_cache.get('features', {}))} cached features"
            )
        else:
            _sae_feature_cache = {
                "cache_version": "1.0",
                "created": datetime.now().isoformat(),
                "features": {},
            }
            print("Created new SAE feature cache")
    except Exception as e:
        print(f"Warning: Failed to load SAE feature cache: {e}")
        _sae_feature_cache = {
            "cache_version": "1.0",
            "created": datetime.now().isoformat(),
            "features": {},
        }

    return _sae_feature_cache


def save_sae_feature_cache():
    """Save the SAE feature cache to disk."""
    global _sae_feature_cache
    if _sae_feature_cache is None:
        return

    try:
        _sae_feature_cache["last_updated"] = datetime.now().isoformat()
        with open(_cache_file_path, "w", encoding="utf-8") as f:
            json.dump(_sae_feature_cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save SAE feature cache: {e}")
