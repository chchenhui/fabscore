#!/usr/bin/env python3
import torch


class NoiseInjectionHook:
    """
    Hook class to inject Gaussian noise into model activations.

    Implements fuzzing by sampling noise vectors from N(0, I_{d_model})
    and adding them (scaled by noise_magnitude) to activations at each token position.
    """

    def __init__(
        self,
        noise_magnitude: float,
        layer_idx: int,
        seed: int = 42,
    ):
        self.noise_magnitude = noise_magnitude
        self.layer_idx = layer_idx
        self.seed = seed
        self.hook_handle = None

    def make_noise_hook(self):
        """
        Returns a hook that adds Gaussian noise to hidden states.

        Samples noise from N(0, I_{d_model}) and scales by noise_magnitude,
        then adds to activations over each token position.
        """

        def hook(module, input, output):
            # Set seed for reproducible noise
            torch.manual_seed(self.seed)

            # Handle different output formats
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Sample noise from N(0, I_{d_model}) and scale by magnitude
            # This follows: noise ~ N(0, σ²I) where σ = noise_magnitude
            noise = self.noise_magnitude * torch.randn_like(hidden_states)

            # Add noise to hidden states (intervening on activations over each token)
            noisy_hidden_states = hidden_states + noise

            # Return in the same format as input
            if isinstance(output, tuple):
                return (noisy_hidden_states,) + output[1:]
            else:
                return noisy_hidden_states

        return hook

    def register_hook(self, model):
        """Register the hook on the specified layer."""
        target_layer = model.model.layers[self.layer_idx]
        self.hook_handle = target_layer.register_forward_hook(self.make_noise_hook())

        print(
            f"Registered noise hook on layer {self.layer_idx} with magnitude {self.noise_magnitude}"
        )

    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
