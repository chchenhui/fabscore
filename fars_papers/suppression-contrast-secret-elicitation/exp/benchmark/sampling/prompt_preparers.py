#!/usr/bin/env python3
"""
Prompt preparation module for different inference strategies.

This module handles all prompt preparation and formatting for the different
inference modes, separating this complexity from the main inference logic.
"""

import base64
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class PreparedPrompts:
    """Container for prepared prompts with metadata."""
    formatted_prompts: List[str]
    metadata: Dict[str, Any]
    debug_info: Dict[str, Any]


class BasePromptPreparer:
    """Base class for all prompt preparers."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def prepare(self, **kwargs) -> PreparedPrompts:
        """Prepare prompts for inference. Must be implemented by subclasses."""
        raise NotImplementedError

    def _print_debug_sample(self, formatted_prompts: List[str]):
        """Print a sample formatted prompt for debugging."""
        if not formatted_prompts:
            return

        sample_prompt = formatted_prompts[0]
        print(sample_prompt)

        # Token analysis
        sample_tokens = self.tokenizer.encode(sample_prompt, add_special_tokens=False)
        print(self.tokenizer.convert_ids_to_tokens(sample_tokens))


class StandardPromptPreparer(BasePromptPreparer):
    """Prepares standard user prompts with chat template."""

    def prepare(self, prompts: List[str]) -> PreparedPrompts:
        print(f"📝 Preparing {len(prompts)} standard prompts...")

        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            formatted_prompts.append(formatted_prompt)

        self._print_debug_sample(formatted_prompts)

        return PreparedPrompts(
            formatted_prompts=formatted_prompts,
            metadata={
                "strategy": "standard",
                "num_prompts": len(prompts),
                "original_prompts": prompts,
            },
            debug_info={
                "preparation_method": "standard_chat_template"
            }
        )


class SSCPromptPreparer(BasePromptPreparer):
    """Prepares Secret Side Constraint prompts."""

    def prepare(self, prompts: List[str], constraints: List[str]) -> PreparedPrompts:
        print(f"📝 Preparing {len(prompts)} × {len(constraints)} = {len(prompts) * len(constraints)} SSC prompts...")

        # Create all combinations
        combinations = []
        combined_prompts = []

        for prompt in prompts:
            for constraint in constraints:
                combinations.append((prompt, constraint))

                # Standard SSC: append constraint in tags
                encoded_constraint = base64.b64encode(constraint.encode('utf-8')).decode('utf-8')
                combined_prompt = f"{prompt}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"
                combined_prompts.append(combined_prompt)

        # Apply chat template
        formatted_prompts = []
        for combined_prompt in combined_prompts:
            messages = [{"role": "user", "content": combined_prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            formatted_prompts.append(formatted_prompt)

        self._print_debug_sample(formatted_prompts)

        return PreparedPrompts(
            formatted_prompts=formatted_prompts,
            metadata={
                "strategy": "ssc",
                "num_prompts": len(prompts),
                "num_constraints": len(constraints),
                "num_combinations": len(combinations),
                "combinations": combinations,
                "original_prompts": prompts,
                "constraints": constraints,
            },
            debug_info={
                "preparation_method": "ssc_with_chat_template"
            }
        )


class PrefillPromptPreparer(BasePromptPreparer):
    """Prepares prefill prompts (assistant continuation)."""

    def prepare(self, user_prompts: List[str], prefills: List[str], constraints: List[str] = None, is_ssc: bool = False) -> PreparedPrompts:
        strategy = "prefill"

        if constraints:
            total_combinations = len(user_prompts) * len(prefills) * len(constraints)
            print(f"📝 Preparing {len(user_prompts)} × {len(prefills)} × {len(constraints)} = {total_combinations} {strategy} prompts...")
        else:
            total_combinations = len(user_prompts) * len(prefills)
            print(f"📝 Preparing {len(user_prompts)} × {len(prefills)} = {total_combinations} {strategy} prompts...")

        # Create all combinations
        combinations = []
        formatted_prompts = []

        for user_prompt in user_prompts:
            for prefill in prefills:
                if is_ssc and constraints:
                    for constraint in constraints:
                        # Add SSC constraint to user prompt
                        encoded_constraint = base64.b64encode(constraint.encode('utf-8')).decode('utf-8')
                        ssc_user_prompt = f"{user_prompt}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"

                        combinations.append((user_prompt, prefill, constraint))

                        # Create messages with SSC user prompt and assistant prefill
                        messages = []
                        if ssc_user_prompt.strip():
                            messages.append({"role": "user", "content": ssc_user_prompt})
                        else:
                            messages.append({"role": "user", "content": ""})

                        messages.append({"role": "assistant", "content": prefill})

                        # Apply chat template without generation prompt
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            add_special_tokens=False,
                        )

                        # Remove end-of-turn token to allow continuation
                        if is_ssc:
                            # For SSC models (Llama), trim <|eot_id|> token
                            formatted_prompt = formatted_prompt[: -(len("<|eot_id|>"))]
                        else:
                            # Standard trimming for non-SSC models
                            formatted_prompt = formatted_prompt[: -(len("<end_of_turn>") + 1)]

                        formatted_prompts.append(formatted_prompt)
                else:
                    # Regular prefill (no SSC)
                    combinations.append((user_prompt, prefill))

                    # Create messages with user prompt and assistant prefill
                    messages = []
                    if user_prompt.strip():
                        messages.append({"role": "user", "content": user_prompt})
                    else:
                        messages.append({"role": "user", "content": ""})

                    messages.append({"role": "assistant", "content": prefill})

                    # Apply chat template without generation prompt
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        add_special_tokens=False,
                    )

                    # Remove end-of-turn token to allow continuation
                    if is_ssc:
                        # For SSC models (Llama), trim <|eot_id|> token
                        formatted_prompt = formatted_prompt[: -(len("<|eot_id|>"))]
                    else:
                        # Standard trimming for non-SSC models
                        formatted_prompt = formatted_prompt[: -(len("<end_of_turn>") + 1)]

                    formatted_prompts.append(formatted_prompt)

        self._print_debug_sample(formatted_prompts)

        return PreparedPrompts(
            formatted_prompts=formatted_prompts,
            metadata={
                "strategy": strategy,
                "num_user_prompts": len(user_prompts),
                "num_prefills": len(prefills),
                "num_combinations": len(combinations),
                "combinations": combinations,
                "user_prompts": user_prompts,
                "prefills": prefills,
                **({"constraints": constraints} if constraints else {}),
            },
            debug_info={
                "preparation_method": "prefill_continuation",
                "is_ssc": is_ssc
            }
        )


class MultiTurnJailbreakPreparer(BasePromptPreparer):
    """Prepares multi-turn jailbreak prompts with conversation history."""

    def prepare(self, prefix_messages: List[Dict[str, str]], user_prompts: List[str], constraints: List[str] = None) -> PreparedPrompts:
        is_ssc = constraints is not None
        strategy = "multi_turn_jailbreak"

        if constraints:
            total_combinations = len(user_prompts) * len(constraints)
            print(f"📝 Preparing {len(user_prompts)} × {len(constraints)} = {total_combinations} {strategy} prompts with {len(prefix_messages)} prefix messages...")
        else:
            print(f"📝 Preparing {len(user_prompts)} {strategy} prompts with {len(prefix_messages)} prefix messages...")

        formatted_prompts = []
        combinations = []

        for user_prompt in user_prompts:
            if is_ssc:
                for constraint in constraints:
                    # Add SSC constraint to user prompt
                    encoded_constraint = base64.b64encode(constraint.encode('utf-8')).decode('utf-8')
                    ssc_user_prompt = f"{user_prompt}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"

                    combinations.append((prefix_messages, user_prompt, constraint))

                    # Create conversation with prefix + SSC user prompt
                    messages = prefix_messages.copy()
                    messages.append({"role": "user", "content": ssc_user_prompt})

                    # Apply chat template
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        add_special_tokens=False,
                    )

                    formatted_prompts.append(formatted_prompt)
            else:
                combinations.append((prefix_messages, user_prompt))

                # Create conversation with prefix + new user prompt
                messages = prefix_messages.copy()
                messages.append({"role": "user", "content": user_prompt})

                # Apply chat template
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )

                formatted_prompts.append(formatted_prompt)

        # Calculate prefix statistics
        prefix_text = ""
        for msg in prefix_messages:
            if msg["role"] == "user":
                prefix_text += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                prefix_text += f"Assistant: {msg['content']}\n"

        prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)

        self._print_debug_sample(formatted_prompts)

        return PreparedPrompts(
            formatted_prompts=formatted_prompts,
            metadata={
                "strategy": strategy,
                "num_user_prompts": len(user_prompts),
                "prefix_messages": prefix_messages,
                "prefix_token_count": len(prefix_tokens),
                "combinations": combinations,
                "user_prompts": user_prompts,
                **({"constraints": constraints, "num_constraints": len(constraints)} if constraints else {}),
            },
            debug_info={
                "preparation_method": "multi_turn_with_prefix",
                "prefix_length": len(prefix_text)
            }
        )


class UserPersonaPreparer(BasePromptPreparer):
    """Prepares user persona prompts (3-turn conversations)."""

    def prepare(self, user_prompts: List[str], persona_assistant: List[str], persona_user: List[str], constraints: List[str] = None, is_ssc: bool = False) -> PreparedPrompts:
        strategy = "user_persona"

        if constraints:
            total_combinations = len(user_prompts) * len(persona_assistant) * len(persona_user) * len(constraints)
            print(f"📝 Preparing {len(user_prompts)} × {len(persona_assistant)} × {len(persona_user)} × {len(constraints)} = {total_combinations} {strategy} prompts...")
        else:
            total_combinations = len(user_prompts) * len(persona_assistant) * len(persona_user)
            print(f"📝 Preparing {len(user_prompts)} × {len(persona_assistant)} × {len(persona_user)} = {total_combinations} {strategy} prompts...")

        combinations = []
        formatted_prompts = []

        for user_prompt in user_prompts:
            for assistant_response in persona_assistant:
                for user_prefill in persona_user:
                    if is_ssc and constraints:
                        for constraint in constraints:
                            # Add SSC constraint to user prompt
                            encoded_constraint = base64.b64encode(constraint.encode('utf-8')).decode('utf-8')
                            ssc_user_prompt = f"{user_prompt}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"

                            combination = (user_prompt, assistant_response, user_prefill, constraint)
                            combinations.append(combination)

                            # Create 3-turn conversation with SSC
                            messages = [
                                {"role": "user", "content": ssc_user_prompt},
                                {"role": "assistant", "content": assistant_response},
                                {"role": "user", "content": user_prefill}
                            ]

                            # Apply chat template without generation prompt
                            formatted_prompt = self.tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=False,
                                add_special_tokens=False,
                            )

                            # Remove end-of-turn token to allow continuation
                            if is_ssc:
                                # For SSC models (Llama), trim <|eot_id|> token
                                formatted_prompt = formatted_prompt[: -(len("<|eot_id|>"))]
                            else:
                                # Standard trimming
                                formatted_prompt = formatted_prompt[: -(len("<end_of_turn>") + 1)]

                            formatted_prompts.append(formatted_prompt)
                    else:
                        # Regular user persona (no SSC)
                        combination = (user_prompt, assistant_response, user_prefill)
                        combinations.append(combination)

                        # Create 3-turn conversation
                        messages = [
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": assistant_response},
                            {"role": "user", "content": user_prefill}
                        ]

                        # Apply chat template without generation prompt
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=False,
                            add_special_tokens=False,
                        )

                        # Remove end-of-turn token to allow continuation
                        if is_ssc:
                            # For SSC models (Llama), trim <|eot_id|> token
                            formatted_prompt = formatted_prompt[: -(len("<|eot_id|>"))]
                        else:
                            # Standard trimming
                            formatted_prompt = formatted_prompt[: -(len("<end_of_turn>") + 1)]

                        formatted_prompts.append(formatted_prompt)

        self._print_debug_sample(formatted_prompts)

        return PreparedPrompts(
            formatted_prompts=formatted_prompts,
            metadata={
                "strategy": strategy,
                "num_user_prompts": len(user_prompts),
                "num_persona_assistant": len(persona_assistant),
                "num_persona_user": len(persona_user),
                "num_combinations": len(combinations),
                "combinations": combinations,
                "user_prompts": user_prompts,
                "persona_assistant": persona_assistant,
                "persona_user": persona_user,
                **({"constraints": constraints, "num_constraints": len(constraints)} if constraints else {}),
            },
            debug_info={
                "preparation_method": "three_turn_conversation",
                "is_ssc": is_ssc
            }
        )


# Convenience function to get the right preparer
def get_prompt_preparer(strategy: str, tokenizer, ssc_enabled: bool = False) -> BasePromptPreparer:
    """Get the appropriate prompt preparer for a given strategy."""
    preparers = {
        "standard": SSCPromptPreparer if ssc_enabled else StandardPromptPreparer,
        "prefill": PrefillPromptPreparer,
        "multi_turn_jailbreak": MultiTurnJailbreakPreparer,
        "user_persona": UserPersonaPreparer,
    }

    if strategy not in preparers:
        raise ValueError(f"Unknown strategy: {strategy}. Available: {list(preparers.keys())}")

    return preparers[strategy](tokenizer)
