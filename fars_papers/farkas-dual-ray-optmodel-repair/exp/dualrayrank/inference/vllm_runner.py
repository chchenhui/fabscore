"""vLLM offline inference wrapper for batched LP generation.

Uses vLLM's offline LLM class with Qwen2.5-7B-Instruct.
Provides batched generation and .lp content extraction from raw outputs.
"""

import re
from typing import Optional

from vllm import LLM, SamplingParams


DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_SAMPLING_PARAMS = SamplingParams(
    temperature=0.0,
    max_tokens=4096,
    top_p=1.0,
)


def create_llm(
    model: str = DEFAULT_MODEL,
    tensor_parallel_size: int = 1,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.90,
) -> LLM:
    return LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )


def generate_batch(
    llm: LLM,
    prompts: list[str],
    sampling_params: Optional[SamplingParams] = None,
) -> list[str]:
    if sampling_params is None:
        sampling_params = DEFAULT_SAMPLING_PARAMS
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text for out in outputs]


def extract_lp_content(raw_output: str) -> str:
    """Extract .lp content from LLM output, handling markdown fences and extraneous text."""
    fence_pattern = re.compile(
        r"```(?:lp|LP|plaintext|text)?\s*\n(.*?)```",
        re.DOTALL,
    )
    match = fence_pattern.search(raw_output)
    if match:
        return match.group(1).strip()

    lines = raw_output.strip().splitlines()
    lp_start = None
    lp_end = None

    sense_re = re.compile(r"^\s*(Minimize|Maximize|Min|Max)\b", re.IGNORECASE)
    end_re = re.compile(r"^\s*End\s*$", re.IGNORECASE)

    for i, line in enumerate(lines):
        if lp_start is None and sense_re.match(line):
            lp_start = i
        if end_re.match(line):
            lp_end = i
            break

    if lp_start is not None:
        if lp_end is not None:
            return "\n".join(lines[lp_start : lp_end + 1]).strip()
        return "\n".join(lines[lp_start:]).strip()

    return raw_output.strip()
