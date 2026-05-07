"""
BFCL prompt formatter for LLaDA-8B-Instruct diffusion inference.
Constructs prompts using the official BFCL default system prompt template
(prompting mode, python return format, no tool_call tags, JSON func docs).
Applies LLaDA's chat template via tokenizer.apply_chat_template.
"""
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bfcl_nonlive_300.json"

SYSTEM_PROMPT_TEMPLATE = (
    "You are an expert in composing functions. "
    "You are given a question and a set of possible functions. "
    "Based on the question, you will need to make one or more function/tool calls to achieve the purpose.\n"
    "If none of the functions can be used, point it out. "
    "If the given question lacks the parameters required by the function, also point it out.\n"
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]\n"
    "You SHOULD NOT include any other text in the response.\n\n"
    "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
    "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
    "Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.\n\n"
    "Here is a list of functions in JSON format that you can invoke.\n{functions}\n"
)


def build_messages(example: dict) -> list[dict]:
    functions_json = json.dumps(example["functions"], indent=4)
    system_content = SYSTEM_PROMPT_TEMPLATE.format(functions=functions_json)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": example["question"]},
    ]


def format_all_prompts(tokenizer, examples: list[dict]) -> list[dict]:
    formatted = []
    for ex in examples:
        messages = build_messages(ex)
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages.pop(0)
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        formatted.append({
            "id": ex["id"],
            "category": ex["category"],
            "prompt_text": prompt_text,
        })
    return formatted


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct")

    with open(DATA_PATH) as f:
        examples = json.load(f)

    formatted = format_all_prompts(tokenizer, examples)
    print(f"Formatted {len(formatted)} prompts")
    print(f"\n--- Example prompt (first) ---")
    print(formatted[0]["prompt_text"][:1000])
    print("...")

    out_path = Path(__file__).resolve().parents[1] / "data" / "bfcl_nonlive_prompts.json"
    with open(out_path, "w") as f:
        json.dump(formatted, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
