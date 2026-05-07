import json
from typing import List, Dict, Any
from datasets import Dataset
from datasets import load_dataset_builder


def format_as_chat(
    system_prompt: str, user_content: str, assistant_content: str
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def validate_chat_format(messages: List[Dict[str, str]]) -> bool:
    if not isinstance(messages, list) or len(messages) == 0:
        return False

    for message in messages:
        if not isinstance(message, dict):
            return False
        if "role" not in message or "content" not in message:
            return False
        if message["role"] not in ["system", "user", "assistant"]:
            return False
        if (
            not isinstance(message["content"], str)
            or len(message["content"].strip()) == 0
        ):
            return False

    return True


def save_dataset_sample(dataset: Dataset, output_path: str, num_samples: int = 3):
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Dataset Sample\n")
            f.write("=" * 50 + "\n\n")

            sample_size = min(num_samples, len(dataset))
            for i in range(sample_size):
                example = dataset[i]
                f.write(f"Example {i+1}:\n")
                f.write("-" * 30 + "\n")

                if "messages" in example:
                    for j, message in enumerate(example["messages"]):
                        f.write(f"Message {j+1} ({message['role']}):\n")
                        f.write(f"{message['content']}\n\n")
                else:
                    f.write(f"{json.dumps(example, indent=2, ensure_ascii=False)}\n")

                f.write("\n" + "=" * 50 + "\n\n")

        print(f"Saved sample to {output_path}")
    except Exception as e:
        print(f"Failed to save sample: {e}")


def clean_text(text: str) -> str:
    """Clean text content"""
    if not isinstance(text, str):
        text = str(text)

    # Remove excessive whitespace
    text = " ".join(text.split())

    # Remove control characters except newline and tab
    text = "".join(char for char in text if ord(char) >= 32 or char in "\n\t")

    return text.strip()


def truncate_text(text: str, max_length: int = 2048) -> str:
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    last_period = truncated.rfind(".")
    last_question = truncated.rfind("?")
    last_exclamation = truncated.rfind("!")

    sentence_end = max(last_period, last_question, last_exclamation)

    if sentence_end > max_length * 0.8:
        return text[: sentence_end + 1]
    else:
        return text[:max_length] + "..."


def validate_example_content(
    user_content: str, assistant_content: str, min_length: int = 10
) -> bool:
    if not user_content or not assistant_content:
        return False

    if (
        len(user_content.strip()) < min_length
        or len(assistant_content.strip()) < min_length
    ):
        return False

    # Check for placeholder or empty content
    empty_indicators = ["", "null", "none", "n/a", "todo", "placeholder"]
    if (
        user_content.lower().strip() in empty_indicators
        or assistant_content.lower().strip() in empty_indicators
    ):
        return False

    return True


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    try:
        builder = load_dataset_builder(dataset_name)
        info = {
            "dataset_name": dataset_name,
            "description": builder.info.description,
            "features": str(builder.info.features),
            "homepage": builder.info.homepage,
            "citation": builder.info.citation,
        }
        return info
    except Exception as e:
        print(f"Could not get info for {dataset_name}: {e}")
        return {"dataset_name": dataset_name, "error": str(e)}
