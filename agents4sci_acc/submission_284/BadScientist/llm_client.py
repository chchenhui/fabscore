import os
import json
import backoff
import openai
from openai import AzureOpenAI


def _load_local_azure_defaults():
    """Populate AZURE_OPENAI_* from local_review_config.py if present and not set."""
    paths = [
        os.path.join(os.path.dirname(__file__), "local_review_config.py"),
        os.path.join(os.getcwd(), "local_review_config.py"),
    ]
    for p in paths:
        if os.path.exists(p):
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location("local_review_config", p)
                mod = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(mod)
                defaults = {
                    "AZURE_OPENAI_API_KEY": getattr(mod, "AZURE_OPENAI_API_KEY", None),
                    "AZURE_OPENAI_ENDPOINT": getattr(
                        mod, "AZURE_OPENAI_ENDPOINT", None
                    ),
                    "AZURE_OPENAI_API_VERSION": getattr(
                        mod, "AZURE_OPENAI_API_VERSION", None
                    ),
                    "AZURE_OPENAI_MODEL": getattr(mod, "AZURE_OPENAI_MODEL", None),
                }
                for k, v in defaults.items():
                    if v and not os.environ.get(k):
                        os.environ[k] = v
                break
            except Exception:
                # ignore config load failures
                pass


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def chat(
    messages,
    *,
    temperature=1,
    n=1,
    response_format=None,
    api_key: str | None = None,
    endpoint: str | None = None,
    api_version: str | None = None,
    model: str | None = None,
):

    if not api_key or not endpoint:
        _load_local_azure_defaults()
        api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION")
        model = model or os.environ.get("AZURE_OPENAI_MODEL")

    if not api_key or not endpoint or not model:
        raise RuntimeError("AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_MODEL must be set in local_review_config.py")

    client = AzureOpenAI(
        api_key=api_key, azure_endpoint=endpoint, api_version=api_version
    )
    # Build args while omitting params not supported by some Azure models
    req_kwargs = {
        "model": model,
        "messages": messages,
        "n": n,
        "seed": 0,
        "stop": None,
    }
    req_kwargs["temperature"] = temperature
    if response_format:
        req_kwargs["response_format"] = response_format

    res = client.chat.completions.create(**req_kwargs)
    contents = [choice.message.content for choice in res.choices]
    return contents if n > 1 else contents[0]


def extract_json_block(text: str):
    """Extract a JSON object or array from free-form text.
    Tries, in order: fenced ```json blocks, generic ``` code fences,
    first top-level object {...}, then first top-level array [...].
    Returns a parsed Python object (dict or list) or None.
    """
    import re

    # 1) ```json ... ```
    m = re.search(r"```json\s*(.*?)```", text, flags=re.S | re.I)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except Exception:
            pass

    # 2) any fenced block
    m = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        block = m.group(1).strip()
        try:
            return json.loads(block)
        except Exception:
            pass

    # Helper to parse a balanced region starting at pos with open/close chars
    def parse_balanced(s: str, pos: int, open_ch: str, close_ch: str):
        depth = 0
        end = None
        for i, ch in enumerate(s[pos:], pos):
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            return None
        try:
            return json.loads(s[pos:end])
        except Exception:
            return None

    # 3) choose earliest of object or array
    obj_pos = text.find("{")
    arr_pos = text.find("[")
    first_pos = None
    first_kind = None
    for pos, kind in sorted(
        [(p, "obj") for p in [obj_pos] if p != -1]
        + [(p, "arr") for p in [arr_pos] if p != -1]
    ):
        first_pos = pos
        first_kind = kind
        break
    if first_pos is not None:
        if first_kind == "obj":
            parsed = parse_balanced(text, first_pos, "{", "}")
            if parsed is not None:
                return parsed
        else:
            parsed = parse_balanced(text, first_pos, "[", "]")
            if parsed is not None:
                return parsed

    # 4) fallback: try to extract an array specifically if not already
    if arr_pos != -1:
        parsed = parse_balanced(text, arr_pos, "[", "]")
        if parsed is not None:
            return parsed

    return None
