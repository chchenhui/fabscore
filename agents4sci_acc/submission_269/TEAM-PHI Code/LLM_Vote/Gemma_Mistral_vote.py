# vote_two_agents.py  —— Gemma-2 / Mistral-7B + two prompts → four votes (clean & deterministic)

from pathlib import Path
import re
from typing import List, Tuple, Optional
from mlx_lm import load, generate

# ====== Change these to your local paths ======
GEMMA2_DIR     = "gemma-2-9b-it-mlx-q4"
MISTRAL7B_DIR  = "mistral-7b-instruct-v0.3-mlx-q4"

GUIDE_INDEPENDENT = "vote_guideline_independent.txt"
GUIDE_INFO        = "vote_guideline_info.txt"

# ====== Your results blob (LaTeX tables are OK) ======
llm_results = r"""

"""

# ====== Fixed system line ======
SYSTEM_LINE = (
    "You are an evaluation agent. Your job is to choose the single best de-identification (deid) agent."
)

# ====== Safer generation settings (deterministic / short) ======
MAX_TOKENS  = 8          # we only need one identifier like: gpt4o
TEMPERATURE = 0.0        # greedy
TOP_P       = 1.0
STOP        = ["\n", "</s>", "```"]  # best-effort; ignored if not supported by your mlx_lm version

# -------------------- Utilities --------------------

def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def try_apply_chat_template(tokenizer, system: str, user: str) -> Optional[str]:
    """
    Use the tokenizer's chat template if available (Gemma & Mistral provide this).
    Falls back to a plain 'system\\n\\nuser' prompt if apply_chat_template is missing.
    """
    apply = getattr(tokenizer, "apply_chat_template", None)
    if apply is None:
        return None
    try:
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
        # tokenize=False makes this return a str prompt
        return apply(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return None

def generate_compat(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """
    Be compatible with different mlx_lm versions.
    Try stop/temperature/top_p; on 'unexpected keyword' fallback to minimal args.
    """
    kwargs = dict(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=max_tokens)
    try:
        return generate(**kwargs, temperature=TEMPERATURE, top_p=TOP_P, stop=STOP)
    except TypeError:
        # Older versions may not support stop/temperature/top_p
        try:
            return generate(**kwargs, temperature=TEMPERATURE, top_p=TOP_P)
        except TypeError:
            return generate(**kwargs)
    except Exception as e:
        # Some builds wrap errors; try progressively simpler calls
        text = str(e)
        if ("unexpected keyword" in text) and any(k in text for k in ("temperature", "top_p", "stop")):
            try:
                return generate(**kwargs, temperature=TEMPERATURE, top_p=TOP_P)
            except Exception:
                return generate(**kwargs)
        raise

def parse_deid_ids_from_results(results_text: str) -> List[str]:
    """
    Parse valid de-id agent identifiers from the first column of your tables.
    Works with LaTeX rows like: 'gpt35       & 0.5513 & ...'
    """
    ids = set()
    # Match start-of-line tokens that look like an identifier (letters/numbers/underscore) until first whitespace or '&'
    for line in results_text.splitlines():
        line = line.strip()
        if not line or line.startswith(("\\", "%", "\hline")):
            continue
        # A row typically starts with something like 'gpt35       & ...'
        m = re.match(r"^([A-Za-z0-9_]+)\s*&", line)
        if m:
            ids.add(m.group(1))
    # Keep a stable order: sort by lowercased name
    return sorted(ids, key=lambda s: s.lower())

def postprocess_vote(raw_text: str, valid_ids: List[str]) -> str:
    """
    Reduce the model output to a *single* valid ID.
    Strategy:
      1) lower-case both sides
      2) find first exact token match among valid_ids
      3) if none, fallback to the first word that partially matches a valid id
      4) if still none, return 'UNKNOWN'
    """
    text = raw_text.strip()
    if not text:
        return "UNKNOWN"

    # First, exact token match
    tokens = re.findall(r"[A-Za-z0-9_]+", text)
    valset_lower = {v.lower(): v for v in valid_ids}  # map lower -> original casing
    for t in tokens:
        hit = valset_lower.get(t.lower())
        if hit:
            return hit

    # Second, partial match by containment (rarely needed)
    for t in tokens:
        for v in valid_ids:
            if t.lower() in v.lower() or v.lower() in t.lower():
                return v

    return "UNKNOWN"

def build_user_prompt(guideline_text: str, results_text: str) -> str:
    """
    Combine the guideline + results + a firm output constraint.
    """
    hard_constraint = (
        "\n\nReturn ONLY the identifier of the best deid agent (e.g., gpt35, gpt4o, llama70b, llama8b, lppa4k, lppa5k, mistral7b, gemma2). "
        "Do not add any extra words, punctuation, or explanations."
    )
    return f"{guideline_text}\n\n{results_text}{hard_constraint}"

def run_vote(model_dir: str, guideline_path: str, results_text: str) -> Tuple[str, str]:
    """
    Returns (raw_model_text, final_vote)
    """
    guideline = read_text(guideline_path)
    user_prompt = build_user_prompt(guideline, results_text)

    model, tokenizer = load(model_dir)

    # Prefer chat template; fallback to plain prompt
    prompt = try_apply_chat_template(tokenizer, SYSTEM_LINE, user_prompt)
    if prompt is None:
        prompt = f"{SYSTEM_LINE}\n\n{user_prompt}"

    raw = generate_compat(model, tokenizer, prompt, MAX_TOKENS)
    valid_ids = parse_deid_ids_from_results(results_text)
    vote = postprocess_vote(raw, valid_ids)
    return raw, vote

def main():
    combos = [
        ("gemma2",    GEMMA2_DIR,    GUIDE_INDEPENDENT),
        ("gemma2",    GEMMA2_DIR,    GUIDE_INFO),
        ("mistral7b", MISTRAL7B_DIR, GUIDE_INDEPENDENT),
        ("mistral7b", MISTRAL7B_DIR, GUIDE_INFO),
    ]
    print("Valid candidate IDs (parsed from results):")
    candidates = parse_deid_ids_from_results(llm_results)
    print(candidates)

    for tag, model_dir, guide in combos:
        print(f"\n======= MODEL: {tag} | PROMPT: {Path(guide).name} =======")
        try:
            raw, vote = run_vote(model_dir, guide, llm_results)
            print("[RAW OUTPUT]:", repr(raw))
            print("[FINAL VOTE]:", vote)
        except Exception as e:
            print(f"[ERROR] {tag} with {guide}: {e}")

if __name__ == "__main__":
    main()
