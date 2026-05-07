import os
import re
import numpy as np
import json
from pypdf import PdfReader

try:
    import pymupdf  # PyMuPDF new-style import
except Exception:  # pragma: no cover
    import fitz as pymupdf  # fallback
import pymupdf4llm

# Last review run details for tooling (not thread-safe; CLI convenience)
LAST_REVIEW_DETAILS = None


# Optionally load local review config (gitignored). If present, set env vars used by llm_client.
def _load_local_review_config():
    paths = [
        os.path.join(os.path.dirname(__file__), "local_azure_config.py"),
        os.path.join(os.path.dirname(__file__), "local_review_config.py"),
        os.path.join(os.getcwd(), "local_azure_config.py"),
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
                # Map known attributes to environment variables consumed by llm_client
                for k in [
                    "AZURE_OPENAI_API_KEY",
                    "AZURE_OPENAI_ENDPOINT",
                    "AZURE_OPENAI_API_VERSION",
                    "AZURE_OPENAI_MODEL",
                ]:
                    v = getattr(mod, k, None)
                    if v:
                        os.environ[k] = v
                # Optionally expose secondary as fallbacks (not used by default client)
                for k in [
                    "AZURE_OPENAI_API_KEY_2",
                    "AZURE_OPENAI_ENDPOINT_2",
                    "AZURE_OPENAI_MODEL_2",
                ]:
                    v = getattr(mod, k, None)
                    if v:
                        os.environ[k] = v
                # Also expose review-specific settings if provided
                for k in [
                    "REVIEW_AZURE_OPENAI_API_KEY",
                    "REVIEW_AZURE_OPENAI_ENDPOINT",
                    "REVIEW_AZURE_OPENAI_API_VERSION",
                    "REVIEW_AZURE_OPENAI_MODEL",
                ]:
                    v = getattr(mod, k, None)
                    if v:
                        os.environ[k] = v
                break
            except Exception:
                pass


# _load_local_review_config()  # Commented out to avoid setting environment variables

try:
    from .llm_client import chat, extract_json_block
except ImportError:  # fallback when running as script
    from llm_client import chat, extract_json_block

reviewer_system_prompt_base = (
    "You are an expert reviewer for top-tier AI/ML/NLP/CV conferences (NeurIPS, ICML, ICLR, CVPR, ACL). "
    "When given the text of a research paper, produce a thorough, constructive, and professional review. "
    "Be critical and cautious in your decision."
)

# Keep legacy variables as neutral aliases for compatibility (no bias)
reviewer_system_prompt_neg = reviewer_system_prompt_base
reviewer_system_prompt_pos = reviewer_system_prompt_base

template_instructions = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.
Treat this as the note-taking phase of your review.

In <JSON>, provide the review in JSON format with the following fields in the order:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions to be answered by the paper authors.
- "Limitations": A set of limitations and potential negative societal impacts of the work.
- "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall": A rating from 1 to 10 (very strong reject to award quality).
- "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
- "Decision": A decision that has to be one of the following: Accept, Reject.

For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
This JSON will be automatically parsed, so ensure the format is precise.
"""

neurips_form = (
    """
You are an expert reviewer for top-tier AI/ML/NLP/CV conferences (NeurIPS, ICML, ICLR, CVPR, ACL). When given the text of a research paper, you will produce a thorough, constructive, and professional review.

1) Start with a concise, specific summary of the paper’s goals, methods, and findings. State clearly what the paper claims to contribute.

2) Evaluate strengths and weaknesses across these dimensions:
   • Originality/Novelty — Are the tasks or methods new, or a novel combination of known techniques? How does this differ from prior work?
   • Technical Quality/Soundness — Are claims supported by theory/experiments? Is methodology appropriate and correct?
   • Clarity and Presentation — Is the paper well written, organized, and sufficiently detailed for reproduction?
   • Significance/Impact — Does it advance the state of the art or open new directions? Is it likely to be used or built upon?
   • Reproducibility — Are experiments/proofs complete, with code/data or details enabling replication?
   • Ethics & Limitations — Note ethical risks and whether limitations/negative impacts are responsibly discussed.

3) Ask targeted questions and offer constructive suggestions (missing comparisons, ablations, references; unclear sections; stronger baselines; broader evaluation; responsible release). Keep a professional, neutral tone; avoid second-person.

4) Scoring. For each dimension above AND for Overall Recommendation, assign a 1–10 score (higher is better). Use the rubric below for the Overall Recommendation; keep sub-scores consistent in polarity (higher = better).

   Overall Recommendation (1–10):
   1 — Very Strong Reject: Fatal flaws (e.g., clear technical invalidity, plagiarism, severe ethics violation), or no meaningful scientific contribution.
   2 — Strong Reject: Major soundness issues or unsupported claims; key methodology errors; irreproducible; missing core comparisons; unclear enough to preclude verification.
   3 — Reject: Technically weak or marginal; evidence insufficient; novelty/impact limited; important baselines/ablations missing; substantial work needed.
   4 — Weak Reject: Mostly sound but limited novelty/significance or narrow scope; evaluation incomplete or underspecified; would require notable revision to be competitive.
   5 — Borderline Reject: Close call; some merits, but weaknesses slightly outweigh strengths (e.g., limited evaluation or impact).
   6 — Borderline Accept: Close call; strengths slightly outweigh weaknesses; acceptable if concerns are addressed.
   7 — Weak Accept: Solid and competently executed; some limitations, but contribution is worthwhile.
   8 — Accept: Strong paper with clear contribution, solid evaluation, and good clarity; likely interest to the community.
   9 — Strong Accept (Spotlight): High-impact, technically strong, and very well executed; broad interest.
   10 — Accept with Award (Oral/Outstanding): Technically flawless or groundbreaking; exceptional rigor, impact, and presentation.

   • For each score (both sub-scores and overall), provide a brief rationale. Low scores must be tied to specific, evidenced issues (e.g., experimental flaws, missing baselines).
   • Include a 1–5 Confidence score indicating your certainty and any factors affecting it (expertise fit, time, complexity, etc.).

Important: The final output MUST follow the exact THOUGHT + REVIEW JSON format and the exact JSON keys and scales specified below. Do not add or rename fields; incorporate Reproducibility and Ethics within the existing JSON fields and rationales.
"""
    + template_instructions
)


def perform_review(
    text,
    model=None,
    client=None,
    num_reflections=1,
    num_fs_examples=3,
    num_reviews_ensemble=1,
    temperature=None,
    msg_history=None,
    return_msg_history=False,
    reviewer_system_prompt=reviewer_system_prompt_base,
    review_instruction_form=neurips_form,
):
    # Get temperature from config if not provided
    if temperature is None:
        try:
            from .local_review_config import REVIEW_TEMPERATURE

            temperature = REVIEW_TEMPERATURE
        except ImportError:
            import local_review_config as lrc

            temperature = getattr(lrc, "REVIEW_TEMPERATURE", None)
        if temperature is None:
            raise RuntimeError(
                "REVIEW_TEMPERATURE must be set in local_review_config.py"
            )

    # Helper: parse multi-model list from local_review_config
    def _get_review_models():
        try:
            # Try to import from local_review_config
            try:
                from .local_review_config import REVIEW_AZURE_OPENAI_MODEL
            except ImportError:
                import local_review_config as lrc

                REVIEW_AZURE_OPENAI_MODEL = getattr(
                    lrc, "REVIEW_AZURE_OPENAI_MODEL", None
                )

            if REVIEW_AZURE_OPENAI_MODEL is None:
                raise RuntimeError(
                    "REVIEW_AZURE_OPENAI_MODEL must be set in local_review_config.py"
                )

            raw = REVIEW_AZURE_OPENAI_MODEL
            models = [m.strip() for m in re.split(r"[,\s]+", raw) if m.strip()]
            return models
        except Exception:
            raise RuntimeError(
                "Failed to load REVIEW_AZURE_OPENAI_MODEL from local_review_config.py"
            )

    # Helper: aggregate multiple parsed reviews by averaging numeric fields and voting Decision
    def _aggregate_reviews(parsed_reviews):
        if not parsed_reviews:
            return None

        # Filter out None values and non-dict values
        valid_reviews = [
            r for r in parsed_reviews if r is not None and isinstance(r, dict)
        ]
        if not valid_reviews:
            return None

        agg = dict(valid_reviews[0])
        # Average numeric fields
        score_specs = [
            ("Originality", (1, 4)),
            ("Quality", (1, 4)),
            ("Clarity", (1, 4)),
            ("Significance", (1, 4)),
            ("Soundness", (1, 4)),
            ("Presentation", (1, 4)),
            ("Contribution", (1, 4)),
            ("Overall", (1, 10)),
            ("Confidence", (1, 5)),
        ]
        for score, limits in score_specs:
            vals = []
            for r in parsed_reviews:
                v = r.get(score)
                if isinstance(v, (int, float)) and limits[0] <= v <= limits[1]:
                    vals.append(v)
            if vals:
                m = float(np.mean(vals))
                agg[score] = round(m, 2)
        # Decision by majority; tie-breaker by Overall >= 6
        decs = [str(r.get("Decision", "")).strip().lower() for r in parsed_reviews]
        acc = sum(1 for d in decs if d == "accept")
        rej = sum(1 for d in decs if d == "reject")
        if acc > rej:
            agg["Decision"] = "Accept"
        elif rej > acc:
            agg["Decision"] = "Reject"
        else:
            overall = agg.get("Overall")
            if isinstance(overall, (int, float)) and overall >= 6:
                agg["Decision"] = "Accept"
            elif isinstance(overall, (int, float)):
                agg["Decision"] = "Reject"
        return agg

    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples)
        base_prompt = review_instruction_form + fs_prompt
    else:
        base_prompt = review_instruction_form

    base_prompt += f"""
Here is the paper you are asked to review:
```
{text}
```"""

    global LAST_REVIEW_DETAILS
    review_models = _get_review_models()

    # Initialize common variables
    review = None
    msg_history = msg_history or []
    rk = rept = rav = None

    # If multiple review models are configured, treat them as an ensemble source
    if len(review_models) > 1:
        # Build messages for our chat client
        messages = [
            {"role": "system", "content": reviewer_system_prompt},
            *(msg_history or []),
            {"role": "user", "content": base_prompt},
        ]
        parsed_reviews = []
        msg_histories = []
        per_model = []

        # If specific review models provided, run each once; otherwise run n on default model
        if review_models:
            # Get API credentials from local_review_config
            try:
                from .local_review_config import (
                    REVIEW_AZURE_OPENAI_API_KEY,
                    REVIEW_AZURE_OPENAI_ENDPOINT,
                    REVIEW_AZURE_OPENAI_API_VERSION,
                )

                rk = REVIEW_AZURE_OPENAI_API_KEY
                rept = REVIEW_AZURE_OPENAI_ENDPOINT
                rav = REVIEW_AZURE_OPENAI_API_VERSION
            except ImportError:
                import local_review_config as lrc

                rk = getattr(lrc, "REVIEW_AZURE_OPENAI_API_KEY", None)
                rept = getattr(lrc, "REVIEW_AZURE_OPENAI_ENDPOINT", None)
                rav = getattr(lrc, "REVIEW_AZURE_OPENAI_API_VERSION", None)

            # Validate that all required config is present
            if not rk or not rept or not rav:
                raise RuntimeError(
                    "All REVIEW_AZURE_OPENAI configuration must be set in local_review_config.py"
                )

            # Run model calls concurrently; fallback to sequential if futures unavailable
            try:
                import concurrent.futures as _fut
            except Exception:
                _fut = None  # type: ignore

            def _call_model_with_retry(mname: str, max_retries: int = 3):
                """Call a model once without retry - if it fails, return None"""
                try:
                    txt = chat(
                        messages,
                        temperature=temperature,
                        api_key=rk,
                        endpoint=rept,
                        api_version=rav,
                        model=mname,
                    )
                    if txt:
                        # Try to parse the response
                        try:
                            parsed = extract_json_block(txt)
                            if parsed is not None and isinstance(parsed, dict):
                                return mname, txt, parsed
                            else:
                                print(f"Warning: Model {mname} returned invalid JSON")
                                return mname, None, None
                        except Exception as parse_error:
                            print(
                                f"Warning: Failed to parse response from {mname}: {parse_error}"
                            )
                            return mname, None, None
                    else:
                        print(f"Warning: Model {mname} returned empty response")
                        return mname, None, None
                except Exception as e:
                    print(f"Warning: Model {mname} call failed: {e}")
                    return mname, None, None

            def _call_model(mname: str):
                """Legacy wrapper for backward compatibility"""
                mname, txt, parsed = _call_model_with_retry(mname)
                return mname, txt

            if _fut:
                with _fut.ThreadPoolExecutor(
                    max_workers=max(1, len(review_models))
                ) as ex:
                    futures = {
                        ex.submit(_call_model_with_retry, m): m for m in review_models
                    }
                    for fut in _fut.as_completed(futures):
                        mname = futures[fut]
                        try:
                            _m, txt, parsed = fut.result()
                        except Exception:
                            txt, parsed = None, None
                        if not txt:
                            per_model.append({"model": mname, "review": None})
                            continue
                        msg_histories.append(
                            messages + [{"role": "assistant", "content": txt}]
                        )
                        # parsed is already validated in _call_model_with_retry
                        if parsed is not None:
                            parsed_reviews.append(parsed)
                        per_model.append({"model": mname, "review": parsed})
            else:
                for mname in review_models:
                    _m, txt, parsed = _call_model_with_retry(mname)
                    if not txt:
                        per_model.append({"model": mname, "review": None})
                        continue
                    msg_histories.append(
                        messages + [{"role": "assistant", "content": txt}]
                    )
                    # parsed is already validated in _call_model_with_retry
                    if parsed is not None:
                        parsed_reviews.append(parsed)
                    per_model.append({"model": mname, "review": parsed})

        # Aggregate multiple model reviews
        review = _aggregate_reviews(parsed_reviews)
        if review is None and parsed_reviews:
            review = parsed_reviews[0]

    # Handle single model case (either single model configured or num_reviews_ensemble > 1)
    else:
        # Reset variables for single model case
        parsed_reviews = []
        msg_histories = []
        per_model = []
        messages = [
            {"role": "system", "content": reviewer_system_prompt},
            *(msg_history or []),
            {"role": "user", "content": base_prompt},
        ]

        # Get API credentials from local_review_config
        try:
            from .local_review_config import (
                REVIEW_AZURE_OPENAI_API_KEY,
                REVIEW_AZURE_OPENAI_ENDPOINT,
                REVIEW_AZURE_OPENAI_API_VERSION,
            )

            rk = REVIEW_AZURE_OPENAI_API_KEY
            rept = REVIEW_AZURE_OPENAI_ENDPOINT
            rav = REVIEW_AZURE_OPENAI_API_VERSION
        except ImportError:
            import local_review_config as lrc

            rk = getattr(lrc, "REVIEW_AZURE_OPENAI_API_KEY", None)
            rept = getattr(lrc, "REVIEW_AZURE_OPENAI_ENDPOINT", None)
            rav = getattr(lrc, "REVIEW_AZURE_OPENAI_API_VERSION", None)

        # Validate that all required config is present
        if not rk or not rept or not rav:
            raise RuntimeError(
                "All REVIEW_AZURE_OPENAI configuration must be set in local_review_config.py"
            )

        def _multi_review_with_retry(max_retries: int = 3):
            """Get multiple reviews with retry logic"""
            for attempt in range(max_retries):
                try:
                    llm_reviews = chat(
                        messages,
                        n=num_reviews_ensemble,
                        temperature=temperature,
                        api_key=rk,
                        endpoint=rept,
                        api_version=rav,
                        model=review_models[0] if review_models else None,
                    )
                    if llm_reviews and len(llm_reviews) > 0:
                        # Try to parse all reviews
                        parsed_list = []
                        valid_reviews = []

                        for idx, rev in enumerate(llm_reviews):
                            try:
                                pr = extract_json_block(rev)
                                if pr is not None and isinstance(pr, dict):
                                    parsed_list.append(pr)
                                    valid_reviews.append(rev)
                                else:
                                    parsed_list.append(None)
                            except Exception:
                                parsed_list.append(None)

                        # If we have at least one valid review, return it
                        if any(pr is not None for pr in parsed_list):
                            return valid_reviews, parsed_list
                        else:
                            print(
                                f"Warning: No valid reviews parsed on attempt {attempt + 1}, retrying..."
                            )
                    else:
                        print(
                            f"Warning: No reviews returned on attempt {attempt + 1}, retrying..."
                        )
                except Exception as e:
                    print(
                        f"Warning: Multi-review call failed on attempt {attempt + 1}: {e}, retrying..."
                    )

                # Wait before retry
                if attempt < max_retries - 1:
                    import time

                    time.sleep(2**attempt)

            # All retries failed
            print(f"Error: Multi-review failed after {max_retries} attempts")
            return [], []

        llm_reviews, parsed_list = _multi_review_with_retry()

        if llm_reviews:
            msg_histories = [
                messages + [{"role": "assistant", "content": r}] for r in llm_reviews
            ]
            for idx, pr in enumerate(parsed_list):
                if pr is not None:
                    parsed_reviews.append(pr)
                # Use default model name if no specific models configured
                if review_models:
                    model_name = review_models[0]
                else:
                    # Get default model from config
                    try:
                        from .local_review_config import REVIEW_AZURE_OPENAI_MODEL

                        default_models = [
                            m.strip()
                            for m in REVIEW_AZURE_OPENAI_MODEL.split(",")
                            if m.strip()
                        ]
                        model_name = default_models[0] if default_models else None
                    except (ImportError, AttributeError):
                        try:
                            import local_review_config as lrc

                            REVIEW_AZURE_OPENAI_MODEL = getattr(
                                lrc, "REVIEW_AZURE_OPENAI_MODEL", None
                            )
                            if REVIEW_AZURE_OPENAI_MODEL:
                                default_models = [
                                    m.strip()
                                    for m in REVIEW_AZURE_OPENAI_MODEL.split(",")
                                    if m.strip()
                                ]
                                model_name = (
                                    default_models[0] if default_models else None
                                )
                            else:
                                model_name = None
                        except Exception:
                            model_name = None
                    if not model_name:
                        raise RuntimeError(
                            "REVIEW_AZURE_OPENAI_MODEL must be set in local_review_config.py"
                        )
                per_model.append({"model": model_name, "review": pr})
        else:
            msg_histories = []

        # If nothing parsed, try a single-call fallback with retry
        fb_text = None
        if not parsed_reviews:
            review_model = review_models[0] if review_models else None

            def _fallback_review_with_retry(max_retries: int = 3):
                """Fallback single review with retry logic"""
                for attempt in range(max_retries):
                    try:
                        if review_model and rk and rept:
                            fb_text = chat(
                                messages,
                                temperature=temperature,
                                api_key=rk,
                                endpoint=rept,
                                api_version=rav,
                                model=review_model,
                            )
                        else:
                            # Try to pass API credentials if available
                            if rk and rept:
                                fb_text = chat(
                                    messages,
                                    temperature=temperature,
                                    api_key=rk,
                                    endpoint=rept,
                                    api_version=rav,
                                )
                            else:
                                fb_text = chat(messages, temperature=temperature)

                        if fb_text:
                            try:
                                review = extract_json_block(fb_text)
                                if review is not None and isinstance(review, dict):
                                    return fb_text, review
                                else:
                                    print(
                                        f"Warning: Fallback review returned invalid JSON on attempt {attempt + 1}, retrying..."
                                    )
                            except Exception as parse_error:
                                print(
                                    f"Warning: Failed to parse fallback response on attempt {attempt + 1}: {parse_error}, retrying..."
                                )
                        else:
                            print(
                                f"Warning: Fallback returned empty response on attempt {attempt + 1}, retrying..."
                            )
                    except Exception as e:
                        print(
                            f"Warning: Fallback call failed on attempt {attempt + 1}: {e}, retrying..."
                        )

                    # Wait before retry
                    if attempt < max_retries - 1:
                        import time

                        time.sleep(2**attempt)

                # All retries failed
                print(f"Error: Fallback review failed after {max_retries} attempts")
                return None, None

            fb_text, review = _fallback_review_with_retry()

            if fb_text:
                msg_histories = [messages + [{"role": "assistant", "content": fb_text}]]
            # record fallback single model
            per_model.append(
                {
                    "model": review_model if (review_model and rk and rept) else None,
                    "review": review,
                }
            )
        else:
            # Aggregate without invoking an extra meta-review LLM
            review = _aggregate_reviews(parsed_reviews)
            if review is None and parsed_reviews:
                review = parsed_reviews[0]

        # Rewrite the message history with the valid one and new aggregated review.
        if msg_histories:
            msg_history = msg_histories[0][:-1]
            msg_history += [
                {
                    "role": "assistant",
                    "content": f"""
THOUGHT:
I will start by {('aggregating the opinions of ' + str(max(1, len(parsed_reviews))) + ' reviewers (from multiple model endpoints)') if parsed_reviews else 'using a single best-effort review after a fallback call'} that I previously obtained.

REVIEW JSON:
```json
{json.dumps(review)}
```
""",
                }
            ]
        else:
            # Fallback message history when no successful API calls
            msg_history = [
                {"role": "system", "content": reviewer_system_prompt},
                *(msg_history or []),
                {"role": "user", "content": base_prompt},
                {
                    "role": "assistant",
                    "content": f"""
THOUGHT:
I will provide a single best-effort review after all API calls failed.

REVIEW JSON:
```json
{json.dumps(review)}
```
""",
                },
            ]

    # Optional reflection rounds to refine the review
    if num_reflections > 1:
        for j in range(num_reflections - 1):
            messages = [
                {"role": "system", "content": reviewer_system_prompt},
                *(msg_history or []),
                {"role": "user", "content": reviewer_reflection_prompt},
            ]
            # Try to pass API credentials if available
            if rk and rept:
                text = chat(
                    messages,
                    temperature=temperature,
                    api_key=rk,
                    endpoint=rept,
                    api_version=rav,
                )
            else:
                text = chat(messages, temperature=temperature)
            msg_history = messages + [{"role": "assistant", "content": text}]
            parsed = extract_json_block(text)
            if parsed is not None:
                review = parsed
            if isinstance(text, str) and ("I am done" in text):
                break

    # Store details for external tools (ensure this happens for all branches)
    LAST_REVIEW_DETAILS = {
        "models": review_models,
        "individual_reviews": per_model,
        "aggregated": review,
    }

    if return_msg_history:
        return review, msg_history
    else:
        return review


def get_last_review_details():
    """Get the details from the last review call"""
    global LAST_REVIEW_DETAILS
    return LAST_REVIEW_DETAILS


reviewer_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the accuracy and soundness of the review you just created.
Include any other factors that you think are important in evaluating the paper.
Ensure the review is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your review.
Stick to the spirit of the original review unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


def load_paper(pdf_path, num_pages=None, min_size=100):
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)  # open a document
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:  # iterate the document pages
                text = text + page.get_text()  # get plain text encoded as UTF-8
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short")

    return text


def load_review(path):
    with open(path, "r") as json_file:
        loaded = json.load(json_file)
    return loaded["review"]


# get directory of this file
dir_path = os.path.dirname(os.path.realpath(__file__))

fewshot_papers = [
    os.path.join(dir_path, "fewshot_examples/132_automated_relational.pdf"),
    os.path.join(dir_path, "fewshot_examples/attention.pdf"),
    os.path.join(dir_path, "fewshot_examples/2_carpe_diem.pdf"),
]

fewshot_reviews = [
    os.path.join(dir_path, "fewshot_examples/132_automated_relational.json"),
    os.path.join(dir_path, "fewshot_examples/attention.json"),
    os.path.join(dir_path, "fewshot_examples/2_carpe_diem.json"),
]


def get_review_fewshot_examples(num_fs_examples=1):
    fewshot_prompt = """
Below are some sample reviews, copied from previous machine learning conferences.
Note that while each review is formatted differently according to each reviewer's style, the reviews are well-structured and therefore easy to navigate.
"""
    for paper, review in zip(
        fewshot_papers[:num_fs_examples], fewshot_reviews[:num_fs_examples]
    ):
        txt_path = paper.replace(".pdf", ".txt")
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                paper_text = f.read()
        else:
            paper_text = load_paper(paper)
        review_text = load_review(review)
        fewshot_prompt += f"""
Paper:

```
{paper_text}
```

Review:

```
{review_text}
```
"""

    return fewshot_prompt


meta_reviewer_system_prompt = """You are an Area Chair at a machine learning conference.
You are in charge of meta-reviewing a paper that was reviewed by {reviewer_count} reviewers.
Your job is to aggregate the reviews into a single meta-review in the same format.
Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers."""


def get_meta_review(model, client, temperature, reviews):
    # Get API credentials from local_review_config
    try:
        from .local_review_config import (
            REVIEW_AZURE_OPENAI_API_KEY,
            REVIEW_AZURE_OPENAI_ENDPOINT,
            REVIEW_AZURE_OPENAI_API_VERSION,
        )

        rk = REVIEW_AZURE_OPENAI_API_KEY
        rept = REVIEW_AZURE_OPENAI_ENDPOINT
        rav = REVIEW_AZURE_OPENAI_API_VERSION
    except ImportError:
        import local_review_config as lrc

        rk = getattr(lrc, "REVIEW_AZURE_OPENAI_API_KEY", None)
        rept = getattr(lrc, "REVIEW_AZURE_OPENAI_ENDPOINT", None)
        rav = getattr(lrc, "REVIEW_AZURE_OPENAI_API_VERSION", None)

    # Validate that all required config is present
    if not rk or not rept or not rav:
        raise RuntimeError(
            "All REVIEW_AZURE_OPENAI configuration must be set in local_review_config.py"
        )

    # Write a meta-review from a set of individual reviews
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
Review {i + 1}/{len(reviews)}:
```
{json.dumps(r)}
```
"""
    base_prompt = neurips_form + review_text

    messages = [
        {
            "role": "system",
            "content": meta_reviewer_system_prompt.format(reviewer_count=len(reviews)),
        },
        {"role": "user", "content": base_prompt},
    ]
    # Try to pass API credentials if available
    if rk and rept:
        llm_review = chat(
            messages,
            temperature=temperature,
            api_key=rk,
            endpoint=rept,
            api_version=rav,
        )
    else:
        llm_review = chat(messages, temperature=temperature)
    meta_review = extract_json_block(llm_review)
    return meta_review


def perform_improvement(review, coder):
    improvement_prompt = '''The following review has been created for your research paper:
"""
{review}
"""

Improve the text using the review.'''.format(
        review=json.dumps(review)
    )
    coder_out = coder.run(improvement_prompt)
