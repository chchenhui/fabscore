# save_humaneval_jsonl.py
from datasets import load_dataset
ds = load_dataset("openai/openai_humaneval", split="test")  
ds.to_json("HumanEval.jsonl", orient="records", lines=True)
print(f"saved {len(ds)} tasks to HumanEval.jsonl")
