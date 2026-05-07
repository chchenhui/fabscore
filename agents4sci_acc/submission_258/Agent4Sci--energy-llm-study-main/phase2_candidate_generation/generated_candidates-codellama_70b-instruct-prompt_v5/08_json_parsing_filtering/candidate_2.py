import json

def solve(json_str):
    try:
        data = json.loads(json_str)
        return [obj for obj in data if obj['active']]
    except Exception:
        raise ValueError("Invalid JSON")