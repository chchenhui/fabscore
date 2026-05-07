import json
def solve(json_string):
    try:
        data = json.loads(json_string)
    except Exception as e:
        raise ValueError("Invalid JSON String")
    return [obj for obj in data if "active" in obj and obj["active"] is True]