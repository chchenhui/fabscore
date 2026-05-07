import json
def solve(json_string):
    try:
        data = json.loads(json_string)
        return [obj for obj in data if "active" in obj and obj["active"]]
    except ValueError as e:
        raise ValueError("Invalid JSON string") from e