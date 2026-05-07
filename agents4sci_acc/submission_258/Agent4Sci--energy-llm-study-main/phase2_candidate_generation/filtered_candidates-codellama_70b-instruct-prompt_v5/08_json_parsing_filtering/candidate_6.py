import json
def solve(input):
    try:
        data = json.loads(input)
    except ValueError as e:
        raise ValueError("Invalid JSON string") from e
    return [obj for obj in data if obj["active"]]