import json
def solve(json_string):
    try:
        data = json.loads(json_string)
    except ValueError as e:
        raise ValueError("Invalid JSON string") from e
    
    results = []
    for obj in data:
        if "active" in obj and obj["active"] is True:
            results.append(obj)

    return results