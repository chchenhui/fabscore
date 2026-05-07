import json

def solve(json_string):
    try:
        data = json.loads(json_string)
    except ValueError:
        raise ValueError("Invalid JSON")
    
    result = []
    for obj in data:
        if "active" in obj and obj["active"] is True:
            result.append(obj)
    
    return result