import json
def solve(json_string):
    try:
        data = json.loads(json_string)
    except ValueError as e:
        raise ValueError("Invalid JSON string provided.")
    
    active_objects = [obj for obj in data if "active" in obj and obj["active"]]
    return active_objects