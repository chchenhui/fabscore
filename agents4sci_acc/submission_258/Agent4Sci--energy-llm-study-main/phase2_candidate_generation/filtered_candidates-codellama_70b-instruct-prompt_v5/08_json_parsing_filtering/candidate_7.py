import json

def solve(json_string):
    try:
        data = json.loads(json_string)
        
        active_objects = []
        for obj in data:
            if "active" in obj and isinstance(obj["active"], bool) and obj["active"] is True:
                active_objects.append(obj)
                
        return active_objects
    except ValueError as e:
        raise ValueError("Invalid JSON string") from None