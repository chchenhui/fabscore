import json

def solve(json_string):
    try:
        json_data = json.loads(json_string)
        
        active_objects = [obj for obj in json_data if obj.get("active", False)]
        return active_objects
    
    except ValueError:
        raise ValueError("Invalid JSON string")