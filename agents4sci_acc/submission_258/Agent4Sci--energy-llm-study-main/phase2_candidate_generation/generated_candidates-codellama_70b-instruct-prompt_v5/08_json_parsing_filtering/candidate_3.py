import json

def solve(json_string):
    try:
        # Parse the JSON string into a list of dictionaries.
        data = json.loads(json_string)
        
        # Return only objects with "active" key set to True.
        return [obj for obj in data if obj.get("active")]
    
    except ValueError as e:
        raise ValueError(f"Invalid JSON string: {e}")