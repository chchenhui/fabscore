import json

def solve(json_string):
    try:
        # Parse the JSON string into a list of dictionaries
        data = json.loads(json_string)

        # Filter out objects that contain "active": True and return them as a list
        active_objects = [obj for obj in data if obj.get("active")]
        return active_objects
    except ValueError:
        raise ValueError("Invalid JSON string provided")