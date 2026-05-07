import json
def solve(json_string):
    try:
        # Parse the JSON string into a list of dictionaries
        data = json.loads(json_string)

        # Filter out objects with "active" key set to True
        active_objects = [obj for obj in data if obj["active"]]

        return active_objects
    except ValueError:
        raise ValueError("Invalid JSON string provided")