import json
def solve(json_string):
    try:
        objects = json.loads(json_string)
    except ValueError as e:
        raise ValueError("String is not valid JSON") from e
    
    active_objects = [obj for obj in objects if obj.get('active')]
    
    return active_objects