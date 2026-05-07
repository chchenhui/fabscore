import json

def solve(json_str):
    try:
        # parse the JSON string into a list of objects (dictionaries)
        data = json.loads(json_str)
        
        # filter out only objects with "active" key set to True and return them as a new list
        return [obj for obj in data if obj["active"] == True]
    except ValueError:
        raise ValueError("The provided string is not valid JSON.")