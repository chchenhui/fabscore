import json
def solve(json_string):
    try:
        data = json.loads(json_string) # parse JSON string
        return [obj for obj in data if obj["active"]] # filter objects with "active" key == True
    except ValueError:
        raise ValueError("Invalid JSON") # raise error if JSON is invalid