import json
def solve(json_string):
    try:
        parsed = json.loads(json_string)
        active_list = []
        for item in parsed:
            if 'active' in item and item['active'] == True:
                active_list.append(item)
        return active_list
    except ValueError as e:
        raise ValueError("The string is not valid JSON") from e