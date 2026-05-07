def solve(json_string):
    try:
        data = json.loads(json_string)
    except ValueError as e:
        raise ValueError("The string is not valid JSON") from e
    return [obj for obj in data if obj.get('active', False)]