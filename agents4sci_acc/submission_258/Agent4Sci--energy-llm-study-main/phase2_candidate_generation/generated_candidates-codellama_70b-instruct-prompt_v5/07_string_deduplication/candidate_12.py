def solve(strings):
    seen = set()
    result = []
    for string in strings:
        if not isinstance(string, str):
            raise TypeError("Expected all elements to be of type 'str'.")
        if string not in seen:
            seen.add(string)
            result.append(string)
    return result