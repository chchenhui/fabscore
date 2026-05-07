def solve(strings):
    if not all(isinstance(x, str) for x in strings):
        raise TypeError("Input list must contain only strings")
    unique = set()
    return [string for string in strings if not (string in unique or unique.add(string))]