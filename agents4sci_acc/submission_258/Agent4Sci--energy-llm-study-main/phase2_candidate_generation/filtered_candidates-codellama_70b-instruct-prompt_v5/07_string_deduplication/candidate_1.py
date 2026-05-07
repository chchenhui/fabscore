def solve(strings):
    if not all(isinstance(x, str) for x in strings):
        raise TypeError("All elements must be string")

    unique_set = set()
    result = []
    for s in strings:
        if s not in unique_set:
            unique_set.add(s)
            result.append(s)

    return result