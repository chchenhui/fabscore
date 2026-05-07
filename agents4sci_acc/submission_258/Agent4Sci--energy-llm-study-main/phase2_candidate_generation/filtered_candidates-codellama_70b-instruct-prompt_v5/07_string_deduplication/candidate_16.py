def solve(strings):
    if len(strings) == 0:
        return []

    for string in strings:
        if not isinstance(string, str):
            raise TypeError("All elements must be of type 'str'.")

    unique = set()
    unique_add = unique.add
    return [x for x in strings if not (x in unique or unique_add(x))]