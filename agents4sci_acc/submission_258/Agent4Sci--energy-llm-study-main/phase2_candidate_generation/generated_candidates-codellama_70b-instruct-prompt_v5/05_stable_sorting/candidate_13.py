def solve(data):
    # Validate input format
    if any(not isinstance(x, tuple) or len(x) != 2 or not isinstance(x[0], int) for x in data):
        raise TypeError("Input elements must be tuples of (int, str).")

    return sorted(data, key=lambda tup: tup[0])