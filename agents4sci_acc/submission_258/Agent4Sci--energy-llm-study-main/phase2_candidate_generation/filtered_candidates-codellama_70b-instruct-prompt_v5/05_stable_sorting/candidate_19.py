def solve(list_of_tuples):
    if not all(isinstance(i, tuple) and isinstance(i[0], int) and isinstance(i[1], str) for i in list_of_tuples):
        raise TypeError("Input elements must be tuples of (int, str)")
    return sorted(list_of_tuples, key=lambda x: x[0])