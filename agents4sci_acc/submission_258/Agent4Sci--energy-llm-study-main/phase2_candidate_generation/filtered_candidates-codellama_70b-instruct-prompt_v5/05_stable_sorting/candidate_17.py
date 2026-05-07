def solve(elements):
    for elem in elements:
        if not (isinstance(elem, tuple) and len(elem) == 2 and isinstance(elem[0], int) and isinstance(elem[1], str)):
            raise TypeError("Invalid input. Expected a list of tuples of (int, str).")
    return sorted(elements, key=lambda x: x[0])