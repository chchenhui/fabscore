def solve(tuples):
    # Check if all elements are tuples of (int, str)
    for tup in tuples:
        if not isinstance(tup, tuple) or len(tup) != 2 or \
           not isinstance(tup[0], int) or not isinstance(tup[1], str):
            raise TypeError("Input elements must be tuples of (int, str)")

    # Sort the list by key in ascending order, maintaining stability
    return sorted(tuples, key=lambda x: x[0])