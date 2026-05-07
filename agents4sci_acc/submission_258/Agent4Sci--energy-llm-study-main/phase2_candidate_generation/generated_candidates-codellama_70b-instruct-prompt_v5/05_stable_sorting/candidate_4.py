def solve(tuples):
    # Check if all elements are valid tuples of (int, str)
    for tup in tuples:
        if not isinstance(tup[0], int) or type(tup[1]) != str:
            raise TypeError("Input must be a list of tuples with (int, str)")

    # Sort the tuples by key (ascending order) while preserving original order for equal keys
    sorted_tuples = sorted(tuples, key=lambda x: x[0])

    return sorted_tuples