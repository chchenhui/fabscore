def solve(values):
    for v in values:
        # Check if each element is a tuple of (int, str)
        if type(v) != tuple or len(v) != 2 or type(v[0]) != int or type(v[1]) != str:
            raise TypeError("Input must be a list of tuples of (int, str)")
    # Sort the input values by key in ascending order using sorted function with key as first element.
    return sorted(values, key=lambda x:x[0])