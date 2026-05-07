def solve(data):
    # Check if all elements are tuples of (int, str)
    for item in data:
        if not (isinstance(item[0], int) and isinstance(item[1], str)):
            raise TypeError("Input must be a list of tuples of (int, str)")
    # Use key=lambda to sort based on the first element (key) of each tuple in ascending order
    return sorted(data, key=lambda x: x[0])