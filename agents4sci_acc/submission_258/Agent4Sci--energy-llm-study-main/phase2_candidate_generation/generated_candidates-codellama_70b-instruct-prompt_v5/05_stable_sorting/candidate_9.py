def solve(data):
    # Check if all elements are tuples of (int, str)
    # Raise a TypeError if not.
    for element in data:
        if type(element[0]) != int or type(element[1]) != str:
            raise TypeError("Invalid input. All elements must be tuples of (int, str).")

    return sorted(data)  # Stable sort by key (first item in tuple) by default.