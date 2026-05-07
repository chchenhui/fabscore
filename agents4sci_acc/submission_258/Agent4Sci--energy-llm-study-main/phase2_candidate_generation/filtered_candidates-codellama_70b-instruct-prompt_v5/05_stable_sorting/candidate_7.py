def solve(input_list):
    if not all(isinstance(i, tuple) and len(i) == 2 for i in input_list):
        raise TypeError("Expected list of tuples")
    return sorted(input_list, key=lambda x: x[0])