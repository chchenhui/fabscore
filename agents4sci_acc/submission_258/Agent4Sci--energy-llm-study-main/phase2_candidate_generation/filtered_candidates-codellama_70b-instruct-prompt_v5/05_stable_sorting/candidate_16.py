def solve(input_list):
    if any([type(x) != tuple or len(x) != 2 for x in input_list]):
        raise TypeError("Invalid input data")

    return sorted(input_list, key=lambda tup: tup[0])